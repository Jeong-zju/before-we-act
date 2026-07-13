"""Runnable staged training for decentralized FE-PC-WAM.

Every stage reads :class:`DecentralizedTransitionDataset`.  Simulator-only
values are consumed solely as supervised targets; model forward calls are
assembled from the dataset's deployable ego stream and previously received
message metadata.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Mapping
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.decentralized_dataset import DecentralizedTransitionDataset
from data.schema import SCHEMA_VERSION
from models.decentralized import (
    EgoLocalWAM,
    EgoLocalWAMConfig,
    LocalIntentionConfig,
    LocalIntentionPosterior,
)
from models.plan_tokenizer import (
    ActionOnlyPlanTokenizer,
    ActionOnlyPlanTokenizerConfig,
    PlanCodeSupport,
    PlanCodeSupportAccumulator,
    compute_action_only_plan_losses,
)
from models.slot_encoder import (
    LocalBeliefSlotEncoder,
    LocalBeliefSlotEncoderConfig,
    compute_local_belief_auxiliary_losses,
)
from train.batch import encode_current_and_future_beliefs
from train.checkpoint import (
    CONTRACT_TAG,
    file_sha256,
    load_checkpoint,
    make_checkpoint,
    require_plan_code_support,
    save_checkpoint,
    upstream_reference,
)
from train.losses import compute_ego_wam_losses, compute_local_intention_losses


@dataclass(frozen=True)
class TrainingConfig:
    stage: str
    data_dir: str
    output: str
    plan_checkpoint: str | None = None
    belief_checkpoint: str | None = None
    wam_checkpoint: str | None = None
    intention_checkpoint: str | None = None

    history: int = 8
    horizon: int = 16
    stride: int = 1
    max_episodes: int = -1
    batch_size: int = 64
    epochs: int = 20
    max_steps: int = -1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    seed: int = 7
    device: str = "auto"

    codebook_size: int = 64
    plan_latent_dim: int = 64
    plan_hidden_dim: int = 256
    min_code_count: int = 1
    residual_std_floor: float = 1e-3
    min_active_codes: int = 4
    min_usage_ratio: float = 0.10
    strict_codebook_health: bool = False
    usage_balance_weight: float = 0.1
    residual_weight: float = 0.05
    residual_dropout: float = 0.2

    slot_dim: int = 128
    belief_hidden_dim: int = 256
    belief_num_heads: int = 4
    belief_history_layers: int = 2
    belief_slot_layers: int = 2
    dropout: float = 0.1

    dynamics_model_dim: int = 512
    dynamics_layers: int = 8
    dynamics_heads: int = 8
    dynamics_ffn_dim: int = 2048
    intention_model_dim: int = 512
    intention_layers: int = 6
    intention_heads: int = 8
    intention_ffn_dim: int = 2048
    message_metadata_dim: int = 4

    robust_oracle_probability: float = 0.45
    robust_inferred_probability: float = 0.35
    robust_corrupt_probability: float = 0.10
    robust_residual_noise_std: float = 0.25

    def __post_init__(self) -> None:
        if self.stage not in {"plan", "belief", "wam", "intention", "wam_robust"}:
            raise ValueError(f"unknown stage {self.stage!r}")
        for name in ("history", "horizon", "stride", "batch_size", "epochs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.min_code_count <= 0 or self.residual_std_floor <= 0:
            raise ValueError("plan support thresholds must be positive")
        if self.min_active_codes <= 0:
            raise ValueError("min_active_codes must be positive")
        if not 0.0 < self.min_usage_ratio <= 1.0:
            raise ValueError("min_usage_ratio must be in (0, 1]")
        probabilities = (
            self.robust_oracle_probability,
            self.robust_inferred_probability,
            self.robust_corrupt_probability,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("robust conditioning probabilities must be in [0, 1]")
        if sum(probabilities) > 1.0 + 1e-8:
            raise ValueError("robust conditioning probabilities cannot sum above one")
        if self.robust_residual_noise_std < 0:
            raise ValueError("robust_residual_noise_std cannot be negative")


class ProgressReporter:
    """In-place progress bars and sparse stage-boundary messages.

    Progress is deliberately separate from :class:`TrainingConfig`: changing
    verbosity must never alter checkpoint lineage or resume compatibility.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_every: int = 50,
        prefix: str = "[training]",
        stream=None,
        position: int = 0,
    ) -> None:
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        self.enabled = bool(enabled)
        self.log_every = int(log_every)
        self.prefix = str(prefix)
        self.stream = sys.stderr if stream is None else stream
        self.position = int(position)

    def child(self, prefix: str) -> "ProgressReporter":
        return ProgressReporter(
            enabled=self.enabled,
            log_every=self.log_every,
            prefix=prefix,
            stream=self.stream,
            position=self.position,
        )

    def emit(self, message: str) -> None:
        if self.enabled:
            print(f"{self.prefix} {message}", file=self.stream, flush=True)

    def metrics_due(self, completed: int, total: int) -> bool:
        return (
            completed == 1
            or completed % self.log_every == 0
            or (total > 0 and completed >= total)
        )

    def progress(
        self,
        *,
        description: str,
        total: int,
        unit: str,
    ):
        return tqdm(
            total=total,
            desc=f"{self.prefix} {description}",
            unit=unit,
            file=self.stream,
            disable=not self.enabled,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
            miniters=1,
            position=self.position,
        )


class _OptimizationProgress:
    """Track one stage's optimization speed, losses, and ETA."""

    def __init__(
        self,
        reporter: ProgressReporter,
        config: TrainingConfig,
        dataset: DecentralizedTransitionDataset,
    ) -> None:
        self.reporter = reporter
        self.config = config
        self.steps_per_epoch = math.ceil(len(dataset) / config.batch_size)
        uncapped_steps = config.epochs * self.steps_per_epoch
        self.total_steps = (
            min(uncapped_steps, config.max_steps)
            if config.max_steps > 0
            else uncapped_steps
        )
        self.started_at = time.monotonic()
        self.processed_samples = 0
        self.epoch_loss = 0.0
        self.epoch_steps = 0
        self.bar = reporter.progress(
            description="training",
            total=self.total_steps,
            unit="step",
        )

    def epoch_start(self, epoch: int) -> None:
        self.epoch_loss = 0.0
        self.epoch_steps = 0
        self.bar.set_postfix_str(
            f"epoch={epoch}/{self.config.epochs}", refresh=False
        )

    def step(
        self,
        *,
        epoch: int,
        loss: torch.Tensor,
        sample_count: int,
    ) -> None:
        loss_value = float(loss.detach().cpu().item())
        self.epoch_loss += loss_value
        self.epoch_steps += 1
        self.processed_samples += int(sample_count)
        completed = self.bar.n + 1
        if self.reporter.metrics_due(completed, self.total_steps):
            elapsed = max(time.monotonic() - self.started_at, 1e-9)
            self.bar.set_postfix_str(
                f"epoch={epoch}/{self.config.epochs} loss={loss_value:.6f} "
                f"samples/s={self.processed_samples / elapsed:.2f}",
                refresh=False,
            )
        self.bar.update(1)

    def epoch_complete(self, epoch: int) -> None:
        if self.epoch_steps == 0:
            self.bar.set_postfix_str(
                f"epoch={epoch}/{self.config.epochs} optimization_steps=0",
                refresh=False,
            )
            return
        self.bar.set_postfix_str(
            f"epoch={epoch}/{self.config.epochs} "
            f"mean_loss={self.epoch_loss / self.epoch_steps:.6f}",
            refresh=False,
        )

    def close(self) -> None:
        self.bar.close()


def smoke_config(config: TrainingConfig) -> TrainingConfig:
    """Return a fast architecture/run while preserving data and path choices."""

    return replace(
        config,
        batch_size=min(config.batch_size, 4),
        epochs=1,
        max_steps=1,
        codebook_size=8,
        plan_latent_dim=8,
        plan_hidden_dim=32,
        slot_dim=16,
        belief_hidden_dim=32,
        belief_num_heads=4,
        belief_history_layers=1,
        belief_slot_layers=1,
        dropout=0.0,
        dynamics_model_dim=32,
        dynamics_layers=1,
        dynamics_heads=4,
        dynamics_ffn_dim=64,
        intention_model_dim=32,
        intention_layers=1,
        intention_heads=4,
        intention_ffn_dim=64,
    )


def train_stage(
    config: TrainingConfig,
    *,
    dataset: DecentralizedTransitionDataset | None = None,
    progress: ProgressReporter | None = None,
) -> Path:
    """Train one stage and return its checkpoint path."""

    progress = progress or ProgressReporter(enabled=False)
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    if dataset is None:
        indexing_started = time.monotonic()
        progress.emit(f"indexing dataset={Path(config.data_dir).resolve()}")
        dataset = DecentralizedTransitionDataset(
            config.data_dir,
            history=config.history,
            horizon=config.horizon,
            stride=config.stride,
            max_episodes=config.max_episodes,
        )
        progress.emit(
            f"indexing complete episodes={len(dataset.paths)} samples={len(dataset)} "
            f"elapsed={format_duration(time.monotonic() - indexing_started)}"
        )
    _validate_supplied_dataset(dataset, config)
    progress.emit(f"device={device.type} samples={len(dataset)}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    handlers: dict[
        str,
        Callable[
            [
                TrainingConfig,
                DecentralizedTransitionDataset,
                torch.device,
                ProgressReporter,
            ],
            Path,
        ],
    ] = {
        "plan": _train_plan,
        "belief": _train_belief,
        "wam": _train_wam,
        "intention": _train_intention,
        "wam_robust": _train_wam_robust,
    }
    destination = handlers[config.stage](config, dataset, device, progress)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        progress.emit(f"peak_cuda_memory={peak_gib:.2f}GiB")
    return destination


def _train_plan(
    config: TrainingConfig,
    dataset: DecentralizedTransitionDataset,
    device: torch.device,
    progress: ProgressReporter,
) -> Path:
    model_config = ActionOnlyPlanTokenizerConfig(
        horizon=config.horizon,
        action_dim=int(dataset.action_dim),
        latent_dim=config.plan_latent_dim,
        hidden_dim=config.plan_hidden_dim,
        codebook_size=config.codebook_size,
        usage_balance_weight=config.usage_balance_weight,
        residual_weight=config.residual_weight,
        residual_dropout=config.residual_dropout,
    )
    model = ActionOnlyPlanTokenizer(model_config).to(device)
    statistics_loader = _loader(dataset, config, shuffle=False)
    action_mean, action_std, observation_count = _action_normalization(
        statistics_loader, progress=progress
    )
    normalization = {
        "action_mean": action_mean,
        "action_std": action_std,
        "action_count": observation_count,
    }
    action_mean_device = action_mean.to(device)
    action_std_device = action_std.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    running: dict[str, float] = {}
    steps = 0
    optimization = _OptimizationProgress(progress, config, dataset)
    model.train()
    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        optimization.epoch_start(epoch)
        for raw_batch in _loader(dataset, config, shuffle=True):
            batch = _to_device(raw_batch, device)
            losses = compute_action_only_plan_losses(
                model,
                {
                    "actions": batch["ego_future_action"],
                    "maneuver": batch["target_maneuver"],
                },
                action_mean=action_mean_device,
                action_std=action_std_device,
            )
            _optimizer_step(optimizer, losses["loss"], model.parameters())
            _accumulate_scalars(running, losses)
            steps += 1
            optimization.step(
                epoch=epoch,
                loss=losses["loss"],
                sample_count=int(batch["ego_id"].shape[0]),
            )
            if _step_limit_reached(config, steps):
                break
        optimization.epoch_complete(epoch)
        if _step_limit_reached(config, steps):
            break
    optimization.close()

    # Re-encode every training window after optimization.  These empirical
    # counts/statistics are the sole candidate-plan support used downstream.
    accumulator = PlanCodeSupportAccumulator(
        model_config.codebook_size, model_config.latent_dim
    )
    model.eval()
    codebook_scan_batches = len(statistics_loader)
    codebook_bar = progress.progress(
        description="codebook scan",
        total=codebook_scan_batches,
        unit="batch",
    )
    try:
        with torch.no_grad():
            for raw_batch in statistics_loader:
                actions = raw_batch["ego_future_action"].to(device)
                encoded = model.encode(
                    _normalize_actions(actions, action_mean_device, action_std_device)
                )
                accumulator.update(encoded["code_indices"], encoded["residual"])
                codebook_bar.update(1)
    finally:
        codebook_bar.close()
    support = accumulator.build(
        min_count=config.min_code_count, std_floor=config.residual_std_floor
    )
    probabilities = support.probabilities
    nonzero = probabilities > 0
    entropy = float(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum().item()
    )
    used_codes = int((support.counts > 0).sum().item())
    active_codes = int(support.active_codes.numel())
    usage_ratio = float(used_codes / support.codebook_size)
    health_failures: list[str] = []
    if active_codes < config.min_active_codes:
        health_failures.append(
            f"active_codes={active_codes} is below min_active_codes={config.min_active_codes}"
        )
    if usage_ratio < config.min_usage_ratio:
        health_failures.append(
            f"usage_ratio={usage_ratio:.6f} is below min_usage_ratio={config.min_usage_ratio:.6f}"
        )
    if health_failures:
        message = " plan codebook health check failed: " + "; ".join(health_failures)
        if config.strict_codebook_health:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    plan_metrics = {
        **_mean_metrics(running, steps),
        "steps": steps,
        "encoded_segments": int(support.counts.sum().item()),
        "used_codes": used_codes,
        "active_codes": active_codes,
        "dead_codes": int(support.codebook_size - used_codes),
        "usage_ratio": usage_ratio,
        "hard_usage_ratio": usage_ratio,
        "entropy": entropy,
        "hard_entropy": entropy,
        "perplexity": float(math.exp(entropy)),
        "hard_perplexity": float(math.exp(entropy)),
        "codebook_health_passed": not health_failures,
        "codebook_health_warnings": health_failures,
        "min_active_codes": config.min_active_codes,
        "min_usage_ratio": config.min_usage_ratio,
    }
    checkpoint = make_checkpoint(
        stage="plan",
        model_class="ActionOnlyPlanTokenizer",
        model_config=model_config,
        model_state_dict=_cpu_state_dict(model),
        training_config=config,
        dataset_metadata=_dataset_metadata(dataset),
        metrics=plan_metrics,
        normalization=normalization,
        plan_code_support=support.to_dict(),
        extra={
            "encoder_input": "ego_future_action_only",
            "hardcoded_plan_codes_allowed": False,
        },
    )
    return save_checkpoint(config.output, checkpoint)


def _train_belief(
    config: TrainingConfig,
    dataset: DecentralizedTransitionDataset,
    device: torch.device,
    progress: ProgressReporter,
) -> Path:
    plan_path, plan_checkpoint, support = _load_plan_upstream(config)
    _validate_plan_dataset_compatibility(plan_checkpoint, dataset, config)
    model_config = LocalBeliefSlotEncoderConfig(
        history=config.history,
        local_dim=dataset.local_history_dim,
        object_dim=3,
        slot_dim=config.slot_dim,
        hidden_dim=config.belief_hidden_dim,
        num_heads=config.belief_num_heads,
        num_history_layers=config.belief_history_layers,
        num_slot_layers=config.belief_slot_layers,
        dropout=config.dropout,
        privileged_aux_dims={
            "self_state": 3 + 3 * int(dataset.spec.joint_dim),
            "object_pose": 3,
            "teammate_pose": 3,
            "task_progress": 1,
            "event_maneuver": 3,
        },
        privileged_aux_roles={
            "self_state": "self",
            "object_pose": "object-belief",
            "teammate_pose": "teammate-belief",
            "task_progress": "task-context",
            "event_maneuver": "task-context",
        },
    )
    model = LocalBeliefSlotEncoder(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    running: dict[str, float] = {}
    steps = 0
    optimization = _OptimizationProgress(progress, config, dataset)
    model.train()
    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        optimization.epoch_start(epoch)
        for raw_batch in _loader(dataset, config, shuffle=True):
            batch = _to_device(raw_batch, device)
            model_batch = _belief_forward_batch(batch)
            losses = compute_local_belief_auxiliary_losses(
                model,
                model_batch,
                _belief_privileged_targets(batch),
            )
            _optimizer_step(optimizer, losses["loss"], model.parameters())
            _accumulate_scalars(running, losses)
            steps += 1
            optimization.step(
                epoch=epoch,
                loss=losses["loss"],
                sample_count=int(batch["ego_id"].shape[0]),
            )
            if _step_limit_reached(config, steps):
                break
        optimization.epoch_complete(epoch)
        if _step_limit_reached(config, steps):
            break
    optimization.close()
    metrics = {**_mean_metrics(running, steps), "steps": steps}
    checkpoint = make_checkpoint(
        stage="belief",
        model_class="LocalBeliefSlotEncoder",
        model_config=model_config,
        model_state_dict=_cpu_state_dict(model),
        training_config=config,
        dataset_metadata=_dataset_metadata(dataset),
        metrics=metrics,
        plan_code_support=support.to_dict(),
        upstream={"plan": upstream_reference(plan_path, plan_checkpoint)},
        extra={
            "slot_role_order": list(LocalBeliefSlotEncoder.ROLE_NAMES),
            "privileged_values_are_forward_inputs": False,
            "auxiliary_target_roles": dict(model_config.privileged_aux_roles),
        },
    )
    return save_checkpoint(config.output, checkpoint)


def _train_wam(
    config: TrainingConfig,
    dataset: DecentralizedTransitionDataset,
    device: torch.device,
    progress: ProgressReporter,
) -> Path:
    plan_path, plan_checkpoint, support = _load_plan_upstream(config)
    belief_path, belief_checkpoint, belief = _load_belief_upstream(config, device)
    tokenizer = _restore_plan_model(plan_checkpoint, device)
    _require_upstream_matches(belief_checkpoint, "plan", plan_path)
    _validate_plan_dataset_compatibility(plan_checkpoint, dataset, config)
    _validate_belief_dataset_compatibility(belief, dataset, config)
    _freeze(tokenizer)
    _freeze(belief)

    model_config = EgoLocalWAMConfig(
        horizon=config.horizon,
        slots_per_agent=LocalBeliefSlotEncoder.NUM_ROLES,
        slot_dim=belief.cfg.slot_dim,
        plan_codebook_size=tokenizer.cfg.codebook_size,
        plan_latent_dim=tokenizer.cfg.latent_dim,
        action_dim_per_agent=int(dataset.action_dim),
        model_dim=config.dynamics_model_dim,
        num_layers=config.dynamics_layers,
        num_heads=config.dynamics_heads,
        ffn_dim=config.dynamics_ffn_dim,
        dropout=config.dropout,
    )
    model = EgoLocalWAM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    normalization = _normalization_to_device(plan_checkpoint["normalization"], device)
    running: dict[str, float] = {}
    steps = 0
    optimization = _OptimizationProgress(progress, config, dataset)
    model.train()
    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        optimization.epoch_start(epoch)
        for raw_batch in _loader(dataset, config, shuffle=True):
            batch = _to_device(raw_batch, device)
            with torch.no_grad():
                beliefs = encode_current_and_future_beliefs(belief, batch)
                ego_plan = _encode_action_plan(
                    tokenizer, batch["ego_future_action"], normalization
                )
                teammate_plan = _encode_action_plan(
                    tokenizer,
                    batch["privileged_teammate_future_action"],
                    normalization,
                )
                branch_plan = _encode_branch_plans(
                    tokenizer, batch["branch_action"], normalization
                )
            losses = compute_ego_wam_losses(
                model,
                ego_slots=beliefs["ego_slots"],
                plan_codes=torch.stack(
                    [ego_plan["code_indices"], teammate_plan["code_indices"]], dim=1
                ),
                plan_residuals=torch.stack(
                    [ego_plan["residual"], teammate_plan["residual"]], dim=1
                ),
                teammate_hypothesis_weight=torch.ones(
                    batch["ego_id"].shape[0], device=device
                ),
                target_ego_slots=beliefs["target_ego_slots"],
                target_ego_actions=batch["ego_future_action"],
                privileged_target_teammate_actions=batch[
                    "privileged_teammate_future_action"
                ],
                target_contact=batch["target_local_contact"],
                target_force=batch["target_local_force"],
                target_progress=batch["target_progress"],
                target_reward=batch["target_reward"],
                target_success=batch["target_success"],
                target_failure_reason=batch["target_failure_reason"],
                target_collision=batch["target_collision"],
                target_force_violation=batch["target_force_violation"],
                branch_plan_codes=branch_plan["code_indices"],
                branch_plan_residuals=branch_plan["residual"],
                branch_returns=batch["branch_return"],
                branch_valid=batch["branch_valid"],
            )
            _optimizer_step(optimizer, losses["loss"], model.parameters())
            _accumulate_scalars(running, losses)
            steps += 1
            optimization.step(
                epoch=epoch,
                loss=losses["loss"],
                sample_count=int(batch["ego_id"].shape[0]),
            )
            if _step_limit_reached(config, steps):
                break
        optimization.epoch_complete(epoch)
        if _step_limit_reached(config, steps):
            break
    optimization.close()
    metrics = {**_mean_metrics(running, steps), "steps": steps}
    checkpoint = make_checkpoint(
        stage="wam",
        model_class="EgoLocalWAM",
        model_config=model_config,
        model_state_dict=_cpu_state_dict(model),
        training_config=config,
        dataset_metadata=_dataset_metadata(dataset),
        metrics=metrics,
        normalization=plan_checkpoint["normalization"],
        plan_code_support=support.to_dict(),
        upstream={
            "plan": upstream_reference(plan_path, plan_checkpoint),
            "belief": upstream_reference(belief_path, belief_checkpoint),
        },
        extra={
            "conditioning": "oracle action-only ego and teammate plan labels",
            "teammate_private_state_input": False,
            "joint_action_target_is_privileged": True,
        },
    )
    return save_checkpoint(config.output, checkpoint)


def _train_intention(
    config: TrainingConfig,
    dataset: DecentralizedTransitionDataset,
    device: torch.device,
    progress: ProgressReporter,
) -> Path:
    plan_path, plan_checkpoint, support = _load_plan_upstream(config)
    belief_path, belief_checkpoint, belief = _load_belief_upstream(config, device)
    wam_path = _required_path(config.wam_checkpoint, "--wam-checkpoint")
    wam_checkpoint = load_checkpoint(wam_path, expected_stage="wam")
    _require_upstream_matches(belief_checkpoint, "plan", plan_path)
    _require_upstream_matches(wam_checkpoint, "plan", plan_path)
    _require_upstream_matches(wam_checkpoint, "belief", belief_path)
    tokenizer = _restore_plan_model(plan_checkpoint, device)
    _validate_plan_dataset_compatibility(plan_checkpoint, dataset, config)
    _validate_belief_dataset_compatibility(belief, dataset, config)
    _freeze(tokenizer)
    _freeze(belief)

    model_config = LocalIntentionConfig(
        slots_per_agent=LocalBeliefSlotEncoder.NUM_ROLES,
        slot_dim=belief.cfg.slot_dim,
        plan_codebook_size=tokenizer.cfg.codebook_size,
        plan_latent_dim=tokenizer.cfg.latent_dim,
        message_metadata_dim=config.message_metadata_dim,
        model_dim=config.intention_model_dim,
        num_layers=config.intention_layers,
        num_heads=config.intention_heads,
        ffn_dim=config.intention_ffn_dim,
        dropout=config.dropout,
    )
    model = LocalIntentionPosterior(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    normalization = _normalization_to_device(plan_checkpoint["normalization"], device)
    active_mask = _active_code_mask(support, device)
    running: dict[str, float] = {}
    steps = 0
    skipped_examples = 0
    optimization = _OptimizationProgress(progress, config, dataset)
    model.train()
    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        optimization.epoch_start(epoch)
        for raw_batch in _loader(dataset, config, shuffle=True):
            batch = _to_device(raw_batch, device)
            with torch.no_grad():
                ego_slots = belief(**_belief_model_kwargs(batch))["slots"]
                ego_plan = _encode_action_plan(
                    tokenizer, batch["ego_future_action"], normalization
                )
                target_plan = _encode_action_plan(
                    tokenizer,
                    batch["privileged_teammate_future_action"],
                    normalization,
                )
            supported_rows = active_mask[target_plan["code_indices"]]
            skipped_examples += int((~supported_rows).sum().item())
            if not supported_rows.any():
                continue
            metadata = torch.zeros(
                int(supported_rows.sum().item()),
                model_config.message_metadata_dim,
                device=device,
            )
            losses = compute_local_intention_losses(
                model,
                ego_slots=ego_slots[supported_rows],
                ego_plan_code=ego_plan["code_indices"][supported_rows],
                ego_plan_residual=ego_plan["residual"][supported_rows],
                agent_id=batch["ego_id"][supported_rows],
                received_message_metadata=metadata,
                target_teammate_code=target_plan["code_indices"][supported_rows],
                target_teammate_residual=target_plan["residual"][supported_rows],
                active_code_mask=active_mask,
            )
            _optimizer_step(optimizer, losses["loss"], model.parameters())
            _accumulate_scalars(running, losses)
            steps += 1
            optimization.step(
                epoch=epoch,
                loss=losses["loss"],
                sample_count=int(supported_rows.sum().item()),
            )
            if _step_limit_reached(config, steps):
                break
        optimization.epoch_complete(epoch)
        if _step_limit_reached(config, steps):
            break
    optimization.close()
    if steps == 0:
        raise RuntimeError(
            "intention stage saw no teammate target in active plan support; "
            "lower --min-code-count or retrain the plan tokenizer"
        )
    metrics = {
        **_mean_metrics(running, steps),
        "steps": steps,
        "skipped_unsupported_examples": skipped_examples,
    }
    checkpoint = make_checkpoint(
        stage="intention",
        model_class="LocalIntentionPosterior",
        model_config=model_config,
        model_state_dict=_cpu_state_dict(model),
        training_config=config,
        dataset_metadata=_dataset_metadata(dataset),
        metrics=metrics,
        normalization=plan_checkpoint["normalization"],
        plan_code_support=support.to_dict(),
        upstream={
            "plan": upstream_reference(plan_path, plan_checkpoint),
            "belief": upstream_reference(belief_path, belief_checkpoint),
            "wam": upstream_reference(wam_path, wam_checkpoint),
        },
        extra={
            "input_boundary": "ego belief + ego plan + agent id + received envelope metadata",
            "teammate_plan_is_target_only": True,
            "active_code_mask": active_mask.cpu(),
        },
    )
    return save_checkpoint(config.output, checkpoint)


def _train_wam_robust(
    config: TrainingConfig,
    dataset: DecentralizedTransitionDataset,
    device: torch.device,
    progress: ProgressReporter,
) -> Path:
    plan_path, plan_checkpoint, support = _load_plan_upstream(config)
    belief_path, belief_checkpoint, belief = _load_belief_upstream(config, device)
    wam_path = _required_path(config.wam_checkpoint, "--wam-checkpoint")
    intention_path = _required_path(config.intention_checkpoint, "--intention-checkpoint")
    wam_checkpoint = load_checkpoint(wam_path, expected_stage="wam")
    intention_checkpoint = load_checkpoint(
        intention_path, expected_stage="intention"
    )
    _require_upstream_matches(belief_checkpoint, "plan", plan_path)
    _require_upstream_matches(wam_checkpoint, "plan", plan_path)
    _require_upstream_matches(wam_checkpoint, "belief", belief_path)
    _require_upstream_matches(intention_checkpoint, "plan", plan_path)
    _require_upstream_matches(intention_checkpoint, "belief", belief_path)
    _require_upstream_matches(intention_checkpoint, "wam", wam_path)
    tokenizer = _restore_plan_model(plan_checkpoint, device)
    intention = LocalIntentionPosterior(
        LocalIntentionConfig(**intention_checkpoint["model_config"])
    ).to(device)
    intention.load_state_dict(intention_checkpoint["model_state_dict"], strict=True)
    model = EgoLocalWAM(EgoLocalWAMConfig(**wam_checkpoint["model_config"])).to(device)
    model.load_state_dict(wam_checkpoint["model_state_dict"], strict=True)
    _validate_plan_dataset_compatibility(plan_checkpoint, dataset, config)
    _validate_belief_dataset_compatibility(belief, dataset, config)
    _freeze(tokenizer)
    _freeze(belief)
    _freeze(intention)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    normalization = _normalization_to_device(plan_checkpoint["normalization"], device)
    active_mask = _active_code_mask(support, device)
    running: dict[str, float] = {}
    mode_counts = {"oracle": 0, "inferred": 0, "missing_prior": 0, "corrupted": 0}
    steps = 0
    optimization = _OptimizationProgress(progress, config, dataset)
    model.train()
    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        optimization.epoch_start(epoch)
        for raw_batch in _loader(dataset, config, shuffle=True):
            batch = _to_device(raw_batch, device)
            with torch.no_grad():
                beliefs = encode_current_and_future_beliefs(belief, batch)
                ego_plan = _encode_action_plan(
                    tokenizer, batch["ego_future_action"], normalization
                )
                true_teammate_plan = _encode_action_plan(
                    tokenizer,
                    batch["privileged_teammate_future_action"],
                    normalization,
                )
                branch_plan = _encode_branch_plans(
                    tokenizer, batch["branch_action"], normalization
                )
                teammate_condition = _robust_teammate_condition(
                    config=config,
                    intention=intention,
                    support=support,
                    active_mask=active_mask,
                    ego_slots=beliefs["ego_slots"],
                    ego_plan=ego_plan,
                    true_teammate_plan=true_teammate_plan,
                    ego_id=batch["ego_id"],
                    device=device,
                )
            for name, value in teammate_condition["mode_counts"].items():
                mode_counts[name] += int(value)
            losses = compute_ego_wam_losses(
                model,
                ego_slots=beliefs["ego_slots"],
                plan_codes=torch.stack(
                    [ego_plan["code_indices"], teammate_condition["code_indices"]],
                    dim=1,
                ),
                plan_residuals=torch.stack(
                    [ego_plan["residual"], teammate_condition["residual"]], dim=1
                ),
                teammate_hypothesis_weight=teammate_condition["weight"],
                target_ego_slots=beliefs["target_ego_slots"],
                target_ego_actions=batch["ego_future_action"],
                privileged_target_teammate_actions=batch[
                    "privileged_teammate_future_action"
                ],
                target_contact=batch["target_local_contact"],
                target_force=batch["target_local_force"],
                target_progress=batch["target_progress"],
                target_reward=batch["target_reward"],
                target_success=batch["target_success"],
                target_failure_reason=batch["target_failure_reason"],
                target_collision=batch["target_collision"],
                target_force_violation=batch["target_force_violation"],
                branch_plan_codes=branch_plan["code_indices"],
                branch_plan_residuals=branch_plan["residual"],
                branch_returns=batch["branch_return"],
                branch_valid=batch["branch_valid"],
            )
            _optimizer_step(optimizer, losses["loss"], model.parameters())
            _accumulate_scalars(running, losses)
            steps += 1
            optimization.step(
                epoch=epoch,
                loss=losses["loss"],
                sample_count=int(batch["ego_id"].shape[0]),
            )
            if _step_limit_reached(config, steps):
                break
        optimization.epoch_complete(epoch)
        if _step_limit_reached(config, steps):
            break
    optimization.close()
    metrics = {
        **_mean_metrics(running, steps),
        "steps": steps,
        **{f"conditioning_{name}": value for name, value in mode_counts.items()},
    }
    checkpoint = make_checkpoint(
        stage="wam_robust",
        model_class="EgoLocalWAM",
        model_config=model.cfg,
        model_state_dict=_cpu_state_dict(model),
        training_config=config,
        dataset_metadata=_dataset_metadata(dataset),
        metrics=metrics,
        normalization=plan_checkpoint["normalization"],
        plan_code_support=support.to_dict(),
        upstream={
            "plan": upstream_reference(plan_path, plan_checkpoint),
            "belief": upstream_reference(belief_path, belief_checkpoint),
            "wam": upstream_reference(wam_path, wam_checkpoint),
            "intention": upstream_reference(intention_path, intention_checkpoint),
        },
        extra={
            "conditioning_modes": ["oracle", "inferred", "missing_prior", "corrupted"],
            "true_teammate_plan_used_as_input_for_non_oracle_rows": False,
            "probabilities": {
                "oracle": config.robust_oracle_probability,
                "inferred": config.robust_inferred_probability,
                "corrupted": config.robust_corrupt_probability,
                "missing_prior": 1.0
                - config.robust_oracle_probability
                - config.robust_inferred_probability
                - config.robust_corrupt_probability,
            },
        },
    )
    return save_checkpoint(config.output, checkpoint)


def _robust_teammate_condition(
    *,
    config: TrainingConfig,
    intention: LocalIntentionPosterior,
    support: PlanCodeSupport,
    active_mask: torch.Tensor,
    ego_slots: torch.Tensor,
    ego_plan: Mapping[str, torch.Tensor],
    true_teammate_plan: Mapping[str, torch.Tensor],
    ego_id: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    batch_size = ego_id.shape[0]
    draws = torch.rand(batch_size, device=device)
    oracle_cut = config.robust_oracle_probability
    inferred_cut = oracle_cut + config.robust_inferred_probability
    corrupt_cut = inferred_cut + config.robust_corrupt_probability
    oracle = draws < oracle_cut
    inferred = (draws >= oracle_cut) & (draws < inferred_cut)
    corrupted = (draws >= inferred_cut) & (draws < corrupt_cut)
    missing = draws >= corrupt_cut

    code = true_teammate_plan["code_indices"].clone()
    residual = true_teammate_plan["residual"].clone()
    # Hypothesis probabilities are external free-energy aggregation weights;
    # they are deliberately not dynamics inputs or supervision signals.
    weight = torch.ones(batch_size, device=device, dtype=ego_slots.dtype)

    metadata = torch.zeros(
        batch_size, intention.cfg.message_metadata_dim, device=device, dtype=ego_slots.dtype
    )
    posterior = intention(
        ego_slots=ego_slots,
        ego_plan_code=ego_plan["code_indices"],
        ego_plan_residual=ego_plan["residual"],
        agent_id=ego_id,
        received_message_metadata=metadata,
    )
    supported_logits = posterior["code_logits"].masked_fill(
        ~active_mask.unsqueeze(0), torch.finfo(posterior["code_logits"].dtype).min
    )
    inferred_code = supported_logits.argmax(dim=-1)
    gather = inferred_code[:, None, None].expand(
        batch_size, 1, intention.cfg.plan_latent_dim
    )
    inferred_residual = posterior["residual_mu_by_code"].gather(1, gather).squeeze(1)
    code[inferred] = inferred_code[inferred]
    residual[inferred] = inferred_residual[inferred]

    prior_sample = support.sample(batch_size, device=device)
    prior_code = prior_sample["code_indices"]
    prior_residual = prior_sample["residual"]
    code[missing] = prior_code[missing]
    residual[missing] = prior_residual[missing]

    # A corrupted packet remains within empirical code support but perturbs the
    # residual.  No private teammate state is introduced to repair it.
    code[corrupted] = prior_code[corrupted]
    residual[corrupted] = prior_residual[corrupted]
    if config.robust_residual_noise_std > 0:
        residual[corrupted] += config.robust_residual_noise_std * torch.randn_like(
            residual[corrupted]
        )
    return {
        "code_indices": code,
        "residual": residual,
        "weight": weight,
        "mode_counts": {
            "oracle": int(oracle.sum().item()),
            "inferred": int(inferred.sum().item()),
            "missing_prior": int(missing.sum().item()),
            "corrupted": int(corrupted.sum().item()),
        },
    }


def _load_plan_upstream(
    config: TrainingConfig,
) -> tuple[Path, dict[str, Any], PlanCodeSupport]:
    path = _required_path(config.plan_checkpoint, "--plan-checkpoint")
    checkpoint = load_checkpoint(path, expected_stage="plan")
    support = PlanCodeSupport.from_dict(require_plan_code_support(checkpoint))
    return path, checkpoint, support


def _load_belief_upstream(
    config: TrainingConfig, device: torch.device
) -> tuple[Path, dict[str, Any], LocalBeliefSlotEncoder]:
    path = _required_path(config.belief_checkpoint, "--belief-checkpoint")
    checkpoint = load_checkpoint(path, expected_stage="belief")
    model = LocalBeliefSlotEncoder(
        LocalBeliefSlotEncoderConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return path, checkpoint, model


def _restore_plan_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> ActionOnlyPlanTokenizer:
    model = ActionOnlyPlanTokenizer(
        ActionOnlyPlanTokenizerConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _validate_plan_dataset_compatibility(
    checkpoint: Mapping[str, Any],
    dataset: DecentralizedTransitionDataset,
    config: TrainingConfig,
) -> None:
    model_config = checkpoint["model_config"]
    if int(model_config["horizon"]) != config.horizon:
        raise ValueError("plan checkpoint horizon does not match this dataset window")
    if int(model_config["action_dim"]) != int(dataset.action_dim):
        raise ValueError("plan checkpoint action_dim does not match the dataset")


def _validate_belief_dataset_compatibility(
    model: LocalBeliefSlotEncoder,
    dataset: DecentralizedTransitionDataset,
    config: TrainingConfig,
) -> None:
    if model.cfg.history != config.history:
        raise ValueError("belief checkpoint history does not match this training run")
    if model.cfg.local_dim != dataset.local_history_dim:
        raise ValueError("belief checkpoint local_dim does not match the  deployable stream")


def _belief_forward_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "local_history": batch["local_history"],
        "history_mask": batch["history_mask"],
        "agent_id": batch["ego_id"],
        "object_observation": batch["object_observation_history"],
        "object_valid": batch["object_valid_history"],
        "object_age": batch["object_age_history"],
        "object_confidence": batch["object_confidence_history"],
    }


def _belief_model_kwargs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_batch = _belief_forward_batch(batch)
    return {
        "local_history": model_batch["local_history"],
        "history_mask": model_batch["history_mask"],
        "agent_id": model_batch["agent_id"],
        "object_observation": model_batch["object_observation"],
        "object_valid": model_batch["object_valid"],
        "object_age": model_batch["object_age"],
        "object_confidence": model_batch["object_confidence"],
    }


def _belief_privileged_targets(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    # Each target is post-transition and reaches only its role-bound loss head.
    # The self probe predicts locally observable proprioception, not an
    # unobservable world pose.  Object/teammate truth is expressed in the ego
    # frame and is used only as an auxiliary simulation label.
    self_state_dim = 3
    local_width = batch["future_model_observation"].shape[-1]
    # The dataset orders base twist, then q/dq/tau, before local force/contact.
    # Private-gates schema: D_model = 17 + 3J. Event fields are task
    # context, not part of the self-state probe.
    if local_width < 17 or (local_width - 17) % 3 != 0:
        raise ValueError("future_model_observation has an invalid  sensor layout")
    self_state_dim += local_width - 17
    return {
        "self_state": batch["future_model_observation"][:, 0, :self_state_dim],
        "object_pose": batch["target_object_pose_ego"][:, 0],
        "teammate_pose": batch["target_teammate_pose_ego"][:, 0],
        "task_progress": batch["target_progress"][:, :1],
        "event_maneuver": F.one_hot(
            batch["target_maneuver"], num_classes=3
        ).to(torch.float32),
    }


def _encode_action_plan(
    tokenizer: ActionOnlyPlanTokenizer,
    actions: torch.Tensor,
    normalization: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return tokenizer.encode(
        _normalize_actions(
            actions, normalization["action_mean"], normalization["action_std"]
        )
    )


def _encode_branch_plans(
    tokenizer: ActionOnlyPlanTokenizer,
    actions: torch.Tensor,
    normalization: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Encode fixed counterfactual plan pairs without exposing their outcomes."""

    if actions.ndim != 5 or actions.shape[2] != 2:
        raise ValueError("branch actions must have shape [B,N,2,H,A]")
    B, N, _, H, A = actions.shape
    encoded = tokenizer.encode(
        _normalize_actions(
            actions.reshape(B * N * 2, H, A),
            normalization["action_mean"],
            normalization["action_std"],
        )
    )
    return {
        "code_indices": encoded["code_indices"].reshape(B, N, 2),
        "residual": encoded["residual"].reshape(B, N, 2, -1),
    }


def _normalize_actions(
    actions: torch.Tensor, action_mean: torch.Tensor, action_std: torch.Tensor
) -> torch.Tensor:
    return (actions - action_mean.view(1, 1, -1)) / action_std.view(
        1, 1, -1
    ).clamp_min(1e-6)


def _action_normalization(
    loader: DataLoader,
    *,
    progress: ProgressReporter,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    total: torch.Tensor | None = None
    total_square: torch.Tensor | None = None
    count = 0
    total_batches = len(loader)
    normalization_bar = progress.progress(
        description="normalization",
        total=total_batches,
        unit="batch",
    )
    try:
        for batch in loader:
            actions = batch["ego_future_action"].to(dtype=torch.float64)
            batch_total = actions.sum(dim=(0, 1))
            batch_square = actions.square().sum(dim=(0, 1))
            total = batch_total if total is None else total + batch_total
            total_square = (
                batch_square if total_square is None else total_square + batch_square
            )
            count += int(actions.shape[0] * actions.shape[1])
            normalization_bar.update(1)
    finally:
        normalization_bar.close()
    if total is None or total_square is None or count == 0:
        raise RuntimeError("cannot compute action normalization from an empty dataset")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(1e-8)
    return mean.float(), variance.sqrt().float(), count


def _normalization_to_device(
    normalization: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    try:
        mean = torch.as_tensor(normalization["action_mean"], device=device)
        std = torch.as_tensor(normalization["action_std"], device=device)
    except KeyError as exc:
        raise ValueError("plan checkpoint is missing  action normalization") from exc
    if mean.ndim != 1 or std.shape != mean.shape or (std <= 0).any():
        raise ValueError("plan checkpoint action normalization is invalid")
    return {"action_mean": mean, "action_std": std}


def _active_code_mask(support: PlanCodeSupport, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(support.codebook_size, dtype=torch.bool, device=device)
    mask[support.active_codes.to(device)] = True
    return mask


def _dataset_metadata(dataset: DecentralizedTransitionDataset) -> dict[str, Any]:
    assert dataset.spec is not None and dataset.action_dim is not None
    dataset_digest = hashlib.sha256()
    for path in dataset.paths:
        dataset_digest.update(path.name.encode("utf-8"))
        dataset_digest.update(file_sha256(path).encode("ascii"))
    manifest_path = dataset.data_dir.parent / "dataset_manifest.json"
    manifest_digest = file_sha256(manifest_path) if manifest_path.is_file() else None
    return {
        "schema_version": SCHEMA_VERSION,
        "data_dir": str(dataset.data_dir.resolve()),
        "episode_count": len(dataset.paths),
        "sample_count": len(dataset),
        "dataset_sha256": dataset_digest.hexdigest(),
        "dataset_manifest_sha256": manifest_digest,
        "fresh_training_required": True,
        "history": dataset.history,
        "horizon": dataset.horizon,
        "stride": dataset.stride,
        "ego_ids": list(dataset.ego_ids),
        "model_observation_dim": dataset.model_observation_dim,
        "local_history_dim": dataset.local_history_dim,
        "action_dim": int(dataset.action_dim),
        "local_observation_spec": {
            "joint_dim": int(dataset.spec.joint_dim),
            "force_dim": int(dataset.spec.force_dim),
            "base_twist_dim": int(dataset.spec.base_twist_dim),
        },
        "input_feature_names": dataset.input_feature_names,
        "deployable_input_keys": sorted(dataset.INPUT_KEYS),
        "local_contact_semantics": dataset.local_contact_semantics,
        "local_force_semantics": dataset.local_force_semantics,
        "local_force_scale_newtons": dataset.local_force_scale_newtons,
        "local_force_units": dataset.local_force_units,
        "local_sensor_provenance": dataset.local_sensor_provenance,
    }


def _loader(
    dataset: DecentralizedTransitionDataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator if shuffle else None,
    )


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    parameters,
) -> None:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite or non-scalar training loss: {loss}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=10.0)
    optimizer.step()


def _accumulate_scalars(
    running: dict[str, float], values: Mapping[str, Any]
) -> None:
    for name, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            running[name] = running.get(name, 0.0) + float(value.detach().cpu().item())


def _mean_metrics(running: Mapping[str, float], steps: int) -> dict[str, float]:
    if steps <= 0:
        raise RuntimeError("training stage completed without an optimization step")
    return {name: value / steps for name, value in running.items()}


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _validate_supplied_dataset(
    dataset: DecentralizedTransitionDataset,
    config: TrainingConfig,
) -> None:
    expected_dir = Path(config.data_dir).resolve()
    if dataset.data_dir.resolve() != expected_dir:
        raise ValueError(
            f"supplied dataset path {dataset.data_dir.resolve()} does not match {expected_dir}"
        )
    for name in ("history", "horizon", "stride"):
        if int(getattr(dataset, name)) != int(getattr(config, name)):
            raise ValueError(
                f"supplied dataset {name}={getattr(dataset, name)} does not match "
                f"training config {name}={getattr(config, name)}"
            )
    expected_episode_count = (
        min(len(dataset.paths), config.max_episodes)
        if config.max_episodes > 0
        else len(dataset.paths)
    )
    if len(dataset.paths) != expected_episode_count:
        raise ValueError("supplied dataset does not match max_episodes")


def _step_limit_reached(config: TrainingConfig, steps: int) -> bool:
    return config.max_steps > 0 and steps >= config.max_steps


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {name: _to_device(item, device) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _required_path(value: str | None, flag: str) -> Path:
    if not value:
        raise ValueError(f"{flag} is required for this stage")
    return Path(value)


def _require_upstream_matches(
    checkpoint: Mapping[str, Any], upstream_name: str, path: str | Path
) -> None:
    reference = checkpoint.get("upstream", {}).get(upstream_name)
    if not isinstance(reference, Mapping):
        raise ValueError(
            f"{checkpoint.get('stage', 'checkpoint')} is missing upstream lineage "
            f"for {upstream_name!r}"
        )
    actual_hash = file_sha256(path)
    if reference.get("sha256") != actual_hash:
        raise ValueError(
            f"{checkpoint.get('stage', 'checkpoint')} was trained against a different "
            f"{upstream_name} checkpoint; expected sha256={reference.get('sha256')}, "
            f"got {actual_hash}"
        )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one decentralized FE-PC-WAM stage"
    )
    parser.add_argument("stage", choices=["plan", "belief", "wam", "intention", "wam_robust"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-checkpoint")
    parser.add_argument("--belief-checkpoint")
    parser.add_argument("--wam-checkpoint")
    parser.add_argument("--intention-checkpoint")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--plan-latent-dim", type=int, default=64)
    parser.add_argument("--plan-hidden-dim", type=int, default=256)
    parser.add_argument("--min-code-count", type=int, default=1)
    parser.add_argument("--residual-std-floor", type=float, default=1e-3)
    parser.add_argument("--min-active-codes", type=int, default=4)
    parser.add_argument("--min-usage-ratio", type=float, default=0.10)
    parser.add_argument("--strict-codebook-health", action="store_true")
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--belief-hidden-dim", type=int, default=256)
    parser.add_argument("--belief-num-heads", type=int, default=4)
    parser.add_argument("--dynamics-model-dim", type=int, default=512)
    parser.add_argument("--dynamics-layers", type=int, default=8)
    parser.add_argument("--dynamics-heads", type=int, default=8)
    parser.add_argument("--dynamics-ffn-dim", type=int, default=2048)
    parser.add_argument("--intention-model-dim", type=int, default=512)
    parser.add_argument("--intention-layers", type=int, default=6)
    parser.add_argument("--intention-heads", type=int, default=8)
    parser.add_argument("--intention-ffn-dim", type=int, default=2048)
    parser.add_argument("--message-metadata-dim", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="refresh loss/throughput postfix after N optimization steps",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="disable progress logs; final JSON remains on stdout",
    )
    return parser


def config_from_arguments(args: argparse.Namespace) -> TrainingConfig:
    names = set(TrainingConfig.__dataclass_fields__)
    values = {name.replace("-", "_"): value for name, value in vars(args).items() if name in names}
    config = TrainingConfig(**values)
    return smoke_config(config) if args.smoke else config


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    config = config_from_arguments(args)
    progress = ProgressReporter(
        enabled=not args.quiet,
        log_every=args.log_every,
        prefix=f"[stage 1/1:{config.stage}]",
    )
    stage_started = time.monotonic()
    progress.emit(
        f"start epochs={config.epochs} batch_size={config.batch_size} "
        f"checkpoint={Path(config.output).resolve()}"
    )
    try:
        destination = train_stage(config, progress=progress)
    except Exception as exc:
        progress.emit(
            f"failed elapsed={format_duration(time.monotonic() - stage_started)} "
            f"error={type(exc).__name__}: {exc}"
        )
        raise
    progress.emit(
        f"completed checkpoint={destination.resolve()} "
        f"elapsed={format_duration(time.monotonic() - stage_started)}"
    )
    checkpoint = load_checkpoint(destination, expected_stage=config.stage)
    summary = {
        "stage": config.stage,
        "output": str(destination.resolve()),
        "contract_tag": CONTRACT_TAG,
        "schema_version": SCHEMA_VERSION,
        "metrics": checkpoint["metrics"],
    }
    print(json.dumps(summary, indent=2, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
