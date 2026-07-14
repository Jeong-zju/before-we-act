"""Validation-aware staged trainer for FE-PC-WAM Research-v2."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from data.research_v2 import ResearchV2Dataset
from models.plan_tokenizer import (
    PlanCodeSupport,
    PlanCodeSupportAccumulator,
    compute_action_only_plan_losses,
)
from models.research_v2 import (
    BeliefEncoderV2,
    BeliefEncoderV2Config,
    BeliefRoleTargetHeads,
    BlockTransitionWorldModelV2,
    DirectParallelWorldModelV2,
    IntentionPosteriorV2,
    PlanDistributionV2Config,
    PlanProposalV2,
    PlanTokenizerV2,
    PlanTokenizerV2Config,
    WorldModelV2Config,
)
from train.batch import encode_current_and_future_beliefs
from train.research_v2_checkpoint import (
    checkpoint_reference,
    load_research_v2_checkpoint,
    make_research_v2_checkpoint,
    save_research_v2_checkpoint,
    sha256_file,
)
from train.research_v2_losses import (
    belief_role_loss,
    plan_distribution_loss_v2,
    world_model_loss_v2,
)


@dataclass(frozen=True)
class ResearchV2TrainingConfig:
    stage: str
    train_dir: str
    val_dir: str
    output_dir: str
    plan_checkpoint: str | None = None
    belief_checkpoint: str | None = None
    world_block_checkpoint: str | None = None
    world_ensemble_checkpoints: tuple[str, ...] = ()
    intention_checkpoint: str | None = None
    history: int = 8
    horizon: int = 16
    stride: int = 1
    batch_size: int = 64
    epochs: int = 20
    max_steps_per_epoch: int = -1
    max_validation_steps: int = 200
    patience: int = 10
    relative_min_delta: float = 0.001
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    num_workers: int = 4
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int = 2
    gradient_accumulation_steps: int = 1
    precision: str = "auto"
    tf32: bool = True
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    fused_optimizer: bool = True
    statistics_max_steps: int = 256
    support_max_steps: int = 256
    communication_price: float | None = None
    seed: int = 7
    device: str = "auto"
    smoke: bool = False
    resume: bool = False
    force_retrain: bool = False

    def __post_init__(self) -> None:
        allowed = {
            "plan",
            "belief",
            "world_direct",
            "world_block",
            "proposal",
            "intention",
            "calibration",
        }
        if self.stage not in allowed:
            raise ValueError(f"unknown Research-v2 stage {self.stage}")
        if min(
            self.history,
            self.horizon,
            self.stride,
            self.batch_size,
            self.epochs,
            self.patience,
            self.gradient_accumulation_steps,
        ) <= 0:
            raise ValueError("training sizes/epochs/patience must be positive")
        if self.num_workers < 0 or self.prefetch_factor <= 0:
            raise ValueError("num_workers must be non-negative and prefetch_factor positive")
        if self.max_validation_steps == 0:
            raise ValueError("max_validation_steps must be -1 or positive")
        if self.statistics_max_steps <= 0 or self.support_max_steps <= 0:
            raise ValueError("statistics/support step limits must be positive")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16, or fp16")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("unsupported torch.compile mode")
        if self.communication_price is not None and (
            not math.isfinite(self.communication_price) or self.communication_price < 0
        ):
            raise ValueError("communication_price must be finite and non-negative")
        if self.resume and self.force_retrain:
            raise ValueError("resume and force_retrain are mutually exclusive")


class _EarlyStop:
    def __init__(self, patience: int, relative_min_delta: float):
        self.patience = patience
        self.relative_min_delta = relative_min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        threshold = self.best * (1.0 - self.relative_min_delta)
        improved = not math.isfinite(self.best) or value < threshold
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def stopped(self) -> bool:
        return self.bad_epochs >= self.patience


def train_research_v2_stage(
    config: ResearchV2TrainingConfig,
    *,
    train_data: ResearchV2Dataset | None = None,
    val_data: ResearchV2Dataset | None = None,
) -> Path:
    _seed(config.seed)
    device = _device(config.device)
    _configure_device(device, config)
    if train_data is None:
        train_data = ResearchV2Dataset(
            config.train_dir,
            history=config.history,
            horizon=config.horizon,
            stride=config.stride,
        )
    if val_data is None:
        val_data = ResearchV2Dataset(
            config.val_dir,
            history=config.history,
            horizon=config.horizon,
            stride=config.stride,
        )
    _require_dataset_compatibility(train_data, val_data)
    manifest = Path(config.train_dir).resolve().parent / "dataset_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Research-v2 dataset root lacks dataset_manifest.json")
    handlers: dict[str, Callable[..., Path]] = {
        "plan": _train_plan,
        "belief": _train_belief,
        "world_direct": lambda *args: _train_world(*args, block=False),
        "world_block": lambda *args: _train_world(*args, block=True),
        "proposal": _train_proposal,
        "intention": _train_intention,
        "calibration": _write_calibration,
    }
    return handlers[config.stage](config, train_data, val_data, device, sha256_file(manifest))


def _train_plan(config, train_data, val_data, device, manifest_hash) -> Path:
    model_cfg = _plan_config(config)
    model = PlanTokenizerV2(model_cfg).to(device)
    mean, std = _action_statistics(train_data, config)
    mean, std = mean.to(device), std.to(device)
    parameters = list(model.parameters())
    optimizer = _optimizer(parameters, config, device)

    def step(run_model, batch, training: bool):
        actions = batch["ego_future_action"]
        maneuver = batch["target_maneuver"]
        maneuver_mask = torch.ones_like(maneuver, dtype=torch.bool)
        branch_rows = batch["branch_group_id"] >= 0
        if branch_rows.any():
            branch_actions = batch["branch_matched_action"][branch_rows, :, 0]
            branch_actions = branch_actions.reshape(
                -1, model_cfg.horizon, model_cfg.action_dim
            )
            actions = torch.cat((actions, branch_actions), dim=0)
            maneuver = torch.cat(
                (
                    maneuver,
                    torch.zeros(
                        branch_actions.shape[0], dtype=torch.long, device=actions.device
                    ),
                ),
                dim=0,
            )
            maneuver_mask = torch.cat(
                (
                    maneuver_mask,
                    torch.zeros(
                        branch_actions.shape[0], dtype=torch.bool, device=actions.device
                    ),
                ),
                dim=0,
            )
        losses = compute_action_only_plan_losses(
            run_model,
            {
                "actions": actions,
                "maneuver": maneuver,
                "maneuver_mask": maneuver_mask,
            },
            action_mean=mean,
            action_std=std,
            include_rate_distortion_metrics=not training,
        )
        losses["branch_action_fraction"] = (
            (~maneuver_mask).float().mean().detach()
        )
        return losses

    def checkpoint(epoch, metrics):
        return _checkpoint(
            config,
            model,
            model_cfg,
            manifest_hash,
            metrics,
            PlanTokenizerV2.__name__,
            ("actions",),
            extra={
                "normalization": {"action_mean": mean.cpu(), "action_std": std.cpu()},
                "local_observation_spec": asdict(train_data.spec),
                "epoch": epoch,
            },
        )

    best = _fit(
        config,
        model,
        train_data,
        val_data,
        device,
        step,
        checkpoint,
        optimizer=optimizer,
        parameters=parameters,
    )
    state = load_research_v2_checkpoint(best, expected_stage="plan", map_location=device)
    if "plan_support" in state.get("extra", {}):
        return best
    model.load_state_dict(state["model_state_dict"])
    support = _estimate_plan_support(model, train_data, mean, std, config, device)
    if not config.smoke and int(support.active_codes.numel()) < 8:
        raise RuntimeError(
            "Research-v2 proposal requires at least 8 active plan codes; "
            "retrain tokenizer or improve mixed-policy data"
        )
    state["extra"]["plan_support"] = support.to_dict()
    state["metrics"]["support_hard_usage"] = _hard_code_metrics(support.counts)
    validation = state["metrics"].get("validation", {})
    full = float(validation.get("loss_action", math.nan))
    code_only = float(validation.get("loss_code_only_action", math.nan))
    baseline = float(validation.get("loss_mean_action_baseline", math.nan))
    state["metrics"]["rate_distortion"] = {
        "normalized_mse_full": full,
        "normalized_mse_code_only": code_only,
        "normalized_mse_mean_action_baseline": baseline,
        "full_relative_improvement_over_mean": _relative_improvement(full, baseline),
        "code_only_relative_improvement_over_mean": _relative_improvement(
            code_only, baseline
        ),
    }
    save_research_v2_checkpoint(best, state)
    return best


def _train_belief(config, train_data, val_data, device, manifest_hash) -> Path:
    plan_path, plan_state, _ = _load_plan(config, device)
    cfg = _belief_config(config, train_data.local_history_dim)
    model = BeliefEncoderV2(cfg).to(device)
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    heads = BeliefRoleTargetHeads(
        cfg.model_dim,
        {
            "self_state": (0, 3),
            "object_pose": (1, 3),
            "teammate_pose": (2, 3),
            "task_progress": (3, 1),
            "maneuver": (3, 3),
        },
    ).to(device)
    parameters = list(model.parameters()) + list(heads.parameters())
    optimizer = _optimizer(parameters, config, device)

    def step(run_model, batch, training):
        # Select checkpoints using the exact EMA representation deployed by
        # every downstream stage and by the runtime, not the transient online
        # encoder that only drives optimizer updates.
        encoder = run_model if training else ema
        belief = encoder(**_belief_kwargs(batch))["belief"]
        targets = {
            "self_state": batch["target_current_self_state"],
            "object_pose": batch["target_current_object_pose"],
            "teammate_pose": batch["target_current_teammate_pose"],
            "task_progress": batch["target_current_task_progress"],
            "maneuver": F.one_hot(batch["target_current_maneuver"], 3).float(),
        }
        losses = belief_role_loss(heads, belief, targets)
        return losses

    def after_optimizer_step():
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.mul_(0.995).add_(source, alpha=0.005)

    def resume_auxiliary(state):
        extra = state.get("extra", {})
        if "training_probe_state" in extra:
            heads.load_state_dict(extra["training_probe_state"])
        if "ema_model_state_dict" in extra:
            ema.load_state_dict(extra["ema_model_state_dict"])

    def checkpoint(epoch, metrics):
        return _checkpoint(
            config,
            model,
            cfg,
            manifest_hash,
            metrics,
            BeliefEncoderV2.__name__,
            BeliefEncoderV2.INPUT_NAMES,
            upstream={"plan": checkpoint_reference(plan_path, plan_state)},
            extra={
                "training_probe_state": _cpu_state(heads),
                "ema_model_state_dict": _cpu_state(ema),
                "deployment_state_dict_key": "ema_model_state_dict",
                "ema_decay": 0.995,
                "epoch": epoch,
            },
        )

    return _fit(
        config,
        model,
        train_data,
        val_data,
        device,
        step,
        checkpoint,
        optimizer=optimizer,
        parameters=parameters,
        after_optimizer_step=after_optimizer_step,
        resume_auxiliary=resume_auxiliary,
    )


def _train_world(config, train_data, val_data, device, manifest_hash, *, block: bool) -> Path:
    plan_path, plan_state, _ = _load_plan(config, device)
    belief_path, belief_state, belief = _load_belief(config, device, use_ema=True)
    _freeze(belief)
    cfg = _world_config(config, belief.cfg.model_dim)
    cls = BlockTransitionWorldModelV2 if block else DirectParallelWorldModelV2
    model = cls(cfg).to(device)
    parameters = list(model.parameters())
    optimizer = _optimizer(parameters, config, device)
    epoch_state = {"index": 0}

    def step(run_model, batch, training):
        with torch.no_grad():
            encoded = encode_current_and_future_beliefs(belief, batch)
        if block and training:
            progress = epoch_state["index"] / max(config.epochs - 1, 1)
            self_ratio = 0.0 if progress < 0.2 else min(0.5, (progress - 0.2) / 0.4 * 0.5)
            blocks = cfg.horizon // cfg.block_length
            teacher = encoded["target_ego_slots"][:, cfg.block_length - 1 :: cfg.block_length]
            teacher_mask = torch.rand(
                batch["ego_id"].shape[0], blocks, device=device
            ) >= self_ratio
            output = run_model.forward_train(
                encoded["ego_slots"],
                batch["ego_future_action"],
                batch["privileged_teammate_future_action"],
                teacher,
                teacher_mask,
            )
            with torch.no_grad():
                consistency = run_model(
                    encoded["ego_slots"],
                    batch["ego_future_action"],
                    batch["privileged_teammate_future_action"],
                )
        else:
            output = run_model(
                encoded["ego_slots"],
                batch["ego_future_action"],
                batch["privileged_teammate_future_action"],
            )
            consistency = None
        losses = world_model_loss_v2(
            output,
            target_belief=encoded["target_ego_slots"],
            target_progress=batch["target_progress"],
            target_contact=batch["target_local_contact"],
            target_force=batch["target_local_force"],
            target_reward=batch["target_reward"].reshape(batch["ego_id"].shape[0], -1),
            target_success=batch["target_success"].reshape(batch["ego_id"].shape[0], -1).max(dim=1).values,
            target_constraint=_trajectory_constraint_target(batch),
            valid_mask=torch.ones_like(batch["target_progress"], dtype=torch.bool),
            consistency_output=consistency,
        )
        branch_losses = _matched_branch_world_loss(run_model, belief, batch, block=block)
        if branch_losses is not None:
            losses["loss"] = losses["loss"] + branch_losses["loss"]
            losses.update(
                {
                    f"branch_{name}": value
                    for name, value in branch_losses.items()
                    if name != "loss"
                }
            )
            losses["selection_metric"] = branch_losses["branch_regret"]
        return losses

    def checkpoint(epoch, metrics):
        return _checkpoint(
            config,
            model,
            cfg,
            manifest_hash,
            metrics,
            cls.__name__,
            cls.INPUT_NAMES,
            upstream={
                "plan": checkpoint_reference(plan_path, plan_state),
                "belief": checkpoint_reference(belief_path, belief_state),
            },
            extra={
                "conditioning": "matched decoded per-step ego/teammate actions",
                "action_prediction_is_primary_target": False,
                "belief_state_dict_key": "ema_model_state_dict",
                "epoch": epoch,
            },
        )

    return _fit(
        config,
        model,
        train_data,
        val_data,
        device,
        step,
        checkpoint,
        optimizer=optimizer,
        parameters=parameters,
        on_epoch_start=lambda epoch: epoch_state.__setitem__("index", epoch),
    )


def _train_proposal(config, train_data, val_data, device, manifest_hash) -> Path:
    plan_path, plan_state, tokenizer = _load_plan(config, device)
    belief_path, belief_state, belief = _load_belief(config, device, use_ema=True)
    world_path, world_state = _load_world(config, device)
    _freeze(tokenizer)
    _freeze(belief)
    support = PlanCodeSupport.from_dict(plan_state["extra"]["plan_support"])
    normalization = plan_state["extra"]["normalization"]
    mean = torch.as_tensor(normalization["action_mean"], device=device)
    std = torch.as_tensor(normalization["action_std"], device=device)
    cfg = _distribution_config(config, belief.cfg.model_dim, tokenizer.cfg.codebook_size, tokenizer.cfg.latent_dim)
    model = PlanProposalV2(cfg).to(device)
    parameters = list(model.parameters())
    optimizer = _optimizer(parameters, config, device)
    active = _active_code_mask(support, device)

    def step(run_model, batch, training):
        del training
        with torch.no_grad():
            state = belief(**_belief_kwargs(batch))["belief"]
            target_actions = batch["ego_future_action"].clone()
            group_rows = batch["branch_group_id"] >= 0
            if group_rows.any():
                branch_return = (
                    batch["branch_target_reward"] * batch["branch_valid_mask"].float()
                ).sum(dim=-1)
                best = _safe_branch_oracle(
                    branch_return,
                    batch["branch_target_constraint"],
                    batch["branch_valid_mask"],
                )
                chosen = batch["branch_matched_action"][
                    torch.arange(best.shape[0], device=device), best, 0
                ]
                target_actions[group_rows] = chosen[group_rows]
            encoded = tokenizer.encode((target_actions - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1))
        output = run_model(state, batch["ego_id"])
        top = _topk_from_distribution(
            output,
            active,
            residual_dim=model.cfg.residual_dim,
            k=min(8, int(active.sum().item())),
        )
        decoded = _decode_plan_batch_differentiable(
            tokenizer, top["topk_codes"], top["topk_residuals"], mean, std
        )
        losses = plan_distribution_loss_v2(
            output,
            target_code=encoded["code_indices"],
            target_residual=encoded["residual"],
            active_code_mask=active,
            diversity_actions=decoded,
        )
        if group_rows.any():
            target_code = encoded["code_indices"]
            covered = (top["topk_codes"] == target_code.unsqueeze(-1)).any(dim=-1)
            coverage = covered[group_rows].float().mean()
            losses["proposal_topk_coverage"] = coverage.detach()
            losses["selection_metric"] = (1.0 - coverage).detach()
        return losses

    def checkpoint(epoch, metrics):
        return _checkpoint(
            config,
            model,
            cfg,
            manifest_hash,
            metrics,
            PlanProposalV2.__name__,
            ("belief", "ego_id"),
            upstream={
                "plan": checkpoint_reference(plan_path, plan_state),
                "belief": checkpoint_reference(belief_path, belief_state),
                "world_block": checkpoint_reference(world_path, world_state),
            },
            extra={
                "active_code_mask": active.cpu(),
                "belief_state_dict_key": "ema_model_state_dict",
                "epoch": epoch,
            },
        )

    return _fit(
        config,
        model,
        train_data,
        val_data,
        device,
        step,
        checkpoint,
        optimizer=optimizer,
        parameters=parameters,
    )


def _train_intention(config, train_data, val_data, device, manifest_hash) -> Path:
    plan_path, plan_state, tokenizer = _load_plan(config, device)
    belief_path, belief_state, belief = _load_belief(config, device, use_ema=True)
    world_path, world_state = _load_world(config, device)
    _freeze(tokenizer)
    _freeze(belief)
    support = PlanCodeSupport.from_dict(plan_state["extra"]["plan_support"])
    normalization = plan_state["extra"]["normalization"]
    mean = torch.as_tensor(normalization["action_mean"], device=device)
    std = torch.as_tensor(normalization["action_std"], device=device)
    cfg = _distribution_config(config, belief.cfg.model_dim, tokenizer.cfg.codebook_size, tokenizer.cfg.latent_dim)
    model = IntentionPosteriorV2(cfg).to(device)
    parameters = list(model.parameters())
    optimizer = _optimizer(parameters, config, device)
    active = _active_code_mask(support, device)

    def step(run_model, batch, training):
        del training
        with torch.no_grad():
            state = belief(**_belief_kwargs(batch))["belief"]
            own = tokenizer.encode((batch["ego_future_action"] - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1))
            peer = tokenizer.encode((batch["privileged_teammate_future_action"] - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1))
        output = run_model(
            state,
            own["code_indices"],
            own["residual"],
            batch["ego_id"],
            torch.zeros(batch["ego_id"].shape[0], model.message_metadata_dim, device=device),
        )
        losses = plan_distribution_loss_v2(
            output,
            target_code=peer["code_indices"],
            target_residual=peer["residual"],
            active_code_mask=active,
        )
        return losses

    def checkpoint(epoch, metrics):
        return _checkpoint(
            config,
            model,
            cfg,
            manifest_hash,
            metrics,
            IntentionPosteriorV2.__name__,
            ("belief", "own_plan_code", "own_plan_residual", "ego_id", "received_message_metadata"),
            upstream={
                "plan": checkpoint_reference(plan_path, plan_state),
                "belief": checkpoint_reference(belief_path, belief_state),
                "world_block": checkpoint_reference(world_path, world_state),
            },
            extra={
                "active_code_mask": active.cpu(),
                "belief_state_dict_key": "ema_model_state_dict",
                "epoch": epoch,
            },
        )

    return _fit(
        config,
        model,
        train_data,
        val_data,
        device,
        step,
        checkpoint,
        optimizer=optimizer,
        parameters=parameters,
    )


def _write_calibration(config, train_data, val_data, device, manifest_hash) -> Path:
    del train_data
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best.pt"
    requested_world_paths = _world_ensemble_paths(config)
    requested_world_hashes = [sha256_file(path) for path in requested_world_paths]
    if config.force_retrain:
        best_path.unlink(missing_ok=True)
    elif config.resume and best_path.is_file():
        existing = load_research_v2_checkpoint(
            best_path, expected_stage="calibration"
        )
        _validate_calibration_resume(
            existing,
            config,
            manifest_hash=manifest_hash,
            world_paths=requested_world_paths,
            world_hashes=requested_world_hashes,
        )
        return best_path
    elif best_path.exists():
        raise FileExistsError(
            f"{best_path} exists; use resume or force_retrain for calibration"
        )

    plan_path, plan_state, tokenizer = _load_plan(config, device)
    belief_path, belief_state, belief = _load_belief(config, device, use_ema=True)
    world_entries = _load_world_ensemble(config, device)
    world_path, world_state, _ = world_entries[0]
    worlds = [entry[2] for entry in world_entries]
    intention_path, intention_state, intention = _load_intention(config, device)
    for module in (tokenizer, belief, intention, *worlds):
        _freeze(module)

    normalization = plan_state["extra"]["normalization"]
    mean = torch.as_tensor(normalization["action_mean"], device=device)
    std = torch.as_tensor(normalization["action_std"], device=device)
    support = PlanCodeSupport.from_dict(plan_state["extra"]["plan_support"])
    active_codes = support.active_codes.to(device=device, dtype=torch.long)
    code_remap = torch.full(
        (tokenizer.cfg.codebook_size,), -1, device=device, dtype=torch.long
    )
    code_remap[active_codes] = torch.arange(active_codes.numel(), device=device)
    precision = _precision_runtime(config, device)

    collected: dict[str, list[torch.Tensor]] = {
        "quantiles": [],
        "return": [],
        "constraint_logits": [],
        "constraint": [],
        "posterior_logits": [],
        "posterior_target": [],
        "residual_ratio": [],
    }
    loader = _loader(val_data, config, False)
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(_select_stage_batch(raw, "calibration"), device)
            with precision.autocast():
                state = belief(**_belief_kwargs(batch))["belief"]
                world_outputs = [
                    world(
                        state,
                        batch["ego_future_action"],
                        batch["privileged_teammate_future_action"],
                    )
                    for world in worlds
                ]
                own = tokenizer.encode(
                    (batch["ego_future_action"] - mean.reshape(1, 1, -1))
                    / std.reshape(1, 1, -1)
                )
                peer = tokenizer.encode(
                    (
                        batch["privileged_teammate_future_action"]
                        - mean.reshape(1, 1, -1)
                    )
                    / std.reshape(1, 1, -1)
                )
                posterior = intention(
                    state,
                    own["code_indices"],
                    own["residual"],
                    batch["ego_id"],
                    torch.zeros(
                        batch["ego_id"].shape[0],
                        intention.message_metadata_dim,
                        device=device,
                    ),
                )
            # Runtime averages calibrated member quantiles and calibrated
            # member constraint probabilities.  Fit that exact ensemble
            # aggregation rather than silently calibrating member zero.
            collected["quantiles"].append(
                torch.stack(
                    [output["return_quantiles"] for output in world_outputs], dim=0
                )
                .mean(dim=0)
                .float()
                .cpu()
            )
            collected["return"].append(
                batch["target_reward"].reshape(batch["ego_id"].shape[0], -1).sum(dim=1).float().cpu()
            )
            collected["constraint_logits"].append(
                torch.stack(
                    [output["constraint_logits"] for output in world_outputs], dim=-1
                )
                .float()
                .cpu()
            )
            collected["constraint"].append(_trajectory_constraint_target(batch).cpu())

            compact_target = code_remap[peer["code_indices"]]
            valid = compact_target >= 0
            if valid.any():
                collected["posterior_logits"].append(
                    posterior["code_logits"][valid][:, active_codes].float().cpu()
                )
                collected["posterior_target"].append(compact_target[valid].cpu())
                target_codes = peer["code_indices"][valid]
                gather = target_codes.reshape(-1, 1, 1).expand(
                    -1, 1, tokenizer.cfg.latent_dim
                )
                mu = posterior["residual_mu_by_code"][valid].gather(1, gather).squeeze(1)
                logvar = (
                    posterior["residual_logvar_by_code"][valid]
                    .gather(1, gather)
                    .squeeze(1)
                )
                ratio = (
                    (peer["residual"][valid] - mu).float().square()
                    / logvar.float().exp().clamp_min(1e-6)
                )
                collected["residual_ratio"].append(ratio.cpu())

    required_arrays = ("quantiles", "return", "constraint_logits", "constraint")
    if any(not collected[name] for name in required_arrays):
        raise RuntimeError("validation loader produced no world calibration samples")
    arrays = {
        name: torch.cat(values, dim=0)
        for name, values in collected.items()
        if values
    }
    posterior_available = all(
        name in arrays
        for name in ("posterior_logits", "posterior_target", "residual_ratio")
    )
    quantile_scale, quantile_bias = _fit_robust_return_affine(
        arrays["quantiles"][:, 1], arrays["return"]
    )
    constraint_temperature, constraint_bias = _fit_binary_temperature(
        arrays["constraint_logits"], arrays["constraint"]
    )
    posterior_temperature = (
        _fit_multiclass_temperature(
            arrays["posterior_logits"], arrays["posterior_target"]
        )
        if posterior_available
        else 1.0
    )
    variance_scale = (
        _fit_variance_scale(arrays["residual_ratio"])
        if posterior_available
        else 1.0
    )
    communication_frozen = config.communication_price is not None
    calibrated_median = arrays["quantiles"][:, 1] * quantile_scale + quantile_bias
    metrics = {
        "validation_only": True,
        "samples": int(arrays["return"].numel()),
        "posterior_samples": (
            int(arrays["posterior_target"].numel()) if posterior_available else 0
        ),
        "return_median_mae_before": float(
            (arrays["quantiles"][:, 1] - arrays["return"]).abs().mean()
        ),
        "return_median_mae_after": float(
            (calibrated_median - arrays["return"]).abs().mean()
        ),
        "constraint_nll_before": _binary_nll(
            arrays["constraint_logits"], arrays["constraint"]
        ),
        "constraint_nll_after": _binary_nll(
            arrays["constraint_logits"] / constraint_temperature + constraint_bias,
            arrays["constraint"],
        ),
        "posterior_nll_before": (
            _multiclass_nll(arrays["posterior_logits"], arrays["posterior_target"])
            if posterior_available
            else None
        ),
        "posterior_nll_after": (
            _multiclass_nll(
                arrays["posterior_logits"] / posterior_temperature,
                arrays["posterior_target"],
            )
            if posterior_available
            else None
        ),
    }
    state = make_research_v2_checkpoint(
        stage="calibration",
        model_class="FrozenCalibrationV2",
        model_config={
            "version": 2,
            "fit_split": "validation",
            "world_ensemble_size": len(world_entries),
        },
        model_state_dict={},
        training_config=config,
        dataset_manifest_sha256=manifest_hash,
        forward_inputs=("return_quantiles", "constraint_logits", "posterior_logits"),
        metrics=metrics,
        upstream={
            "plan": checkpoint_reference(plan_path, plan_state),
            "belief": checkpoint_reference(belief_path, belief_state),
            "world_block": checkpoint_reference(world_path, world_state),
            "intention": checkpoint_reference(intention_path, intention_state),
            **{
                f"world_block_member_{index:02d}": checkpoint_reference(path, state)
                for index, (path, state, _) in enumerate(world_entries)
            },
        },
        extra={
            "quantile_scale": quantile_scale,
            "quantile_bias": quantile_bias,
            "constraint_temperature": constraint_temperature,
            "constraint_logit_bias": constraint_bias,
            "posterior_temperature": posterior_temperature,
            "posterior_variance_scale": variance_scale,
            "communication_price_frozen": communication_frozen,
            "communication_price": config.communication_price,
            "communication_price_method": (
                "configured_fixed"
                if communication_frozen
                else "not_fitted_no_counterfactual_communication_labels"
            ),
            "belief_state_dict_key": "ema_model_state_dict",
            "world_ensemble_size": len(world_entries),
            "world_ensemble_sha256": requested_world_hashes,
            "world_ensemble_aggregation": (
                "mean_quantiles_and_mean_calibrated_member_probabilities"
            ),
        },
    )
    return save_research_v2_checkpoint(best_path, state)


@dataclass
class _PrecisionRuntime:
    device_type: str
    dtype: torch.dtype | None
    scaler: Any

    def autocast(self):
        if self.dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


def _fit(
    config,
    model,
    train_data,
    val_data,
    device,
    step,
    checkpoint_builder,
    *,
    optimizer,
    parameters,
    after_optimizer_step: Callable[[], None] | None = None,
    resume_auxiliary: Callable[[Mapping[str, Any]], None] | None = None,
    on_epoch_start: Callable[[int], None] | None = None,
) -> Path:
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    trainer_path = root / "trainer_state.pt"
    if config.force_retrain:
        for path in (best_path, last_path, trainer_path):
            path.unlink(missing_ok=True)
    elif not config.resume and any(path.exists() for path in (best_path, last_path, trainer_path)):
        raise FileExistsError(
            f"{root} already contains training artifacts; use resume or force_retrain"
        )

    stopper = _EarlyStop(config.patience, config.relative_min_delta)
    precision = _precision_runtime(config, device)
    start_epoch = 0
    if config.resume and trainer_path.is_file() and last_path.is_file():
        last_state = load_research_v2_checkpoint(
            last_path, expected_stage=config.stage, map_location=device
        )
        _validate_resume_checkpoint(last_state, config)
        trainer = torch.load(trainer_path, map_location=device, weights_only=False)
        if int(trainer.get("epoch", -1)) != int(last_state["metrics"]["epoch"]):
            raise ValueError("trainer_state.pt and last.pt refer to different epochs")
        model.load_state_dict(last_state["model_state_dict"])
        optimizer.load_state_dict(trainer["optimizer_state_dict"])
        if precision.scaler.is_enabled() and trainer.get("scaler_state_dict"):
            precision.scaler.load_state_dict(trainer["scaler_state_dict"])
        stopper.best = float(trainer["early_stop_best"])
        stopper.bad_epochs = int(trainer["early_stop_bad_epochs"])
        _restore_rng_state(trainer.get("rng_state", {}), device)
        if resume_auxiliary is not None:
            resume_auxiliary(last_state)
        start_epoch = int(trainer["epoch"])
        if bool(trainer.get("completed", False)):
            if not best_path.is_file():
                raise FileNotFoundError("completed trainer state lacks best.pt")
            _status(
                f"[{config.stage}] resume reused completed checkpoint "
                f"{best_path} (epoch {start_epoch})"
            )
            return best_path

    execution_model = _maybe_compile(model, config, device)
    train_loader = _loader(train_data, config, True)
    val_loader = _loader(val_data, config, False)
    for epoch in range(start_epoch, config.epochs):
        _status(f"[{config.stage}] epoch {epoch + 1}/{config.epochs} started")
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        _set_loader_epoch(train_loader, epoch)
        train_metrics = _epoch(
            execution_model,
            train_loader,
            device,
            step,
            True,
            config,
            precision,
            optimizer=optimizer,
            parameters=parameters,
            after_optimizer_step=after_optimizer_step,
        )
        val_metrics = _epoch(
            execution_model,
            val_loader,
            device,
            step,
            False,
            config,
            precision,
        )
        metrics = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": val_metrics,
            "selection_metric": val_metrics.get("selection_metric", val_metrics["loss"]),
        }
        state = checkpoint_builder(epoch, metrics)
        improved = stopper.update(metrics["selection_metric"])
        save_research_v2_checkpoint(last_path, state)
        if improved:
            _atomic_copy(last_path, best_path)
        _save_trainer_state(
            trainer_path,
            epoch=epoch + 1,
            optimizer=optimizer,
            precision=precision,
            stopper=stopper,
            device=device,
            completed=False,
        )
        if stopper.stopped:
            _status(
                f"[{config.stage}] early stopping after epoch {epoch + 1}; "
                f"best validation metric={stopper.best:.6g}"
            )
            break
        _status(
            f"[{config.stage}] epoch {epoch + 1}/{config.epochs} completed: "
            f"train_loss={train_metrics['loss']:.6g}, "
            f"validation_metric={metrics['selection_metric']:.6g}"
        )
    if not best_path.is_file():
        raise RuntimeError("training produced no best checkpoint")
    _save_trainer_state(
        trainer_path,
        epoch=int(load_research_v2_checkpoint(last_path)["metrics"]["epoch"]),
        optimizer=optimizer,
        precision=precision,
        stopper=stopper,
        device=device,
        completed=True,
    )
    return best_path


def _epoch(
    model,
    loader,
    device,
    step,
    training,
    config,
    precision,
    *,
    optimizer=None,
    parameters: Sequence[torch.nn.Parameter] = (),
    after_optimizer_step: Callable[[], None] | None = None,
) -> dict[str, Any]:
    model.train(training)
    totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    count = 0
    optimizer_steps = 0
    pending_microbatches = 0
    hard_code_counts: torch.Tensor | None = None
    if training:
        if optimizer is None or not parameters:
            raise ValueError("training epoch requires optimizer and parameters")
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, raw in enumerate(loader):
            batch = _to_device(_select_stage_batch(raw, config.stage), device)
            with precision.autocast():
                losses = step(model, batch, training)
            if "loss" not in losses or not torch.isfinite(losses["loss"]).all():
                raise FloatingPointError(f"non-finite {config.stage} loss")
            code_indices = losses.get("code_indices")
            soft_usage = losses.get("soft_code_usage")
            if torch.is_tensor(code_indices) and torch.is_tensor(soft_usage):
                codebook_size = int(soft_usage.numel())
                batch_counts = torch.bincount(
                    code_indices.detach().reshape(-1).long().cpu(),
                    minlength=codebook_size,
                )
                if hard_code_counts is None:
                    hard_code_counts = torch.zeros_like(batch_counts)
                hard_code_counts += batch_counts
            if training:
                precision.scaler.scale(
                    losses["loss"] / config.gradient_accumulation_steps
                ).backward()
                pending_microbatches += 1
                is_last = batch_index + 1 == len(loader)
                if (
                    pending_microbatches >= config.gradient_accumulation_steps
                    or is_last
                ):
                    _optimizer_step(
                        optimizer,
                        parameters,
                        precision,
                        config,
                        pending_microbatches,
                    )
                    optimizer_steps += 1
                    pending_microbatches = 0
                    if after_optimizer_step is not None:
                        after_optimizer_step()
            for name, value in losses.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value.detach().item())
                    metric_counts[name] = metric_counts.get(name, 0) + 1
            count += 1
            if (batch_index + 1) % 100 == 0:
                phase = "train" if training else "validation"
                total_batches = len(loader)
                _status(
                    f"[{config.stage}:{phase}] batch {batch_index + 1}/"
                    f"{total_batches}, loss={float(losses['loss'].detach()):.6g}"
                )
    if count == 0:
        raise RuntimeError("empty training/validation loader")
    metrics = {name: value / metric_counts[name] for name, value in totals.items()}
    metrics["microbatches"] = float(count)
    if training:
        metrics["optimizer_steps"] = float(optimizer_steps)
    if hard_code_counts is not None:
        metrics.update(_hard_code_metrics(hard_code_counts))
    return metrics


def _hard_code_metrics(counts: torch.Tensor) -> dict[str, Any]:
    counts = torch.as_tensor(counts, dtype=torch.long, device="cpu").reshape(-1)
    total = int(counts.sum().item())
    if total <= 0:
        raise ValueError("hard code metrics require at least one encoded sample")
    probability = counts.double() / float(total)
    used = probability > 0
    entropy = -(probability[used] * probability[used].log()).sum()
    return {
        "hard_code_samples": total,
        "hard_codes_used": int(used.sum().item()),
        "hard_usage_ratio": float(used.float().mean().item()),
        "hard_entropy": float(entropy.item()),
        "hard_perplexity": float(entropy.exp().item()),
        "hard_max_code_fraction": float(probability.max().item()),
        "hard_code_counts": counts.tolist(),
    }


def _relative_improvement(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline) or baseline <= 0:
        return math.nan
    return float((baseline - value) / baseline)


def _checkpoint(
    config,
    model,
    model_config,
    manifest_hash,
    metrics,
    model_class,
    forward_inputs,
    *,
    upstream=None,
    extra=None,
):
    return make_research_v2_checkpoint(
        stage=config.stage,
        model_class=model_class,
        model_config=model_config,
        model_state_dict=_cpu_state(model),
        training_config=config,
        dataset_manifest_sha256=manifest_hash,
        forward_inputs=forward_inputs,
        metrics=metrics,
        upstream=upstream,
        extra=extra,
    )


def _load_plan(config, device):
    path = _required(config.plan_checkpoint, "plan checkpoint")
    state = load_research_v2_checkpoint(path, expected_stage="plan", map_location=device)
    model = PlanTokenizerV2(PlanTokenizerV2Config(**state["model_config"])).to(device)
    model.load_state_dict(state["model_state_dict"])
    return path, state, model


def _load_belief(config, device, *, use_ema: bool = False):
    path = _required(config.belief_checkpoint, "belief checkpoint")
    state = load_research_v2_checkpoint(path, expected_stage="belief", map_location=device)
    model = BeliefEncoderV2(BeliefEncoderV2Config(**state["model_config"])).to(device)
    model_state = state["model_state_dict"]
    if use_ema:
        extra = state.get("extra", {})
        deployment_key = extra.get("deployment_state_dict_key", "ema_model_state_dict")
        if deployment_key not in extra:
            raise ValueError(
                f"belief checkpoint declares deployment weights {deployment_key!r} "
                "but does not contain them"
            )
        model_state = extra[deployment_key]
    model.load_state_dict(model_state)
    return path, state, model


def _load_world(config, device):
    path = _required(config.world_block_checkpoint, "world-block checkpoint")
    state = load_research_v2_checkpoint(path, expected_stage="world_block", map_location=device)
    return path, state


def _world_ensemble_paths(config) -> tuple[Path, ...]:
    configured = tuple(getattr(config, "world_ensemble_checkpoints", ()))
    if configured:
        paths = tuple(_required(value, "world ensemble checkpoint") for value in configured)
        if config.world_block_checkpoint is not None and (
            paths[0].resolve() != Path(config.world_block_checkpoint).resolve()
        ):
            raise ValueError(
                "world_block_checkpoint must be the first world ensemble member"
            )
        return paths
    return (_required(config.world_block_checkpoint, "world-block checkpoint"),)


def _load_world_ensemble(config, device):
    entries = []
    reference_config = None
    reference_dataset = None
    expected_upstream = {
        "plan": sha256_file(_required(config.plan_checkpoint, "plan checkpoint")),
        "belief": sha256_file(
            _required(config.belief_checkpoint, "belief checkpoint")
        ),
    }
    for path in _world_ensemble_paths(config):
        state = load_research_v2_checkpoint(
            path, expected_stage="world_block", map_location=device
        )
        if reference_config is None:
            reference_config = state["model_config"]
            reference_dataset = state["dataset_manifest_sha256"]
        elif (
            state["model_config"] != reference_config
            or state["dataset_manifest_sha256"] != reference_dataset
        ):
            raise ValueError("world ensemble members have incompatible contracts")
        for name, expected_hash in expected_upstream.items():
            reference = state.get("upstream", {}).get(name)
            if reference is None or reference.get("sha256") != expected_hash:
                raise ValueError(
                    f"world ensemble member has stale/missing {name} lineage"
                )
        model = BlockTransitionWorldModelV2(
            WorldModelV2Config(**state["model_config"])
        ).to(device)
        model.load_state_dict(state["model_state_dict"])
        entries.append((path, state, model))
    return entries


def _load_intention(config, device):
    path = _required(config.intention_checkpoint, "intention checkpoint")
    state = load_research_v2_checkpoint(path, expected_stage="intention", map_location=device)
    model = IntentionPosteriorV2(
        PlanDistributionV2Config(**state["model_config"])
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    return path, state, model


def _fit_robust_return_affine(prediction, target):
    x = prediction.detach().double().reshape(-1)
    y = target.detach().double().reshape(-1)
    if x.numel() < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("return calibration requires finite validation predictions")
    x10, x90 = torch.quantile(x, torch.tensor([0.1, 0.9], dtype=x.dtype))
    y10, y90 = torch.quantile(y, torch.tensor([0.1, 0.9], dtype=y.dtype))
    spread = float(x90 - x10)
    scale = 1.0 if abs(spread) < 1e-8 else float((y90 - y10) / (x90 - x10))
    scale = float(np.clip(scale, 0.05, 20.0))
    bias = float(torch.median(y - scale * x))
    return scale, bias


def _fit_binary_temperature(logits, target):
    logits = logits.detach().double()
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    target = target.detach().double().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != target.shape[0] or logits.numel() == 0:
        raise ValueError("binary calibration arrays are empty or mismatched")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (log_temperature, bias), max_iter=75, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(math.log(0.05), math.log(20.0)).exp()
        ensemble_probability = (logits / temperature + bias).sigmoid().mean(dim=-1)
        loss = F.binary_cross_entropy(ensemble_probability, target)
        loss = loss + 1e-6 * (log_temperature.square() + bias.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(
        log_temperature.detach().clamp(math.log(0.05), math.log(20.0)).exp()
    )
    return temperature, float(bias.detach().clamp(-20.0, 20.0))


def _fit_multiclass_temperature(logits, target):
    logits = logits.detach().double()
    target = target.detach().long().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != target.shape[0] or logits.shape[0] == 0:
        raise ValueError("posterior calibration arrays are empty or mismatched")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (log_temperature,), max_iter=75, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(math.log(0.05), math.log(20.0)).exp()
        loss = F.cross_entropy(logits / temperature, target)
        loss = loss + 1e-6 * log_temperature.square()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(
        log_temperature.detach().clamp(math.log(0.05), math.log(20.0)).exp()
    )


def _fit_variance_scale(ratio):
    values = ratio.detach().double().reshape(-1)
    values = values[torch.isfinite(values) & (values >= 0)]
    if values.numel() == 0:
        raise ValueError("posterior variance calibration has no finite residuals")
    lower, upper = torch.quantile(
        values, torch.tensor([0.01, 0.99], dtype=values.dtype)
    )
    scale = float(values.clamp(lower, upper).mean())
    return float(np.clip(scale, 0.05, 20.0))


def _binary_nll(logits, target):
    logits = logits.detach().float()
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    target = target.detach().float().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != target.shape[0]:
        raise ValueError("binary NLL arrays are mismatched")
    return float(F.binary_cross_entropy(logits.sigmoid().mean(dim=-1), target))


def _multiclass_nll(logits, target):
    return float(F.cross_entropy(logits.detach().float(), target.detach().long()))


def _plan_config(config):
    return PlanTokenizerV2Config(
        horizon=config.horizon,
        hidden_dim=32 if config.smoke else 256,
        codebook_size=8 if config.smoke else 64,
        residual_dim=16,
    )


def _belief_config(config, local_dim):
    return BeliefEncoderV2Config(
        history=config.history,
        local_dim=local_dim,
        model_dim=16 if config.smoke else 256,
        num_heads=4 if config.smoke else 8,
        temporal_layers=1 if config.smoke else 3,
        role_layers=1 if config.smoke else 3,
        ffn_dim=32 if config.smoke else 1024,
        dropout=0.0 if config.smoke else 0.1,
    )


def _world_config(config, belief_dim):
    return WorldModelV2Config(
        horizon=config.horizon,
        block_length=4,
        belief_dim=belief_dim,
        model_dim=32 if config.smoke else 512,
        context_layers=1 if config.smoke else 4,
        transition_layers=1 if config.smoke else 6,
        heads=4 if config.smoke else 8,
        ffn_dim=64 if config.smoke else 2048,
        dropout=0.0 if config.smoke else 0.1,
    )


def _distribution_config(config, belief_dim, codebook_size, residual_dim):
    return PlanDistributionV2Config(
        belief_dim=belief_dim,
        codebook_size=codebook_size,
        residual_dim=residual_dim,
        model_dim=16 if config.smoke else 256,
        layers=1 if config.smoke else 4,
        heads=4 if config.smoke else 8,
        ffn_dim=32 if config.smoke else 1024,
        dropout=0.0 if config.smoke else 0.1,
    )


def _belief_kwargs(batch):
    return {
        "local_history": batch["local_history"],
        "history_mask": batch["history_mask"],
        "ego_id": batch["ego_id"],
        "object_observation": batch["object_observation_history"],
        "object_valid": batch["object_valid_history"],
        "object_age": batch["object_age_history"],
        "object_confidence": batch["object_confidence_history"],
    }


def _trajectory_constraint_target(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Use the same force/collision/private-event semantics as matched branches."""

    batch_size = int(batch["ego_id"].shape[0])
    sources = (
        batch["target_force_violation"],
        batch["target_collision"],
        batch["target_private_event_error"],
    )
    return torch.stack(
        [value.reshape(batch_size, -1).amax(dim=1) for value in sources], dim=0
    ).amax(dim=0).to(dtype=torch.float32)


def _safe_branch_oracle(
    realized_return: torch.Tensor,
    constraint: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Choose a valid safe branch first, then maximize realized return."""

    if realized_return.ndim != 2 or constraint.shape != realized_return.shape:
        raise ValueError("branch return/constraint must both have shape [B,N]")
    if valid_mask.shape[:2] != realized_return.shape:
        raise ValueError("branch valid mask must have shape [B,N,H]")
    valid = valid_mask.any(dim=-1)
    safe = valid & (constraint <= 0.5)
    safe_scores = realized_return.masked_fill(~safe, -torch.inf)
    safe_best = safe_scores.argmax(dim=-1)
    minimum_constraint = constraint.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
    least_unsafe = valid & (constraint <= minimum_constraint + 1e-6)
    fallback_scores = realized_return.masked_fill(~least_unsafe, -torch.inf)
    fallback_best = fallback_scores.argmax(dim=-1)
    no_valid = ~valid.any(dim=-1)
    if no_valid.any():
        fallback_best = torch.where(no_valid, realized_return.argmax(dim=-1), fallback_best)
    return torch.where(safe.any(dim=-1), safe_best, fallback_best)


def _topk_from_distribution(output, active_mask, *, residual_dim: int, k: int):
    mask = active_mask.to(output["code_logits"].device, dtype=torch.bool).reshape(1, -1)
    logits = output["code_logits"].masked_fill(~mask, -torch.inf)
    if int(mask.sum().item()) < k:
        raise ValueError(f"proposal requires {k} active codes")
    _, codes = logits.topk(k, dim=-1)
    residuals = output["residual_mu_by_code"].gather(
        1, codes.unsqueeze(-1).expand(-1, -1, residual_dim)
    )
    return {**output, "topk_codes": codes, "topk_residuals": residuals}


def _decode_plan_batch_differentiable(
    tokenizer, codes, residuals, action_mean, action_std
):
    """Decode frozen code embeddings while retaining residual-head gradients."""

    leading = codes.shape
    flat_codes = codes.reshape(-1)
    flat_residuals = residuals.reshape(-1, residuals.shape[-1])
    decoded = tokenizer.decode(
        tokenizer.vq.embedding(flat_codes), flat_residuals
    )["recon_actions"]
    decoded = (
        decoded * action_std.reshape(1, 1, -1)
        + action_mean.reshape(1, 1, -1)
    )
    return decoded.reshape(
        *leading, tokenizer.cfg.horizon, tokenizer.cfg.action_dim
    )


def _matched_branch_world_loss(model, belief, batch, *, block: bool):
    """Train WAM on outcomes paired with the exact forced joint actions."""

    active_rows = batch["branch_group_id"] >= 0
    if not active_rows.any():
        return None
    B = int(active_rows.sum().item())
    N = batch["branch_matched_action"].shape[1]
    selected = {name: value[active_rows] for name, value in batch.items() if torch.is_tensor(value)}

    def repeat_history(value):
        return value[:, None].expand(B, N, *value.shape[1:]).reshape(B * N, *value.shape[1:])

    flat = {
        "local_history": repeat_history(selected["local_history"]),
        "history_mask": repeat_history(selected["history_mask"]),
        "ego_id": repeat_history(selected["ego_id"]),
        "object_observation_history": repeat_history(selected["object_observation_history"]),
        "object_valid_history": repeat_history(selected["object_valid_history"]),
        "object_confidence_history": repeat_history(selected["object_confidence_history"]),
        "object_age_history": repeat_history(selected["object_age_history"]),
        "future_model_observation": selected["branch_future_model_observation"].reshape(
            B * N, model.cfg.horizon, -1
        ),
        "ego_future_action": selected["branch_matched_action"][:, :, 0].reshape(
            B * N, model.cfg.horizon, model.cfg.action_dim
        ),
        "future_object_observation": selected["branch_future_object_observation"].reshape(
            B * N, model.cfg.horizon, -1
        ),
        "future_object_valid": selected["branch_future_object_valid"].reshape(
            B * N, model.cfg.horizon
        ),
        "future_object_confidence": selected["branch_future_object_confidence"].reshape(
            B * N, model.cfg.horizon
        ),
        "future_object_age": selected["branch_future_object_age"].reshape(
            B * N, model.cfg.horizon
        ),
    }
    with torch.no_grad():
        encoded = encode_current_and_future_beliefs(belief, flat)
    peer_action = selected["branch_matched_action"][:, :, 1].reshape(
        B * N, model.cfg.horizon, model.cfg.action_dim
    )
    output = model(encoded["ego_slots"], flat["ego_future_action"], peer_action)
    valid = selected["branch_valid_mask"].reshape(B * N, model.cfg.horizon)
    identifiers = selected["branch_group_id"][:, None].expand(B, N).reshape(B * N)
    return world_model_loss_v2(
        output,
        target_belief=encoded["target_ego_slots"],
        target_progress=selected["branch_target_progress"].reshape(B * N, model.cfg.horizon),
        target_contact=selected["branch_target_contact"].reshape(B * N, model.cfg.horizon),
        target_force=selected["branch_target_force"].reshape(B * N, model.cfg.horizon),
        target_reward=selected["branch_target_reward"].reshape(B * N, model.cfg.horizon),
        target_success=selected["branch_target_success"].reshape(B * N),
        target_constraint=selected["branch_target_constraint"].reshape(B * N),
        valid_mask=valid,
        branch_group_id=identifiers,
    )


class _FileGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle at file and in-file level while keeping HDF5 reads local."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        max_batches: int,
        drop_last: bool,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_batches = int(max_batches)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.ranges: list[tuple[int, int]] = []
        start = 0
        previous = dataset.index[0].file_idx
        for index, sample in enumerate(dataset.index[1:], start=1):
            if sample.file_idx != previous:
                self.ranges.append((start, index))
                start = index
                previous = sample.file_idx
        self.ranges.append((start, len(dataset.index)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            total += 1
        return min(total, self.max_batches) if self.max_batches > 0 else total

    def __iter__(self) -> Iterator[list[int]]:
        # Validation uses a fixed randomized file subset when capped so its
        # estimate is not biased toward the first episode filenames.
        epoch = self.epoch if self.shuffle else 0
        rng = random.Random(self.seed + 1_000_003 * epoch)
        ranges = list(self.ranges)
        uncapped_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            uncapped_batches += 1
        if self.shuffle or (self.max_batches > 0 and self.max_batches < uncapped_batches):
            rng.shuffle(ranges)
        pending: list[int] = []
        yielded = 0
        torch_generator = torch.Generator().manual_seed(self.seed + 1_000_003 * epoch)
        for start, stop in ranges:
            if self.shuffle:
                order = torch.randperm(stop - start, generator=torch_generator).tolist()
                indices = [start + offset for offset in order]
            else:
                indices = list(range(start, stop))
            pending.extend(indices)
            while len(pending) >= self.batch_size:
                yield pending[: self.batch_size]
                del pending[: self.batch_size]
                yielded += 1
                if self.max_batches > 0 and yielded >= self.max_batches:
                    return
        if pending and not self.drop_last and (
            self.max_batches <= 0 or yielded < self.max_batches
        ):
            yield pending


def _action_statistics(dataset, config):
    total = torch.zeros(int(dataset.action_dim), dtype=torch.float64)
    square = torch.zeros_like(total)
    count = 0
    loader = _loader(
        dataset,
        config,
        False,
        max_steps=-1 if config.smoke else config.statistics_max_steps,
        drop_last=False,
        collate_keys={
            "ego_future_action",
            "branch_group_id",
            "branch_matched_action",
        },
    )
    for batch in loader:
        values = [
            batch["ego_future_action"].double().reshape(-1, int(dataset.action_dim))
        ]
        branch_rows = batch["branch_group_id"] >= 0
        if branch_rows.any():
            values.append(
                batch["branch_matched_action"][branch_rows, :, 0]
                .double()
                .reshape(-1, int(dataset.action_dim))
            )
        value = torch.cat(values, dim=0)
        total += value.sum(dim=0)
        square += value.square().sum(dim=0)
        count += value.shape[0]
    if count == 0:
        raise RuntimeError("cannot estimate action statistics from an empty dataset")
    mean = total / count
    variance = square / count - mean.square()
    return mean.float(), variance.clamp_min(1e-6).sqrt().float()


def _estimate_plan_support(model, dataset, mean, std, config, device):
    accumulator = PlanCodeSupportAccumulator(model.cfg.codebook_size, model.cfg.latent_dim)
    loader = _loader(
        dataset,
        config,
        False,
        max_steps=-1 if config.smoke else config.support_max_steps,
        drop_last=False,
        collate_keys={
            "ego_future_action",
            "privileged_teammate_future_action",
            "branch_group_id",
            "branch_matched_action",
        },
    )
    model.eval()
    with torch.no_grad():
        for raw in loader:
            actions = raw["ego_future_action"].to(device, non_blocking=device.type == "cuda")
            encoded = model.encode(
                (actions - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
            )
            accumulator.update(encoded["code_indices"], encoded["residual"])
            peer_actions = raw["privileged_teammate_future_action"].to(
                device, non_blocking=device.type == "cuda"
            )
            peer_encoded = model.encode(
                (peer_actions - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
            )
            accumulator.update(
                peer_encoded["code_indices"], peer_encoded["residual"]
            )
            branch_rows = raw["branch_group_id"] >= 0
            if branch_rows.any():
                branch_actions = raw["branch_matched_action"][branch_rows, :, 0].to(
                    device, non_blocking=device.type == "cuda"
                )
                branch_actions = branch_actions.reshape(
                    -1, model.cfg.horizon, model.cfg.action_dim
                )
                branch_encoded = model.encode(
                    (branch_actions - mean.reshape(1, 1, -1))
                    / std.reshape(1, 1, -1)
                )
                accumulator.update(
                    branch_encoded["code_indices"], branch_encoded["residual"]
                )
    return accumulator.build(min_count=1, std_floor=1e-3)


def _loader(
    dataset,
    config,
    shuffle,
    *,
    max_steps: int | None = None,
    drop_last=None,
    collate_keys: set[str] | None = None,
):
    selected_keys = tuple(
        sorted(collate_keys or _STAGE_BATCH_KEYS[config.stage])
    )
    loader_dataset = dataset.project(selected_keys)
    if max_steps is None:
        max_steps = (
            config.max_steps_per_epoch if shuffle else config.max_validation_steps
        )
    if drop_last is None:
        drop_last = bool(shuffle and len(dataset) >= config.batch_size)
    sampler = _FileGroupedBatchSampler(
        loader_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        seed=config.seed + (0 if shuffle else 104_729),
        max_batches=max_steps,
        drop_last=drop_last,
    )
    generator = torch.Generator().manual_seed(config.seed)
    pin_memory = (
        config.pin_memory
        if config.pin_memory is not None
        else torch.cuda.is_available() and config.device != "cpu"
    )
    persistent = (
        config.persistent_workers
        if config.persistent_workers is not None
        else config.num_workers > 0
    )
    kwargs = dict(
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent and config.num_workers > 0),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(
        loader_dataset,
        **kwargs,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _optimizer(parameters, config, device):
    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if config.fused_optimizer and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def _optimizer_step(optimizer, parameters, precision, config, pending,):
    precision.scaler.unscale_(optimizer)
    if pending < config.gradient_accumulation_steps:
        correction = config.gradient_accumulation_steps / float(pending)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
    torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
    precision.scaler.step(optimizer)
    precision.scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _precision_runtime(config, device) -> _PrecisionRuntime:
    precision = config.precision
    if precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp32"
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use fp32 for CPU smoke tests")
    dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    scaler_enabled = precision == "fp16" and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    except TypeError:  # PyTorch < 2.3 compatibility.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    return _PrecisionRuntime(device.type, dtype, scaler)


def _maybe_compile(model, config, device):
    if not config.compile_model or config.smoke or device.type != "cuda":
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_model requires torch.compile")
    compiled = torch.compile(model, mode=config.compile_mode, dynamic=False)
    if hasattr(model, "forward_train"):
        compiled.forward_train = torch.compile(  # type: ignore[attr-defined]
            model.forward_train, mode=config.compile_mode, dynamic=False
        )
    return compiled


def _set_loader_epoch(loader, epoch):
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


_COMMON_BELIEF_KEYS = {
    "local_history",
    "history_mask",
    "ego_id",
    "object_observation_history",
    "object_valid_history",
    "object_confidence_history",
    "object_age_history",
}
_STAGE_BATCH_KEYS = {
    "plan": {
        "ego_future_action",
        "target_maneuver",
        "branch_group_id",
        "branch_matched_action",
    },
    "belief": _COMMON_BELIEF_KEYS
    | {
        "target_current_self_state",
        "target_current_object_pose",
        "target_current_teammate_pose",
        "target_current_task_progress",
        "target_current_maneuver",
    },
    "world_direct": _COMMON_BELIEF_KEYS
    | {
        "future_model_observation",
        "ego_future_action",
        "future_object_observation",
        "future_object_valid",
        "future_object_confidence",
        "future_object_age",
        "privileged_teammate_future_action",
        "target_progress",
        "target_local_contact",
        "target_local_force",
        "target_reward",
        "target_success",
        "target_force_violation",
        "target_collision",
        "target_private_event_error",
        "branch_group_id",
        "branch_matched_action",
        "branch_valid_mask",
        "branch_future_model_observation",
        "branch_future_object_observation",
        "branch_future_object_valid",
        "branch_future_object_confidence",
        "branch_future_object_age",
        "branch_target_progress",
        "branch_target_contact",
        "branch_target_force",
        "branch_target_reward",
        "branch_target_success",
        "branch_target_constraint",
    },
    "proposal": _COMMON_BELIEF_KEYS
    | {
        "ego_future_action",
        "branch_group_id",
        "branch_matched_action",
        "branch_valid_mask",
        "branch_target_reward",
        "branch_target_constraint",
    },
    "intention": _COMMON_BELIEF_KEYS
    | {"ego_future_action", "privileged_teammate_future_action"},
    "calibration": _COMMON_BELIEF_KEYS
    | {
        "ego_future_action",
        "privileged_teammate_future_action",
        "target_reward",
        "target_force_violation",
        "target_collision",
        "target_private_event_error",
    },
}
_STAGE_BATCH_KEYS["world_block"] = _STAGE_BATCH_KEYS["world_direct"]


def _select_stage_batch(batch, stage):
    required = _STAGE_BATCH_KEYS[stage]
    missing = required - set(batch)
    if missing:
        raise KeyError(f"{stage} batch misses {sorted(missing)}")
    return {name: batch[name] for name in required}


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {name: _to_device(item, device) for name, item in value.items()}
    return value


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _save_trainer_state(
    path,
    *,
    epoch,
    optimizer,
    precision,
    stopper,
    device,
    completed,
):
    payload = {
        "format": 1,
        "epoch": int(epoch),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": precision.scaler.state_dict(),
        "early_stop_best": float(stopper.best),
        "early_stop_bad_epochs": int(stopper.bad_epochs),
        "rng_state": _capture_rng_state(device),
        "completed": bool(completed),
    }
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _capture_rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, device):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _validate_resume_checkpoint(checkpoint, config):
    stored = checkpoint.get("training_config", {})
    current = asdict(config)
    ignored = {
        "output_dir",
        "plan_checkpoint",
        "belief_checkpoint",
        "world_block_checkpoint",
        "intention_checkpoint",
        "device",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "compile_model",
        "compile_mode",
        "epochs",
        "resume",
        "force_retrain",
    }
    mismatches = {
        key: {"stored": stored.get(key), "requested": value}
        for key, value in current.items()
        if key not in ignored and stored.get(key) != value
    }
    if mismatches:
        raise ValueError(f"cannot resume with changed training configuration: {mismatches}")
    completed_epochs = int(checkpoint.get("metrics", {}).get("epoch", 0))
    if config.epochs < completed_epochs:
        raise ValueError(
            f"requested epochs={config.epochs} is below completed epochs={completed_epochs}"
        )
    manifest = Path(config.train_dir).resolve().parent / "dataset_manifest.json"
    if checkpoint["dataset_manifest_sha256"] != sha256_file(manifest):
        raise ValueError("cannot resume after dataset_manifest.json changed")
    upstream_fields = {
        "plan": config.plan_checkpoint,
        "belief": config.belief_checkpoint,
        "world_block": config.world_block_checkpoint,
        "intention": config.intention_checkpoint,
    }
    for name, reference in checkpoint.get("upstream", {}).items():
        current_path = upstream_fields.get(name)
        if current_path is None or sha256_file(current_path) != reference.get("sha256"):
            raise ValueError(f"cannot resume after {name} upstream checkpoint changed")


def _validate_calibration_resume(
    checkpoint,
    config,
    *,
    manifest_hash: str,
    world_paths: Sequence[Path],
    world_hashes: list[str],
):
    if checkpoint["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("cannot reuse calibration after dataset manifest changed")
    stored = checkpoint.get("training_config", {})
    current = asdict(config)
    ignored = {
        "output_dir",
        "plan_checkpoint",
        "belief_checkpoint",
        "world_block_checkpoint",
        "world_ensemble_checkpoints",
        "intention_checkpoint",
        "device",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "compile_model",
        "compile_mode",
        "epochs",
        "resume",
        "force_retrain",
    }
    mismatches = {
        key: {"stored": stored.get(key), "requested": value}
        for key, value in current.items()
        if key not in ignored and stored.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"cannot reuse calibration with changed configuration: {mismatches}"
        )
    upstream_paths = {
        "plan": _required(config.plan_checkpoint, "plan checkpoint"),
        "belief": _required(config.belief_checkpoint, "belief checkpoint"),
        "world_block": world_paths[0],
        "intention": _required(config.intention_checkpoint, "intention checkpoint"),
    }
    upstream = checkpoint.get("upstream", {})
    for name, path in upstream_paths.items():
        if name not in upstream or upstream[name].get("sha256") != sha256_file(path):
            raise ValueError(f"cannot reuse calibration after {name} changed")
    for index, (path, expected_hash) in enumerate(zip(world_paths, world_hashes)):
        name = f"world_block_member_{index:02d}"
        if (
            sha256_file(path) != expected_hash
            or name not in upstream
            or upstream[name].get("sha256") != expected_hash
        ):
            raise ValueError("cannot reuse calibration after world ensemble changed")
    if checkpoint.get("extra", {}).get("world_ensemble_sha256") != world_hashes:
        raise ValueError("cannot reuse calibration after world ensemble changed")


def _cpu_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _required(value, label):
    if value is None:
        raise ValueError(f"{label} is required")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but torch.cuda.is_available() is false; "
            "check the NVIDIA driver/PyTorch CUDA build or use --device cpu only "
            "for a smoke test"
        )
    return device


def _configure_device(device, config):
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.tf32)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if config.tf32 else "highest")


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_dataset_compatibility(train_data, val_data):
    fields = ("history", "horizon", "local_history_dim", "action_dim", "spec")
    for name in fields:
        if getattr(train_data, name) != getattr(val_data, name):
            raise ValueError(f"train/validation Research-v2 contract differs for {name}")


def _active_code_mask(support: PlanCodeSupport, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(support.codebook_size, dtype=torch.bool, device=device)
    mask[support.active_codes.to(device)] = True
    return mask
