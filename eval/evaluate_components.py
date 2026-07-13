"""Held-out component evaluation for the decentralized FE-PC-WAM stack.

The evaluator deliberately mirrors the staged training forwards while keeping
all modules frozen.  Simulator-only values are used as loss targets, never as
belief/intention inputs.  The robust WAM is measured under oracle action-plan
conditioning so its dynamics errors are not conflated with intention errors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import torch
from torch.utils.data import DataLoader

from data.decentralized_dataset import DecentralizedTransitionDataset
from data.schema import (
    LEGACY_CONTACT_SEMANTICS,
    LEGACY_FORCE_SEMANTICS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
)
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
    require_plan_code_support,
)
from train.losses import compute_ego_wam_losses, compute_local_intention_losses


REPORT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ComponentEvaluationConfig:
    data_dir: str | Path
    plan_checkpoint: str | Path
    belief_checkpoint: str | Path
    wam_checkpoint: str | Path
    intention_checkpoint: str | Path
    split: str = "validation"
    batch_size: int = 64
    max_batches: int = -1
    device: str = "auto"
    num_workers: int = 0
    stride: int = 1
    max_episodes: int = -1
    ece_bins: int = 15
    output: str | Path | None = None

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")
        if self.batch_size <= 0 or self.num_workers < 0 or self.stride <= 0:
            raise ValueError("batch_size/stride must be positive and num_workers non-negative")
        if self.max_batches == 0 or self.max_batches < -1:
            raise ValueError("max_batches must be -1 or positive")
        if self.max_episodes == 0 or self.max_episodes < -1:
            raise ValueError("max_episodes must be -1 or positive")
        if self.ece_bins <= 0:
            raise ValueError("ece_bins must be positive")


class _WeightedScalars:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, int] = {}

    def add(self, values: Mapping[str, Any], names: tuple[str, ...], weight: int) -> None:
        for name in names:
            if name not in values:
                continue
            value = values[name]
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(f"metric {name!r} must be a scalar tensor")
            number = float(value.detach().cpu().item())
            if not math.isfinite(number):
                raise FloatingPointError(f"metric {name!r} is non-finite")
            self.sums[name] = self.sums.get(name, 0.0) + number * weight
            self.weights[name] = self.weights.get(name, 0) + weight

    def means(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.weights[name]
            for name in self.sums
            if self.weights[name] > 0
        }


def evaluate_components(config: ComponentEvaluationConfig) -> dict[str, Any]:
    """Evaluate frozen components on a validation or test dataset."""

    device = _resolve_device(config.device)
    paths = {
        "plan": Path(config.plan_checkpoint).resolve(),
        "belief": Path(config.belief_checkpoint).resolve(),
        "wam_robust": Path(config.wam_checkpoint).resolve(),
        "intention": Path(config.intention_checkpoint).resolve(),
    }
    initial_hashes = {name: file_sha256(path) for name, path in paths.items()}
    checkpoints, models, support, normalization = _load_stack(paths, device)
    held_out_evidence = _validate_held_out_directory(config, checkpoints)

    plan = models["plan"]
    belief = models["belief"]
    wam = models["wam_robust"]
    intention = models["intention"]
    assert isinstance(plan, ActionOnlyPlanTokenizer)
    assert isinstance(belief, LocalBeliefSlotEncoder)
    assert isinstance(wam, EgoLocalWAM)
    assert isinstance(intention, LocalIntentionPosterior)

    dataset = DecentralizedTransitionDataset(
        config.data_dir,
        history=belief.cfg.history,
        horizon=plan.cfg.horizon,
        stride=config.stride,
        max_episodes=config.max_episodes,
    )
    _validate_dataset_compatibility(dataset, plan, belief, wam, intention)
    _validate_dataset_sensor_contract(dataset, checkpoints)
    checkpoint_contact_semantics = str(
        checkpoints["plan"].get("dataset", {}).get(
            "local_contact_semantics", LEGACY_CONTACT_SEMANTICS
        )
    )
    checkpoint_force_semantics = str(
        checkpoints["plan"].get("dataset", {}).get(
            "local_force_semantics", LEGACY_FORCE_SEMANTICS
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    active_mask = torch.zeros(plan.cfg.codebook_size, dtype=torch.bool, device=device)
    active_mask[support.active_codes.to(device)] = True
    plan_counts = torch.zeros(plan.cfg.codebook_size, dtype=torch.long)
    plan_supported = 0
    reconstruction_square_sum = 0.0
    reconstruction_absolute_sum = 0.0
    reconstruction_normalized_square_sum = 0.0
    reconstruction_elements = 0

    plan_metrics = _WeightedScalars()
    belief_metrics = _WeightedScalars()
    wam_metrics = _WeightedScalars()
    intention_metrics = _WeightedScalars()
    plan_loss_names = (
        "loss",
        "loss_action",
        "loss_vq",
        "loss_usage_balance",
        "loss_residual",
        "loss_auxiliary_trajectory",
        "loss_maneuver",
        "soft_perplexity",
    )
    belief_loss_names = (
        "loss",
        "loss_aux_self_state",
        "loss_aux_object_pose",
        "loss_aux_teammate_pose",
        "loss_aux_task_progress",
        "loss_aux_event_maneuver",
    )
    wam_loss_names = (
        "loss",
        "loss_slots",
        "loss_ego_actions",
        "loss_privileged_teammate_actions",
        "loss_contact",
        "loss_force",
        "loss_progress",
        "loss_step_reward",
        "loss_return_quantiles",
        "loss_quantile_crossing",
        "loss_terminal_success",
        "loss_terminal_failure",
        "loss_collision_risk",
        "loss_force_violation_risk",
        "loss_completion_time",
        "loss_branch_ranking",
    )
    intention_loss_names = ("loss", "loss_code", "loss_residual_nll")

    intention_evaluated = 0
    intention_unsupported = 0
    intention_correct = 0
    intention_brier_sum = 0.0
    intention_confidence_sum = 0.0
    true_counts = torch.zeros(plan.cfg.codebook_size, dtype=torch.long)
    pred_counts = torch.zeros(plan.cfg.codebook_size, dtype=torch.long)
    true_positive = torch.zeros(plan.cfg.codebook_size, dtype=torch.long)
    calibration_count = torch.zeros(config.ece_bins, dtype=torch.long)
    calibration_confidence = torch.zeros(config.ece_bins, dtype=torch.float64)
    calibration_correct = torch.zeros(config.ece_bins, dtype=torch.float64)

    batches_evaluated = 0
    samples_evaluated = 0
    state_versions = _capture_state_versions(models)
    action_mean = normalization["action_mean"]
    action_std = normalization["action_std"]

    with torch.inference_mode():
        for raw_batch in loader:
            if config.max_batches > 0 and batches_evaluated >= config.max_batches:
                break
            batch = _to_device(raw_batch, device)
            batch_size = int(batch["ego_id"].shape[0])

            plan_losses = compute_action_only_plan_losses(
                plan,
                {
                    "actions": batch["ego_future_action"],
                    "maneuver": batch["target_maneuver"],
                },
                action_mean=action_mean,
                action_std=action_std,
            )
            plan_metrics.add(plan_losses, plan_loss_names, batch_size)
            ego_codes = plan_losses["code_indices"]
            plan_counts += torch.bincount(
                ego_codes.detach().cpu(), minlength=plan.cfg.codebook_size
            )
            plan_supported += int(active_mask[ego_codes].sum().item())

            normalized_actions = _normalize_actions(
                batch["ego_future_action"], action_mean, action_std
            )
            reconstructed_normalized = plan_losses["recon_actions"]
            reconstructed_actions = (
                reconstructed_normalized * action_std.view(1, 1, -1)
                + action_mean.view(1, 1, -1)
            )
            action_error = reconstructed_actions - batch["ego_future_action"]
            normalized_error = reconstructed_normalized - normalized_actions
            reconstruction_square_sum += float(action_error.square().sum().item())
            reconstruction_absolute_sum += float(action_error.abs().sum().item())
            reconstruction_normalized_square_sum += float(
                normalized_error.square().sum().item()
            )
            reconstruction_elements += int(action_error.numel())

            belief_targets = {
                name: value
                for name, value in _belief_privileged_targets(batch).items()
                if name in belief.cfg.privileged_aux_dims
            }
            belief_losses = compute_local_belief_auxiliary_losses(
                belief,
                _belief_forward_batch(batch),
                belief_targets,
            )
            belief_metrics.add(belief_losses, belief_loss_names, batch_size)

            beliefs = encode_current_and_future_beliefs(belief, batch)
            ego_plan = {
                "code_indices": ego_codes,
                "residual": plan_losses["residual"],
            }
            teammate_plan = plan.encode(
                _normalize_actions(
                    batch["privileged_teammate_future_action"], action_mean, action_std
                )
            )
            branch_actions = batch["branch_action"]
            branch_batch, branch_count, _, branch_horizon, branch_action_dim = (
                branch_actions.shape
            )
            encoded_branches = plan.encode(
                _normalize_actions(
                    branch_actions.reshape(
                        branch_batch * branch_count * 2,
                        branch_horizon,
                        branch_action_dim,
                    ),
                    action_mean,
                    action_std,
                )
            )
            wam_losses = compute_ego_wam_losses(
                wam,
                ego_slots=beliefs["ego_slots"],
                plan_codes=torch.stack(
                    [ego_plan["code_indices"], teammate_plan["code_indices"]], dim=1
                ),
                plan_residuals=torch.stack(
                    [ego_plan["residual"], teammate_plan["residual"]], dim=1
                ),
                teammate_hypothesis_weight=torch.ones(
                    batch_size, device=device, dtype=beliefs["ego_slots"].dtype
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
                branch_plan_codes=encoded_branches["code_indices"].reshape(
                    branch_batch, branch_count, 2
                ),
                branch_plan_residuals=encoded_branches["residual"].reshape(
                    branch_batch, branch_count, 2, -1
                ),
                branch_returns=batch["branch_return"],
                branch_valid=batch["branch_valid"],
            )
            wam_metrics.add(wam_losses, wam_loss_names, batch_size)

            supported_rows = active_mask[teammate_plan["code_indices"]]
            supported_count = int(supported_rows.sum().item())
            intention_unsupported += batch_size - supported_count
            if supported_count:
                metadata = torch.zeros(
                    supported_count,
                    intention.cfg.message_metadata_dim,
                    device=device,
                    dtype=beliefs["ego_slots"].dtype,
                )
                intention_losses = compute_local_intention_losses(
                    intention,
                    ego_slots=beliefs["ego_slots"][supported_rows],
                    ego_plan_code=ego_plan["code_indices"][supported_rows],
                    ego_plan_residual=ego_plan["residual"][supported_rows],
                    agent_id=batch["ego_id"][supported_rows],
                    received_message_metadata=metadata,
                    target_teammate_code=teammate_plan["code_indices"][supported_rows],
                    target_teammate_residual=teammate_plan["residual"][supported_rows],
                    active_code_mask=active_mask,
                )
                intention_metrics.add(
                    intention_losses, intention_loss_names, supported_count
                )
                probabilities = intention_losses["supported_code_probabilities"]
                targets = teammate_plan["code_indices"][supported_rows]
                confidence, predictions = probabilities.max(dim=-1)
                correct = predictions.eq(targets)
                one_hot = torch.nn.functional.one_hot(
                    targets, num_classes=plan.cfg.codebook_size
                ).to(probabilities.dtype)
                brier = (probabilities - one_hot).square().sum(dim=-1)

                intention_evaluated += supported_count
                intention_correct += int(correct.sum().item())
                intention_brier_sum += float(brier.sum().item())
                intention_confidence_sum += float(confidence.sum().item())
                true_counts += torch.bincount(
                    targets.detach().cpu(), minlength=plan.cfg.codebook_size
                )
                pred_counts += torch.bincount(
                    predictions.detach().cpu(), minlength=plan.cfg.codebook_size
                )
                matched = targets[correct]
                true_positive += torch.bincount(
                    matched.detach().cpu(), minlength=plan.cfg.codebook_size
                )
                bin_index = (confidence * config.ece_bins).long().clamp_max(
                    config.ece_bins - 1
                )
                calibration_count += torch.bincount(
                    bin_index.detach().cpu(), minlength=config.ece_bins
                )
                calibration_confidence.scatter_add_(
                    0,
                    bin_index.detach().cpu(),
                    confidence.detach().cpu().to(torch.float64),
                )
                calibration_correct.scatter_add_(
                    0,
                    bin_index.detach().cpu(),
                    correct.detach().cpu().to(torch.float64),
                )

            samples_evaluated += batch_size
            batches_evaluated += 1

    if batches_evaluated == 0 or samples_evaluated == 0:
        raise RuntimeError("component evaluation processed no held-out samples")
    _assert_state_unchanged(models, state_versions)
    final_hashes = {name: file_sha256(path) for name, path in paths.items()}
    if final_hashes != initial_hashes:
        raise RuntimeError("a checkpoint file changed during read-only evaluation")

    plan_usage = _code_usage(plan_counts)
    plan_result = {
        **plan_metrics.means(),
        "code_counts": plan_counts.tolist(),
        **plan_usage,
        "active_support_coverage": plan_supported / samples_evaluated,
        "action_reconstruction_mse": reconstruction_square_sum
        / reconstruction_elements,
        "action_reconstruction_mae": reconstruction_absolute_sum
        / reconstruction_elements,
        "action_reconstruction_mse_normalized": reconstruction_normalized_square_sum
        / reconstruction_elements,
    }
    intention_result: dict[str, Any] = {
        **intention_metrics.means(),
        "examples_total": samples_evaluated,
        "examples_evaluated": intention_evaluated,
        "skipped_unsupported_examples": intention_unsupported,
        "active_support_coverage": intention_evaluated / samples_evaluated,
        "accuracy": None,
        "macro_f1": None,
        "macro_f1_active_support": None,
        "brier_score": None,
        "ece": None,
        "ece_bins": config.ece_bins,
        "mean_confidence": None,
    }
    if intention_evaluated:
        false_positive = pred_counts - true_positive
        false_negative = true_counts - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = torch.where(
            denominator > 0,
            2.0 * true_positive.to(torch.float64) / denominator.clamp_min(1),
            torch.zeros_like(denominator, dtype=torch.float64),
        )
        observed_classes = denominator > 0
        active_codes = support.active_codes.to(dtype=torch.long)
        nonempty_bins = calibration_count > 0
        bin_accuracy = calibration_correct[nonempty_bins] / calibration_count[
            nonempty_bins
        ]
        bin_confidence = calibration_confidence[nonempty_bins] / calibration_count[
            nonempty_bins
        ]
        bin_weight = calibration_count[nonempty_bins].to(torch.float64) / float(
            intention_evaluated
        )
        intention_result.update(
            {
                "accuracy": intention_correct / intention_evaluated,
                "macro_f1": float(f1[observed_classes].mean().item()),
                "macro_f1_active_support": float(f1[active_codes].mean().item()),
                "brier_score": intention_brier_sum / intention_evaluated,
                "ece": float(
                    (bin_weight * (bin_accuracy - bin_confidence).abs()).sum().item()
                ),
                "mean_confidence": intention_confidence_sum / intention_evaluated,
            }
        )

    report: dict[str, Any] = {
        "report_format_version": REPORT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_tag": CONTRACT_TAG,
        "schema_version": SCHEMA_VERSION,
        "evaluation": {
            **asdict(config),
            "data_dir": str(Path(config.data_dir).resolve()),
            "plan_checkpoint": str(paths["plan"]),
            "belief_checkpoint": str(paths["belief"]),
            "wam_checkpoint": str(paths["wam_robust"]),
            "intention_checkpoint": str(paths["intention"]),
            "output": str(Path(config.output).resolve()) if config.output else None,
            "resolved_device": str(device),
            "batches_evaluated": batches_evaluated,
            "samples_evaluated": samples_evaluated,
            "dataset_truncated_by_max_batches": samples_evaluated < len(dataset),
            "models_in_eval_mode": True,
            "gradients_enabled": False,
            "weights_unchanged": True,
        },
        "dataset": {
            "split": config.split,
            "held_out_from_checkpoint_training_data": held_out_evidence["verified"],
            "held_out_verification": held_out_evidence,
            "episode_count": len(dataset.paths),
            "window_count": len(dataset),
            "history": dataset.history,
            "horizon": dataset.horizon,
            "stride": dataset.stride,
            "ego_ids": list(dataset.ego_ids),
            "action_dim": int(dataset.action_dim),
            "local_history_dim": dataset.local_history_dim,
            "deployable_input_keys": sorted(dataset.INPUT_KEYS),
            "local_contact_semantics": dataset.local_contact_semantics,
            "local_force_semantics": dataset.local_force_semantics,
            "local_force_units": dataset.local_force_units,
            "local_force_scale_newtons": dataset.local_force_scale_newtons,
            "local_sensor_provenance": dataset.local_sensor_provenance,
        },
        "checkpoints": {
            name: {
                "path": str(paths[name]),
                "sha256": initial_hashes[name],
                "stage": checkpoints[name]["stage"],
            }
            for name in paths
        },
        "protocol": {
            "plan_encoder_input": "normalized ego future actions only",
            "belief_forward_inputs": "ego-local deployable history and local object estimate",
            "wam_conditioning": "oracle action-only ego and teammate plan labels",
            "intention_message_metadata": "all-zero (no previously received message)",
            "intention_metric_population": "held-out teammate targets within training active-code support",
            "local_contact_semantics": {
                "held_out_dataset": dataset.local_contact_semantics,
                "checkpoint_training_data": checkpoint_contact_semantics,
                "strict_local": checkpoint_contact_semantics
                == STRICT_LOCAL_CONTACT_SEMANTICS,
            },
            "local_force_semantics": {
                "held_out_dataset": dataset.local_force_semantics,
                "checkpoint_training_data": checkpoint_force_semantics,
                "strict_local": checkpoint_force_semantics
                == STRICT_LOCAL_FORCE_SEMANTICS,
                "held_out_scale_newtons": dataset.local_force_scale_newtons,
                "checkpoint_scale_newtons": checkpoints["plan"].get(
                    "dataset", {}
                ).get("local_force_scale_newtons"),
            },
        },
        "plan": plan_result,
        "belief": belief_metrics.means(),
        "wam_robust": wam_metrics.means(),
        "intention": intention_result,
    }
    if config.output:
        _write_json_atomic(Path(config.output), report)
    return report


def _load_stack(
    paths: Mapping[str, Path], device: torch.device
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, torch.nn.Module],
    PlanCodeSupport,
    dict[str, torch.Tensor],
]:
    checkpoints = {
        "plan": load_checkpoint(paths["plan"], expected_stage="plan"),
        "belief": load_checkpoint(paths["belief"], expected_stage="belief"),
        "wam_robust": load_checkpoint(
            paths["wam_robust"], expected_stage="wam_robust"
        ),
        "intention": load_checkpoint(
            paths["intention"], expected_stage="intention"
        ),
    }
    _validate_lineage(checkpoints, paths)
    support = PlanCodeSupport.from_dict(
        require_plan_code_support(checkpoints["plan"])
    )
    plan = ActionOnlyPlanTokenizer(
        ActionOnlyPlanTokenizerConfig(**checkpoints["plan"]["model_config"])
    ).to(device)
    belief = LocalBeliefSlotEncoder(
        LocalBeliefSlotEncoderConfig(**checkpoints["belief"]["model_config"])
    ).to(device)
    wam = EgoLocalWAM(
        EgoLocalWAMConfig(**checkpoints["wam_robust"]["model_config"])
    ).to(device)
    intention = LocalIntentionPosterior(
        LocalIntentionConfig(**checkpoints["intention"]["model_config"])
    ).to(device)
    models: dict[str, torch.nn.Module] = {
        "plan": plan,
        "belief": belief,
        "wam_robust": wam,
        "intention": intention,
    }
    for name, model in models.items():
        model.load_state_dict(checkpoints[name]["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    normalization = _normalization_to_device(checkpoints["plan"], plan, device)
    _validate_stack_compatibility(plan, belief, wam, intention, support)
    return checkpoints, models, support, normalization


def _validate_lineage(
    checkpoints: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path]
) -> None:
    for checkpoint_name, upstream_name in (
        ("belief", "plan"),
        ("intention", "plan"),
        ("intention", "belief"),
        ("wam_robust", "plan"),
        ("wam_robust", "belief"),
        ("wam_robust", "intention"),
    ):
        reference = checkpoints[checkpoint_name].get("upstream", {}).get(upstream_name)
        if not isinstance(reference, Mapping):
            raise ValueError(
                f"{checkpoint_name} checkpoint is missing {upstream_name} lineage"
            )
        actual = file_sha256(paths[upstream_name])
        if reference.get("sha256") != actual:
            raise ValueError(
                f"{checkpoint_name} was trained against a different {upstream_name} "
                "checkpoint"
            )
    intention_wam = checkpoints["intention"].get("upstream", {}).get("wam")
    robust_wam = checkpoints["wam_robust"].get("upstream", {}).get("wam")
    if not isinstance(intention_wam, Mapping) or not isinstance(robust_wam, Mapping):
        raise ValueError("intention/wam_robust checkpoints are missing base-WAM lineage")
    if intention_wam.get("sha256") != robust_wam.get("sha256"):
        raise ValueError("intention and wam_robust do not share the same base WAM")


def _validate_held_out_directory(
    config: ComponentEvaluationConfig,
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evaluation_dir = Path(config.data_dir).resolve()
    split_key = "val" if config.split == "validation" else "test"
    manifest_path = evaluation_dir.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"held-out evaluation requires dataset manifest {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("held-out dataset manifest has the wrong schema version")
    split_state = manifest.get("splits", {}).get(split_key)
    if not isinstance(split_state, Mapping):
        raise ValueError(f"dataset manifest has no {split_key!r} split")
    declared_path = split_state.get("path")
    if not isinstance(declared_path, str) or Path(declared_path).resolve() != evaluation_dir:
        raise ValueError(
            f"data_dir is not the manifest-declared {split_key} split"
        )
    episode_paths = sorted(evaluation_dir.glob("episode_*.hdf5"))
    declared_episodes = int(split_state.get("episodes", -1))
    if declared_episodes != len(episode_paths):
        raise ValueError(
            "held-out episode count differs from the dataset manifest"
        )
    for path in episode_paths:
        with h5py.File(path, "r") as file:
            actual_split = str(file["metadata"].attrs.get("split", ""))
        if actual_split != split_key:
            raise ValueError(
                f"{path} metadata split {actual_split!r} != {split_key!r}"
            )
    episode_hashes = [
        {"name": path.name, "sha256": file_sha256(path)} for path in episode_paths
    ]
    episode_set_sha256 = hashlib.sha256(
        json.dumps(
            episode_hashes, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    for name, checkpoint in checkpoints.items():
        dataset_metadata = checkpoint.get("dataset")
        training_dir = (
            dataset_metadata.get("data_dir")
            if isinstance(dataset_metadata, Mapping)
            else None
        )
        if not isinstance(training_dir, str) or not training_dir:
            raise ValueError(
                f"{name} checkpoint has no training data_dir; held-out status cannot be verified"
            )
        if Path(training_dir).resolve() == evaluation_dir:
            raise ValueError(
                f"evaluation data_dir matches the {name} checkpoint training data; "
                "use a validation or test split"
            )
    return {
        "verified": True,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_split": split_key,
        "declared_path": str(evaluation_dir),
        "episode_count": len(episode_paths),
        "episode_set_sha256": episode_set_sha256,
        "episode_metadata_split_verified": True,
    }


def _validate_dataset_sensor_contract(
    dataset: DecentralizedTransitionDataset,
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        "local_contact_semantics": dataset.local_contact_semantics,
        "local_force_semantics": dataset.local_force_semantics,
        "local_force_units": dataset.local_force_units,
        "local_force_scale_newtons": dataset.local_force_scale_newtons,
        "local_sensor_provenance": dataset.local_sensor_provenance,
    }
    for name, checkpoint in checkpoints.items():
        metadata = checkpoint.get("dataset")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{name} checkpoint lacks dataset contract metadata")
        actual = {
            "local_contact_semantics": metadata.get(
                "local_contact_semantics", LEGACY_CONTACT_SEMANTICS
            ),
            "local_force_semantics": metadata.get(
                "local_force_semantics", LEGACY_FORCE_SEMANTICS
            ),
            "local_force_units": metadata.get("local_force_units"),
            "local_force_scale_newtons": metadata.get(
                "local_force_scale_newtons"
            ),
            "local_sensor_provenance": metadata.get(
                "local_sensor_provenance"
            ),
        }
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if key == "local_force_scale_newtons" and expected_value is not None:
                matches = actual_value is not None and math.isclose(
                    float(actual_value), float(expected_value)
                )
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(
                    f"{name} checkpoint and held-out dataset differ for {key}"
                )


def _validate_stack_compatibility(
    plan: ActionOnlyPlanTokenizer,
    belief: LocalBeliefSlotEncoder,
    wam: EgoLocalWAM,
    intention: LocalIntentionPosterior,
    support: PlanCodeSupport,
) -> None:
    if support.codebook_size != plan.cfg.codebook_size:
        raise ValueError("plan support and tokenizer codebook sizes differ")
    if wam.cfg.horizon != plan.cfg.horizon:
        raise ValueError("robust WAM and tokenizer horizons differ")
    if wam.cfg.plan_codebook_size != plan.cfg.codebook_size:
        raise ValueError("robust WAM and tokenizer codebook sizes differ")
    if intention.cfg.plan_codebook_size != plan.cfg.codebook_size:
        raise ValueError("intention and tokenizer codebook sizes differ")
    if wam.cfg.plan_latent_dim != plan.cfg.latent_dim:
        raise ValueError("robust WAM and tokenizer latent dimensions differ")
    if intention.cfg.plan_latent_dim != plan.cfg.latent_dim:
        raise ValueError("intention and tokenizer latent dimensions differ")
    expected_slots = (LocalBeliefSlotEncoder.NUM_ROLES, belief.cfg.slot_dim)
    if (wam.cfg.slots_per_agent, wam.cfg.slot_dim) != expected_slots:
        raise ValueError("robust WAM and belief slot interfaces differ")
    if (intention.cfg.slots_per_agent, intention.cfg.slot_dim) != expected_slots:
        raise ValueError("intention and belief slot interfaces differ")


def _validate_dataset_compatibility(
    dataset: DecentralizedTransitionDataset,
    plan: ActionOnlyPlanTokenizer,
    belief: LocalBeliefSlotEncoder,
    wam: EgoLocalWAM,
    intention: LocalIntentionPosterior,
) -> None:
    if dataset.horizon != plan.cfg.horizon:
        raise ValueError("dataset and plan checkpoint horizons differ")
    if int(dataset.action_dim) != plan.cfg.action_dim:
        raise ValueError("dataset and plan checkpoint action dimensions differ")
    if dataset.history != belief.cfg.history:
        raise ValueError("dataset and belief checkpoint history lengths differ")
    if dataset.local_history_dim != belief.cfg.local_dim:
        raise ValueError("dataset and belief checkpoint local input widths differ")
    if belief.cfg.object_dim != 3:
        raise ValueError(" held-out dataset requires the three-dimensional object estimate")
    if wam.cfg.action_dim_per_agent != int(dataset.action_dim):
        raise ValueError("dataset and robust WAM action dimensions differ")
    if intention.cfg.message_metadata_dim < 0:
        raise ValueError("intention message metadata width is invalid")


def _normalization_to_device(
    checkpoint: Mapping[str, Any],
    plan: ActionOnlyPlanTokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("plan checkpoint is missing action normalization")
    try:
        mean = torch.as_tensor(
            normalization["action_mean"], device=device, dtype=torch.float32
        )
        std = torch.as_tensor(
            normalization["action_std"], device=device, dtype=torch.float32
        )
    except KeyError as exc:
        raise ValueError("plan checkpoint is missing action normalization") from exc
    if mean.shape != (plan.cfg.action_dim,) or std.shape != mean.shape:
        raise ValueError("plan checkpoint action normalization has the wrong shape")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("plan checkpoint action normalization must be finite and positive")
    return {"action_mean": mean, "action_std": std}


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


def _belief_privileged_targets(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    local_width = batch["future_model_observation"].shape[-1]
    if local_width < 17 or (local_width - 17) % 3 != 0:
        raise ValueError("future_model_observation has an invalid  sensor layout")
    self_state_dim = 3 + local_width - 17
    return {
        "self_state": batch["future_model_observation"][:, 0, :self_state_dim],
        "object_pose": batch["target_object_pose_ego"][:, 0],
        "teammate_pose": batch["target_teammate_pose_ego"][:, 0],
        "task_progress": batch["target_progress"][:, :1],
        "event_maneuver": torch.nn.functional.one_hot(
            batch["target_maneuver"], num_classes=3
        ).to(torch.float32),
    }


def _normalize_actions(
    actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (actions - mean.view(1, 1, -1)) / std.view(1, 1, -1).clamp_min(1e-6)


def _code_usage(counts: torch.Tensor) -> dict[str, Any]:
    total = int(counts.sum().item())
    if total <= 0:
        raise RuntimeError("cannot summarize an empty plan-code evaluation")
    probabilities = counts.to(torch.float64) / total
    nonzero = probabilities > 0
    entropy = float(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum().item()
    )
    used = int(nonzero.sum().item())
    return {
        "encoded_segments": total,
        "used_codes": used,
        "usage_ratio": used / counts.numel(),
        "entropy": entropy,
        "perplexity": math.exp(entropy),
    }


def _capture_state_versions(
    models: Mapping[str, torch.nn.Module],
) -> dict[tuple[str, str], int]:
    versions: dict[tuple[str, str], int] = {}
    for model_name, model in models.items():
        for state_name, tensor in model.state_dict(keep_vars=True).items():
            versions[(model_name, state_name)] = tensor._version
    return versions


def _assert_state_unchanged(
    models: Mapping[str, torch.nn.Module], versions: Mapping[tuple[str, str], int]
) -> None:
    for model_name, model in models.items():
        for state_name, tensor in model.state_dict(keep_vars=True).items():
            if tensor._version != versions[(model_name, state_name)]:
                raise RuntimeError(
                    f"{model_name}.{state_name} changed during read-only evaluation"
                )


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {name: _to_device(item, device) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen FE-PC-WAM components on held-out data"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--plan-checkpoint", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--wam-checkpoint", required=True, help="wam_robust.pt")
    parser.add_argument("--intention-checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--output")
    return parser


def config_from_arguments(args: argparse.Namespace) -> ComponentEvaluationConfig:
    return ComponentEvaluationConfig(**vars(args))


def main() -> None:
    config = config_from_arguments(build_argument_parser().parse_args())
    report = evaluate_components(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
