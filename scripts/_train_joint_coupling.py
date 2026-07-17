"""Internal joint-coupling stage used by the Joint WAM training entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from eval.joint_wam_training import (  # noqa: E402
    evaluate_joint_wam_offline,
    joint_wam_offline_acceptance_report,
)
from train.action_prior import load_action_prior_checkpoint  # noqa: E402
from train.action_flow_checkpointing import load_action_flow_checkpoint  # noqa: E402
from train.joint_wam import (  # noqa: E402
    JointWAMTrainConfig,
    train_joint_wam_stage,
)
from train.joint_wam_checkpointing import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    GENERATED_ACTION_WORLD_TARGET_SOURCE,
    load_joint_wam_checkpoint,
    save_joint_wam_checkpoint,
)
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_ar_losses import RWMLossWeights  # noqa: E402
from train.rwm_u_checkpointing import load_rwm_u_member_checkpoint  # noqa: E402
from train.trajectory_dataset import InMemoryProprioSequenceDataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam/joint_wam.yaml",
    )
    parser.add_argument("--world-model-checkpoint-dir", type=Path)
    parser.add_argument("--action-prior-checkpoint-dir", type=Path)
    parser.add_argument("--action-flow-checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-eval-batches", type=int, default=-1)
    parser.add_argument("--max-episodes-per-split", type=int, default=-1)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    settings = _settings(config, args)
    formal_run = _formal_run(args)
    formal_checkpoint = (ROOT / config["checkpoint"]["directory"]).resolve()
    if not formal_run and settings["checkpoint_dir"] == formal_checkpoint:
        raise ValueError(
            "debug limits may not write the formal checkpoint; use a separate "
            "--checkpoint-dir"
        )
    device = _device(args.device)
    _seed(settings["seed"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    parent_manifest = settings["world_model_checkpoint"] / "dataset_manifest.json"
    if _sha256(settings["split_manifest"]) != _sha256(parent_manifest):
        raise ValueError(
            "training data split must match the initialization manifest"
        )

    source_before = _source_fingerprints(settings)
    joint, world_model_metadata = load_rwm_u_member_checkpoint(
        settings["world_model_checkpoint"],
        0,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    frozen_teacher, frozen_metadata = load_rwm_u_member_checkpoint(
        settings["world_model_checkpoint"],
        0,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    if (
        frozen_metadata["normalization"].sha256()
        != world_model_metadata["normalization"].sha256()
    ):
        raise RuntimeError("Joint and frozen-teacher normalization differs")
    flow, j1_metadata = load_action_flow_checkpoint(
        settings["action_flow_checkpoint"],
        world_model_checkpoint=settings["world_model_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    if settings["require_action_flow_passed"] and not bool(
        j1_metadata["metrics"].get("passed", False)
    ):
        raise RuntimeError("action-flow warm-up checks have not passed")
    action_prior, _ = load_action_prior_checkpoint(
        settings["action_prior_checkpoint"],
        world_model_checkpoint=settings["world_model_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
        expected_normalization_sha256=world_model_metadata["normalization"].sha256(),
    )
    if int(config["initialization"]["member_index"]) != 0:
        raise ValueError("Joint WAM is locked to member_index=0")
    if bool(config["initialization"]["online_load_ensemble"]):
        raise ValueError("Joint WAM may not load the ensemble online")
    anchor_prior_delta_at_start = _maximum_delta(
        action_prior.state_dict(), flow.anchor_prior.state_dict()
    )
    if anchor_prior_delta_at_start != 0.0:
        raise RuntimeError("embedded anchor differs from the accepted action prior")

    joint_initial = _cpu_state_dict(joint)
    teacher_initial = _cpu_state_dict(frozen_teacher)
    flow_initial = _cpu_state_dict(flow)
    anchor_initial = _cpu_state_dict(flow.anchor_prior)
    partitions = _manifest_partitions(settings["split_manifest"])
    if args.max_episodes_per_split > 0:
        partitions = {
            name: paths[: args.max_episodes_per_split]
            for name, paths in partitions.items()
        }
    if any(not partitions[name] for name in ("train", "validation", "test")):
        raise RuntimeError("Joint WAM requires non-empty manifest partitions")

    stage_configs = _stage_configs(config, settings, args.max_steps)
    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=3 + len(stage_configs) + 2 + 1,
    ) as progress:
        datasets: dict[str, InMemoryProprioSequenceDataset] = {}
        complete: dict[str, np.ndarray] = {}
        selected: dict[str, np.ndarray] = {}
        for name in ("train", "validation", "test"):
            phase = progress.add_phase(f"preload {name} chunks", len(partitions[name]))
            datasets[name] = InMemoryProprioSequenceDataset(
                paths=partitions[name],
                history_horizon=joint.config.history_horizon,
                forecast_horizon=flow.config.horizon,
                state_dim=joint.config.state_dim,
                action_dim=joint.config.action_dim,
                allow_legacy_wam=False,
                planning_discount=settings["quality_discount"],
                action_prior_behavior_weights=settings["behavior_weights"],
                action_prior_require_success=settings["require_success"],
                action_prior_min_return_quantile=(
                    settings["min_return_quantile"] if name == "train" else 0.0
                ),
                progress=phase.advance,
            )
            complete[name] = datasets[name].complete_forecast_indices()
            selected[name] = datasets[name].complete_forecast_indices(
                require_positive_action_quality=True
            )
            phase.finish(
                f"{len(complete[name])} complete / {len(selected[name])} action-supervised"
            )
        # World/risk objectives consume every complete chunk.  Action losses use
        # action_quality_weights inside the trainer and therefore still exclude
        # failed/low-quality behavior from policy supervision.
        train_loader = DataLoader(
            Subset(datasets["train"], complete["train"].tolist()),
            batch_size=settings["batch_size"],
            shuffle=True,
            num_workers=settings["num_workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=settings["num_workers"] > 0,
        )
        evaluation_loaders = {
            name: DataLoader(
                Subset(datasets[name], complete[name].tolist()),
                batch_size=settings["batch_size"],
                shuffle=False,
                num_workers=settings["num_workers"],
                pin_memory=device.type == "cuda",
                persistent_workers=settings["num_workers"] > 0,
            )
            for name in ("validation", "test")
        }
        positive_weights = _positive_weights(settings["world_model_checkpoint"])
        history: list[dict[str, Any]] = []
        completed_steps = 0
        stage_summaries: list[dict[str, Any]] = []
        for stage_index, stage_config in enumerate(stage_configs):
            phase = progress.add_phase(
                f"train Joint WAM {stage_config.scope}", stage_config.max_steps
            )
            stage_history, stage_steps = train_joint_wam_stage(
                flow,
                joint,
                frozen_teacher,
                train_loader,
                device=device,
                config=stage_config,
                seed=settings["seed"] + stage_index,
                positive_weights=positive_weights,
                progress=phase.advance,
            )
            for item in stage_history:
                item["stage_index"] = stage_index
                item["global_step"] = completed_steps + int(item["step"])
            history.extend(stage_history)
            completed_steps += stage_steps
            final = stage_history[-1] if stage_history else {}
            stage_summaries.append(
                {
                    "name": settings["stages"][stage_index]["name"],
                    "scope": stage_config.scope,
                    "steps": stage_steps,
                    "final_loss": final.get("loss"),
                    "final_world_loss": final.get("world_loss"),
                    "final_generated_consistency": final.get(
                        "generated_consistency_loss"
                    ),
                }
            )
            phase.finish(
                f"{stage_steps} steps, final loss {float(final.get('loss', 0.0)):.5f}"
            )

        offline: dict[str, Any] = {}
        for name in ("validation", "test"):
            maximum = (
                min(len(evaluation_loaders[name]), args.max_eval_batches)
                if args.max_eval_batches > 0
                else len(evaluation_loaders[name])
            )
            phase = progress.add_phase(f"evaluate {name} Joint WAM", maximum)
            offline[name] = evaluate_joint_wam_offline(
                joint,
                frozen_teacher,
                flow,
                evaluation_loaders[name],
                device=device,
                execution_steps=int(config["action_chunk"]["execution_steps"]),
                solver_steps=int(config["action_chunk"]["solver_steps"]),
                solver=str(config["action_chunk"]["solver"]),
                anchor_residual_scale=float(
                    config["runtime"]["anchor_residual_scale"]
                ),
                normalized_action_clip=float(
                    config["runtime"]["normalized_action_clip"]
                ),
                fixed_actions={
                    int(index): float(value)
                    for index, value in config["runtime"]["fixed_actions"].items()
                },
                max_batches=args.max_eval_batches,
                progress=phase.advance,
            )
            phase.finish(
                f"world NRMSE {offline[name]['expert_action_world_state_nrmse']:.5f}"
            )

        joint_delta = _maximum_delta(joint_initial, joint.state_dict())
        shared_history_delta = _named_parameter_delta(
            joint_initial,
            joint.state_dict(),
            prefixes=("features.", "transition_encoder.", "belief_gru."),
        )
        world_delta = _named_parameter_delta(
            joint_initial,
            joint.state_dict(),
            prefixes=("decoder.", "heads."),
        )
        teacher_delta = _maximum_delta(teacher_initial, frozen_teacher.state_dict())
        flow_delta = _maximum_delta(flow_initial, flow.state_dict())
        anchor_delta = _maximum_delta(anchor_initial, flow.anchor_prior.state_dict())
        branch_gradient_maxima = {
            name: max((float(item.get(name, 0.0)) for item in history), default=0.0)
            for name in (
                "action_to_flow_gradient_norm",
                "action_to_backbone_gradient_norm",
                "world_to_flow_gradient_norm",
                "world_to_backbone_gradient_norm",
                "consistency_to_flow_gradient_norm",
                "consistency_to_backbone_gradient_norm",
            )
        }
        source_immutable = source_before == _source_fingerprints(settings)
        metrics: dict[str, Any] = {
            "format_version": "wam.joint_wam.metrics/1",
            "model": "joint_wam",
            "completed_steps": completed_steps,
            "stage_summaries": stage_summaries,
            "loss_history": history,
            "branch_gradient_maxima": branch_gradient_maxima,
            "offline": offline,
            "data": {
                name: {
                    "episodes": len(partitions[name]),
                    "complete_chunks": int(len(complete[name])),
                    "action_supervised_chunks": int(len(selected[name])),
                }
                for name in datasets
            },
            "member_0_parameter_delta": joint_delta,
            "shared_history_parameter_delta": shared_history_delta,
            "world_parameter_delta": world_delta,
            "action_flow_parameter_delta": flow_delta,
            "frozen_teacher_parameter_delta": teacher_delta,
            "anchor_prior_parameter_delta": anchor_delta,
            "anchor_prior_delta_at_start": anchor_prior_delta_at_start,
            "source_checkpoints_immutable": source_immutable,
            "generated_action_world_target_source": (
                GENERATED_ACTION_WORLD_TARGET_SOURCE
            ),
            "generated_action_demo_state_is_ground_truth": False,
            "online_loaded_member_indices": [0],
            "online_ensemble_loaded": False,
            "formal_run": formal_run,
            "debug_limits": {
                "max_steps": args.max_steps,
                "max_eval_batches": args.max_eval_batches,
                "max_episodes_per_split": args.max_episodes_per_split,
                "batch_size_override": args.batch_size,
            },
            "effective_batch_size": settings["batch_size"],
            "effective_num_workers": settings["num_workers"],
            "num_workers_override": args.num_workers,
        }
        manifest = {
            "source": str(settings["split_manifest"]),
            "split_seed": int(config["data"]["split_seed"]),
            "partitions": {
                name: [str(path.resolve()) for path in paths]
                for name, paths in partitions.items()
            },
            "partition_seeds": {
                name: sorted({int(seed) for seed in datasets[name].episode_seeds})
                for name in datasets
            },
            "complete_action_chunk_only": True,
            "action_supervision_uses_quality_weights": True,
            "world_supervision_uses_all_complete_chunks": True,
            "generated_action_world_target_source": (
                GENERATED_ACTION_WORLD_TARGET_SOURCE
            ),
            "action_flow_on_policy_seeds": sorted(
                {
                    int(seed)
                    for seed in j1_metadata["dataset_manifest"].get(
                        "action_flow_on_policy_seeds", []
                    )
                }
            ),
            "generated_or_relabel_seeds": [],
            "smoke_subset": args.max_episodes_per_split > 0,
        }
        metrics["action_flow_warmup"] = j1_metadata["metrics"]
        phase = progress.add_phase("save and reload Joint WAM", 2)
        save_joint_wam_checkpoint(
            settings["checkpoint_dir"],
            joint,
            flow,
            world_model_metadata["normalization"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            source_fingerprints=source_before,
        )
        phase.advance({"batch": 1})
        reloaded_joint, reloaded_flow, _ = load_joint_wam_checkpoint(
            settings["checkpoint_dir"],
            device=device,
            expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
        )
        joint_reload_difference = _maximum_delta(
            joint.state_dict(), reloaded_joint.state_dict()
        )
        flow_reload_difference = _maximum_delta(
            flow.state_dict(), reloaded_flow.state_dict()
        )
        reload_difference = max(joint_reload_difference, flow_reload_difference)
        limits = config["offline_acceptance"]
        split_acceptance = {
            name: joint_wam_offline_acceptance_report(
                split_metrics,
                member_0_parameter_delta=joint_delta,
                shared_history_parameter_delta=shared_history_delta,
                world_parameter_delta=world_delta,
                action_flow_parameter_delta=flow_delta,
                anchor_prior_parameter_delta=anchor_delta,
                frozen_teacher_parameter_delta=teacher_delta,
                source_checkpoints_immutable=source_immutable,
                checkpoint_reload_max_abs_diff=reload_difference,
                branch_gradient_maxima=branch_gradient_maxima,
                maximum_expert_world_nrmse=float(
                    limits["maximum_expert_world_nrmse"]
                ),
                maximum_generated_teacher_state_nrmse=float(
                    limits["maximum_generated_teacher_state_nrmse"]
                ),
            )
            for name, split_metrics in offline.items()
        }
        formal_checks = _formal_acceptance_checks(
            args,
            completed_steps=completed_steps,
            configured_steps=sum(
                int(stage["steps"]) for stage in settings["stages"]
            ),
            smoke_subset=bool(manifest["smoke_subset"]),
        )
        acceptance = {
            "format_version": "wam.joint_wam.offline_acceptance/1",
            "model": "joint_wam",
            "passed": all(item["passed"] for item in split_acceptance.values())
            and all(formal_checks.values()),
            "splits": split_acceptance,
            "formal_checks": formal_checks,
        }
        metrics["checkpoint_reload"] = {
            "joint_member_max_abs_diff": joint_reload_difference,
            "action_flow_max_abs_diff": flow_reload_difference,
            "max_abs_diff": reload_difference,
            "strict": True,
        }
        metrics["offline_acceptance"] = acceptance
        metrics["passed"] = bool(acceptance["passed"])
        save_joint_wam_checkpoint(
            settings["checkpoint_dir"],
            joint,
            flow,
            world_model_metadata["normalization"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            source_fingerprints=source_before,
        )
        phase.advance({"batch": 2})
        phase.finish(f"reload max diff {reload_difference:.3g}")
    summary = {
        "checkpoint": str(settings["checkpoint_dir"]),
        "passed": metrics["passed"],
        "completed_steps": completed_steps,
        "test": offline["test"],
        "member_0_parameter_delta": joint_delta,
        "shared_history_parameter_delta": shared_history_delta,
        "world_parameter_delta": world_delta,
        "action_flow_parameter_delta": flow_delta,
        "frozen_teacher_parameter_delta": teacher_delta,
        "anchor_prior_parameter_delta": anchor_delta,
        "branch_gradient_maxima": branch_gradient_maxima,
        "checkpoint_reload_max_abs_diff": reload_difference,
        "offline_acceptance": acceptance,
    }
    print(json.dumps(summary, indent=2))
    return 0 if metrics["passed"] else 2


def _stage_configs(
    config: Mapping[str, Any], settings: Mapping[str, Any], max_steps: int
) -> list[JointWAMTrainConfig]:
    training = config["joint_training"]
    losses = training["loss_weights"]
    raw_stages = settings["stages"]
    ratio_schedule = ((0.0, 0.25), (0.25, 0.50), (0.50, 1.0))
    remaining = max_steps
    result: list[JointWAMTrainConfig] = []
    for index, raw in enumerate(raw_stages):
        steps = int(raw["steps"])
        if max_steps > 0:
            steps = min(steps, remaining)
            remaining -= steps
        if steps <= 0:
            break
        ratio_start, ratio_end = ratio_schedule[min(index, len(ratio_schedule) - 1)]
        result.append(
            JointWAMTrainConfig(
                scope=str(raw["scope"]),
                epochs=1,
                flow_learning_rate=float(raw["action_learning_rate"]),
                member_learning_rate=float(raw["member_learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                flow_gradient_clip_norm=float(training["gradient_clip_norm"]),
                member_gradient_clip_norm=float(training["gradient_clip_norm"]),
                use_amp=bool(training["use_amp"]),
                max_steps=steps,
                warm_start_probability=float(training["warm_start_probability"]),
                warm_start_noise_std=float(training["warm_start_noise_std"]),
                cold_noise_std=float(training["cold_noise_std"]),
                cold_zero_probability=float(training["cold_zero_probability"]),
                execution_steps=int(config["action_chunk"]["execution_steps"]),
                action_endpoint_weight=float(losses["action_endpoint"]),
                action_smoothness_weight=float(losses["action_smoothness"]),
                world_horizon=int(config["action_chunk"]["horizon"]),
                world_horizon_decay=float(training["horizon_decay"]),
                world_loss_weights=RWMLossWeights(
                    state_mean=float(losses["state_mean"]),
                    state_nll=float(losses["state_nll"]),
                    gripper_closed=float(losses["gripper_closed"]),
                    reward=float(losses["reward"]),
                    done=float(losses["done"]),
                    terminal=float(losses["terminal"]),
                    auxiliary=float(losses["auxiliary"]),
                ),
                solver_steps=int(config["action_chunk"]["solver_steps"]),
                solver=str(config["action_chunk"]["solver"]),
                normalized_action_clip=float(training["normalized_action_clip"]),
                anchor_residual_scale=float(training["anchor_residual_scale"]),
                generated_warm_start_probability=float(
                    training["generated_warm_start_probability"]
                ),
                generated_action_ratio_start=ratio_start,
                generated_action_ratio_end=ratio_end,
                fixed_actions=tuple(
                    (int(index), float(value))
                    for index, value in config["runtime"]["fixed_actions"].items()
                ),
                action_loss_weight=float(losses["action_flow"]),
                world_loss_weight=float(losses["expert_world"]),
                generated_consistency_weight=float(
                    losses["generated_consistency"]
                ),
                generated_state_weight=1.0,
                generated_risk_weight=float(losses["generated_risk"]),
                generated_progress_weight=float(losses["generated_progress"]),
                gradient_audit_interval=int(
                    training["branch_gradient_audit_interval"]
                ),
            )
        )
    if not result:
        raise ValueError("Joint WAM has no enabled training stages")
    return result


def _settings(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config["joint_training"]
    stages = list(training["stages"])
    if [stage["scope"] for stage in stages] != [
        "flow_only",
        "world_heads",
        "full_joint",
    ]:
        raise ValueError("training must progressively unfreeze flow/heads/full model")
    if config["checkpoint"].get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format does not match the implementation")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    return {
        "world_model_checkpoint": (
            args.world_model_checkpoint_dir
            or ROOT / config["initialization"]["world_model_checkpoint"]
        ).resolve(),
        "action_prior_checkpoint": (
            args.action_prior_checkpoint_dir
            or ROOT / config["initialization"]["action_prior_checkpoint"]
        ).resolve(),
        "action_flow_checkpoint": (
            args.action_flow_checkpoint_dir
            or ROOT / config["checkpoint"]["warmup_directory"]
        ).resolve(),
        "split_manifest": (ROOT / config["data"]["split_manifest"]).resolve(),
        "checkpoint_dir": (
            args.checkpoint_dir or ROOT / config["checkpoint"]["directory"]
        ).resolve(),
        "require_action_flow_passed": True,
        "batch_size": int(
            training["batch_size"] if args.batch_size is None else args.batch_size
        ),
        "num_workers": int(
            training["num_workers"] if args.num_workers is None else args.num_workers
        ),
        "seed": int(training["seed"]),
        "quality_discount": float(training["action_quality_discount"]),
        "behavior_weights": dict(training["action_quality_behavior_weights"]),
        "require_success": bool(training["action_quality_require_success"]),
        "min_return_quantile": float(
            training["action_quality_min_return_quantile"]
        ),
        "stages": stages,
    }


def _formal_run(args: argparse.Namespace) -> bool:
    limits = {
        "max_steps": int(args.max_steps),
        "max_eval_batches": int(args.max_eval_batches),
        "max_episodes_per_split": int(args.max_episodes_per_split),
    }
    invalid = [name for name, value in limits.items() if value == 0 or value < -1]
    if invalid:
        raise ValueError(f"debug limits must be -1 or positive: {invalid}")
    return all(value == -1 for value in limits.values()) and args.batch_size is None


def _formal_acceptance_checks(
    args: argparse.Namespace,
    *,
    completed_steps: int,
    configured_steps: int,
    smoke_subset: bool,
) -> dict[str, bool]:
    formal_run = _formal_run(args)
    return {
        "no_debug_limits": formal_run,
        "all_configured_training_steps_completed": completed_steps
        == configured_steps,
        "full_dataset_partitions_used": not smoke_subset,
        "full_validation_and_test_evaluated": args.max_eval_batches == -1,
    }


def _manifest_partitions(path: Path) -> dict[str, list[Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_unit") != "episode_seed":
        raise ValueError("world-model ensemble manifest is not episode-seed grouped")
    partitions = payload.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("world-model ensemble manifest has no partitions")
    return {
        name: [Path(str(item)).resolve() for item in partitions[name]]
        for name in ("train", "validation", "test")
    }


def _positive_weights(path: Path) -> dict[str, float]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    label_stats = metrics.get("outcome_label_stats")
    if not isinstance(label_stats, Mapping):
        raise ValueError("world-model metrics has no outcome_label_stats")
    return {
        name: float(label_stats[name]["positive_weight"])
        for name in ("done", "success", "failure")
    }


def _source_fingerprints(settings: Mapping[str, Any]) -> dict[str, str]:
    world_model = settings["world_model_checkpoint"]
    paths: dict[str, Path] = {
        f"world_model_member_{index}": world_model / f"members/member_{index:02d}.safetensors"
        for index in range(5)
    }
    paths.update(
        {
            "world_model_schema": world_model / "schema.json",
            "world_model_normalization": world_model / "normalization.npz",
            "world_model_dataset_manifest": world_model / "dataset_manifest.json",
            "configured_split_manifest": settings["split_manifest"],
            "action_prior": settings["action_prior_checkpoint"]
            / "action_prior.safetensors",
            "warmup_action_flow": settings["action_flow_checkpoint"] / "action_flow.safetensors",
            "warmup_schema": settings["action_flow_checkpoint"] / "schema.json",
        }
    )
    return {name: _sha256(path) for name, path in paths.items()}


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _maximum_delta(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> float:
    if before.keys() != after.keys():
        return float("inf")
    differences: list[float] = []
    for name, first in before.items():
        second = after[name].detach().cpu()
        if torch.is_floating_point(first):
            differences.append(float((first.detach().cpu() - second).abs().max()))
        else:
            differences.append(
                0.0 if torch.equal(first.detach().cpu(), second) else float("inf")
            )
    return max(differences, default=0.0)


def _named_parameter_delta(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...],
) -> float:
    names = [name for name in before if name.startswith(prefixes)]
    if not names:
        raise ValueError(f"no parameters match prefixes {prefixes}")
    return _maximum_delta(
        {name: before[name] for name in names},
        {name: after[name] for name in names},
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Joint WAM config root must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(config_path: Path, seed: int) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    sources = [
        config_path.resolve(),
        ROOT / "train/joint_wam.py",
        ROOT / "train/joint_wam_checkpointing.py",
        ROOT / "eval/joint_wam_training.py",
        ROOT / "scripts/_train_joint_coupling.py",
    ]
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "source_files_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
