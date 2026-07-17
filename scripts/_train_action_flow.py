"""Internal action-flow warm-up used by the Joint WAM training entrypoint."""

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
from envs.runtime import RunnerConfig, SimulationRunner  # noqa: E402
from envs.two_robot_carry_env import (  # noqa: E402
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)
from eval.action_flow import evaluate_action_flow_offline  # noqa: E402
from models.wam import (  # noqa: E402
    ActionChunkConfig,
    StatefulActionFlow,
    StatefulActionFlowConfig,
    shift_action_chunk_warm_start,
)
from policies import ActionPriorPolicy  # noqa: E402
from train.action_prior import world_model_member_fingerprint  # noqa: E402
from train.action_prior import load_action_prior_checkpoint  # noqa: E402
from train.action_flow import (  # noqa: E402
    ActionFlowDistillationBuffer,
    ActionFlowOnPolicyTrainConfig,
    ActionFlowTrainConfig,
    action_prior_teacher_chunk,
    fine_tune_action_flow_on_policy,
    train_action_flow,
)
from train.action_flow_checkpointing import (  # noqa: E402
    load_action_flow_checkpoint,
    save_action_flow_checkpoint,
)
from train.progress import TrainingProgress  # noqa: E402
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
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-eval-batches", type=int, default=-1)
    parser.add_argument("--max-episodes-per-split", type=int, default=-1)
    parser.add_argument("--on-policy-rounds", type=int)
    parser.add_argument("--on-policy-episodes-per-suite", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    settings = _settings(config, args)
    device = _device(args.device)
    _seed(settings["seed"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    world_model_before = world_model_member_fingerprint(settings["world_model_checkpoint"], 0)
    prior_before = _sha256(settings["action_prior_checkpoint"] / "action_prior.safetensors")
    member, world_model_metadata = load_rwm_u_member_checkpoint(
        settings["world_model_checkpoint"],
        0,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    action_prior, _ = load_action_prior_checkpoint(
        settings["action_prior_checkpoint"],
        world_model_checkpoint=settings["world_model_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
        expected_normalization_sha256=world_model_metadata["normalization"].sha256(),
    )
    if int(config["initialization"]["member_index"]) != 0:
        raise ValueError("Joint WAM is locked to member_index=0")
    flow = StatefulActionFlow(
        StatefulActionFlowConfig(
            feature_dim=member.planning_feature_dim,
            action_dim=member.config.action_dim,
            horizon=int(config["action_chunk"]["horizon"]),
            **settings["model"],
        ),
        world_model_metadata["normalization"],
    ).to(device)
    flow.set_anchor_from_prior(action_prior)
    member_initial = {
        name: value.detach().cpu().clone()
        for name, value in member.state_dict().items()
    }
    flow_initial = {
        name: value.detach().cpu().clone()
        for name, value in flow.state_dict().items()
    }
    partitions = _manifest_partitions(settings["split_manifest"])
    if args.max_episodes_per_split > 0:
        partitions = {
            name: paths[: args.max_episodes_per_split]
            for name, paths in partitions.items()
        }
    if any(not partitions[name] for name in ("train", "validation", "test")):
        raise RuntimeError("action-flow warm-up requires non-empty partitions")

    on_policy_rounds = int(settings["on_policy"]["rounds"])
    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=7 + 2 * on_policy_rounds,
    ) as progress:
        datasets: dict[str, InMemoryProprioSequenceDataset] = {}
        selected: dict[str, np.ndarray] = {}
        complete: dict[str, np.ndarray] = {}
        for name in ("train", "validation", "test"):
            phase = progress.add_phase(f"preload {name} action chunks", len(partitions[name]))
            datasets[name] = InMemoryProprioSequenceDataset(
                paths=partitions[name],
                history_horizon=member.config.history_horizon,
                forecast_horizon=flow.config.horizon,
                state_dim=member.config.state_dim,
                action_dim=member.config.action_dim,
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
                f"{len(complete[name])} complete / {len(selected[name])} selected"
            )
        train_loader = DataLoader(
            Subset(datasets["train"], selected["train"].tolist()),
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
        train_steps = len(train_loader) * settings["epochs"]
        if args.max_steps > 0:
            train_steps = min(train_steps, args.max_steps)
        phase = progress.add_phase("train stateful action flow", train_steps)
        loss_history, completed_steps = train_action_flow(
            flow,
            member,
            train_loader,
            device=device,
            config=ActionFlowTrainConfig(
                epochs=settings["epochs"],
                learning_rate=settings["learning_rate"],
                weight_decay=settings["weight_decay"],
                gradient_clip_norm=settings["gradient_clip_norm"],
                warm_start_probability=settings["warm_start_probability"],
                warm_start_noise_std=settings["warm_start_noise_std"],
                cold_noise_std=settings["cold_noise_std"],
                cold_zero_probability=settings["cold_zero_probability"],
                endpoint_weight=settings["endpoint_weight"],
                smoothness_weight=settings["smoothness_weight"],
                execution_steps=int(config["action_chunk"]["execution_steps"]),
                action_prior_distillation_steps=settings[
                    "action_prior_distillation_steps"
                ],
                action_prior_distillation_blend=settings[
                    "action_prior_distillation_blend"
                ],
                use_amp=settings["use_amp"],
                max_steps=args.max_steps,
            ),
            seed=settings["seed"],
            action_prior=action_prior,
            progress=phase.advance,
        )
        phase.finish(
            f"{completed_steps} steps, final loss {loss_history[-1]['loss']:.5f}"
        )
        on_policy_metrics: list[dict[str, Any]] = []
        on_policy_seeds: list[int] = []
        for round_index in range(on_policy_rounds):
            episode_count = 2 * int(settings["on_policy"]["episodes_per_suite"])
            phase = progress.add_phase(
                f"collect on-policy round {round_index + 1}", episode_count
            )
            buffer, rollout_metrics = _collect_on_policy_distillation(
                flow,
                member,
                action_prior,
                config=config,
                settings=settings["on_policy"],
                round_index=round_index,
                progress=phase.advance,
            )
            on_policy_seeds.extend(rollout_metrics["seeds"])
            phase.finish(
                f"{len(buffer)} requests, success {rollout_metrics['success_rate']:.1%}"
            )
            fine_tune_steps = (
                (len(buffer) + int(settings["on_policy"]["batch_size"]) - 1)
                // int(settings["on_policy"]["batch_size"])
            ) * int(settings["on_policy"]["epochs"])
            phase = progress.add_phase(
                f"fine-tune action flow round {round_index + 1}", fine_tune_steps
            )
            fine_history, fine_steps = fine_tune_action_flow_on_policy(
                flow,
                buffer,
                device=device,
                config=ActionFlowOnPolicyTrainConfig(
                    epochs=int(settings["on_policy"]["epochs"]),
                    batch_size=int(settings["on_policy"]["batch_size"]),
                    learning_rate=float(settings["on_policy"]["learning_rate"]),
                    weight_decay=float(settings["on_policy"]["weight_decay"]),
                    gradient_clip_norm=float(
                        settings["on_policy"]["gradient_clip_norm"]
                    ),
                    cold_replay_probability=float(
                        settings["on_policy"]["cold_replay_probability"]
                    ),
                    warm_start_noise_std=float(
                        settings["on_policy"]["warm_start_noise_std"]
                    ),
                    endpoint_weight=float(settings["on_policy"]["endpoint_weight"]),
                    smoothness_weight=float(
                        settings["on_policy"]["smoothness_weight"]
                    ),
                    solver_steps=int(config["action_chunk"]["solver_steps"]),
                    solver_endpoint_weight=float(
                        settings["on_policy"]["solver_endpoint_weight"]
                    ),
                    offline_replay_weight=float(
                        settings["on_policy"]["offline_replay_weight"]
                    ),
                    use_amp=bool(settings["on_policy"]["use_amp"]),
                ),
                seed=settings["seed"] + 1000 + round_index,
                replay_loader=train_loader,
                world_model=member,
                action_prior=action_prior,
                progress=phase.advance,
            )
            phase.finish(f"{fine_steps} steps, final loss {fine_history[-1]['loss']:.5f}")
            on_policy_metrics.append(
                {
                    **rollout_metrics,
                    "requests": len(buffer),
                    "fine_tune_steps": fine_steps,
                    "final_loss": fine_history[-1],
                }
            )
        offline: dict[str, Any] = {}
        for name in ("validation", "test"):
            maximum = (
                min(len(evaluation_loaders[name]), args.max_eval_batches)
                if args.max_eval_batches > 0
                else len(evaluation_loaders[name])
            )
            phase = progress.add_phase(f"evaluate {name} action flow", maximum)
            offline[name] = evaluate_action_flow_offline(
                flow,
                member,
                evaluation_loaders[name],
                device=device,
                execution_steps=int(config["action_chunk"]["execution_steps"]),
                solver_steps=int(config["action_chunk"]["solver_steps"]),
                solver=str(config["action_chunk"]["solver"]),
                max_batches=args.max_eval_batches,
                progress=phase.advance,
                action_prior=action_prior,
            )
            phase.finish(
                f"cold RMSE {offline[name]['cold_action_chunk_rmse']:.5f}"
            )
        member_delta = _maximum_delta(member_initial, member.state_dict())
        flow_delta = _maximum_delta(flow_initial, flow.state_dict())
        anchor_prior_delta = _maximum_delta(
            action_prior.state_dict(), flow.anchor_prior.state_dict()
        )
        source_immutable = (
            world_model_before
            == world_model_member_fingerprint(settings["world_model_checkpoint"], 0)
            and prior_before
            == _sha256(settings["action_prior_checkpoint"] / "action_prior.safetensors")
        )
        metrics: dict[str, Any] = {
            "format_version": "wam.action_flow_warmup.metrics/1",
            "stage": "action_flow_warmup",
            "completed_steps": completed_steps,
            "loss_history": loss_history,
            "on_policy_distillation": {
                "rounds": on_policy_metrics,
                "seeds": on_policy_seeds,
                "privileged_state_exposed": False,
            },
            "offline": offline,
            "data": {
                name: {
                    "episodes": len(partitions[name]),
                    "complete_chunks": int(len(complete[name])),
                    "selected_action_chunks": int(len(selected[name])),
                }
                for name in datasets
            },
            "member_0_parameter_delta": member_delta,
            "action_flow_parameter_delta": flow_delta,
            "anchor_prior_parameter_delta": anchor_prior_delta,
            "source_checkpoints_immutable": source_immutable,
            "online_loaded_member_indices": [0],
            "online_ensemble_loaded": False,
        }
        manifest = {
            "source": str(settings["split_manifest"]),
            "split_seed": int(config["data"]["split_seed"]),
            "partitions": {
                name: [str(path.resolve()) for path in paths]
                for name, paths in partitions.items()
            },
            "complete_action_chunk_only": True,
            "action_flow_on_policy_seeds": on_policy_seeds,
            "smoke_subset": args.max_episodes_per_split > 0,
        }
        phase = progress.add_phase("save and reload action flow", 2)
        save_action_flow_checkpoint(
            settings["checkpoint_dir"],
            flow,
            world_model_checkpoint=settings["world_model_checkpoint"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            normalization_sha256=world_model_metadata["normalization"].sha256(),
        )
        phase.advance({"batch": 1})
        reloaded, _ = load_action_flow_checkpoint(
            settings["checkpoint_dir"],
            world_model_checkpoint=settings["world_model_checkpoint"],
            device=device,
            expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
        )
        reload_difference = _maximum_delta(flow.state_dict(), reloaded.state_dict())
        metrics["checkpoint_reload_max_abs_diff"] = reload_difference
        metrics["passed"] = bool(
            member_delta == 0.0
            and flow_delta > 0.0
            and anchor_prior_delta == 0.0
            and source_immutable
            and reload_difference == 0.0
            and all(
                item["generated_action_non_finite"] == 0
                and item["generated_action_out_of_bounds"] == 0
                and item["generated_action_world_non_finite"] == 0
                and item["generated_action_demo_state_is_ground_truth"] is False
                for item in offline.values()
            )
        )
        save_action_flow_checkpoint(
            settings["checkpoint_dir"],
            flow,
            world_model_checkpoint=settings["world_model_checkpoint"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            normalization_sha256=world_model_metadata["normalization"].sha256(),
        )
        phase.advance({"batch": 2})
        phase.finish(f"reload max diff {reload_difference:.3g}")
    summary = {
        "checkpoint": str(settings["checkpoint_dir"]),
        "passed": metrics["passed"],
        "test": offline["test"],
        "member_0_parameter_delta": member_delta,
        "action_flow_parameter_delta": flow_delta,
        "anchor_prior_parameter_delta": anchor_prior_delta,
        "checkpoint_reload_max_abs_diff": reload_difference,
    }
    print(json.dumps(summary, indent=2))
    return 0 if metrics["passed"] else 2


def _collect_on_policy_distillation(
    flow: StatefulActionFlow,
    member: Any,
    action_prior: Any,
    *,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    round_index: int,
    progress: Any,
) -> tuple[ActionFlowDistillationBuffer, dict[str, Any]]:
    """Collect scheduled warm-start pairs on successful teacher histories."""

    del flow
    buffer = ActionFlowDistillationBuffer()
    horizon = int(config["action_chunk"]["horizon"])
    execution_steps = int(config["action_chunk"]["execution_steps"])
    chunk_contract = ActionChunkConfig(
        action_dim=int(config["data"]["action_dim"]),
        horizon=horizon,
        execution_steps=execution_steps,
        solver_steps=int(config["action_chunk"]["solver_steps"]),
        warm_start_mode=str(config["action_chunk"]["warm_start_mode"]),
    )
    fixed_actions = {
        int(index): float(value)
        for index, value in config["runtime"].get("fixed_actions", {}).items()
    }
    count = int(settings["episodes_per_suite"])
    seeds_by_suite = {
        "standard": range(
            int(settings["standard_seed_start"]) + round_index * count,
            int(settings["standard_seed_start"]) + (round_index + 1) * count,
        ),
        "challenge": range(
            int(settings["challenge_seed_start"]) + round_index * count,
            int(settings["challenge_seed_start"]) + (round_index + 1) * count,
        ),
    }
    successes: list[bool] = []
    used_seeds: list[int] = []
    for suite, seeds in seeds_by_suite.items():
        overrides = (
            {}
            if suite == "standard"
            else dict(config["evaluation"]["challenge_environment"])
        )
        env = TwoRobotCooperativeStopEnv(
            CooperativeStopEnvConfig(include_camera_images=False, **overrides)
        )
        try:
            for episode_index, seed in enumerate(seeds):
                callback_step = 0
                previous_target: torch.Tensor | None = None

                @torch.inference_mode()
                def record(
                    features: torch.Tensor,
                    hidden: torch.Tensor,
                    current_state: torch.Tensor,
                ) -> None:
                    nonlocal callback_step, previous_target
                    if callback_step % execution_steps == 0:
                        targets = action_prior_teacher_chunk(
                            member,
                            action_prior,
                            hidden,
                            current_state,
                            steps=horizon,
                        ).clone()
                        for index, value in fixed_actions.items():
                            targets[..., index] = value
                        initial_actions = (
                            None
                            if previous_target is None
                            else shift_action_chunk_warm_start(
                                previous_target[0],
                                chunk_contract,
                                executed_steps=execution_steps,
                            ).unsqueeze(0)
                        )
                        buffer.add(features, targets, initial_actions)
                        previous_target = targets.detach()
                    callback_step += 1

                policy = ActionPriorPolicy(
                    member,
                    action_prior,
                    fixed_actions=fixed_actions,
                    distillation_callback=record,
                )
                summary = SimulationRunner(
                    env,
                    policy,
                    RunnerConfig(expose_privileged_state_to_policy=False),
                ).run_episode(
                    seed=seed,
                    episode_index=episode_index,
                    randomize=bool(config["evaluation"]["randomize"]),
                )
                success = bool(summary.final_info.get("success", False))
                successes.append(success)
                used_seeds.append(seed)
                progress({"episode": len(successes), "success": int(success)})
        finally:
            env.close()
    return buffer, {
        "round": round_index + 1,
        "seeds": used_seeds,
        "episodes": len(successes),
        "success_rate": float(np.mean(successes)),
    }


def _settings(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config["action_flow"]["training"]
    on_policy = dict(config["action_flow"]["on_policy_distillation"])
    if args.on_policy_rounds is not None:
        on_policy["rounds"] = args.on_policy_rounds
    if args.on_policy_episodes_per_suite is not None:
        on_policy["episodes_per_suite"] = args.on_policy_episodes_per_suite
    if int(on_policy["rounds"]) < 0 or int(on_policy["episodes_per_suite"]) <= 0:
        raise ValueError("on-policy rounds must be non-negative and episodes positive")
    return {
        "world_model_checkpoint": (
            args.world_model_checkpoint_dir
            or ROOT / config["initialization"]["world_model_checkpoint"]
        ).resolve(),
        "action_prior_checkpoint": (
            args.action_prior_checkpoint_dir
            or ROOT / config["initialization"]["action_prior_checkpoint"]
        ).resolve(),
        "split_manifest": (ROOT / config["data"]["split_manifest"]).resolve(),
        "checkpoint_dir": (
            args.checkpoint_dir or ROOT / config["checkpoint"]["warmup_directory"]
        ).resolve(),
        "model": dict(config["action_flow"]["model"]),
        "epochs": int(args.epochs or training["epochs"]),
        "batch_size": int(args.batch_size or training["batch_size"]),
        "num_workers": int(
            training["num_workers"] if args.num_workers is None else args.num_workers
        ),
        "seed": int(training["seed"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "use_amp": bool(training["use_amp"]),
        "warm_start_probability": float(training["warm_start_probability"]),
        "warm_start_noise_std": float(training["warm_start_noise_std"]),
        "cold_noise_std": float(training["cold_noise_std"]),
        "cold_zero_probability": float(training["cold_zero_probability"]),
        "endpoint_weight": float(training["endpoint_weight"]),
        "smoothness_weight": float(training["smoothness_weight"]),
        "action_prior_distillation_steps": int(
            training["action_prior_distillation_steps"]
        ),
        "action_prior_distillation_blend": float(
            training["action_prior_distillation_blend"]
        ),
        "quality_discount": float(training["quality_discount"]),
        "behavior_weights": dict(training["action_quality_behavior_weights"]),
        "require_success": bool(training["action_quality_require_success"]),
        "min_return_quantile": float(training["action_quality_min_return_quantile"]),
        "on_policy": on_policy,
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


def _maximum_delta(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> float:
    if before.keys() != after.keys():
        return float("inf")
    differences = []
    for name, first in before.items():
        second = after[name].detach().cpu()
        if torch.is_floating_point(first):
            differences.append(float((first.detach().cpu() - second).abs().max()))
        else:
            differences.append(0.0 if torch.equal(first.detach().cpu(), second) else float("inf"))
    return max(differences, default=0.0)


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
        ROOT / "models/wam/stateful_action_flow.py",
        ROOT / "train/action_flow.py",
        ROOT / "train/action_flow_checkpointing.py",
        ROOT / "eval/action_flow.py",
        ROOT / "scripts/_train_action_flow.py",
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
