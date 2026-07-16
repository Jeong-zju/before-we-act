"""Train Phase 3 action-prior/value heads on a frozen accepted Phase 2 RWM-U."""

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
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from models.wam import WAMPlanningHeadConfig, WAMPlanningHeads  # noqa: E402
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_u_checkpointing import load_rwm_u_checkpoint  # noqa: E402
from train.trajectory_dataset import (  # noqa: E402
    InMemoryProprioSequenceDataset,
    discover_episode_paths,
    split_episode_paths,
)
from train.wam_mppi_checkpointing import (  # noqa: E402
    load_wam_mppi_heads_checkpoint,
    save_wam_mppi_heads_checkpoint,
)
from train.wam_mppi_trainer import (  # noqa: E402
    PlanningHeadsTrainConfig,
    evaluate_planning_heads,
    train_planning_heads,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/wam/phase3_wam_mppi_v1.yaml"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--phase2-checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-eval-batches", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-refresh-hz", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    settings = _settings(config, args)
    _validate_settings(settings)
    device = _device(args.device)
    _seed(settings["seed"])
    ensemble, phase2_metadata = load_rwm_u_checkpoint(
        settings["phase2_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    member = ensemble.members[0]
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(
            feature_dim=member.planning_feature_dim,
            action_dim=member.config.action_dim,
            **settings["planning_heads"],
        )
    ).to(device)
    paths = discover_episode_paths(settings["data_dir"])
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]
    split_paths = split_episode_paths(paths, seed=settings["split_seed"])
    if any(not split_paths[name] for name in ("train", "validation", "test")):
        raise RuntimeError("Phase 3 requires non-empty train/validation/test splits")

    datasets: dict[str, InMemoryProprioSequenceDataset] = {}
    total_stages = 8
    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=total_stages,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        for split_name in ("train", "validation", "test"):
            partition = split_paths[split_name]
            phase = progress.add_phase(f"preload {split_name} planning data", len(partition))
            dataset = InMemoryProprioSequenceDataset(
                paths=partition,
                history_horizon=member.config.history_horizon,
                forecast_horizon=1,
                state_dim=member.config.state_dim,
                action_dim=member.config.action_dim,
                allow_legacy_wam=False,
                planning_discount=settings["discount"],
                action_prior_behavior_weights=settings["behavior_weights"],
                action_prior_require_success=settings["require_success"],
                action_prior_min_return_quantile=(
                    settings["min_return_quantile"]
                    if split_name == "train"
                    else 0.0
                ),
                progress=phase.advance,
            )
            datasets[split_name] = dataset
            phase.finish(
                f"{len(dataset)} fragments, {dataset.nbytes / 2**20:.1f} MiB"
            )

        loaders = {
            name: DataLoader(
                dataset,
                batch_size=settings["batch_size"],
                shuffle=name == "train",
                num_workers=settings["num_workers"],
                pin_memory=device.type == "cuda",
                persistent_workers=settings["num_workers"] > 0,
            )
            for name, dataset in datasets.items()
        }
        train_steps = len(loaders["train"]) * settings["epochs"]
        if args.max_steps > 0:
            train_steps = min(train_steps, args.max_steps)
        train_phase = progress.add_phase("train action prior + value", train_steps)
        loss_history, completed_steps = train_planning_heads(
            heads,
            member,
            loaders["train"],
            device=device,
            config=PlanningHeadsTrainConfig(
                epochs=settings["epochs"],
                learning_rate=settings["learning_rate"],
                weight_decay=settings["weight_decay"],
                gradient_clip_norm=settings["gradient_clip_norm"],
                action_prior_weight=settings["action_prior_weight"],
                value_weight=settings["value_weight"],
                max_steps=args.max_steps,
            ),
            progress=train_phase.advance,
        )
        train_phase.finish(
            f"{completed_steps} steps, final loss {loss_history[-1]:.5f}"
        )

        evaluations: dict[str, Any] = {}
        for split_name in ("train", "validation", "test"):
            eval_phase = progress.add_phase(
                f"evaluate {split_name} planning heads",
                _effective_batches(loaders[split_name], args.max_eval_batches),
            )
            evaluations[split_name] = evaluate_planning_heads(
                heads,
                member,
                loaders[split_name],
                device=device,
                max_batches=args.max_eval_batches,
                progress=eval_phase.advance,
            )
            eval_phase.finish(
                f"value MAE {evaluations[split_name]['value_mae']:.4f}"
            )

        save_phase = progress.add_phase("save and strict reload", 2)
        metrics: dict[str, Any] = {
            "format_version": "wam.phase3.planning_heads.metrics/1",
            "completed_steps": completed_steps,
            "loss_history": loss_history,
            "evaluation": evaluations,
            "planning_data": datasets["train"].planning_metadata,
        }
        manifest = {
            "split_seed": settings["split_seed"],
            "partitions": {
                name: [str(path.resolve()) for path in partition]
                for name, partition in split_paths.items()
            },
            "smoke_subset": args.max_episodes > 0,
        }
        save_wam_mppi_heads_checkpoint(
            settings["checkpoint_dir"],
            heads,
            phase2_checkpoint=settings["phase2_checkpoint"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            normalization_sha256=phase2_metadata["normalization"].sha256(),
        )
        save_phase.advance({"batch": 1})
        reloaded, _ = load_wam_mppi_heads_checkpoint(
            settings["checkpoint_dir"],
            phase2_checkpoint=settings["phase2_checkpoint"],
            device=device,
            expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            expected_normalization_sha256=phase2_metadata["normalization"].sha256(),
        )
        reload_difference = _max_parameter_difference(heads, reloaded)
        metrics["checkpoint_reload_max_abs_diff"] = reload_difference
        save_wam_mppi_heads_checkpoint(
            settings["checkpoint_dir"],
            heads,
            phase2_checkpoint=settings["phase2_checkpoint"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            normalization_sha256=phase2_metadata["normalization"].sha256(),
        )
        save_phase.advance({"batch": 2})
        save_phase.finish(f"reload max diff {reload_difference:.3g}")

    print(
        json.dumps(
            {
                "checkpoint": str(settings["checkpoint_dir"].resolve()),
                "test": evaluations["test"],
                "checkpoint_reload_max_abs_diff": reload_difference,
            },
            indent=2,
        )
    )
    return 0


def _settings(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = config["data"]
    phase2 = config["phase2"]
    training = config["training"]
    checkpoint = config["checkpoint"]
    return {
        "data_dir": (args.data_dir or ROOT / data["directory"]).resolve(),
        "phase2_checkpoint": (
            args.phase2_checkpoint_dir or ROOT / phase2["checkpoint"]
        ).resolve(),
        "checkpoint_dir": (
            args.checkpoint_dir or ROOT / checkpoint["directory"]
        ).resolve(),
        "split_seed": int(data["split_seed"]),
        "planning_heads": dict(config["planning_heads"]),
        "epochs": int(args.epochs or training["epochs"]),
        "batch_size": int(args.batch_size or training["batch_size"]),
        "num_workers": int(
            training["num_workers"] if args.num_workers is None else args.num_workers
        ),
        "seed": int(training["seed"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "discount": float(training["discount"]),
        "value_target": str(training["value_target"]),
        "action_prior_weight": float(training["action_prior_weight"]),
        "value_weight": float(training["value_weight"]),
        "behavior_weights": dict(training["action_prior_behavior_weights"]),
        "require_success": bool(training["action_prior_require_success"]),
        "min_return_quantile": float(
            training["action_prior_min_return_quantile"]
        ),
    }


def _validate_settings(settings: Mapping[str, Any]) -> None:
    if settings["batch_size"] <= 0 or settings["num_workers"] < 0:
        raise ValueError("invalid DataLoader settings")
    if not settings["phase2_checkpoint"].is_dir():
        raise FileNotFoundError(settings["phase2_checkpoint"])
    if settings["value_target"] != "discounted_monte_carlo":
        raise ValueError("Phase 3 V1 supports only discounted_monte_carlo value targets")


def _effective_batches(loader: DataLoader, maximum: int) -> int:
    return min(len(loader), maximum) if maximum > 0 else len(loader)


def _max_parameter_difference(
    first: WAMPlanningHeads, second: WAMPlanningHeads
) -> float:
    return max(
        float((first.state_dict()[name] - value).abs().max().cpu())
        for name, value in second.state_dict().items()
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 3 config root must be a mapping")
    return payload


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
        ROOT / "models/wam/planning_heads.py",
        ROOT / "policies/wam_mppi_policy.py",
        ROOT / "train/wam_mppi_trainer.py",
        ROOT / "scripts/train_phase3_planning_heads.py",
    ]
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "source_files_sha256": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
