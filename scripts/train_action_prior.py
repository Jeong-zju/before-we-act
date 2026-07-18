"""Train the action-prior baseline on the frozen world-model belief."""

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
from models.wam import ActionPrior, ActionPriorConfig  # noqa: E402
from train.action_prior import (  # noqa: E402
    ActionPriorTrainConfig,
    evaluate_action_prior,
    load_action_prior_checkpoint,
    save_action_prior_checkpoint,
    train_action_prior,
)
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_ar_checkpointing import load_wam_checkpoint  # noqa: E402
from train.trajectory_dataset import (  # noqa: E402
    InMemoryProprioSequenceDataset,
    discover_episode_paths,
    split_episode_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/wam/action_prior.yaml"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--world-model-checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-eval-batches", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    settings = _settings(config, args)
    device = _device(args.device)
    _seed(settings["seed"])
    world_model, world_model_metadata = load_wam_checkpoint(
        settings["world_model_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    prior = ActionPrior(
        ActionPriorConfig(
            feature_dim=world_model.planning_feature_dim,
            action_dim=world_model.config.action_dim,
            **settings["model"],
        )
    ).to(device)
    paths = discover_episode_paths(settings["data_dir"])
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]
    partitions = split_episode_paths(paths, seed=settings["split_seed"])
    if any(not partitions[name] for name in ("train", "validation", "test")):
        raise RuntimeError("action-prior training requires non-empty splits")

    with TrainingProgress(enabled=not args.no_progress, total_stages=8) as progress:
        datasets: dict[str, InMemoryProprioSequenceDataset] = {}
        for name in ("train", "validation", "test"):
            phase = progress.add_phase(f"preload {name} action-prior data", len(partitions[name]))
            datasets[name] = InMemoryProprioSequenceDataset(
                paths=partitions[name],
                history_horizon=world_model.config.history_horizon,
                forecast_horizon=1,
                state_dim=world_model.config.state_dim,
                action_dim=world_model.config.action_dim,
                allow_legacy_wam=False,
                planning_discount=settings["discount"],
                action_prior_behavior_weights=settings["behavior_weights"],
                action_prior_require_success=settings["require_success"],
                action_prior_min_return_quantile=(
                    settings["min_return_quantile"] if name == "train" else 0.0
                ),
                progress=phase.advance,
            )
            phase.finish(
                f"{len(datasets[name])} fragments, {datasets[name].nbytes / 2**20:.1f} MiB"
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
        phase = progress.add_phase("train action prior", train_steps)
        loss_history, completed_steps = train_action_prior(
            prior,
            world_model,
            loaders["train"],
            device=device,
            config=ActionPriorTrainConfig(
                epochs=settings["epochs"],
                learning_rate=settings["learning_rate"],
                weight_decay=settings["weight_decay"],
                gradient_clip_norm=settings["gradient_clip_norm"],
                max_steps=args.max_steps,
            ),
            progress=phase.advance,
        )
        phase.finish(f"{completed_steps} steps, final loss {loss_history[-1]:.5f}")
        evaluations: dict[str, Any] = {}
        for name in ("train", "validation", "test"):
            maximum = min(len(loaders[name]), args.max_eval_batches) if args.max_eval_batches > 0 else len(loaders[name])
            phase = progress.add_phase(f"evaluate {name} action prior", maximum)
            evaluations[name] = evaluate_action_prior(
                prior,
                world_model,
                loaders[name],
                device=device,
                max_batches=args.max_eval_batches,
                progress=phase.advance,
            )
            phase.finish(f"selected RMSE {evaluations[name]['selected_action_rmse']:.5f}")
        phase = progress.add_phase("save and strict reload", 2)
        metrics = {
            "format_version": "wam.action_prior.metrics/1",
            "completed_steps": completed_steps,
            "loss_history": loss_history,
            "evaluation": evaluations,
            "planning_data": datasets["train"].planning_metadata,
        }
        manifest = {
            "split_seed": settings["split_seed"],
            "partitions": {
                name: [str(path.resolve()) for path in partition]
                for name, partition in partitions.items()
            },
            "smoke_subset": args.max_episodes > 0,
        }
        save_action_prior_checkpoint(
            settings["checkpoint_dir"],
            prior,
            world_model_checkpoint=settings["world_model_checkpoint"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            normalization_sha256=world_model_metadata["normalization"].sha256(),
        )
        phase.advance({"batch": 1})
        reloaded, _ = load_action_prior_checkpoint(
            settings["checkpoint_dir"],
            world_model_checkpoint=settings["world_model_checkpoint"],
            device=device,
            expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            expected_normalization_sha256=world_model_metadata["normalization"].sha256(),
        )
        difference = max(
            float((prior.state_dict()[name] - value).abs().max().cpu())
            for name, value in reloaded.state_dict().items()
        )
        metrics["checkpoint_reload_max_abs_diff"] = difference
        save_action_prior_checkpoint(
            settings["checkpoint_dir"],
            prior,
            world_model_checkpoint=settings["world_model_checkpoint"],
            experiment_config=config,
            dataset_manifest=manifest,
            metrics=metrics,
            provenance=_provenance(args.config, settings["seed"]),
            schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            normalization_sha256=world_model_metadata["normalization"].sha256(),
        )
        phase.advance({"batch": 2})
        phase.finish(f"reload max diff {difference:.3g}")
    print(
        json.dumps(
            {
                "checkpoint": str(settings["checkpoint_dir"]),
                "test": evaluations["test"],
                "checkpoint_reload_max_abs_diff": difference,
            },
            indent=2,
        )
    )
    return 0


def _settings(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config["training"]
    return {
        "data_dir": (args.data_dir or ROOT / config["data"]["directory"]).resolve(),
        "world_model_checkpoint": (
            args.world_model_checkpoint_dir or ROOT / config["world_model"]["checkpoint"]
        ).resolve(),
        "checkpoint_dir": (
            args.checkpoint_dir or ROOT / config["checkpoint"]["directory"]
        ).resolve(),
        "split_seed": int(config["data"]["split_seed"]),
        "model": dict(config["model"]),
        "epochs": int(args.epochs or training["epochs"]),
        "batch_size": int(args.batch_size or training["batch_size"]),
        "num_workers": int(training["num_workers"] if args.num_workers is None else args.num_workers),
        "seed": int(training["seed"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "discount": float(training["discount"]),
        "behavior_weights": dict(training["behavior_weights"]),
        "require_success": bool(training["require_success"]),
        "min_return_quantile": float(training["min_return_quantile"]),
    }


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
        raise ValueError("action-prior config root must be a mapping")
    return payload


def _provenance(config_path: Path, seed: int) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    sources = [
        config_path.resolve(),
        ROOT / "models/wam/action_prior.py",
        ROOT / "policies/action_prior.py",
        ROOT / "train/action_prior.py",
        ROOT / "scripts/train_action_prior.py",
    ]
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "source_files_sha256": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
