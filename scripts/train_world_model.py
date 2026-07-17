"""Train the recurrent world-action model (RWM-AR)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from models.wam import RWMARConfig, RWMARWorldModel, WorldModelSequenceInputs  # noqa: E402
from train.rwm_ar_checkpointing import (  # noqa: E402
    load_wam_checkpoint,
    save_wam_checkpoint,
)
from train.rwm_ar_losses import RWMLossWeights  # noqa: E402
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_ar_trainer import (  # noqa: E402
    RWMTrainConfig,
    build_optimizer,
    evaluate_wam_loss,
    fit_wam_label_stats,
    fit_wam_normalization,
    make_positive_weights,
    train_curriculum_stage,
)
from train.trajectory_dataset import (  # noqa: E402
    InMemoryProprioSequenceDataset,
    ProprioSequenceDataset,
    discover_episode_paths,
    split_episode_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam/world_model.yaml",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--overfit-refine-epochs", type=int)
    parser.add_argument("--overfit-refine-learning-rate", type=float)
    parser.add_argument("--forecast-curriculum", type=int, nargs="+")
    parser.add_argument("--curriculum-epochs", type=int, nargs="+")
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument(
        "--max-steps-per-stage",
        type=int,
        default=-1,
        help="Limit optimizer steps in each curriculum stage; useful for smoke tests.",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=-1,
        help="Restrict the training split to the first N sequence fragments.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use-amp", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--preload-data", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-refresh-hz", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    resolved = _resolve_settings(args, config)
    _validate_settings(resolved)
    _seed_everything(resolved["seed"])
    device = _resolve_device(resolved["device"])
    paths = discover_episode_paths(resolved["data_dir"])
    if resolved["max_episodes"] > 0:
        paths = paths[: resolved["max_episodes"]]
    split_paths = split_episode_paths(paths, seed=resolved["split_seed"])
    if not split_paths["train"] or not split_paths["validation"]:
        raise RuntimeError("recurrent world model requires non-empty train and validation splits")

    curriculum = tuple(resolved["forecast_curriculum"])
    curriculum_epochs = tuple(resolved["curriculum_epochs"])
    training_schedule = [
        {
            "horizon": horizon,
            "epochs": epochs,
            "name": f"H={horizon}",
            "learning_rate": resolved["learning_rate"],
        }
        for horizon, epochs in zip(curriculum, curriculum_epochs, strict=True)
    ]
    if resolved["overfit_samples"] > 0 and resolved["overfit_refine_epochs"] > 0:
        training_schedule.append(
            {
                "horizon": 1,
                "epochs": resolved["overfit_refine_epochs"],
                "name": "H=1 overfit refine",
                "learning_rate": resolved["overfit_refine_learning_rate"],
            }
        )
    training_partitions = ("train", "validation")
    preload_stages = (
        sum(bool(split_paths[name]) for name in training_partitions)
        if resolved["preload_data"]
        else 0
    )
    total_stages = preload_stages + 2 + 2 * len(training_schedule) + 1
    datasets: dict[str, Dataset | None] = {}
    metrics: dict[str, Any] = {
        "format_version": "wam.world_model.training/1",
        "curriculum": [],
    }
    checkpoint_dir = Path(resolved["checkpoint_dir"])

    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=total_stages,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        try:
            for split_name in training_partitions:
                partition = split_paths[split_name]
                if not partition:
                    datasets[split_name] = None
                    continue
                dataset_partition = partition
                if split_name == "train" and resolved["overfit_samples"] > 0:
                    dataset_partition = partition[
                        : min(len(partition), resolved["overfit_samples"])
                    ]
                preload_phase = (
                    progress.add_phase(f"preload {split_name}", len(dataset_partition))
                    if resolved["preload_data"]
                    else None
                )
                dataset = _build_dataset(
                    dataset_partition,
                    resolved,
                    forecast_horizon=max(curriculum),
                    progress=(preload_phase.advance if preload_phase else None),
                )
                datasets[split_name] = dataset
                if preload_phase is not None:
                    assert isinstance(dataset, InMemoryProprioSequenceDataset)
                    preload_phase.finish(
                        f"{len(dataset)} fragments, {dataset.nbytes / 2**20:.1f} MiB"
                    )

            train_dataset = datasets["train"]
            validation_dataset = datasets["validation"]
            assert train_dataset is not None and validation_dataset is not None
            if resolved["overfit_samples"] > 0:
                count = min(resolved["overfit_samples"], len(train_dataset))
                train_dataset = Subset(train_dataset, range(count))
                metrics["overfit_samples"] = count
            stats_loader = _build_loader(
                train_dataset, resolved, shuffle=False, device=device
            )
            train_loader = _build_loader(
                train_dataset, resolved, shuffle=True, device=device
            )
            validation_loader = _build_loader(
                validation_dataset, resolved, shuffle=False, device=device
            )

            statistics_phase = progress.add_phase(
                "dataset statistics", len(stats_loader)
            )
            stats = fit_wam_normalization(
                stats_loader,
                state_dim=resolved["state_dim"],
                action_dim=resolved["action_dim"],
                yaw_indices=tuple(resolved["yaw_indices"]),
                std_floor=resolved["normalization_std_floor"],
                progress=statistics_phase.advance,
            )
            statistics_phase.finish(f"normalization {stats.sha256()[:12]}")

            labels_phase = progress.add_phase("outcome label stats", len(stats_loader))
            label_stats = fit_wam_label_stats(
                stats_loader, progress=labels_phase.advance
            )
            labels_phase.finish(
                ", ".join(
                    f"{name} w+={values['positive_weight']:.1f}"
                    for name, values in label_stats.items()
                )
            )
            metrics["outcome_label_stats"] = label_stats

            model_config = RWMARConfig(
                state_dim=resolved["state_dim"],
                action_dim=resolved["action_dim"],
                history_horizon=resolved["history_horizon"],
                train_forecast_horizon=max(curriculum),
                planning_horizon=resolved["planning_horizon"],
                encoder_hidden_dim=resolved["encoder_hidden_dim"],
                gru_hidden_dim=resolved["gru_hidden_dim"],
                gru_layers=resolved["gru_layers"],
                dropout=resolved["dropout"],
                min_log_std=resolved["min_log_std"],
                max_log_std=resolved["max_log_std"],
                yaw_indices=tuple(resolved["yaw_indices"]),
                gripper_closed_indices=tuple(resolved["gripper_closed_indices"]),
            )
            model = RWMARWorldModel(model_config, stats).to(device)
            train_config = _train_config(resolved)
            optimizer = build_optimizer(model, train_config)
            positive_weights = make_positive_weights(label_stats, device)

            for stage in training_schedule:
                horizon = int(stage["horizon"])
                epochs = int(stage["epochs"])
                learning_rate = float(stage["learning_rate"])
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                possible_steps = epochs * len(train_loader)
                if train_config.max_steps > 0:
                    possible_steps = min(possible_steps, train_config.max_steps)
                train_phase = progress.add_phase(
                    f"train RWM-AR {stage['name']}", max(possible_steps, 1)
                )
                loss_history, completed_steps = train_curriculum_stage(
                    model,
                    train_loader,
                    optimizer,
                    horizon=horizon,
                    epochs=epochs,
                    config=train_config,
                    positive_weights=positive_weights,
                    device=device,
                    completed_steps=0,
                    progress=train_phase.advance,
                )
                final_loss = loss_history[-1] if loss_history else float("nan")
                train_phase.finish(
                    f"{completed_steps} steps, final loss {final_loss:.5f}"
                )
                validation_batches = len(validation_loader)
                if resolved["validation_max_batches"] > 0:
                    validation_batches = min(
                        validation_batches, resolved["validation_max_batches"]
                    )
                validation_phase = progress.add_phase(
                    f"validate RWM-AR {stage['name']}", max(validation_batches, 1)
                )
                validation_metrics = evaluate_wam_loss(
                    model,
                    validation_loader,
                    horizon=horizon,
                    config=train_config,
                    positive_weights=positive_weights,
                    device=device,
                    max_batches=resolved["validation_max_batches"],
                    progress=validation_phase.advance,
                )
                validation_phase.finish(
                    f"loss {float(validation_metrics.get('total', float('nan'))):.5f}"
                )
                metrics["curriculum"].append(
                    {
                        "horizon": horizon,
                        "stage": stage["name"],
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "optimizer_steps": completed_steps,
                        "loss_history": loss_history,
                        "validation": validation_metrics,
                    }
                )

            save_phase = progress.add_phase("save and strict reload", 2)
            manifest = _dataset_manifest(split_paths, resolved)
            provenance = _provenance(args, resolved, device)
            save_wam_checkpoint(
                checkpoint_dir,
                model,
                stats,
                experiment_config=config,
                dataset_manifest=manifest,
                metrics=metrics,
                provenance=provenance,
                schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            )
            save_phase.advance({"batch": 1})
            reload_model, _ = load_wam_checkpoint(
                checkpoint_dir,
                device=device,
                expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            )
            reload_consistency = _check_reload_consistency(
                model, reload_model, validation_loader, device
            )
            metrics["checkpoint_reload"] = reload_consistency
            # Persist the result added after the initial checkpoint write.
            save_wam_checkpoint(
                checkpoint_dir,
                model,
                stats,
                experiment_config=config,
                dataset_manifest=manifest,
                metrics=metrics,
                provenance=provenance,
                schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            )
            save_phase.advance({"batch": 2})
            save_phase.finish(f"max_abs_diff {reload_consistency['max_abs_diff']:.3g}")
        finally:
            for dataset in datasets.values():
                close = getattr(dataset, "close", None)
                if close is not None:
                    close()

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_dir.resolve()),
                "curriculum": [
                    {
                        "stage": item.get("stage", f"H={item['horizon']}"),
                        "horizon": item["horizon"],
                        "optimizer_steps": item["optimizer_steps"],
                        "final_loss": (
                            item["loss_history"][-1] if item["loss_history"] else None
                        ),
                        "validation_loss": item["validation"].get("total"),
                    }
                    for item in metrics["curriculum"]
                ],
                "checkpoint_reload": metrics["checkpoint_reload"],
            },
            indent=2,
        )
    )
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def _resolve_settings(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    data = config["data"]
    model = config["model"]
    features = config["state_features"]
    training = config["training"]
    evaluation = config["evaluation"]
    checkpoint = config["checkpoint"]
    return {
        "data_dir": args.data_dir or ROOT / data["directory"],
        "checkpoint_dir": args.checkpoint_dir or ROOT / checkpoint["directory"],
        "device": args.device,
        "state_dim": int(data["state_dim"]),
        "action_dim": int(data["action_dim"]),
        "history_horizon": int(data["history_horizon"]),
        "planning_horizon": int(data["planning_horizon"]),
        "split_seed": int(data["split_seed"]),
        "yaw_indices": tuple(features["yaw_indices"]),
        "gripper_closed_indices": tuple(features["gripper_closed_indices"]),
        "encoder_hidden_dim": int(model["encoder_hidden_dim"]),
        "gru_hidden_dim": int(model["gru_hidden_dim"]),
        "gru_layers": int(model["gru_layers"]),
        "dropout": float(model["dropout"]),
        "min_log_std": float(model["min_log_std"]),
        "max_log_std": float(model["max_log_std"]),
        "ensemble_size": int(model["ensemble_size"]),
        "predict_delta": bool(features["predict_delta"]),
        "training_mode": str(training["mode"]),
        "privileged_inputs": bool(training["privileged_inputs"]),
        "forecast_curriculum": tuple(
            int(item)
            for item in (args.forecast_curriculum or training["forecast_curriculum"])
        ),
        "curriculum_epochs": tuple(
            int(item)
            for item in (args.curriculum_epochs or training["curriculum_epochs"])
        ),
        "batch_size": int(args.batch_size or training["batch_size"]),
        "num_workers": int(
            args.num_workers
            if args.num_workers is not None
            else training["num_workers"]
        ),
        "learning_rate": float(
            args.learning_rate
            if args.learning_rate is not None
            else training["learning_rate"]
        ),
        "weight_decay": float(
            args.weight_decay
            if args.weight_decay is not None
            else training["weight_decay"]
        ),
        "overfit_refine_epochs": int(
            args.overfit_refine_epochs
            if args.overfit_refine_epochs is not None
            else training.get("overfit_refine_epochs", 0)
        ),
        "overfit_refine_learning_rate": float(
            args.overfit_refine_learning_rate
            if args.overfit_refine_learning_rate is not None
            else training.get("overfit_refine_learning_rate", training["learning_rate"])
        ),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "horizon_decay": float(training["horizon_decay"]),
        "normalization_std_floor": float(training["normalization_std_floor"]),
        "use_amp": bool(training["use_amp"] if args.use_amp is None else args.use_amp),
        "loss_weights": dict(training["loss_weights"]),
        "validation_max_batches": int(evaluation["validation_max_batches"]),
        "preload_data": bool(
            training["preload_data"] if args.preload_data is None else args.preload_data
        ),
        "max_episodes": int(args.max_episodes),
        "max_steps_per_stage": int(args.max_steps_per_stage),
        "overfit_samples": int(args.overfit_samples),
        "seed": int(args.seed if args.seed is not None else training["seed"]),
    }


def _validate_settings(settings: Mapping[str, Any]) -> None:
    if len(settings["forecast_curriculum"]) != len(settings["curriculum_epochs"]):
        raise ValueError(
            "forecast_curriculum and curriculum_epochs must have equal length"
        )
    if tuple(sorted(settings["forecast_curriculum"])) != tuple(
        settings["forecast_curriculum"]
    ):
        raise ValueError("forecast_curriculum must be non-decreasing")
    if any(value <= 0 for value in settings["forecast_curriculum"]):
        raise ValueError("forecast_curriculum values must be positive")
    if any(value <= 0 for value in settings["curriculum_epochs"]):
        raise ValueError("curriculum_epochs values must be positive")
    if settings["normalization_std_floor"] <= 0.0:
        raise ValueError("normalization_std_floor must be positive")
    if settings["learning_rate"] <= 0.0 or settings["weight_decay"] < 0.0:
        raise ValueError(
            "learning_rate must be positive and weight_decay non-negative"
        )
    if settings["overfit_refine_epochs"] < 0:
        raise ValueError("overfit_refine_epochs must be non-negative")
    if settings["overfit_refine_learning_rate"] <= 0.0:
        raise ValueError("overfit_refine_learning_rate must be positive")
    if settings["ensemble_size"] != 1:
        raise ValueError(
            "recurrent world model requires ensemble_size=1; ensemble belongs to world-model ensemble"
        )
    if not settings["predict_delta"]:
        raise ValueError("recurrent world model requires state_features.predict_delta=true")
    if settings["training_mode"] != "autoregressive":
        raise ValueError("recurrent world model requires training.mode=autoregressive")
    if settings["privileged_inputs"]:
        raise ValueError("recurrent world model inputs cannot include privileged state")
    for name in ("batch_size", "state_dim", "action_dim", "history_horizon"):
        if settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("max_episodes", "max_steps_per_stage", "overfit_samples"):
        if settings[name] == 0 or settings[name] < -1:
            raise ValueError(f"{name} must be -1 or positive")


def _build_dataset(
    paths: tuple[Path, ...],
    settings: Mapping[str, Any],
    *,
    forecast_horizon: int,
    progress: Any | None,
) -> Dataset:
    common = {
        "paths": paths,
        "history_horizon": settings["history_horizon"],
        "forecast_horizon": forecast_horizon,
        "state_dim": settings["state_dim"],
        "action_dim": settings["action_dim"],
        "allow_legacy_wam": False,
    }
    if settings["preload_data"]:
        return InMemoryProprioSequenceDataset(**common, progress=progress)
    return ProprioSequenceDataset(**common)


def _build_loader(
    dataset: Dataset,
    settings: Mapping[str, Any],
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    workers = int(settings["num_workers"])
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(settings["seed"]),
    )


def _train_config(settings: Mapping[str, Any]) -> RWMTrainConfig:
    loss = settings["loss_weights"]
    return RWMTrainConfig(
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        gradient_clip_norm=settings["gradient_clip_norm"],
        horizon_decay=settings["horizon_decay"],
        use_amp=settings["use_amp"],
        max_steps=settings["max_steps_per_stage"],
        loss_weights=RWMLossWeights(
            state_mean=float(loss["state_mean"]),
            state_nll=float(loss["state_nll"]),
            gripper_closed=float(loss["gripper_closed"]),
            reward=float(loss["reward"]),
            done=float(loss["done"]),
            terminal=float(loss["terminal"]),
            auxiliary=float(loss["auxiliary"]),
        ),
    )


@torch.no_grad()
def _check_reload_consistency(
    model: RWMARWorldModel,
    reloaded: RWMARWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    raw = next(iter(loader))
    batch = {
        name: value.to(device)
        for name, value in raw.items()
        if isinstance(value, torch.Tensor)
    }
    history = WorldModelSequenceInputs(
        batch["states"], batch["past_actions"], batch["valid_mask"]
    )
    horizon = min(16, batch["candidate_actions"].shape[1])
    first = model.predict(history, batch["candidate_actions"][:, :horizon])
    second = reloaded.predict(history, batch["candidate_actions"][:, :horizon])
    differences = {
        name: float((getattr(first, name) - getattr(second, name)).abs().max().cpu())
        for name in first.__dataclass_fields__
    }
    maximum = max(differences.values(), default=0.0)
    if maximum != 0.0:
        raise RuntimeError(f"checkpoint reload changed outputs by {maximum}")
    return {
        "passed": True,
        "strict": True,
        "max_abs_diff": maximum,
        "fields": differences,
    }


def _dataset_manifest(
    split_paths: Mapping[str, tuple[Path, ...]], settings: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": PROPRIO_WAM_SCHEMA_VERSION,
        "split_unit": "episode_seed",
        "split_seed": settings["split_seed"],
        "partitions": {
            name: [str(path.resolve()) for path in paths]
            for name, paths in split_paths.items()
        },
    }


def _provenance(
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "config_path": str(args.config.resolve()),
        "seed": settings["seed"],
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "external_checkpoints": [],
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
