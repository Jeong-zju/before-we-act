"""Train independent episode-bootstrap RWM-U members and ablations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TRAINING_SOURCE_PATHS = (
    "scripts/train_world_model_ensemble.py",
    "models/wam/api.py",
    "models/wam/config.py",
    "models/wam/ensemble.py",
    "models/wam/heads.py",
    "models/wam/normalizer.py",
    "models/wam/recurrent_dynamics.py",
    "models/wam/rollout.py",
    "models/wam/state_features.py",
    "train/rwm_ar_losses.py",
    "train/rwm_ar_trainer.py",
    "train/rwm_u_checkpointing.py",
    "train/rwm_u_trainer.py",
    "train/trajectory_dataset.py",
)

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from models.wam import (  # noqa: E402
    RWMARConfig,
    RWMARWorldModel,
    RWMUEnsemble,
    RWMUEnsembleConfig,
    RWMURiskConfig,
    WorldModelSequenceInputs,
)
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_ar_checkpointing import load_wam_checkpoint  # noqa: E402
from train.rwm_ar_losses import RWMLossWeights  # noqa: E402
from train.rwm_ar_trainer import (  # noqa: E402
    RWMTrainConfig,
    build_optimizer,
    evaluate_wam_loss,
    fit_wam_label_stats,
    make_positive_weights,
    train_curriculum_stage,
)
from train.rwm_u_checkpointing import (  # noqa: E402
    load_rwm_u_checkpoint,
    load_rwm_u_member_weights,
    load_teacher_forcing_weights,
    save_rwm_u_checkpoint,
    save_rwm_u_member_weights,
    save_teacher_forcing_weights,
)
from train.rwm_u_trainer import (  # noqa: E402
    EpisodeBootstrap,
    ensemble_parameter_diversity,
    make_episode_bootstrap,
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
        "--config", type=Path, default=ROOT / "configs/wam/world_model_ensemble.yaml"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--world-model-checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--forecast-curriculum", type=int, nargs="+")
    parser.add_argument("--curriculum-epochs", type=int, nargs="+")
    parser.add_argument("--validation-max-batches", type=int)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--max-steps-per-stage", type=int, default=-1)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use-amp", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--preload-data", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--skip-teacher-forcing-ablation", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only completed members whose run signature matches",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-refresh-hz", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    settings = _resolve_settings(args, config)
    _validate_settings(settings)
    device = _resolve_device(settings["device"])
    _seed_everything(settings["seed"])
    paths = discover_episode_paths(settings["data_dir"])
    if settings["max_episodes"] > 0:
        paths = paths[: settings["max_episodes"]]
    split_paths = split_episode_paths(paths, seed=settings["split_seed"])
    if not split_paths["train"] or not split_paths["validation"]:
        raise RuntimeError("world-model ensemble requires non-empty train and validation splits")

    source_model, source_metadata = load_wam_checkpoint(
        settings["world_model_checkpoint_dir"],
        device="cpu",
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    stats = source_metadata["normalization"]
    if (
        source_model.config.state_dim != settings["state_dim"]
        or source_model.config.action_dim != settings["action_dim"]
    ):
        raise ValueError("recurrent world model checkpoint dimensions do not match world-model ensemble config")
    _validate_source_data_provenance(
        source_metadata["dataset_manifest"],
        split_paths,
        smoke_subset=settings["max_episodes"] > 0,
    )
    del source_model

    checkpoint_dir = Path(settings["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_signature = _run_signature(config, settings, split_paths, stats.sha256())
    partial = _load_partial_state(checkpoint_dir, args.resume, run_signature)
    completed_members = {int(index) for index in partial.get("members", {})}
    teacher_complete = bool(partial.get("teacher_forcing"))
    teacher_enabled = bool(settings["teacher_forcing_ablation"])
    pending_model_count = settings["ensemble_size"] - len(completed_members)
    if teacher_enabled and not teacher_complete:
        pending_model_count += 1
    preload_stages = 2 if settings["preload_data"] else 0
    total_stages = (
        preload_stages
        + 1
        + pending_model_count * 2 * len(settings["forecast_curriculum"])
        + 1
    )
    datasets: dict[str, Dataset] = {}
    members: list[RWMARWorldModel] = []
    bootstraps: list[EpisodeBootstrap] = []

    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=total_stages,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        try:
            for split_name in ("train", "validation"):
                partition = split_paths[split_name]
                preload_phase = (
                    progress.add_phase(f"preload {split_name}", len(partition))
                    if settings["preload_data"]
                    else None
                )
                dataset = _build_dataset(
                    partition,
                    settings,
                    forecast_horizon=max(settings["forecast_curriculum"]),
                    progress=preload_phase.advance if preload_phase else None,
                )
                datasets[split_name] = dataset
                if preload_phase is not None:
                    assert isinstance(dataset, InMemoryProprioSequenceDataset)
                    preload_phase.finish(
                        f"{len(dataset)} fragments, {dataset.nbytes / 2**20:.1f} MiB"
                    )

            train_dataset = datasets["train"]
            validation_dataset = datasets["validation"]
            stats_loader = _build_loader(
                train_dataset,
                settings,
                shuffle=False,
                device=device,
                seed=settings["seed"],
            )
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
            positive_weights = make_positive_weights(label_stats, device)
            validation_loader = _build_loader(
                validation_dataset,
                settings,
                shuffle=False,
                device=device,
                seed=settings["seed"],
            )
            member_config = _member_config(settings)
            train_config = _train_config(settings)

            for member_index in range(settings["ensemble_size"]):
                member_seed = _member_seed(settings, member_index)
                bootstrap = make_episode_bootstrap(train_dataset, seed=member_seed)
                bootstraps.append(bootstrap)
                _seed_everything(member_seed)
                model = RWMARWorldModel(member_config, stats)
                if member_index in completed_members:
                    model = load_rwm_u_member_weights(
                        checkpoint_dir, member_index, model, device="cpu"
                    )
                    members.append(model.cpu().eval())
                    continue
                train_subset = Subset(train_dataset, bootstrap.sample_indices)
                train_loader = _build_loader(
                    train_subset,
                    settings,
                    shuffle=True,
                    device=device,
                    seed=member_seed,
                )
                model, member_metrics = _train_model(
                    model,
                    train_loader,
                    validation_loader,
                    settings,
                    train_config,
                    positive_weights,
                    device,
                    progress,
                    label=f"RWM-U member {member_index + 1}/{settings['ensemble_size']}",
                    teacher_forcing=False,
                )
                model.cpu().eval()
                save_rwm_u_member_weights(checkpoint_dir, member_index, model)
                partial.setdefault("members", {})[str(member_index)] = member_metrics
                _save_partial_state(checkpoint_dir, run_signature, partial)
                members.append(model)

            teacher_model: RWMARWorldModel | None = None
            if teacher_enabled:
                teacher_seed = _member_seed(settings, 0)
                _seed_everything(teacher_seed)
                teacher_model = RWMARWorldModel(member_config, stats)
                if teacher_complete:
                    teacher_model = load_teacher_forcing_weights(
                        checkpoint_dir, teacher_model, device="cpu"
                    )
                else:
                    teacher_subset = Subset(
                        train_dataset, bootstraps[0].sample_indices
                    )
                    teacher_loader = _build_loader(
                        teacher_subset,
                        settings,
                        shuffle=True,
                        device=device,
                        seed=teacher_seed,
                    )
                    teacher_model, teacher_metrics = _train_model(
                        teacher_model,
                        teacher_loader,
                        validation_loader,
                        settings,
                        train_config,
                        positive_weights,
                        device,
                        progress,
                        label="teacher-forcing ablation",
                        teacher_forcing=True,
                    )
                    teacher_model.cpu().eval()
                    save_teacher_forcing_weights(checkpoint_dir, teacher_model)
                    partial["teacher_forcing"] = teacher_metrics
                    _save_partial_state(checkpoint_dir, run_signature, partial)

            ensemble = RWMUEnsemble(
                members,
                RWMUEnsembleConfig(
                    ensemble_size=settings["ensemble_size"], bootstrap=True
                ),
                stats,
                risk_config=_risk_config(settings),
            ).cpu()
            training_metrics = {
                "format_version": "wam.world_model_ensemble.training/1",
                "outcome_label_stats": label_stats,
                "members": [
                    partial["members"][str(index)]
                    for index in range(settings["ensemble_size"])
                ],
                "teacher_forcing": partial.get("teacher_forcing"),
                "parameter_diversity": ensemble_parameter_diversity(ensemble),
            }
            save_phase = progress.add_phase("save and strict reload ensemble", 2)
            manifest = _dataset_manifest(split_paths, settings)
            bootstrap_manifest = _bootstrap_manifest(
                bootstraps, split_paths["train"]
            )
            provenance = _provenance(args, settings, device, run_signature)
            save_rwm_u_checkpoint(
                checkpoint_dir,
                ensemble,
                stats,
                teacher_forcing_model=teacher_model,
                experiment_config=config,
                dataset_manifest=manifest,
                bootstrap_manifest=bootstrap_manifest,
                metrics=training_metrics,
                provenance=provenance,
                schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            )
            save_phase.advance({"batch": 1})
            reloaded, _ = load_rwm_u_checkpoint(
                checkpoint_dir,
                device="cpu",
                expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            )
            reload_metrics = _check_reload_consistency(
                ensemble, reloaded, validation_loader
            )
            training_metrics["checkpoint_reload"] = reload_metrics
            save_rwm_u_checkpoint(
                checkpoint_dir,
                ensemble,
                stats,
                teacher_forcing_model=teacher_model,
                experiment_config=config,
                dataset_manifest=manifest,
                bootstrap_manifest=bootstrap_manifest,
                metrics=training_metrics,
                provenance=provenance,
                schema_version=PROPRIO_WAM_SCHEMA_VERSION,
            )
            save_phase.advance({"batch": 2})
            save_phase.finish(f"max_abs_diff {reload_metrics['max_abs_diff']:.3g}")
        finally:
            for dataset in datasets.values():
                close = getattr(dataset, "close", None)
                if close is not None:
                    close()

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_dir.resolve()),
                "ensemble_size": settings["ensemble_size"],
                "bootstrap": True,
                "teacher_forcing_ablation": teacher_enabled,
                "parameter_diversity": training_metrics["parameter_diversity"],
                "checkpoint_reload": training_metrics["checkpoint_reload"],
            },
            indent=2,
        )
    )
    return 0


def _train_model(
    model: RWMARWorldModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    settings: Mapping[str, Any],
    train_config: RWMTrainConfig,
    positive_weights: Mapping[str, torch.Tensor],
    device: torch.device,
    progress: TrainingProgress,
    *,
    label: str,
    teacher_forcing: bool,
) -> tuple[RWMARWorldModel, dict[str, Any]]:
    model.to(device)
    optimizer = build_optimizer(model, train_config)
    stages: list[dict[str, Any]] = []
    for horizon, epochs in zip(
        settings["forecast_curriculum"],
        settings["curriculum_epochs"],
        strict=True,
    ):
        possible_steps = epochs * len(train_loader)
        if train_config.max_steps > 0:
            possible_steps = min(possible_steps, train_config.max_steps)
        train_phase = progress.add_phase(
            f"train {label} H={horizon}", max(possible_steps, 1)
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
            progress=train_phase.advance,
            teacher_forcing=teacher_forcing,
        )
        final_loss = loss_history[-1] if loss_history else float("nan")
        train_phase.finish(f"{completed_steps} steps, final loss {final_loss:.5f}")
        validation_batches = len(validation_loader)
        if settings["validation_max_batches"] > 0:
            validation_batches = min(
                validation_batches, settings["validation_max_batches"]
            )
        validation_phase = progress.add_phase(
            f"validate {label} H={horizon}", max(validation_batches, 1)
        )
        validation = evaluate_wam_loss(
            model,
            validation_loader,
            horizon=horizon,
            config=train_config,
            positive_weights=positive_weights,
            device=device,
            max_batches=settings["validation_max_batches"],
            progress=validation_phase.advance,
            teacher_forcing=teacher_forcing,
        )
        validation_phase.finish(
            f"loss {float(validation.get('total', float('nan'))):.5f}"
        )
        stages.append(
            {
                "horizon": horizon,
                "epochs": epochs,
                "optimizer_steps": completed_steps,
                "loss_history": loss_history,
                "validation": validation,
                "teacher_forcing": teacher_forcing,
            }
        )
    return model, {"stages": stages}


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
    initialization = config["initialization"]
    return {
        "data_dir": args.data_dir or ROOT / data["directory"],
        "checkpoint_dir": args.checkpoint_dir or ROOT / checkpoint["directory"],
        "world_model_checkpoint_dir": args.world_model_checkpoint_dir
        or ROOT / initialization["checkpoint"],
        "device": args.device,
        "state_dim": int(data["state_dim"]),
        "action_dim": int(data["action_dim"]),
        "history_horizon": int(data["history_horizon"]),
        "planning_horizon": int(data["planning_horizon"]),
        "split_seed": int(data["split_seed"]),
        "yaw_indices": tuple(features["yaw_indices"]),
        "gripper_closed_indices": tuple(features["gripper_closed_indices"]),
        "predict_delta": bool(features["predict_delta"]),
        "family": str(model["family"]),
        "ensemble_size": int(args.ensemble_size or model["ensemble_size"]),
        "bootstrap": bool(model["bootstrap"]),
        "encoder_hidden_dim": int(model["encoder_hidden_dim"]),
        "gru_hidden_dim": int(model["gru_hidden_dim"]),
        "gru_layers": int(model["gru_layers"]),
        "dropout": float(model["dropout"]),
        "min_log_std": float(model["min_log_std"]),
        "max_log_std": float(model["max_log_std"]),
        "risk": dict(config["risk"]),
        "forecast_curriculum": tuple(
            int(value)
            for value in (
                args.forecast_curriculum or training["forecast_curriculum"]
            )
        ),
        "curriculum_epochs": tuple(
            int(value)
            for value in (args.curriculum_epochs or training["curriculum_epochs"])
        ),
        "batch_size": int(args.batch_size or training["batch_size"]),
        "num_workers": int(
            training["num_workers"] if args.num_workers is None else args.num_workers
        ),
        "seed": int(training["seed"] if args.seed is None else args.seed),
        "member_seed_stride": int(training["member_seed_stride"]),
        "learning_rate": float(
            training["learning_rate"]
            if args.learning_rate is None
            else args.learning_rate
        ),
        "weight_decay": float(
            training["weight_decay"] if args.weight_decay is None else args.weight_decay
        ),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "horizon_decay": float(training["horizon_decay"]),
        "use_amp": bool(training["use_amp"] if args.use_amp is None else args.use_amp),
        "preload_data": bool(
            training["preload_data"]
            if args.preload_data is None
            else args.preload_data
        ),
        "teacher_forcing_ablation": bool(training["teacher_forcing_ablation"])
        and not args.skip_teacher_forcing_ablation,
        "training_mode": str(training["mode"]),
        "privileged_inputs": bool(training["privileged_inputs"]),
        "loss_weights": dict(training["loss_weights"]),
        "validation_max_batches": int(
            evaluation["validation_max_batches"]
            if args.validation_max_batches is None
            else args.validation_max_batches
        ),
        "max_episodes": int(args.max_episodes),
        "max_steps_per_stage": int(args.max_steps_per_stage),
    }


def _validate_settings(settings: Mapping[str, Any]) -> None:
    if settings["family"] != "rwm_u" or settings["ensemble_size"] < 2:
        raise ValueError("world-model ensemble requires model.family=rwm_u and ensemble_size>=2")
    if not settings["bootstrap"]:
        raise ValueError("world-model ensemble requires episode bootstrap sampling")
    if not settings["predict_delta"] or settings["training_mode"] != "autoregressive":
        raise ValueError("world-model ensemble requires autoregressive delta prediction")
    if settings["privileged_inputs"]:
        raise ValueError("world-model ensemble runtime inputs cannot include privileged state")
    if len(settings["forecast_curriculum"]) != len(settings["curriculum_epochs"]):
        raise ValueError("forecast curriculum and epoch lists must have equal length")
    if tuple(sorted(settings["forecast_curriculum"])) != settings["forecast_curriculum"]:
        raise ValueError("forecast curriculum must be non-decreasing")
    for name in (
        "state_dim",
        "action_dim",
        "history_horizon",
        "batch_size",
        "member_seed_stride",
    ):
        if settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if settings["max_episodes"] == 0 or settings["max_episodes"] < -1:
        raise ValueError("max_episodes must be -1 or positive")
    if settings["max_steps_per_stage"] == 0 or settings["max_steps_per_stage"] < -1:
        raise ValueError("max_steps_per_stage must be -1 or positive")
    if (
        settings["validation_max_batches"] == 0
        or settings["validation_max_batches"] < -1
    ):
        raise ValueError("validation_max_batches must be -1 or positive")


def _member_config(settings: Mapping[str, Any]) -> RWMARConfig:
    return RWMARConfig(
        state_dim=settings["state_dim"],
        action_dim=settings["action_dim"],
        history_horizon=settings["history_horizon"],
        train_forecast_horizon=max(settings["forecast_curriculum"]),
        planning_horizon=settings["planning_horizon"],
        encoder_hidden_dim=settings["encoder_hidden_dim"],
        gru_hidden_dim=settings["gru_hidden_dim"],
        gru_layers=settings["gru_layers"],
        dropout=settings["dropout"],
        min_log_std=settings["min_log_std"],
        max_log_std=settings["max_log_std"],
        yaw_indices=settings["yaw_indices"],
        gripper_closed_indices=settings["gripper_closed_indices"],
    )


def _risk_config(settings: Mapping[str, Any]) -> RWMURiskConfig:
    return RWMURiskConfig(**{name: float(value) for name, value in settings["risk"].items()})


def _train_config(settings: Mapping[str, Any]) -> RWMTrainConfig:
    loss = settings["loss_weights"]
    return RWMTrainConfig(
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        gradient_clip_norm=settings["gradient_clip_norm"],
        horizon_decay=settings["horizon_decay"],
        use_amp=settings["use_amp"],
        max_steps=settings["max_steps_per_stage"],
        loss_weights=RWMLossWeights(**{name: float(value) for name, value in loss.items()}),
    )


def _build_dataset(
    paths: Sequence[Path],
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
    seed: int,
) -> DataLoader:
    workers = int(settings["num_workers"])
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def _check_reload_consistency(
    original: RWMUEnsemble, reloaded: RWMUEnsemble, loader: DataLoader
) -> dict[str, Any]:
    raw = next(iter(loader))
    history = WorldModelSequenceInputs(
        raw["states"], raw["past_actions"], raw["valid_mask"]
    )
    horizon = min(16, raw["candidate_actions"].shape[1])
    first = original.predict(history, raw["candidate_actions"][:, :horizon])
    second = reloaded.predict(history, raw["candidate_actions"][:, :horizon])
    differences = {
        name: float((getattr(first, name) - getattr(second, name)).abs().max())
        for name in first.__dataclass_fields__
    }
    maximum = max(differences.values(), default=0.0)
    if maximum != 0.0:
        raise RuntimeError(f"RWM-U checkpoint reload changed outputs by {maximum}")
    return {"passed": True, "strict": True, "max_abs_diff": maximum, "fields": differences}


def _member_seed(settings: Mapping[str, Any], member_index: int) -> int:
    return int(settings["seed"] + (member_index + 1) * settings["member_seed_stride"])


def _dataset_manifest(
    split_paths: Mapping[str, Sequence[Path]], settings: Mapping[str, Any]
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


def _validate_source_data_provenance(
    source_manifest: Mapping[str, Any],
    split_paths: Mapping[str, Sequence[Path]],
    *,
    smoke_subset: bool,
) -> None:
    if smoke_subset:
        return
    reference = source_manifest.get("partitions")
    if not isinstance(reference, Mapping):
        raise ValueError("recurrent world model checkpoint has no auditable dataset partitions")
    current = {
        name: [str(path.resolve()) for path in paths]
        for name, paths in split_paths.items()
    }
    normalized_reference = {
        name: [str(Path(path).resolve()) for path in reference.get(name, [])]
        for name in current
    }
    if current != normalized_reference:
        raise ValueError(
            "world-model ensemble data split differs from the recurrent world model normalization provenance; "
            "retrain recurrent world model or explicitly run a limited --max-episodes smoke test"
        )


def _bootstrap_manifest(
    bootstraps: Sequence[EpisodeBootstrap], train_paths: Sequence[Path]
) -> dict[str, Any]:
    members = []
    for index, bootstrap in enumerate(bootstraps):
        item = bootstrap.manifest()
        item["member_index"] = index
        item["drawn_episode_paths"] = [
            str(train_paths[position].resolve()) for position in bootstrap.episode_draws
        ]
        members.append(item)
    return {
        "format_version": "wam.rwm_u.bootstrap/1",
        "sampling_unit": "episode",
        "with_replacement": True,
        "members": members,
    }


def _run_signature(
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    split_paths: Mapping[str, Sequence[Path]],
    normalization_hash: str,
) -> str:
    payload = {
        "config": config,
        "settings": {
            name: value
            for name, value in settings.items()
            if name not in {"device", "checkpoint_dir"}
        },
        "paths": {
            name: [str(path.resolve()) for path in values]
            for name, values in split_paths.items()
        },
        "normalization_sha256": normalization_hash,
    }
    return hashlib.sha256(
        json.dumps(_plain(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_partial_state(
    checkpoint_dir: Path, resume: bool, run_signature: str
) -> dict[str, Any]:
    path = checkpoint_dir / "partial_training_metrics.json"
    if not resume:
        return {"members": {}}
    if not path.is_file():
        raise FileNotFoundError("--resume requires partial_training_metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_signature") != run_signature:
        raise ValueError("resume run signature does not match config/data/normalization")
    return dict(payload.get("state", {}))


def _save_partial_state(
    checkpoint_dir: Path, run_signature: str, state: Mapping[str, Any]
) -> None:
    payload = {
        "format_version": "wam.world_model_ensemble.partial/1",
        "run_signature": run_signature,
        "state": state,
    }
    (checkpoint_dir / "partial_training_metrics.json").write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _provenance(
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    device: torch.device,
    run_signature: str,
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
    try:
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_dirty = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_files_sha256": _training_source_hashes(args.config),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "config_path": str(args.config.resolve()),
        "run_signature": run_signature,
        "seed": settings["seed"],
        "member_seed_stride": settings["member_seed_stride"],
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "external_checkpoints": [str(settings["world_model_checkpoint_dir"])],
    }


def _training_source_hashes(config_path: Path) -> dict[str, str]:
    """Fingerprint the files that define training behavior for dirty-tree runs."""

    paths = [config_path.resolve(), *(ROOT / value for value in _TRAINING_SOURCE_PATHS)]
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"training provenance source is missing: {resolved}")
        try:
            name = str(resolved.relative_to(ROOT))
        except ValueError:
            name = str(resolved)
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return dict(sorted(hashes.items()))


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


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
