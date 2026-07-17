"""Evaluate RWM-AR rollouts and emit an auditable report."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from safetensors.torch import load_file
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from eval.rwm_ar_open_loop import (  # noqa: E402
    RecursiveBaseline,
    evaluate_open_loop,
)
from models import OneStepMLPWorldModel, OneStepMLPWorldModelConfig  # noqa: E402
from models.wam import (  # noqa: E402
    NormalizationStats,
    RWMARWorldModel,
    WorldModelSequenceInputs,
)
from models.wam.rollout import wrap_to_pi  # noqa: E402
from train.rwm_ar_checkpointing import load_wam_checkpoint  # noqa: E402
from train.progress import TrainingProgress  # noqa: E402
from train.trajectory_dataset import (  # noqa: E402
    InMemoryProprioSequenceDataset,
    ProprioSequenceDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam/world_model.yaml",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints/world_model",
    )
    parser.add_argument(
        "--baseline-output-dir",
        type=Path,
        default=ROOT / "outputs/baselines",
    )
    parser.add_argument("--overfit-checkpoint-dir", type=Path)
    parser.add_argument(
        "--overfit-only",
        action="store_true",
        help="audit only --overfit-checkpoint-dir without requiring a full checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/world_model",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument(
        "--preload-data", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-refresh-hz", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = _resolve_device(args.device)
    if args.overfit_only:
        return _run_overfit_only(args, config, device)
    horizons = tuple(
        sorted(
            {
                *(int(item) for item in config["evaluation"]["open_loop_horizons"]),
                int(config["evaluation"]["acceptance_rollout_horizon"]),
            }
        )
    )
    model, metadata = load_wam_checkpoint(
        args.checkpoint_dir,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    stats = metadata["normalization"]
    baseline_mlp = _load_baseline_mlp(args.baseline_output_dir, device)
    partitions = metadata["dataset_manifest"]["partitions"]
    evaluation_paths = {
        name: tuple(Path(path) for path in partitions[name])
        for name in ("validation", "test")
    }
    if args.max_episodes > 0:
        evaluation_paths = {
            name: paths[: args.max_episodes] for name, paths in evaluation_paths.items()
        }
    preload_stages = 2 if args.preload_data else 0
    overfit_stages = 2 if args.overfit_checkpoint_dir is not None else 0
    total_stages = preload_stages + 2 + overfit_stages + 1
    datasets: dict[str, Dataset] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=total_stages,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        try:
            for split_name, paths in evaluation_paths.items():
                preload = (
                    progress.add_phase(f"preload {split_name}", len(paths))
                    if args.preload_data
                    else None
                )
                dataset = _build_dataset(
                    paths,
                    history_horizon=model.config.history_horizon,
                    forecast_horizon=max(horizons),
                    preload=args.preload_data,
                    progress=preload.advance if preload else None,
                )
                datasets[split_name] = dataset
                if preload is not None:
                    assert isinstance(dataset, InMemoryProprioSequenceDataset)
                    preload.finish(
                        f"{len(dataset)} fragments, {dataset.nbytes / 2**20:.1f} MiB"
                    )

            validation_loader = _loader(datasets["validation"], args, device)
            validation_total = _effective_batches(validation_loader, args.max_batches)
            validation_progress = progress.add_phase(
                "evaluate validation", validation_total
            )
            validation_metrics, _ = evaluate_open_loop(
                model,
                validation_loader,
                stats,
                device=device,
                horizons=horizons,
                baseline_mlp=baseline_mlp,
                calibrate_thresholds=True,
                max_batches=args.max_batches,
                progress=validation_progress.advance,
            )
            validation_progress.finish(
                _metric_summary(validation_metrics, horizon=max(horizons))
            )
            thresholds = {
                label: float(values["threshold"])
                for label, values in validation_metrics["outcome_calibration"].items()
            }

            test_loader = _loader(datasets["test"], args, device)
            test_total = _effective_batches(test_loader, args.max_batches)
            test_progress = progress.add_phase("evaluate test", test_total)
            test_metrics, example = evaluate_open_loop(
                model,
                test_loader,
                stats,
                device=device,
                horizons=horizons,
                baseline_mlp=baseline_mlp,
                classification_thresholds=thresholds,
                max_batches=args.max_batches,
                progress=test_progress.advance,
            )
            test_progress.finish(_metric_summary(test_metrics, horizon=max(horizons)))

            overfit_result: dict[str, Any] | None = None
            if args.overfit_checkpoint_dir is not None:
                overfit_result = _evaluate_overfit_checkpoint(
                    args,
                    config,
                    device,
                    progress,
                )

            output_progress = progress.add_phase("write report and rollout", 3)
            acceptance = _acceptance_report(
                test_metrics,
                metadata["metrics"],
                overfit_result,
                config,
                full_test_split=(args.max_batches == -1 and args.max_episodes == -1),
            )
            report = {
                "format_version": "wam.world_model.open_loop/1",
                "checkpoint": str(args.checkpoint_dir.resolve()),
                "baseline_reference": str(args.baseline_output_dir.resolve()),
                "validation": validation_metrics,
                "test": test_metrics,
                "overfit": overfit_result,
                "acceptance": acceptance,
            }
            metrics_path = args.output_dir / "open_loop_metrics.json"
            metrics_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            output_progress.advance({"batch": 1})
            np.savez(args.output_dir / "rollout_example.npz", **example)
            output_progress.advance({"batch": 2})
            _write_rollout_svg(args.output_dir / "rollout_example.svg", example)
            output_progress.advance({"batch": 3})
            output_progress.finish(
                "world-model acceptance " + ("passed" if acceptance["passed"] else "not passed")
            )
        finally:
            for dataset in datasets.values():
                close = getattr(dataset, "close", None)
                if close is not None:
                    close()

    print(
        json.dumps(
            {
                "metrics": str((args.output_dir / "open_loop_metrics.json").resolve()),
                "acceptance": acceptance,
            },
            indent=2,
        )
    )
    return 0 if acceptance["passed"] else 2


def _run_overfit_only(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    device: torch.device,
) -> int:
    if args.overfit_checkpoint_dir is None:
        raise ValueError("--overfit-only requires --overfit-checkpoint-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=3,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        result = _evaluate_overfit_checkpoint(args, config, device, progress)
        output = progress.add_phase("write overfit audit", 1)
        report = {
            "format_version": "wam.world_model.overfit_audit/1",
            "overfit": result,
        }
        path = args.output_dir / "overfit_metrics.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        output.advance({"batch": 1})
        output.finish("overfit " + ("passed" if result["passed"] else "not passed"))
    print(json.dumps({"metrics": str(path.resolve()), "overfit": result}, indent=2))
    return 0 if result["passed"] else 2


def _evaluate_overfit_checkpoint(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    device: torch.device,
    progress: TrainingProgress,
) -> dict[str, Any]:
    assert args.overfit_checkpoint_dir is not None
    model, metadata = load_wam_checkpoint(
        args.overfit_checkpoint_dir,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    sample_count = int(metadata["metrics"].get("overfit_samples", 0))
    paths = tuple(
        Path(path) for path in metadata["dataset_manifest"]["partitions"]["train"]
    )
    paths = paths[: min(len(paths), max(sample_count, 1))]
    preload = progress.add_phase("preload overfit train", len(paths))
    dataset = _build_dataset(
        paths,
        history_horizon=model.config.history_horizon,
        forecast_horizon=16,
        preload=True,
        progress=preload.advance,
    )
    assert isinstance(dataset, InMemoryProprioSequenceDataset)
    preload.finish(f"{len(dataset)} fragments")
    try:
        subset: Dataset = Subset(dataset, range(min(sample_count, len(dataset))))
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        phase = progress.add_phase("evaluate overfit train", max(len(loader), 1))
        metrics, _ = evaluate_open_loop(
            model,
            loader,
            metadata["normalization"],
            device=device,
            horizons=(1, 16),
            max_batches=-1,
            progress=phase.advance,
        )
        distribution = _overfit_distribution_diagnostics(model, loader, device)
        one_step = metrics["models"]["rwm_ar"]["exact_horizon"]["1"]
        nrmse = float(one_step["continuous_state_nrmse"])
        closed_rmse = float(one_step["gripper_closed_rmse"])
        phase.finish(f"continuous NRMSE {nrmse:.5f}, closed RMSE {closed_rmse:.5f}")
        nrmse_limit = float(config["evaluation"]["acceptance_overfit_nrmse_max"])
        closed_limit = float(
            config["evaluation"]["acceptance_overfit_closed_rmse_max"]
        )
        return {
            "checkpoint": str(args.overfit_checkpoint_dir.resolve()),
            "samples": sample_count,
            "metrics": metrics,
            "distribution_diagnostics": distribution,
            "criterion": {
                "continuous_state_nrmse_max": nrmse_limit,
                "gripper_closed_rmse_max": closed_limit,
            },
            "passed": (
                100 <= sample_count <= 500
                and nrmse <= nrmse_limit
                and closed_rmse <= closed_limit
                and distribution["all_finite"]
            ),
        }
    finally:
        dataset.close()


@torch.no_grad()
def _overfit_distribution_diagnostics(
    model: RWMARWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    log_stds: list[torch.Tensor] = []
    normalized_errors: list[torch.Tensor] = []
    model.eval()
    for raw_batch in loader:
        batch = {
            name: value.to(device, non_blocking=True)
            for name, value in raw_batch.items()
            if isinstance(value, torch.Tensor)
        }
        horizon = min(16, int(batch["candidate_actions"].shape[1]))
        predictions = model.predict(
            WorldModelSequenceInputs(
                batch["states"], batch["past_actions"], batch["valid_mask"]
            ),
            batch["candidate_actions"][:, :horizon],
        )
        error = predictions.next_state_mean - batch["target_states"][:, :horizon]
        for yaw_index in model.config.yaw_indices:
            error[..., yaw_index] = wrap_to_pi(error[..., yaw_index])
        continuous = model.continuous_state_mask
        valid = batch["forecast_mask"][:, :horizon].bool()
        log_stds.append(
            predictions.normalized_delta_log_std[..., continuous][valid]
            .float()
            .cpu()
            .reshape(-1)
        )
        normalized_errors.append(
            (error[..., continuous] / model.delta_std[continuous])[valid]
            .float()
            .abs()
            .cpu()
            .reshape(-1)
        )
    if not log_stds or not normalized_errors:
        raise ValueError("overfit distribution diagnostics received no valid values")
    log_std = torch.cat(log_stds)
    error = torch.cat(normalized_errors)
    finite = torch.isfinite(log_std).all() and torch.isfinite(error).all()
    finite_log_std = log_std[torch.isfinite(log_std)]
    finite_error = error[torch.isfinite(error)]
    if finite_log_std.numel() == 0 or finite_error.numel() == 0:
        return {"all_finite": False, "values": int(log_std.numel())}
    quantiles = torch.tensor([0.01, 0.5, 0.99])
    log_q = torch.quantile(finite_log_std, quantiles)
    error_q = torch.quantile(finite_error, quantiles)
    return {
        "all_finite": bool(finite),
        "values": int(log_std.numel()),
        "normalized_delta_log_std": {
            "min": float(finite_log_std.min()),
            "p01": float(log_q[0]),
            "p50": float(log_q[1]),
            "p99": float(log_q[2]),
            "max": float(finite_log_std.max()),
            "fraction_at_min_bound": float(
                torch.isclose(
                    finite_log_std,
                    torch.tensor(model.config.min_log_std),
                    atol=1e-6,
                )
                .float()
                .mean()
            ),
            "fraction_at_max_bound": float(
                torch.isclose(
                    finite_log_std,
                    torch.tensor(model.config.max_log_std),
                    atol=1e-6,
                )
                .float()
                .mean()
            ),
        },
        "absolute_normalized_delta_error": {
            "p01": float(error_q[0]),
            "p50": float(error_q[1]),
            "p99": float(error_q[2]),
            "max": float(finite_error.max()),
        },
    }


def _acceptance_report(
    test: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    overfit: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    *,
    full_test_split: bool,
) -> dict[str, Any]:
    models = test["models"]
    one_step = {
        name: values["exact_horizon"]["1"]["continuous_state_nrmse"]
        for name, values in models.items()
    }
    beats_baselines = (
        one_step["rwm_ar"] < one_step["constant_velocity"]
        and one_step["rwm_ar"] < one_step["baseline_mlp_recursive"]
    )
    stability = models["rwm_ar"]["exact_horizon"]["16"]
    violation_limit = float(config["evaluation"]["acceptance_constraint_violation_max"])
    stable = (
        stability["finite_rollout_rate"] == 1.0
        and stability["state_constraint_violation_rate"] <= violation_limit
    )
    reload_check = training_metrics.get("checkpoint_reload", {})
    checks = {
        "overfit_100_to_500_fragments": {
            "passed": bool(overfit and overfit.get("passed")),
            "evidence": overfit,
        },
        "one_step_beats_constant_velocity_and_baseline_mlp": {
            "passed": beats_baselines,
            "continuous_state_nrmse": one_step,
        },
        "rollout_16_steps_is_finite_and_bounded": {
            "passed": stable,
            "finite_rollout_rate": stability["finite_rollout_rate"],
            "state_constraint_violation_rate": stability[
                "state_constraint_violation_rate"
            ],
            "constraint_violation_limit": violation_limit,
        },
        "strict_checkpoint_reload_is_elementwise_identical": {
            "passed": bool(
                reload_check.get("passed") and reload_check.get("max_abs_diff") == 0.0
            ),
            "evidence": reload_check,
        },
    }
    return {
        "passed": full_test_split and all(item["passed"] for item in checks.values()),
        "full_test_split_evaluated": full_test_split,
        "checks": checks,
        "note": "world-model acceptance only; this is not a ensemble uncertainty acceptance or world-model ensemble uncertainty claim.",
    }


def _build_dataset(
    paths: tuple[Path, ...],
    *,
    history_horizon: int,
    forecast_horizon: int,
    preload: bool,
    progress: Any | None,
) -> Dataset:
    common = {
        "paths": paths,
        "history_horizon": history_horizon,
        "forecast_horizon": forecast_horizon,
        "allow_legacy_wam": False,
    }
    if preload:
        return InMemoryProprioSequenceDataset(**common, progress=progress)
    return ProprioSequenceDataset(**common)


def _loader(
    dataset: Dataset, args: argparse.Namespace, device: torch.device
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def _load_baseline_mlp(path: Path, device: torch.device) -> RecursiveBaseline:
    config = OneStepMLPWorldModelConfig(
        **json.loads((path / "mlp_config.json").read_text())
    )
    model = OneStepMLPWorldModel(config)
    incompatible = model.load_state_dict(
        load_file(path / "mlp.safetensors", device=str(device)), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"baseline MLP strict load failed: {incompatible}")
    model.to(device).eval()
    return RecursiveBaseline(
        model=model,
        stats=NormalizationStats.load(path / "normalization.npz"),
    )


def _write_rollout_svg(path: Path, example: Mapping[str, np.ndarray]) -> None:
    width, height = 1100, 720
    margin_left, margin_right = 72, 24
    panel_height, panel_gap = 135, 24
    channels = (
        (4, "robot 0 vy"),
        (15, "robot 1 vy"),
        (0, "robot 0 x"),
        (11, "robot 1 x"),
    )
    colors = {
        "actual": "#f8fafc",
        "rwm_ar": "#22d3ee",
        "baseline_mlp_recursive": "#f59e0b",
        "constant_velocity": "#a78bfa",
    }
    names = [name for name in colors if name in example]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<text x="72" y="34" fill="#e2e8f0" font-family="monospace" font-size="20">World-model open-loop rollout</text>',
    ]
    plot_width = width - margin_left - margin_right
    for panel, (channel, label) in enumerate(channels):
        top = 58 + panel * (panel_height + panel_gap)
        values = np.concatenate(
            [np.asarray(example[name])[:, channel] for name in names]
        )
        finite = values[np.isfinite(values)]
        lower = float(finite.min()) if finite.size else -1.0
        upper = float(finite.max()) if finite.size else 1.0
        padding = max((upper - lower) * 0.08, 1e-5)
        lower -= padding
        upper += padding
        parts.extend(
            (
                f'<rect x="{margin_left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#111827" stroke="#334155"/>',
                f'<text x="{margin_left + 8}" y="{top + 18}" fill="#94a3b8" font-family="monospace" font-size="13">{escape(label)} [{lower:.3g}, {upper:.3g}]</text>',
            )
        )
        for name in names:
            series = np.asarray(example[name])[:, channel]
            points = []
            for index, value in enumerate(series):
                x = margin_left + plot_width * index / max(len(series) - 1, 1)
                y = top + panel_height * (upper - float(value)) / (upper - lower)
                points.append(f"{x:.2f},{y:.2f}")
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[name]}" stroke-width="{3 if name == "actual" else 2}" opacity="0.95"/>'
            )
    legend_x = 72
    for name in names:
        parts.extend(
            (
                f'<line x1="{legend_x}" y1="{height - 20}" x2="{legend_x + 25}" y2="{height - 20}" stroke="{colors[name]}" stroke-width="3"/>',
                f'<text x="{legend_x + 31}" y="{height - 15}" fill="#cbd5e1" font-family="monospace" font-size="13">{escape(name)}</text>',
            )
        )
        legend_x += 210
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _metric_summary(metrics: Mapping[str, Any], *, horizon: int) -> str:
    value = metrics["models"]["rwm_ar"]["exact_horizon"][str(horizon)]
    return f"H={horizon} continuous NRMSE {value['continuous_state_nrmse']:.5f}"


def _effective_batches(loader: DataLoader, maximum: int) -> int:
    return max(min(len(loader), maximum) if maximum > 0 else len(loader), 1)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    for name in ("max_batches", "max_episodes"):
        value = getattr(args, name)
        if value == 0 or value < -1:
            raise ValueError(f"{name} must be -1 or positive")


if __name__ == "__main__":
    raise SystemExit(main())
