"""Calibrate RWM-U uncertainty and emit an auditable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from eval.uncertainty import (  # noqa: E402
    OODActionPerturbation,
    evaluate_rwm_u,
    fit_variance_calibration,
)
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_u_checkpointing import (  # noqa: E402
    load_rwm_u_checkpoint,
    load_teacher_forcing_ablation,
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
    parser.add_argument("--baseline-mlp-nrmse", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/joint_wam/world_model",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--validation-max-batches", type=int)
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
    config = _load_config(args.config)
    settings = _resolve_settings(args, config)
    device = _resolve_device(args.device)
    ensemble, metadata = load_rwm_u_checkpoint(
        settings["checkpoint_dir"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    try:
        teacher_model = load_teacher_forcing_ablation(
            settings["checkpoint_dir"], device=device
        )
    except FileNotFoundError:
        teacher_model = None
    stats = metadata["normalization"]
    paths = discover_episode_paths(settings["data_dir"])
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]
    split_paths = split_episode_paths(paths, seed=settings["split_seed"])
    if not split_paths["validation"] or not split_paths["test"]:
        raise RuntimeError("uncertainty acceptance requires non-empty validation and test splits")
    _validate_checkpoint_data_provenance(
        metadata["dataset_manifest"],
        split_paths,
        smoke_subset=args.max_episodes > 0,
    )

    horizons = settings["horizons"]
    preload_stages = 2 if args.preload_data else 0
    total_stages = preload_stages + 3
    datasets: dict[str, Dataset] = {}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=total_stages,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        try:
            for split_name in ("validation", "test"):
                partition = split_paths[split_name]
                preload_phase = (
                    progress.add_phase(f"preload {split_name}", len(partition))
                    if args.preload_data
                    else None
                )
                dataset = _build_dataset(
                    partition,
                    settings,
                    forecast_horizon=max(horizons),
                    preload=args.preload_data,
                    progress=preload_phase.advance if preload_phase else None,
                )
                datasets[split_name] = dataset
                if preload_phase is not None:
                    assert isinstance(dataset, InMemoryProprioSequenceDataset)
                    preload_phase.finish(
                        f"{len(dataset)} fragments, {dataset.nbytes / 2**20:.1f} MiB"
                    )

            validation_loader = _build_loader(
                datasets["validation"], args, device
            )
            test_loader = _build_loader(datasets["test"], args, device)
            calibration_batches = _effective_batches(
                validation_loader, settings["validation_max_batches"]
            )
            calibration_phase = progress.add_phase(
                "calibrate validation uncertainty", calibration_batches
            )
            calibration = fit_variance_calibration(
                ensemble,
                validation_loader,
                device=device,
                horizon=max(horizons),
                max_batches=settings["validation_max_batches"],
                scale_min=settings["variance_scale_min"],
                scale_max=settings["variance_scale_max"],
                progress=calibration_phase.advance,
            )
            calibration_phase.finish(
                f"{calibration['forecast_values']} state values"
            )
            _write_json(output_dir / "uncertainty_calibration.json", calibration)

            test_phase = progress.add_phase(
                "evaluate uncertainty and OOD",
                _effective_batches(test_loader, args.max_batches),
            )
            evaluation = evaluate_rwm_u(
                ensemble,
                test_loader,
                stats,
                device=device,
                calibration=calibration,
                horizons=horizons,
                teacher_forcing_model=teacher_model,
                ood=OODActionPerturbation(
                    scale=settings["ood_action_scale"],
                    offset_std=settings["ood_action_offset_std"],
                    action_low=settings["ood_action_low"],
                    action_high=settings["ood_action_high"],
                ),
                event_horizon=settings["event_horizon"],
                event_progress_min=settings["event_progress_min"],
                event_slowdown_min=settings["event_slowdown_min"],
                event_asymmetry_min=settings["event_asymmetry_min"],
                event_ambiguity_fraction=settings["event_ambiguity_fraction"],
                max_batches=args.max_batches,
                progress=test_phase.advance,
            )
            acceptance = _acceptance_report(
                evaluation,
                metadata,
                settings,
                full_test_split_evaluated=(
                    args.max_batches < 0 and args.max_episodes < 0
                ),
                teacher_forcing_available=teacher_model is not None,
            )
            test_phase.finish("uncertainty acceptance " + ("passed" if acceptance["passed"] else "failed"))

            report_phase = progress.add_phase("write uncertainty acceptance report", 2)
            report = {
                "format_version": "wam.world_model_ensemble.uncertainty/1",
                "checkpoint": str(Path(settings["checkpoint_dir"]).resolve()),
                "baseline_mlp_h20_nrmse": settings["baseline_mlp_nrmse"],
                "calibration": calibration,
                "test": evaluation,
                "acceptance": acceptance,
            }
            _write_json(output_dir / "uncertainty_metrics.json", report)
            report_phase.advance({"batch": 1})
            (output_dir / "uncertainty_report.md").write_text(
                _markdown_report(report), encoding="utf-8"
            )
            report_phase.advance({"batch": 2})
            report_phase.finish("JSON and Markdown written")
        finally:
            for dataset in datasets.values():
                close = getattr(dataset, "close", None)
                if close is not None:
                    close()

    print(json.dumps({"output": str(output_dir.resolve()), "acceptance": acceptance}, indent=2))
    return 0 if acceptance["passed"] else 2


def _acceptance_report(
    evaluation: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    full_test_split_evaluated: bool,
    teacher_forcing_available: bool,
) -> dict[str, Any]:
    horizon = str(settings["acceptance_horizon"])
    exact = evaluation["exact_horizon"][horizon]
    ood = evaluation["ood"]["exact_horizon"][horizon]
    event = evaluation["event_aligned"]
    ensemble_nrmse = float(exact["ensemble_mean_continuous_nrmse"])
    mlp_nrmse = float(settings["baseline_mlp_nrmse"])
    mlp_improvement = (mlp_nrmse - ensemble_nrmse) / mlp_nrmse
    member_nrmse = float(exact["member0_continuous_nrmse"])
    teacher_nrmse = exact["teacher_forcing_continuous_nrmse"]
    ar_improvement = (
        (float(teacher_nrmse) - member_nrmse) / float(teacher_nrmse)
        if teacher_nrmse not in (None, 0.0)
        else None
    )
    diversity = checkpoint_metadata["metrics"].get("parameter_diversity", {})
    checks = {
        "h20_vs_baseline_mlp": {
            "passed": mlp_improvement >= settings["acceptance_mlp_improvement_min"],
            "ensemble_nrmse": ensemble_nrmse,
            "baseline_mlp_nrmse": mlp_nrmse,
            "relative_improvement": mlp_improvement,
            "minimum": settings["acceptance_mlp_improvement_min"],
        },
        "autoregressive_vs_teacher_forcing": {
            "passed": bool(
                teacher_forcing_available
                and ar_improvement is not None
                and ar_improvement
                >= settings["acceptance_ar_over_teacher_forcing_min"]
            ),
            "member0_nrmse": member_nrmse,
            "teacher_forcing_nrmse": teacher_nrmse,
            "relative_improvement": ar_improvement,
            "minimum": settings["acceptance_ar_over_teacher_forcing_min"],
        },
        "uncertainty_error_correlation": {
            "passed": bool(
                exact["uncertainty_error_spearman"] is not None
                and exact["uncertainty_error_spearman"]
                >= settings["acceptance_uncertainty_error_spearman_min"]
            ),
            "spearman": exact["uncertainty_error_spearman"],
            "minimum": settings["acceptance_uncertainty_error_spearman_min"],
        },
        "ood_identification": {
            "passed": bool(
                ood["auroc"] is not None
                and ood["auroc"] >= settings["acceptance_ood_auroc_min"]
                and ood["epistemic_ratio"]
                >= settings["acceptance_ood_epistemic_ratio_min"]
            ),
            "auroc": ood["auroc"],
            "auroc_minimum": settings["acceptance_ood_auroc_min"],
            "epistemic_ratio": ood["epistemic_ratio"],
            "ratio_minimum": settings["acceptance_ood_epistemic_ratio_min"],
        },
        "event_aligned_no_average_braking": {
            "passed": bool(
                event["available"]
                and event["member_dominant_agent_accuracy"]
                >= settings["acceptance_event_dominant_agent_accuracy_min"]
                and event["member_ambiguous_braking_rate"]
                <= settings["acceptance_event_ambiguous_rate_max"]
            ),
            **event,
            "accuracy_minimum": settings["acceptance_event_dominant_agent_accuracy_min"],
            "ambiguous_rate_maximum": settings["acceptance_event_ambiguous_rate_max"],
        },
        "independent_members": {
            "passed": bool(diversity.get("passed", False)),
            **diversity,
        },
        "full_test_split": {"passed": full_test_split_evaluated},
    }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "horizon": int(horizon),
        "full_test_split_evaluated": full_test_split_evaluated,
        "checks": checks,
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    gate = report["acceptance"]
    rows = []
    for name, check in gate["checks"].items():
        detail = ", ".join(
            f"{key}={value:.5g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in check.items()
            if key != "passed" and value is not None
        )
        rows.append(f"| {name} | {'通过' if check['passed'] else '未通过'} | {detail} |")
    return "\n".join(
        (
            "# World-model ensemble uncertainty / uncertainty acceptance report",
            "",
            f"结论：**uncertainty acceptance {'通过' if gate['passed'] else '未通过'}**。",
            "",
            "| 判据 | 结论 | 关键数据 |",
            "|---|---|---|",
            *rows,
            "",
            "阈值仅由版本化配置定义；variance scale 仅使用 validation 拟合，test 不参与校准。",
        )
    )


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def _resolve_settings(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    data = config["data"]
    evaluation = config["evaluation"]
    return {
        "data_dir": args.data_dir or ROOT / data["directory"],
        "checkpoint_dir": args.checkpoint_dir or ROOT / config["checkpoint"]["directory"],
        "baseline_mlp_nrmse": float(
            evaluation["baseline_mlp_h20_nrmse"]
            if args.baseline_mlp_nrmse is None
            else args.baseline_mlp_nrmse
        ),
        "state_dim": int(data["state_dim"]),
        "action_dim": int(data["action_dim"]),
        "history_horizon": int(data["history_horizon"]),
        "split_seed": int(data["split_seed"]),
        "horizons": tuple(int(value) for value in evaluation["open_loop_horizons"]),
        "validation_max_batches": int(
            evaluation["validation_max_batches"]
            if args.validation_max_batches is None
            else args.validation_max_batches
        ),
        **{
            name: evaluation[name]
            for name in (
                "variance_scale_min",
                "variance_scale_max",
                "event_horizon",
                "event_progress_min",
                "event_slowdown_min",
                "event_asymmetry_min",
                "event_ambiguity_fraction",
                "ood_action_scale",
                "ood_action_offset_std",
                "ood_action_low",
                "ood_action_high",
                "acceptance_horizon",
                "acceptance_mlp_improvement_min",
                "acceptance_ar_over_teacher_forcing_min",
                "acceptance_uncertainty_error_spearman_min",
                "acceptance_ood_auroc_min",
                "acceptance_ood_epistemic_ratio_min",
                "acceptance_event_dominant_agent_accuracy_min",
                "acceptance_event_ambiguous_rate_max",
            )
        },
    }


def _build_dataset(
    paths: Sequence[Path],
    settings: Mapping[str, Any],
    *,
    forecast_horizon: int,
    preload: bool,
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
    if preload:
        return InMemoryProprioSequenceDataset(**common, progress=progress)
    return ProprioSequenceDataset(**common)


def _validate_checkpoint_data_provenance(
    checkpoint_manifest: Mapping[str, Any],
    split_paths: Mapping[str, Sequence[Path]],
    *,
    smoke_subset: bool,
) -> None:
    if smoke_subset:
        return
    reference = checkpoint_manifest.get("partitions")
    if not isinstance(reference, Mapping):
        raise ValueError("RWM-U checkpoint has no auditable dataset partitions")
    current = {
        name: [str(path.resolve()) for path in paths]
        for name, paths in split_paths.items()
    }
    normalized_reference = {
        name: [str(Path(path).resolve()) for path in reference.get(name, [])]
        for name in current
    }
    if current != normalized_reference:
        raise ValueError("evaluation data split differs from RWM-U checkpoint provenance")


def _build_loader(
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


def _effective_batches(loader: DataLoader, maximum: int) -> int:
    return max(min(len(loader), maximum) if maximum > 0 else len(loader), 1)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    for name in ("max_batches", "max_episodes", "validation_max_batches"):
        value = getattr(args, name)
        if value is None:
            continue
        if value == 0 or value < -1:
            raise ValueError(f"{name} must be -1 or positive")


if __name__ == "__main__":
    raise SystemExit(main())
