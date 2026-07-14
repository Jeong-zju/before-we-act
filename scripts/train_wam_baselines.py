"""Train and evaluate the Phase 0 proprioceptive WAM baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:  # Keep --no-progress usable in minimal/headless installs.
    Console = None
    Progress = None
from safetensors.torch import save_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import (  # noqa: E402
    ActionPrior,
    ActionPriorConfig,
    LinearWorldModel,
    LinearWorldModelConfig,
    WorldActionModel,
    WorldActionModelConfig,
)
from train.baselines import (  # noqa: E402
    BaselineTrainConfig,
    evaluate_baseline,
    fit_normalization,
    train_baseline,
)
from train.trajectory_dataset import (  # noqa: E402
    ProprioSequenceDataset,
    discover_episode_paths,
    split_episode_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train linear dynamics, one-step MLP, and action-prior baselines "
            "without reading privileged state."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("linear", "mlp", "action_prior"),
        default=("linear", "mlp", "action_prior"),
    )
    parser.add_argument("--history-horizon", type=int, default=32)
    parser.add_argument("--forecast-horizon", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=22)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--reward-weight", type=float, default=0.1)
    parser.add_argument("--done-weight", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hdf5-cache-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars for CI or redirected logs.",
    )
    parser.add_argument(
        "--allow-legacy-wam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the previous WAM HDF5 layout for baseline comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    paths = discover_episode_paths(args.data_dir)
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]
    split_paths = split_episode_paths(paths, seed=args.seed)
    datasets = {
        name: _build_dataset(args, partition)
        for name, partition in split_paths.items()
    }
    if datasets["train"] is None:
        raise RuntimeError("episode split produced an empty training partition")
    loaders = {
        name: _build_loader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=name == "train",
            seed=args.seed,
        )
        for name, dataset in datasets.items()
    }
    train_loader = loaders["train"]
    assert train_loader is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_config = BaselineTrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        reward_weight=args.reward_weight,
        done_weight=args.done_weight,
    )
    metrics: dict[str, Any] = {}
    with TrainingProgress(enabled=not args.no_progress) as progress:
        normalization_progress = progress.add_phase(
            "dataset statistics", len(train_loader)
        )
        stats = fit_normalization(
            train_loader,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            progress=normalization_progress.advance,
        )
        normalization_progress.finish("normalization ready")
        stats.save(str(args.output_dir / "normalization.npz"))

        for model_name in dict.fromkeys(args.models):
            model, model_config = _build_model(model_name, args)
            train_steps = args.epochs * len(train_loader)
            if args.max_steps > 0:
                train_steps = min(train_steps, args.max_steps)
            training_progress = progress.add_phase(
                f"train {model_name}", train_steps
            )
            loss_history = train_baseline(
                model,
                train_loader,
                stats,
                train_config,
                device,
                progress=training_progress.advance,
            )
            training_progress.finish(
                f"final loss {loss_history[-1]:.5f}"
            )
            split_metrics: dict[str, dict[str, float | int]] = {}
            for split_name, loader in loaders.items():
                if loader is None:
                    split_metrics[split_name] = {"samples": 0}
                    continue
                evaluation_progress = progress.add_phase(
                    f"eval {model_name}/{split_name}", len(loader)
                )
                split_metrics[split_name] = evaluate_baseline(
                    model,
                    loader,
                    stats,
                    device,
                    progress=evaluation_progress.advance,
                )
                evaluation_progress.finish(
                    _metric_summary(split_metrics[split_name])
                )
            metrics[model_name] = {
                "loss_history": loss_history,
                "splits": split_metrics,
            }
            state_dict = {
                key: value.detach().cpu().contiguous()
                for key, value in model.state_dict().items()
            }
            save_file(
                state_dict,
                args.output_dir / f"{model_name}.safetensors",
                metadata={
                    "baseline": model_name,
                    "format_version": "wam.phase0/1",
                },
            )
            (args.output_dir / f"{model_name}_config.json").write_text(
                json.dumps(asdict(model_config), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    run_config = {
        key: _json_value(value) for key, value in vars(args).items()
    }
    run_config["resolved_device"] = str(device)
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "split_unit": "episode_seed",
        "partitions": {
            name: [str(path.resolve()) for path in partition]
            for name, partition in split_paths.items()
        },
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    metrics_path = args.output_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"metrics": str(metrics_path), "results": metrics}, indent=2))

    for dataset in datasets.values():
        if dataset is not None:
            dataset.close()
    return 0


def _build_dataset(
    args: argparse.Namespace, paths: tuple[Path, ...]
) -> ProprioSequenceDataset | None:
    if not paths:
        return None
    return ProprioSequenceDataset(
        paths=paths,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        allow_legacy_wam=args.allow_legacy_wam,
        hdf5_cache_size=args.hdf5_cache_size,
    )


def _build_loader(
    dataset: ProprioSequenceDataset | None,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader | None:
    if dataset is None:
        return None
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _build_model(
    name: str, args: argparse.Namespace
) -> tuple[torch.nn.Module, Any]:
    if name == "linear":
        config = LinearWorldModelConfig(args.state_dim, args.action_dim)
        return LinearWorldModel(config), config
    if name == "mlp":
        config = WorldActionModelConfig(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            predict_delta=False,
        )
        return WorldActionModel(config), config
    if name == "action_prior":
        config = ActionPriorConfig(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=max(1, args.hidden_layers - 1),
        )
        return ActionPrior(config), config
    raise ValueError(f"unknown baseline {name!r}")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "history_horizon",
        "forecast_horizon",
        "state_dim",
        "action_dim",
        "hidden_dim",
        "hidden_layers",
        "batch_size",
        "epochs",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0 or args.hdf5_cache_size < 0:
        raise ValueError("num_workers and hdf5_cache_size must be non-negative")
    if args.max_episodes == 0 or args.max_episodes < -1:
        raise ValueError("max_episodes must be -1 or positive")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


class TrainingProgress:
    """One Rich display shared by statistics, optimization, and evaluation."""

    def __init__(self, *, enabled: bool) -> None:
        self._progress: Any | None = None
        if not enabled:
            return
        if Progress is None or Console is None:
            print(
                "Progress display unavailable: install 'rich' to enable it.",
                file=sys.stderr,
            )
            return
        self._progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold cyan]{task.description:<32}"),
            BarColumn(bar_width=None, complete_style="cyan"),
            MofNCompleteColumn(),
            TextColumn("[magenta]{task.fields[detail]}", justify="left"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(stderr=True),
            refresh_per_second=10,
        )

    def __enter__(self) -> "TrainingProgress":
        if self._progress is not None:
            self._progress.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def add_phase(self, description: str, total: int) -> "TrainingProgressPhase":
        if total <= 0:
            raise ValueError("progress phase total must be positive")
        task_id = None
        if self._progress is not None:
            task_id = self._progress.add_task(description, total=total, detail="starting")
        return TrainingProgressPhase(self._progress, task_id, total)


class TrainingProgressPhase:
    def __init__(self, progress: Any | None, task_id: Any | None, total: int) -> None:
        self.progress = progress
        self.task_id = task_id
        self.total = int(total)

    def advance(self, values: Any) -> None:
        if self.progress is None or self.task_id is None:
            return
        detail = _progress_detail(values)
        self.progress.update(self.task_id, advance=1, detail=detail)

    def finish(self, detail: str = "done") -> None:
        if self.progress is not None and self.task_id is not None:
            self.progress.update(
                self.task_id,
                completed=self.total,
                detail=detail,
            )


def _progress_detail(values: Any) -> str:
    if not isinstance(values, dict):
        return str(values)
    if "loss" in values:
        return (
            f"epoch {int(values['epoch'])}/{int(values['epochs'])} "
            f"loss {float(values['loss']):.5f}"
        )
    if "samples" in values:
        return f"{int(values['samples'])} samples"
    if "batch" in values:
        return f"batch {int(values['batch'])}"
    return "running"


def _metric_summary(metrics: dict[str, float | int]) -> str:
    for key in ("state_nrmse", "action_rmse", "state_rmse"):
        if key in metrics:
            return f"{key} {float(metrics[key]):.5f}"
    return f"{int(metrics.get('samples', 0))} samples"


if __name__ == "__main__":
    raise SystemExit(main())
