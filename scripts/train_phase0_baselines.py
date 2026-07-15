"""Train and evaluate the accepted Phase 0 proprioceptive baselines."""

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
    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Column
    from rich.text import Text
except ImportError:  # Keep --no-progress usable in minimal/headless installs.
    Console = None
    Group = None
    Live = None
    Progress = None
    Column = None
    Text = None
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
    OneStepMLPWorldModel,
    OneStepMLPWorldModelConfig,
)
from train.phase0_baselines import (  # noqa: E402
    BaselineTrainConfig,
    evaluate_baseline,
    fit_binary_label_stats,
    fit_normalization,
    train_baseline,
)
from train.trajectory_dataset import (  # noqa: E402
    InMemoryOneStepDataset,
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
    parser.add_argument(
        "--outcome-weight",
        type=float,
        default=0.1,
        help="Weight for the positive-weighted success and failure BCE losses.",
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hdf5-cache-size", type=int, default=8)
    parser.add_argument(
        "--preload-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preload the minimal one-step baseline tensors into RAM once and reuse "
            "them across statistics, models, epochs, and evaluation (default: true)."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars for CI or redirected logs.",
    )
    parser.add_argument(
        "--progress-refresh-hz",
        type=float,
        default=4.0,
        help="Maximum Rich display refresh rate (default: 4 Hz).",
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_config = BaselineTrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        reward_weight=args.reward_weight,
        done_weight=args.done_weight,
        outcome_weight=args.outcome_weight,
    )
    model_names = tuple(dict.fromkeys(args.models))
    trains_world_model = any(name != "action_prior" for name in model_names)
    evaluation_stages = sum(bool(partition) for partition in split_paths.values())
    preload_stages = evaluation_stages if args.preload_data else 0
    label_stats_stages = int(trains_world_model)
    total_stages = (
        preload_stages
        + 1
        + label_stats_stages
        + len(model_names) * (1 + evaluation_stages)
    )
    metrics: dict[str, Any] = {}
    datasets: dict[str, InMemoryOneStepDataset | ProprioSequenceDataset | None] = {}
    with TrainingProgress(
        enabled=not args.no_progress,
        refresh_per_second=args.progress_refresh_hz,
        total_stages=total_stages,
    ) as progress:
        for split_name, partition in split_paths.items():
            if not partition:
                datasets[split_name] = None
                continue
            preload_progress = (
                progress.add_phase(f"preload {split_name}", len(partition))
                if args.preload_data
                else None
            )
            dataset = _build_dataset(
                args,
                partition,
                progress=(
                    preload_progress.advance if preload_progress is not None else None
                ),
            )
            datasets[split_name] = dataset
            if preload_progress is not None:
                assert isinstance(dataset, InMemoryOneStepDataset)
                preload_progress.finish(
                    f"{len(dataset)} samples, {dataset.nbytes / 2**20:.1f} MiB"
                )

        if datasets["train"] is None:
            raise RuntimeError("episode split produced an empty training partition")
        loaders = {
            name: _build_loader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=name == "train",
                seed=args.seed,
                pin_memory=device.type == "cuda",
            )
            for name, dataset in datasets.items()
        }
        train_loader = loaders["train"]
        assert train_loader is not None
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

        binary_label_stats: dict[str, dict[str, float | int | bool]] = {}
        if trains_world_model:
            label_progress = progress.add_phase(
                "outcome label stats", len(train_loader)
            )
            binary_label_stats = fit_binary_label_stats(
                train_loader,
                progress=label_progress.advance,
            )
            label_progress.finish(
                ", ".join(
                    f"{name} w+={values['positive_weight']:.1f}"
                    for name, values in binary_label_stats.items()
                )
            )
            (args.output_dir / "outcome_label_stats.json").write_text(
                json.dumps(
                    {
                        "format_version": "wam.phase0.outcome-labels/1",
                        "source_split": "train",
                        "labels": binary_label_stats,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        for model_name in model_names:
            model, model_config = _build_model(model_name, args)
            train_steps = args.epochs * len(train_loader)
            if args.max_steps > 0:
                train_steps = min(train_steps, args.max_steps)
            training_progress = progress.add_phase(f"train {model_name}", train_steps)
            loss_history = train_baseline(
                model,
                train_loader,
                stats,
                train_config,
                device,
                binary_label_stats=(
                    binary_label_stats if model_name != "action_prior" else None
                ),
                progress=training_progress.advance,
            )
            training_progress.finish(f"final loss {loss_history[-1]:.5f}")
            split_metrics: dict[str, dict[str, Any]] = {}
            calibration: dict[str, Any] | None = None
            classification_thresholds: dict[str, float] | None = None
            evaluation_order = (
                ("validation", "train", "test")
                if model_name != "action_prior"
                else tuple(loaders)
            )
            if model_name != "action_prior" and loaders["validation"] is None:
                raise RuntimeError(
                    "outcome threshold calibration requires a validation split"
                )
            for split_name in evaluation_order:
                loader = loaders[split_name]
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
                    classification_thresholds=classification_thresholds,
                    calibrate_thresholds=(
                        model_name != "action_prior" and split_name == "validation"
                    ),
                    progress=evaluation_progress.advance,
                )
                if model_name != "action_prior" and split_name == "validation":
                    calibration = split_metrics[split_name].pop("calibration")
                    classification_thresholds = {
                        label: float(values["threshold"])
                        for label, values in calibration.items()
                    }
                evaluation_progress.finish(_metric_summary(split_metrics[split_name]))
            metrics[model_name] = {
                "loss_history": loss_history,
                "splits": split_metrics,
            }
            if model_name != "action_prior":
                assert calibration is not None
                calibration_payload = {
                    "format_version": "wam.phase0.outcome-calibration/1",
                    "model": model_name,
                    "source_split": "validation",
                    "objective": "max_f1",
                    "tie_break": "higher_recall",
                    "labels": calibration,
                }
                metrics[model_name]["training_binary_label_stats"] = binary_label_stats
                metrics[model_name]["outcome_calibration"] = calibration_payload
                (args.output_dir / f"{model_name}_outcome_calibration.json").write_text(
                    json.dumps(calibration_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            state_dict = {
                key: value.detach().cpu().contiguous()
                for key, value in model.state_dict().items()
            }
            save_file(
                state_dict,
                args.output_dir / f"{model_name}.safetensors",
                metadata={
                    "baseline": model_name,
                    "format_version": "wam.phase0/2",
                },
            )
            (args.output_dir / f"{model_name}_config.json").write_text(
                json.dumps(asdict(model_config), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    run_config = {key: _json_value(value) for key, value in vars(args).items()}
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
    args: argparse.Namespace,
    paths: tuple[Path, ...],
    *,
    progress: Any | None = None,
) -> InMemoryOneStepDataset | ProprioSequenceDataset | None:
    if not paths:
        return None
    if args.preload_data:
        return InMemoryOneStepDataset(
            paths=paths,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            allow_legacy_wam=args.allow_legacy_wam,
            progress=progress,
        )
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
    dataset: InMemoryOneStepDataset | ProprioSequenceDataset | None,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
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
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _build_model(name: str, args: argparse.Namespace) -> tuple[torch.nn.Module, Any]:
    if name == "linear":
        config = LinearWorldModelConfig(args.state_dim, args.action_dim)
        return LinearWorldModel(config), config
    if name == "mlp":
        config = OneStepMLPWorldModelConfig(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            predict_delta=False,
        )
        return OneStepMLPWorldModel(config), config
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
    if args.progress_refresh_hz <= 0:
        raise ValueError("progress_refresh_hz must be positive")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


class TrainingProgress:
    """A retained Rich block for every training pipeline stage."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_stages: int,
        refresh_per_second: float = 4.0,
    ) -> None:
        self._console: Any | None = None
        self._progress: Any | None = None
        self._task_id: Any | None = None
        self._active_phase: TrainingProgressPhase | None = None
        self._phases: list[TrainingProgressPhase] = []
        self.refresh_per_second = float(refresh_per_second)
        self.total_stages = int(total_stages)
        self.stage_index = 0
        if refresh_per_second <= 0:
            raise ValueError("refresh_per_second must be positive")
        if self.total_stages <= 0:
            raise ValueError("total_stages must be positive")
        if not enabled:
            return
        if Progress is None or Console is None or Live is None or Group is None:
            print(
                "Progress display unavailable: install 'rich' to enable it.",
                file=sys.stderr,
            )
            return
        self._console = Console(stderr=True)

    def _make_progress(self) -> Any:
        assert Progress is not None
        assert Column is not None
        description_column = Column(ratio=2, min_width=4, no_wrap=True)
        bar_column = Column(ratio=3, min_width=3, no_wrap=True)
        detail_column = Column(ratio=2, min_width=4, no_wrap=True)
        return Progress(
            SpinnerColumn(
                style="bold cyan",
                finished_text="[bold green]✓[/bold green]",
            ),
            TextColumn("[bold blue]{task.fields[stage_text]}"),
            TextColumn(
                "[bold cyan]{task.description}",
                table_column=description_column,
            ),
            BarColumn(
                bar_width=None,
                complete_style="cyan",
                finished_style="green",
                table_column=bar_column,
            ),
            MofNCompleteColumn(),
            TextColumn(
                "[magenta]{task.fields[detail]}",
                table_column=detail_column,
            ),
            TimeElapsedColumn(),
            console=self._console,
            expand=True,
            auto_refresh=False,
        )

    def __enter__(self) -> "TrainingProgress":
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._active_phase is not None:
            self._active_phase.stop_display()
            self._active_phase = None

    def add_phase(self, description: str, total: int) -> "TrainingProgressPhase":
        if total <= 0:
            raise ValueError("progress phase total must be positive")
        self.stage_index += 1
        if self.stage_index > self.total_stages:
            raise RuntimeError(
                f"training created more than {self.total_stages} planned stages"
            )
        if self._active_phase is not None and not self._active_phase.finished:
            raise RuntimeError("cannot start a new progress stage before finishing it")
        stable_description = _clean_progress_text(description)
        stage_text = _stage_text(self.stage_index, self.total_stages)
        loss_history: list[float] = []
        progress = self._make_progress() if self._console is not None else None
        task_id: Any | None = None
        if progress is not None:
            self._progress = progress
            task_id = progress.add_task(
                stable_description,
                total=total,
                stage_text=stage_text,
                detail="starting",
            )
            self._task_id = task_id
        phase = TrainingProgressPhase(
            owner=self,
            progress=progress,
            task_id=task_id,
            total=total,
            stage_index=self.stage_index,
            description=description,
            loss_history=loss_history,
            show_loss_chart=stable_description.startswith("train "),
        )
        self._phases.append(phase)
        self._active_phase = phase
        phase.start_display()
        return phase

    def report_completion(
        self,
        *,
        stage_index: int,
        description: str,
        detail: str,
    ) -> None:
        message = (
            f"✓ completed stage {stage_index}/{self.total_stages} | "
            f"{_clean_progress_text(description)} | "
            f"{_clean_progress_text(detail)}"
        )
        if self._console is None:
            print(message, file=sys.stderr)

    def phase_finished(self, phase: "TrainingProgressPhase") -> None:
        if self._active_phase is phase:
            self._active_phase = None


class TrainingProgressPhase:
    def __init__(
        self,
        *,
        owner: TrainingProgress,
        progress: Any | None,
        task_id: Any | None,
        total: int,
        stage_index: int,
        description: str,
        loss_history: list[float],
        show_loss_chart: bool,
    ) -> None:
        self.owner = owner
        self.progress = progress
        self.task_id = task_id
        self.total = int(total)
        self.stage_index = int(stage_index)
        self.description = str(description)
        self.loss_history = loss_history
        self.show_loss_chart = bool(show_loss_chart)
        self.live: Any | None = None
        self.finished = False

    def start_display(self) -> None:
        if self.progress is None or Live is None:
            return
        self.live = Live(
            _PhaseDisplay(self),
            console=self.owner._console,
            refresh_per_second=self.owner.refresh_per_second,
            transient=False,
            vertical_overflow="visible",
        )
        self.live.start(refresh=True)

    def stop_display(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None
            if self.owner._console is not None and not self.owner._console.is_terminal:
                self.owner._console.line()

    def advance(self, values: Any) -> None:
        if isinstance(values, dict) and "loss" in values:
            loss = float(values["loss"])
            if np.isfinite(loss):
                self.loss_history.append(loss)
        if self.progress is None or self.task_id is None:
            return
        detail = _progress_detail(values)
        self.progress.update(self.task_id, advance=1, detail=detail)

    def finish(self, detail: str = "done") -> None:
        if self.finished:
            return
        if self.progress is not None and self.task_id is not None:
            self.progress.update(
                self.task_id,
                completed=self.total,
                detail=_clean_progress_text(detail),
            )
            if self.live is not None:
                self.live.refresh()
        self.stop_display()
        self.owner.report_completion(
            stage_index=self.stage_index,
            description=self.description,
            detail=detail,
        )
        self.finished = True
        self.owner.phase_finished(self)


class _PhaseDisplay:
    """Render a one-line progress row and an optional full-width loss chart."""

    def __init__(self, phase: TrainingProgressPhase) -> None:
        self.phase = phase

    def __rich_console__(self, _console: Any, _options: Any) -> Any:
        assert Group is not None
        renderables: list[Any] = [
            self.phase.progress.make_tasks_table(self.phase.progress.tasks)
        ]
        if self.phase.show_loss_chart:
            renderables.append(_AdaptiveLossPointChart(self.phase.loss_history))
        yield Group(*renderables)


class _AdaptiveLossPointChart:
    """Render the complete batch-loss history as a Braille point chart."""

    def __init__(self, values: Any, *, height: int = 5) -> None:
        self.values = values
        self.height = int(height)

    def __rich_console__(self, _console: Any, options: Any) -> Any:
        width = max(int(options.max_width), 0)
        if Text is None or width <= 0:
            yield ""
            return
        yield _loss_point_chart(self.values, width=width, height=self.height)


def _loss_point_chart(values: Any, *, width: int, height: int = 5) -> Any:
    """Build a fixed-height chart with 2x4 Braille subpixels per character."""

    assert Text is not None
    if width <= 0 or height <= 0:
        return Text("")
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    label_width = 10 if width >= 30 else 0
    axis_width = 1
    plot_width = max(width - label_width - axis_width, 1)
    pixel_width = plot_width * 2
    pixel_height = height * 4
    masks = np.zeros((height, plot_width), dtype=np.uint8)

    lower = 0.0
    upper = 0.0
    if finite.size:
        # The x-axis always spans step 1 through the current step. As training
        # grows, the complete history is re-binned to the available subpixels,
        # matching the expanding-domain behavior of experiment dashboards.
        sampled = _downsample_loss(finite, pixel_width)
        if finite.size >= 20:
            observed_lower, observed_upper = np.quantile(finite, (0.01, 0.99))
            observed_lower = float(observed_lower)
            observed_upper = float(observed_upper)
        else:
            observed_lower = float(finite.min())
            observed_upper = float(finite.max())
        center = 0.5 * (observed_lower + observed_upper)
        padding = max(
            0.05 * (observed_upper - observed_lower),
            abs(center) * 1e-6,
            1e-8,
        )
        lower = observed_lower - padding
        upper = observed_upper + padding
        x_positions = np.rint(
            np.linspace(0, pixel_width - 1, len(sampled))
        ).astype(np.int64)
        plotted = np.clip(sampled, lower, upper)
        y_positions = np.rint(
            (upper - plotted) / (upper - lower) * (pixel_height - 1)
        ).astype(np.int64)
        for x_position, y_position in zip(x_positions, y_positions, strict=True):
            cell_x, dot_x = divmod(int(x_position), 2)
            cell_y, dot_y = divmod(int(y_position), 4)
            masks[cell_y, cell_x] |= _BRAILLE_BITS[dot_y][dot_x]

    chart = Text()
    for row_index, row in enumerate(masks):
        if label_width:
            if row_index == 0 and finite.size:
                label = f"{upper:9.3g} "
            elif row_index == height - 1 and finite.size:
                label = f"{lower:9.3g} "
            else:
                label = " " * label_width
            chart.append(label, style="dim")
            chart.append("┤", style="dim")
        else:
            chart.append("│", style="dim")
        for mask in row:
            if mask:
                chart.append(chr(0x2800 + int(mask)), style="yellow")
            else:
                chart.append(" ")
        if row_index < height - 1:
            chart.append("\n")
    chart.append("\n")
    if label_width:
        chart.append(" " * label_width, style="dim")
    footer = list("─" * plot_width)
    step_range = f" steps 1→{len(finite)} "
    if len(step_range) + 2 <= plot_width:
        footer[1 : 1 + len(step_range)] = step_range
    chart.append("└" + "".join(footer), style="dim")
    return chart


_BRAILLE_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)


def _downsample_loss(values: np.ndarray, width: int) -> np.ndarray:
    """Average chronological bins so at most one point occupies each x column."""

    if values.size <= width:
        return values
    boundaries = np.linspace(0, values.size, width + 1, dtype=np.int64)
    return np.asarray(
        [
            values[boundaries[index] : boundaries[index + 1]].mean()
            for index in range(width)
        ],
        dtype=np.float64,
    )


def _clean_progress_text(value: str) -> str:
    return " ".join(str(value).split())


def _stage_text(stage_index: int, total_stages: int) -> str:
    width = len(str(total_stages))
    return f"stage={stage_index:0{width}d}/{total_stages}"


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
