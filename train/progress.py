"""Shared adaptive Rich progress blocks for WAM training pipelines."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

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
        TimeRemainingColumn,
    )
    from rich.table import Column
    from rich.text import Text
except ImportError:
    Console = Group = Live = Progress = Column = Text = None


class TrainingProgress:
    """Retain one terminal block for every completed pipeline stage."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_stages: int,
        refresh_per_second: float = 4.0,
    ) -> None:
        if refresh_per_second <= 0.0 or total_stages <= 0:
            raise ValueError("refresh rate and total stages must be positive")
        self.refresh_per_second = float(refresh_per_second)
        self.total_stages = int(total_stages)
        self.stage_index = 0
        self._console: Any | None = None
        self._active_phase: TrainingProgressPhase | None = None
        self._phases: list[TrainingProgressPhase] = []
        if not enabled:
            return
        if any(item is None for item in (Console, Group, Live, Progress, Column, Text)):
            print("Progress display unavailable: install 'rich'.", file=sys.stderr)
            return
        self._console = Console(stderr=True)

    def __enter__(self) -> "TrainingProgress":
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._active_phase is not None:
            self._active_phase.stop_display()
            self._active_phase = None

    def _make_progress(self) -> Any:
        description = Column(ratio=2, min_width=4, no_wrap=True)
        bar = Column(ratio=3, min_width=3, no_wrap=True)
        detail = Column(ratio=2, min_width=4, no_wrap=True)
        return Progress(
            SpinnerColumn(
                style="bold cyan", finished_text="[bold green]✓[/bold green]"
            ),
            TextColumn("[bold blue]{task.fields[stage_text]}"),
            TextColumn("[bold cyan]{task.description}", table_column=description),
            BarColumn(
                bar_width=None,
                complete_style="cyan",
                finished_style="green",
                table_column=bar,
            ),
            MofNCompleteColumn(),
            TextColumn("[magenta]{task.fields[detail]}", table_column=detail),
            TextColumn("[dim]elapsed"),
            TimeElapsedColumn(),
            TextColumn("[dim]remaining"),
            TimeRemainingColumn(),
            console=self._console,
            expand=True,
            auto_refresh=False,
        )

    def add_phase(
        self, description: str, total: int, *, show_loss_chart: bool | None = None
    ) -> "TrainingProgressPhase":
        if total <= 0:
            raise ValueError("progress phase total must be positive")
        if self._active_phase is not None and not self._active_phase.finished:
            raise RuntimeError("finish the active stage before starting another")
        self.stage_index += 1
        if self.stage_index > self.total_stages:
            raise RuntimeError("created more progress stages than planned")
        stable_description = clean_progress_text(description)
        progress = self._make_progress() if self._console is not None else None
        task_id = None
        if progress is not None:
            task_id = progress.add_task(
                stable_description,
                total=total,
                stage_text=stage_text(self.stage_index, self.total_stages),
                detail="starting",
            )
        phase = TrainingProgressPhase(
            owner=self,
            progress=progress,
            task_id=task_id,
            total=total,
            stage_index=self.stage_index,
            description=description,
            show_loss_chart=(
                stable_description.startswith("train ")
                if show_loss_chart is None
                else show_loss_chart
            ),
        )
        self._phases.append(phase)
        self._active_phase = phase
        phase.start_display()
        return phase

    def phase_finished(self, phase: "TrainingProgressPhase") -> None:
        if self._active_phase is phase:
            self._active_phase = None

    def report_completion(self, phase: "TrainingProgressPhase", detail: str) -> None:
        if self._console is None:
            print(
                f"✓ completed stage {phase.stage_index}/{self.total_stages} | "
                f"{clean_progress_text(phase.description)} | "
                f"{clean_progress_text(detail)}",
                file=sys.stderr,
            )


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
        show_loss_chart: bool,
    ) -> None:
        self.owner = owner
        self.progress = progress
        self.task_id = task_id
        self.total = int(total)
        self.stage_index = int(stage_index)
        self.description = str(description)
        self.show_loss_chart = bool(show_loss_chart)
        self.loss_history: list[float] = []
        self.live: Any | None = None
        self.finished = False

    def start_display(self) -> None:
        if self.progress is None:
            return
        self.live = Live(
            PhaseDisplay(self),
            console=self.owner._console,
            refresh_per_second=self.owner.refresh_per_second,
            transient=False,
            vertical_overflow="visible",
        )
        self.live.start(refresh=True)

    def stop_display(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        self.live = None
        if self.owner._console is not None and not self.owner._console.is_terminal:
            self.owner._console.line()

    def advance(self, values: Any) -> None:
        if isinstance(values, dict) and "loss" in values:
            loss = float(values["loss"])
            if np.isfinite(loss):
                self.loss_history.append(loss)
        if self.progress is not None and self.task_id is not None:
            self.progress.update(
                self.task_id, advance=1, detail=progress_detail(values)
            )

    def finish(self, detail: str = "done") -> None:
        if self.finished:
            return
        if self.progress is not None and self.task_id is not None:
            self.progress.update(
                self.task_id, completed=self.total, detail=clean_progress_text(detail)
            )
            if self.live is not None:
                self.live.refresh()
        self.stop_display()
        self.owner.report_completion(self, detail)
        self.finished = True
        self.owner.phase_finished(self)


class PhaseDisplay:
    def __init__(self, phase: TrainingProgressPhase) -> None:
        self.phase = phase

    def __rich_console__(self, _console: Any, _options: Any) -> Any:
        renderables: list[Any] = [
            self.phase.progress.make_tasks_table(self.phase.progress.tasks)
        ]
        if self.phase.show_loss_chart:
            renderables.append(AdaptiveLossPointChart(self.phase.loss_history))
        yield Group(*renderables)


class AdaptiveLossPointChart:
    def __init__(self, values: Any, *, height: int = 5) -> None:
        self.values = values
        self.height = int(height)

    def __rich_console__(self, _console: Any, options: Any) -> Any:
        width = max(int(options.max_width), 0)
        yield loss_point_chart(self.values, width=width, height=self.height)


def loss_point_chart(values: Any, *, width: int, height: int = 5) -> Any:
    """baseline-compatible complete-history Braille point chart."""

    if Text is None or width <= 0 or height <= 0:
        return ""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    label_width = 10 if width >= 30 else 0
    plot_width = max(width - label_width - 1, 1)
    pixel_width = plot_width * 2
    pixel_height = height * 4
    masks = np.zeros((height, plot_width), dtype=np.uint8)
    lower = upper = 0.0
    if finite.size:
        sampled = _downsample_loss(finite, pixel_width)
        if finite.size >= 20:
            lower, upper = map(float, np.quantile(finite, (0.01, 0.99)))
        else:
            lower, upper = float(finite.min()), float(finite.max())
        center = 0.5 * (lower + upper)
        padding = max(0.05 * (upper - lower), abs(center) * 1e-6, 1e-8)
        lower -= padding
        upper += padding
        xs = np.rint(np.linspace(0, pixel_width - 1, len(sampled))).astype(np.int64)
        ys = np.rint(
            (upper - np.clip(sampled, lower, upper))
            / (upper - lower)
            * (pixel_height - 1)
        ).astype(np.int64)
        for x, y in zip(xs, ys, strict=True):
            cell_x, dot_x = divmod(int(x), 2)
            cell_y, dot_y = divmod(int(y), 4)
            masks[cell_y, cell_x] |= _BRAILLE_BITS[dot_y][dot_x]
    chart = Text()
    for row_index, row in enumerate(masks):
        if label_width:
            label = (
                f"{upper:9.3g} "
                if row_index == 0 and finite.size
                else f"{lower:9.3g} "
                if row_index == height - 1 and finite.size
                else " " * label_width
            )
            chart.append(label, style="dim")
            chart.append("┤", style="dim")
        else:
            chart.append("│", style="dim")
        for mask in row:
            chart.append(chr(0x2800 + int(mask)) if mask else " ", style="yellow")
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
    if values.size <= width:
        return values
    boundaries = np.linspace(0, values.size, width + 1, dtype=np.int64)
    return np.asarray(
        [
            values[boundaries[index] : boundaries[index + 1]].mean()
            for index in range(width)
        ]
    )


def clean_progress_text(value: str) -> str:
    return " ".join(str(value).split())


def stage_text(index: int, total: int) -> str:
    width = len(str(total))
    return f"stage={index:0{width}d}/{total}"


def progress_detail(values: Any) -> str:
    if not isinstance(values, dict):
        return str(values)
    if "loss" in values:
        if "epoch" in values:
            detail = f"epoch {int(values['epoch'])}"
            if "epochs" in values:
                detail += f"/{int(values['epochs'])}"
        elif "step" in values:
            detail = f"step {int(values['step'])}"
        else:
            detail = "train"
        detail += f" loss {float(values['loss']):.5f}"
        if "state_mean_mse" in values:
            detail += f" mean {float(values['state_mean_mse']):.4g}"
        if "state_nll" in values:
            detail += f" nll {float(values['state_nll']):.4g}"
        if "action_nll" in values:
            detail += f" prior {float(values['action_nll']):.4g}"
        if "value_huber" in values:
            detail += f" value {float(values['value_huber']):.4g}"
        if "flow_loss" in values:
            detail += f" flow {float(values['flow_loss']):.4g}"
        if "world_loss" in values:
            detail += f" world {float(values['world_loss']):.4g}"
        return detail
    if "samples" in values:
        return f"{int(values['samples'])} samples"
    if "batch" in values:
        return f"batch {int(values['batch'])}"
    if "episode" in values:
        detail = f"episode {int(values['episode'])}"
        if "episodes" in values:
            detail += f"/{int(values['episodes'])}"
        return detail
    return "running"


__all__ = [
    "AdaptiveLossPointChart",
    "PhaseDisplay",
    "TrainingProgress",
    "TrainingProgressPhase",
    "loss_point_chart",
]
