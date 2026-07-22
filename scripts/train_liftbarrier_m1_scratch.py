"""Train LiftBarrier M1 through the dedicated scratch-only curriculum."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml
from rich.console import Console
from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam import AffineActionCodecConfig  # noqa: E402
from models.wam_multimodal import (  # noqa: E402
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
)
from train.generic_m1_trajectory_dataset import (  # noqa: E402
    GENERIC_M1_DATASET_PROTOCOL,
    GenericM1ManifestIndex,
    GenericM1WindowDataset,
)
from train.m1_scratch_builder import (  # noqa: E402
    ScratchM1BuildConfig,
    build_scratch_m1,
)
from train.m1_scratch_checkpointing import (  # noqa: E402
    save_scratch_m1_checkpoint,
)
from train.m1_scratch_training import (  # noqa: E402
    ScratchM1StageConfig,
    build_scratch_optimizer,
    scratch_stage_required_keys,
    train_scratch_m1_stage,
    validate_scratch_stage_order,
)
from train.m1_training import M1FlowObjectiveConfig  # noqa: E402


CONFIG_FORMAT = "wam.multimodal.m1.scratch_config/1"
CONSOLE = Console(stderr=True)


@dataclass(frozen=True)
class ScratchInputPipelineConfig:
    """Resolved high-throughput controls recorded in checkpoint provenance."""

    batch_size: int
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    multiprocessing_context: str
    in_order: bool
    hdf5_cache_size: int
    preload_to_ram: bool
    preload_shared_memory: bool
    preload_max_available_fraction: float
    precision: str
    torch_float32_matmul_precision: str
    allow_tf32: bool
    cudnn_benchmark: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 16D LiftBarrier M1 from one random task-side initialization. "
            "No legacy model, action prior, or checkpoint is accepted."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m1_liftbarrier_scratch.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--steps-scale",
        type=float,
        default=1.0,
        help="Scale every stage length; intended for controlled smoke runs.",
    )
    parser.add_argument(
        "--skip-hdf5-sha256",
        action="store_true",
        help="Skip expensive episode hashes; HDF5 schema is still validated.",
    )
    parser.add_argument(
        "--no-rich-progress",
        action="store_true",
        help="Disable Rich progress rendering; checkpoint evidence is unchanged.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < float(args.steps_scale) <= 1.0:
        raise ValueError("--steps-scale must be in (0,1]")
    config_path = args.config.expanduser().resolve(strict=True)
    config = _load_yaml(config_path)
    if config.get("format_version") != CONFIG_FORMAT:
        raise ValueError("unsupported scratch M1 config format")
    data_cfg = _mapping(config, "data")
    if data_cfg.get("dataset_protocol") != GENERIC_M1_DATASET_PROTOCOL:
        raise ValueError("scratch LiftBarrier requires generic trajectory data")
    progress_enabled = not args.no_rich_progress
    device = _device(args.device)
    training_cfg = _mapping(config, "training")
    pipeline = _resolve_input_pipeline(training_cfg, device=device)
    _configure_compute(pipeline, device=device)
    manifest = _load_manifest(
        _path(data_cfg["manifest"]),
        verify_hdf5_sha256=not args.skip_hdf5_sha256,
        progress_enabled=progress_enabled,
    )
    codec_cfg = _load_codec(config, manifest)
    build_cfg = ScratchM1BuildConfig.from_dict(_mapping(config, "build"))
    _validate_dimensions(config, manifest, build_cfg)
    stages = tuple(
        _scaled_stage(value, float(args.steps_scale))
        for value in _sequence(training_cfg, "stages")
    )
    validate_scratch_stage_order(stages)
    _print_stage_plan(stages, pipeline=pipeline)

    train_dataset = GenericM1WindowDataset(
        manifest,
        split="train",
        state_history=int(data_cfg["state_history"]),
        action_chunk=int(data_cfg["action_horizon"]),
        cameras=tuple(data_cfg["camera_order"]),
        visual_history=int(data_cfg["visual_history_frames"]),
        future_horizons=tuple(data_cfg["future_visual_horizons"]),
        hdf5_cache_size=pipeline.hdf5_cache_size,
    )
    preload_keys = GenericM1WindowDataset.SAMPLE_KEYS
    preload_report = _preload_training_split(
        train_dataset,
        preload_keys,
        pipeline=pipeline,
        progress_enabled=progress_enabled,
    )

    with CONSOLE.status(
        "[bold cyan]Loading and strictly verifying frozen DINOv3 teacher…",
        spinner="dots",
        refresh_per_second=8,
    ) if progress_enabled else nullcontext():
        vision = _build_vision_encoder(_mapping(config, "vision"))
        bundle = build_scratch_m1(
            build_cfg,
            manifest.load_normalization(),
            codec_cfg,
            vision_encoder=vision,
        )
        bundle.to(device)

    weight_decays = {float(stage.weight_decay) for stage in stages}
    if len(weight_decays) != 1:
        raise ValueError("all scratch stages must share one optimizer weight_decay")
    optimizer = build_scratch_optimizer(
        bundle.model,
        bundle.action_flow,
        weight_decay=weight_decays.pop(),
    )
    flow_objective = M1FlowObjectiveConfig(**_mapping(config, "flow_objective"))
    if flow_objective.policy_fixed_action_dims:
        raise ValueError("scratch LiftBarrier must supervise all 16 action dimensions")

    reports: list[dict[str, Any]] = []
    progress = _training_progress(progress_enabled)
    with progress:
        overall_task = progress.add_task(
            "[bold]Stages",
            total=len(stages),
            status=f"0/{len(stages)} ready",
        )
        for stage_index, stage in enumerate(stages):
            sample_keys = scratch_stage_required_keys(bundle.model, stage)
            loader = _stage_loader(
                train_dataset,
                sample_keys,
                pipeline=pipeline,
                device=device,
                seed=build_cfg.seed + stage_index,
            )
            if len(loader) == 0:
                raise RuntimeError("scratch training loader has no complete batch")
            stage_number = stage_index + 1
            stage_task = progress.add_task(
                f"[cyan][{stage_number}/{len(stages)}][/cyan] {stage.name}",
                total=stage.steps,
                status="starting",
            )
            progress.update(
                overall_task,
                status=f"{stage_number}/{len(stages)} {stage.name}",
            )

            def update_stage(
                completed: int,
                metrics: Mapping[str, float],
                *,
                task_id: int = stage_task,
            ) -> None:
                status = (
                    f"loss={metrics['total']:.5g} "
                    f"grad={metrics['gradient_norm']:.4g}"
                )
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / (1024**3)
                    status += f" gpu={allocated:.2f}GiB"
                progress.update(
                    task_id,
                    completed=completed,
                    status=status,
                )

            report = train_scratch_m1_stage(
                bundle.model,
                bundle.action_flow,
                loader,
                optimizer,
                stage,
                device=device,
                flow_objective=flow_objective,
                seed=build_cfg.seed + stage_index,
                precision=pipeline.precision,
                progress=update_stage,
            )
            history = report.pop("history")
            report["final_metrics"] = history[-1]
            report["sample_keys"] = sorted(sample_keys)
            reports.append(report)
            progress.update(stage_task, status="complete")
            progress.advance(overall_task)
            del loader
            gc.collect()
        progress.update(overall_task, status=f"{len(stages)}/{len(stages)} complete")

    train_dataset.clear_ram_preload()
    train_dataset.close()

    checkpoint_cfg = _mapping(config, "checkpoint")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _path(checkpoint_cfg["output"])
    )
    save_scratch_m1_checkpoint(
        output,
        bundle,
        dataset_lineage=manifest.checkpoint_lineage("train"),
        stage_state={
            "completed_stage_count": len(stages),
            "stages": [stage.to_dict() for stage in stages],
            "optimizer_state_policy": "preserved_across_all_stages",
        },
        metrics={"stage_reports": reports},
        provenance={
            "config_path": str(config_path),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "legacy_checkpoint_loaded": False,
            "input_pipeline": asdict(pipeline),
            "ram_preload": preload_report,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "checkpoint": str(output),
                "initialization_mode": "scratch",
                "action_dim": bundle.action_codec.action_dim,
                "action_codec_sha256": bundle.action_codec.semantic_sha256,
                "completed_stages": [stage.name for stage in stages],
                "input_pipeline": asdict(pipeline),
                "ram_preload": preload_report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_manifest(
    path: Path,
    *,
    verify_hdf5_sha256: bool,
    progress_enabled: bool,
) -> GenericM1ManifestIndex:
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=CONSOLE,
        disable=not progress_enabled,
    ) as progress:
        task = progress.add_task("Audit manifest/HDF5", total=None)

        def update(current: int, total: int) -> None:
            progress.update(task, total=total, completed=current)

        manifest = GenericM1ManifestIndex.from_path(
            path,
            verify_hdf5_sha256=verify_hdf5_sha256,
            verify_hdf5_contract=True,
            verify_normalization=True,
            progress_callback=update,
        )
        progress.update(task, completed=len(manifest.episodes))
        return manifest


def _resolve_input_pipeline(
    training: Mapping[str, Any],
    *,
    device: torch.device,
) -> ScratchInputPipelineConfig:
    raw_workers = training.get("num_workers", "auto")
    if raw_workers == "auto":
        cpu_count = os.cpu_count() or 1
        num_workers = min(12, max(2, cpu_count // 2))
    else:
        num_workers = int(raw_workers)
    if num_workers < 0:
        raise ValueError("scratch num_workers must be non-negative or 'auto'")

    batch_size = int(training["batch_size"])
    prefetch_factor = int(training.get("prefetch_factor", 2))
    hdf5_cache_size = int(training.get("hdf5_cache_size", 8))
    if batch_size <= 0 or prefetch_factor <= 0 or hdf5_cache_size < 0:
        raise ValueError("invalid scratch batch/prefetch/HDF5 cache controls")
    persistent = _boolean(training, "persistent_workers", default=True)
    pin_memory = _boolean(training, "pin_memory", default=True)
    in_order = _boolean(training, "in_order", default=True)
    context = str(training.get("multiprocessing_context", "spawn"))
    if context not in {"spawn", "forkserver", "fork"}:
        raise ValueError("unsupported scratch multiprocessing_context")
    preload = _boolean(training, "preload_to_ram", default=False)
    preload_shared = _boolean(
        training,
        "preload_shared_memory",
        default=num_workers > 0,
    )
    if preload and num_workers > 0 and context != "fork" and not preload_shared:
        raise ValueError(
            "spawn/forkserver RAM preload must use shared memory to avoid "
            "one dataset copy per worker"
        )
    max_fraction = float(training.get("preload_max_available_fraction", 0.5))
    if not 0.0 < max_fraction < 1.0:
        raise ValueError("preload_max_available_fraction must be in (0,1)")
    precision = str(training.get("precision", "fp32"))
    if precision not in {"fp32", "bf16"}:
        raise ValueError("scratch precision must be 'fp32' or 'bf16'")
    matmul_precision = str(
        training.get("torch_float32_matmul_precision", "high")
    )
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("invalid torch_float32_matmul_precision")

    return ScratchInputPipelineConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent and num_workers > 0,
        pin_memory=pin_memory and device.type == "cuda",
        multiprocessing_context=context,
        in_order=in_order,
        hdf5_cache_size=hdf5_cache_size,
        preload_to_ram=preload,
        preload_shared_memory=preload_shared,
        preload_max_available_fraction=max_fraction,
        precision=precision,
        torch_float32_matmul_precision=matmul_precision,
        allow_tf32=_boolean(training, "allow_tf32", default=True),
        cudnn_benchmark=_boolean(training, "cudnn_benchmark", default=True),
    )


def _configure_compute(
    pipeline: ScratchInputPipelineConfig,
    *,
    device: torch.device,
) -> None:
    torch.set_float32_matmul_precision(pipeline.torch_float32_matmul_precision)
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = pipeline.allow_tf32
    torch.backends.cudnn.allow_tf32 = pipeline.allow_tf32
    torch.backends.cudnn.benchmark = pipeline.cudnn_benchmark


def _preload_training_split(
    dataset: GenericM1WindowDataset,
    sample_keys: frozenset[str],
    *,
    pipeline: ScratchInputPipelineConfig,
    progress_enabled: bool,
) -> dict[str, Any]:
    if not pipeline.preload_to_ram:
        return {
            "enabled": False,
            "strategy": "parallel_hdf5_workers_with_os_page_cache",
        }

    estimated = dataset.estimate_ram_preload_bytes(sample_keys)
    available = _available_memory_bytes()
    allowed = int(available * pipeline.preload_max_available_fraction)
    if estimated > allowed:
        raise MemoryError(
            "scratch RAM preload exceeds the configured available-memory budget: "
            f"required={decimal(estimated)}, allowed={decimal(allowed)}, "
            f"available={decimal(available)}"
        )
    shared_available: int | None = None
    if pipeline.preload_shared_memory:
        shared_available = shutil.disk_usage("/dev/shm").free
        shared_allowed = int(shared_available * 0.9)
        if estimated > shared_allowed:
            raise MemoryError(
                "scratch shared-RAM preload exceeds /dev/shm budget: "
                f"required={decimal(estimated)}, allowed={decimal(shared_allowed)}"
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[loaded]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        disable=not progress_enabled,
    ) as progress:
        task = progress.add_task(
            "Preload train split to shared RAM",
            total=len(dataset.records),
            loaded="0 B",
        )

        def update(current: int, total: int, loaded: int) -> None:
            progress.update(
                task,
                total=total,
                completed=current,
                loaded=decimal(loaded),
            )

        report = dataset.preload_to_ram(
            sample_keys,
            shared_memory=pipeline.preload_shared_memory,
            progress_callback=update,
        )
    return {
        **report,
        "strategy": "decompressed_worker_shared_ram",
        "estimated_bytes": estimated,
        "available_memory_bytes_before_preload": available,
        "shared_memory_available_bytes_before_preload": shared_available,
        "max_available_fraction": pipeline.preload_max_available_fraction,
    }


def _stage_loader(
    dataset: GenericM1WindowDataset,
    sample_keys: frozenset[str],
    *,
    pipeline: ScratchInputPipelineConfig,
    device: torch.device,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    kwargs: dict[str, Any] = {
        "batch_size": pipeline.batch_size,
        "shuffle": True,
        "num_workers": pipeline.num_workers,
        "pin_memory": pipeline.pin_memory and device.type == "cuda",
        "drop_last": True,
        "generator": torch.Generator().manual_seed(int(seed)),
        "in_order": pipeline.in_order,
    }
    if pipeline.num_workers > 0:
        kwargs.update(
            prefetch_factor=pipeline.prefetch_factor,
            persistent_workers=pipeline.persistent_workers,
            multiprocessing_context=pipeline.multiprocessing_context,
            worker_init_fn=_data_worker_init,
        )
    return DataLoader(dataset.project(sample_keys), **kwargs)


def _data_worker_init(worker_id: int) -> None:
    del worker_id
    torch.set_num_threads(1)


def _training_progress(enabled: bool) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        disable=not enabled,
        refresh_per_second=8,
    )


def _print_stage_plan(
    stages: tuple[ScratchM1StageConfig, ...],
    *,
    pipeline: ScratchInputPipelineConfig,
) -> None:
    table = Table(title=f"LiftBarrier M1 scratch curriculum — {len(stages)} stages")
    table.add_column("Stage", justify="right")
    table.add_column("Name")
    table.add_column("Objective")
    table.add_column("Steps", justify="right")
    for index, stage in enumerate(stages, start=1):
        table.add_row(
            f"{index}/{len(stages)}",
            stage.name,
            stage.objective,
            f"{stage.steps:,}",
        )
    CONSOLE.print(table)
    CONSOLE.print(
        "[dim]pipeline:[/dim] "
        f"batch={pipeline.batch_size}, workers={pipeline.num_workers}, "
        f"prefetch={pipeline.prefetch_factor}, pin_memory={pipeline.pin_memory}, "
        f"persistent={pipeline.persistent_workers}, "
        f"RAM preload={pipeline.preload_to_ram}, "
        f"precision={pipeline.precision}, TF32={pipeline.allow_tf32}"
    )


def _available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    return page_size * available_pages


def _boolean(
    values: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"scratch training {key!r} must be a boolean")
    return value


def _load_codec(
    config: Mapping[str, Any], manifest: GenericM1ManifestIndex
) -> AffineActionCodecConfig:
    codec_section = _mapping(config, "action_codec")
    codec = AffineActionCodecConfig.load(_path(codec_section["path"]))
    expected = str(codec_section["expected_semantic_sha256"])
    if codec.sha256() != expected:
        raise ValueError("action codec semantic hash differs from scratch config")
    if manifest.action_codec is None:
        raise ValueError(
            "training manifest has no applied action codec; regenerate it with "
            "scripts/prepare_robofactory_m1_training_artifacts.py"
        )
    if manifest.action_codec.semantic_sha256 != expected:
        raise ValueError("training manifest action codec differs from scratch config")
    if manifest.action_storage_domain != codec.raw_domain:
        raise ValueError("training manifest raw action domain differs from codec")
    if manifest.action_domain != codec.encoded_domain:
        raise ValueError("training manifest model action domain differs from codec")
    return codec


def _validate_dimensions(
    config: Mapping[str, Any],
    manifest: GenericM1ManifestIndex,
    build: ScratchM1BuildConfig,
) -> None:
    data = _mapping(config, "data")
    declared = (int(data["state_dim"]), int(data["action_dim"]))
    observed = (manifest.state_dim, manifest.action_dim)
    model = (build.world.state_dim, build.world.action_dim)
    if len({declared, observed, model}) != 1:
        raise ValueError(
            f"scratch data/model dimensions differ: config={declared}, "
            f"manifest={observed}, model={model}"
        )
    if tuple(data["camera_order"]) != manifest.camera_order:
        raise ValueError("scratch camera order differs from training manifest")
    if tuple(build.latent_wam.task_vocabulary) != manifest.task_order:
        raise ValueError("scratch task vocabulary differs from training manifest")


def _build_vision_encoder(config: Mapping[str, Any]) -> FrozenDINOv3Encoder:
    if config.get("frozen") is not True:
        raise ValueError("scratch M1 requires frozen=true for DINOv3")
    return FrozenDINOv3Encoder(
        FrozenDINOv3Config(
            encoder_name=str(config["encoder_name"]),
            model_id=str(config["model_id"]),
            revision=str(config["revision"]),
            config_path=_path(config["config_path"]),
            weights_path=_path(config["weights_path"]),
            expected_config_sha256=str(config["expected_config_sha256"]),
            expected_weights_sha256=str(config["expected_weights_sha256"]),
            preprocess_id=str(config["preprocess_id"]),
            input_size=int(config["input_size"]),
            inference_batch_size=int(config["inference_batch_size"]),
        )
    )


def _scaled_stage(value: Any, scale: float) -> ScratchM1StageConfig:
    if not isinstance(value, Mapping):
        raise ValueError("scratch training stage must be a mapping")
    payload = dict(value)
    payload["steps"] = max(1, round(int(payload["steps"]) * scale))
    return ScratchM1StageConfig.from_dict(payload)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"scratch M1 config {key!r} must be a mapping")
    return dict(result)


def _sequence(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise ValueError(f"scratch M1 config {key!r} must be a non-empty list")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("scratch M1 config must contain a mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
