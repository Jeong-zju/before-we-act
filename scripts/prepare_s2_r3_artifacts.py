#!/usr/bin/env python3
"""Fit the S2-R3 train-only DINO PCA and future-target statistics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _append_jsonl,
    _atomic_torch_save,
    _load_yaml,
    _mapping,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    S2_ARTIFACT_FORMAT,
    file_sha256,
    load_s2_artifact,
    project_dino_grid,
)
from train.s2_grouped_trajectory import (  # noqa: E402
    S2_FUTURE_HORIZONS,
    S2GroupedTrajectoryDataset,
    grouped_s2_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-log", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    artifact_config = _mapping(raw, "artifacts")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (ROOT / str(artifact_config["pca_statistics"])).resolve()
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else None
    )
    if output.is_file():
        load_s2_artifact(output, device=torch.device("cpu"))
        _stage(progress_log, "artifact_reuse", f"verified existing {output}")
        print(json.dumps({"artifact": str(output), "sha256": file_sha256(output)}))
        return 0

    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S2-R3 artifact fitting requires exactly one visible GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S2-R3 artifact fitting requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(int(_mapping(raw, "training")["seed"]))

    _stage(progress_log, "dataset_validation", "opening five train manifests")
    dataset = _dataset(raw, split="train")
    pca_config = _mapping(raw, "pca")
    windows_per_task = int(pca_config.get("windows_per_task", 16))
    if windows_per_task <= 0:
        raise ValueError("pca.windows_per_task must be positive")
    indices = _fit_indices(dataset, windows_per_task)
    selection_sha256 = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    _stage(progress_log, "dinov3_load", "loading frozen verified DINOv3")
    vision = _vision(raw).to(device).eval()
    grid_height = int(pca_config.get("grid_height", 2))
    grid_width = int(pca_config.get("grid_width", 2))
    raw_features: list[torch.Tensor] = []
    for position, index in enumerate(indices, start=1):
        grouped = grouped_s2_batch(default_collate([dataset[index]]))
        valid = grouped["valid_agent_mask"][0]
        images = grouped["agent_observations"][0, valid].to(device)
        encoded = vision.forward_spatial_grid(
            images,
            grid_height=grid_height,
            grid_width=grid_width,
        ).spatial_tokens
        raw_features.append(encoded.float().reshape(-1, 1024).cpu())
        if position == 1 or position % 10 == 0 or position == len(indices):
            _progress(
                progress_log,
                phase="pca_samples",
                completed=position,
                total=len(indices),
            )
    feature_matrix = torch.cat(raw_features, dim=0).to(device)
    if feature_matrix.shape[0] < 256:
        raise RuntimeError("S2 PCA fitting produced fewer than 256 patch samples")
    pca_mean = feature_matrix.mean(dim=0)
    centered = feature_matrix - pca_mean
    _stage(
        progress_log,
        "pca_fit",
        f"fitting 1024->256 PCA from {feature_matrix.shape[0]} train patches",
    )
    _, _, components = torch.pca_lowrank(
        centered,
        q=256,
        center=False,
        niter=4,
    )
    projected = centered @ components
    projected_std = projected.std(dim=0, unbiased=False).clamp_min(1e-6)
    temporary_artifact: dict[str, object] = {
        "pca_mean": pca_mean,
        "pca_components": components,
        "pca_projected_std": projected_std,
    }
    del feature_matrix, centered, projected, raw_features

    state_values: list[list[torch.Tensor]] = [
        [] for _ in S2_FUTURE_HORIZONS
    ]
    visual_values: list[list[torch.Tensor]] = [
        [] for _ in S2_FUTURE_HORIZONS
    ]
    for position, index in enumerate(indices, start=1):
        grouped = grouped_s2_batch(default_collate([dataset[index]]))
        valid_agents = grouped["valid_agent_mask"].to(device)
        future_valid = grouped["future_agent_visual_valid_mask"].to(device)
        current_images = grouped["agent_observations"].to(device)
        future_images = grouped["future_agent_observations"].to(device)
        current = torch.zeros(
            1,
            4,
            grid_height * grid_width,
            256,
            device=device,
        )
        current[valid_agents] = project_dino_grid(
            vision.forward_spatial_grid(
                current_images[valid_agents],
                grid_height=grid_height,
                grid_width=grid_width,
            ).spatial_tokens.float(),
            temporary_artifact,
        )
        future = torch.zeros(
            1,
            4,
            len(S2_FUTURE_HORIZONS),
            grid_height * grid_width,
            256,
            device=device,
        )
        future[future_valid] = project_dino_grid(
            vision.forward_spatial_grid(
                future_images[future_valid],
                grid_height=grid_height,
                grid_width=grid_width,
            ).spatial_tokens.float(),
            temporary_artifact,
        )
        visual_delta = future - current[:, :, None]
        state_delta = grouped["future_state_delta"]
        state_valid = grouped["future_state_valid_mask"]
        for horizon_index in range(len(S2_FUTURE_HORIZONS)):
            state_mask = state_valid[:, :, horizon_index]
            visual_mask = future_valid[:, :, horizon_index]
            state_values[horizon_index].append(
                state_delta[:, :, horizon_index][state_mask].float().cpu()
            )
            visual_values[horizon_index].append(
                visual_delta[:, :, horizon_index][visual_mask]
                .reshape(-1, 256)
                .float()
                .cpu()
            )
        if position == 1 or position % 10 == 0 or position == len(indices):
            _progress(
                progress_log,
                phase="target_statistics",
                completed=position,
                total=len(indices),
            )

    state_mean, state_std = _statistics(state_values, width=18)
    visual_mean, visual_std = _statistics(visual_values, width=256)
    payload = {
        "format_version": S2_ARTIFACT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_split": "train",
        "future_horizons": list(S2_FUTURE_HORIZONS),
        "grid_height": grid_height,
        "grid_width": grid_width,
        "pca_mean": pca_mean.cpu(),
        "pca_components": components.cpu(),
        "pca_projected_std": projected_std.cpu(),
        "state_delta_mean": state_mean,
        "state_delta_std": state_std,
        "visual_delta_mean": visual_mean,
        "visual_delta_std": visual_std,
        "fit": {
            "windows_per_task": windows_per_task,
            "windows": len(indices),
            "selection_sha256": selection_sha256,
            "task_vocabulary": list(dataset.task_vocabulary),
        },
        "data": {
            "manifests": [
                {
                    "task_id": contract.task_id,
                    "path": str(contract.manifest_path),
                    "sha256": contract.manifest_sha256,
                }
                for contract in dataset.contracts
            ],
            "dataset_summary": dataset.summary(),
        },
        "vision": {
            "weights_sha256": vision.artifact_sha256,
            "config_sha256": vision.config_sha256,
            "encoder_name": vision.encoder_name,
        },
    }
    _atomic_torch_save(payload, output)
    dataset.close()
    _stage(progress_log, "complete", f"wrote {output}")
    print(json.dumps({"artifact": str(output), "sha256": file_sha256(output)}))
    return 0


def _dataset(
    config: Mapping[str, object],
    *,
    split: str,
) -> S2GroupedTrajectoryDataset:
    data = _mapping(config, "data")
    manifests = [
        (ROOT / str(value)).resolve(strict=True)
        for value in data["manifests"]  # type: ignore[index]
    ]
    return S2GroupedTrajectoryDataset(
        manifests,
        split=split,
        stride=int(data.get("stride", 1)),
        hdf5_cache_size=int(data.get("hdf5_cache_size", 4)),
    )


def _fit_indices(
    dataset: S2GroupedTrajectoryDataset,
    windows_per_task: int,
) -> list[int]:
    result: list[int] = []
    for task_index in range(len(dataset.contracts)):
        indices = dataset.task_indices(task_index)
        count = min(windows_per_task, len(indices))
        if count == 1:
            offsets = [len(indices) // 2]
        else:
            offsets = [
                round(position * (len(indices) - 1) / (count - 1))
                for position in range(count)
            ]
        result.extend(indices.start + offset for offset in offsets)
    return result


def _statistics(
    values: list[list[torch.Tensor]],
    *,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = []
    stds = []
    for horizon_values in values:
        nonempty = [value for value in horizon_values if value.numel()]
        if not nonempty:
            raise RuntimeError("S2 target statistics contain an empty horizon")
        concatenated = torch.cat(nonempty, dim=0).reshape(-1, width)
        means.append(concatenated.mean(dim=0))
        stds.append(concatenated.std(dim=0, unbiased=False).clamp_min(1e-6))
    return torch.stack(means), torch.stack(stds)


def _stage(path: Path | None, stage: str, detail: str) -> None:
    payload = {
        "event": "startup_stage",
        "stage": stage,
        "detail": detail,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _append_jsonl(path, payload)


def _progress(
    path: Path | None,
    *,
    phase: str,
    completed: int,
    total: int,
) -> None:
    payload = {
        "event": "prepare_progress",
        "phase": phase,
        "completed": completed,
        "total": total,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if path is not None:
        _append_jsonl(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
