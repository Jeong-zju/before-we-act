#!/usr/bin/env python3
"""Extend the frozen R3 PCA artifact with train-only shared-view statistics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_s2_r3_artifacts import (  # noqa: E402
    _dataset,
    _encode_valid_spatial_grid,
    _fit_indices,
    _progress,
    _stage,
    _statistics,
    _validate_complete_agent_cameras,
)
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _atomic_torch_save,
    _load_yaml,
    _mapping,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    file_sha256,
    load_s2_artifact,
    project_dino_grid,
)
from train.s2_grouped_trajectory import (  # noqa: E402
    S2_FUTURE_HORIZONS,
    grouped_s2_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-log", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    artifacts = _mapping(raw, "artifacts")
    source = (
        args.source.expanduser().resolve(strict=True)
        if args.source is not None
        else (
            ROOT / str(artifacts["r3_pca_statistics"])
        ).resolve(strict=True)
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            ROOT / str(artifacts["pca_statistics"])
        ).resolve()
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else None
    )
    source_sha256 = file_sha256(source)
    if output.is_file():
        existing = load_s2_artifact(output, device=torch.device("cpu"))
        extension = existing.get("r4_shared_extension")
        if (
            isinstance(extension, Mapping)
            and extension.get("source_r3_artifact_sha256") == source_sha256
            and _valid_shared_statistics(existing)
        ):
            _stage(progress_log, "r4_artifact_reuse", f"verified {output}")
            print(
                json.dumps(
                    {"artifact": str(output), "sha256": file_sha256(output)}
                )
            )
            return 0
        raise ValueError(
            "existing R4 artifact does not match the frozen R3 PCA source"
        )

    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S2-R4 artifact fitting requires one visible GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S2-R4 artifact fitting requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(int(_mapping(raw, "training")["seed"]))

    artifact = load_s2_artifact(source, device=device)
    dataset = _dataset(raw, split="train")
    _validate_complete_agent_cameras(dataset)
    pca = _mapping(raw, "pca")
    windows_per_task = int(pca.get("windows_per_task", 16))
    indices = _fit_indices(dataset, windows_per_task)
    grid_height = int(pca.get("grid_height", 2))
    grid_width = int(pca.get("grid_width", 2))
    grid_tokens = grid_height * grid_width
    vision = _vision(raw).to(device).eval()
    values: list[list[torch.Tensor]] = [
        [] for _ in S2_FUTURE_HORIZONS
    ]
    _stage(
        progress_log,
        "r4_shared_statistics",
        "fitting global-slot delta normalization on frozen train windows",
    )
    for position, index in enumerate(indices, start=1):
        grouped = grouped_s2_batch(default_collate([dataset[index]]))
        current_valid = grouped["shared_observation_valid_mask"].to(device)
        future_valid = grouped["future_shared_visual_valid_mask"].to(device)
        current = torch.zeros(
            1,
            grid_tokens,
            256,
            device=device,
        )
        future = torch.zeros(
            1,
            len(S2_FUTURE_HORIZONS),
            grid_tokens,
            256,
            device=device,
        )
        current_features = _encode_valid_spatial_grid(
            vision,
            grouped["shared_observation"].to(device),
            current_valid,
            grid_height=grid_height,
            grid_width=grid_width,
        )
        if current_features.numel():
            current[current_valid] = project_dino_grid(
                current_features,
                artifact,
            )
        future_features = _encode_valid_spatial_grid(
            vision,
            grouped["future_shared_observations"].to(device),
            future_valid,
            grid_height=grid_height,
            grid_width=grid_width,
        )
        if future_features.numel():
            future[future_valid] = project_dino_grid(
                future_features,
                artifact,
            )
        delta = future - current[:, None]
        for horizon_index in range(len(S2_FUTURE_HORIZONS)):
            mask = future_valid[:, horizon_index]
            values[horizon_index].append(
                delta[:, horizon_index][mask]
                .reshape(-1, 256)
                .float()
                .cpu()
            )
        if position == 1 or position % 10 == 0 or position == len(indices):
            _progress(
                progress_log,
                phase="r4_shared_statistics",
                completed=position,
                total=len(indices),
            )

    mean, std = _statistics(values, width=256)
    payload = {
        key: value.detach().cpu()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in artifact.items()
    }
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shared_visual_delta_mean": mean,
            "shared_visual_delta_std": std,
            "r4_shared_extension": {
                "source_r3_artifact": str(source),
                "source_r3_artifact_sha256": source_sha256,
                "fit_split": "train",
                "windows_per_task": windows_per_task,
                "windows": len(indices),
                "target": "global_shared_view_delta",
            },
        }
    )
    _atomic_torch_save(payload, output)
    dataset.close()
    _stage(progress_log, "r4_artifact_complete", f"wrote {output}")
    print(
        json.dumps({"artifact": str(output), "sha256": file_sha256(output)})
    )
    return 0


def _valid_shared_statistics(value: Mapping[str, object]) -> bool:
    for key in ("shared_visual_delta_mean", "shared_visual_delta_std"):
        tensor = value.get(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != (len(S2_FUTURE_HORIZONS), 256)
            or not bool(torch.isfinite(tensor).all())
        ):
            return False
    std = value["shared_visual_delta_std"]
    return isinstance(std, torch.Tensor) and bool(std.gt(0.0).all())


if __name__ == "__main__":
    raise SystemExit(main())
