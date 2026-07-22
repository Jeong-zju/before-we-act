#!/usr/bin/env python3
"""Audit an M1 data protocol and materialize representative window samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.m1_data_protocol import (  # noqa: E402
    build_m1_window_dataset,
    load_m1_data_manifest,
    m1_data_protocol_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation", "test"),
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--state-history", type=int, default=32)
    parser.add_argument("--action-chunk", type=int, default=8)
    parser.add_argument("--visual-history", type=int, default=2)
    parser.add_argument(
        "--future-horizons", nargs="+", type=int, default=(1, 2, 4, 8)
    )
    parser.add_argument("--camera", action="append", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hdf5-cache-size", type=int, default=2)
    parser.add_argument(
        "--skip-hdf5-hashes",
        action="store_true",
        help="Skip byte hashing for a quick smoke; HDF5 contracts are still audited.",
    )
    parser.add_argument(
        "--skip-normalization-hash",
        action="store_true",
        help="Skip normalization file/hash validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_m1_data_manifest(
        args.manifest,
        verify_hdf5_sha256=not args.skip_hdf5_hashes,
        verify_hdf5_contract=True,
        verify_normalization=not args.skip_normalization_hash,
        progress_callback=_progress,
    )
    report: dict[str, Any] = {
        "passed": True,
        "manifest": str(manifest.manifest_path),
        "manifest_sha256": manifest.manifest_sha256,
        "protocol": m1_data_protocol_evidence(manifest),
        "camera_order": list(manifest.camera_order),
        "task_order": list(manifest.task_order),
        "splits": {},
    }
    for split in dict.fromkeys(args.splits):
        dataset = build_m1_window_dataset(
            manifest,
            split=split,
            state_history=args.state_history,
            action_chunk=args.action_chunk,
            cameras=args.camera,
            visual_history=args.visual_history,
            future_horizons=args.future_horizons,
            stride=args.stride,
            hdf5_cache_size=args.hdf5_cache_size,
        )
        try:
            indices = tuple(dict.fromkeys((0, len(dataset) - 1)))
            samples = [_sample_summary(dataset[index]) for index in indices]
            report["splits"][split] = {
                "window_summary": dataset.window_summary(),
                "sample_lineage": [
                    {
                        "path": str(dataset.sample_lineage(index).path),
                        "decision_t": dataset.sample_lineage(index).decision_t,
                    }
                    for index in indices
                ],
                "samples": samples,
                "checkpoint_lineage": dataset.checkpoint_lineage(),
            }
        finally:
            dataset.close()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _sample_summary(sample: dict[str, torch.Tensor]) -> dict[str, Any]:
    expected = {
        "states",
        "state_valid_mask",
        "past_actions",
        "past_action_valid_mask",
        "images",
        "task_index",
        "action_targets",
        "future_states",
        "future_images",
        "future_image_novelty_mask",
        "future_horizons",
    }
    if set(sample) != expected:
        raise RuntimeError(f"sample key contract drifted: {sorted(sample)}")
    for name, value in sample.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"sample tensor {name!r} contains NaN/Inf")
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in sorted(sample.items())
    }


def _progress(current: int, total: int) -> None:
    if current == 0 or current == total or current % 10 == 0:
        print(f"verify HDF5 {current}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
