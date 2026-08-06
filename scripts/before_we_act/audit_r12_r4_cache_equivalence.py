#!/usr/bin/env python3
"""Prove cached post-DINO grids equal native-RGB online recomputation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from before_we_act.data.full_episode_windows import FULL_EPISODE_PROTOCOL
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.spatial_observation import (
    R12SpatialObservationEncoder,
    locked_r12_full_episode_observation,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-index", required=True)
    parser.add_argument("--vision-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    index = json.loads(Path(args.full_index).read_text())
    if (
        index.get("protocol_variant") != FULL_EPISODE_PROTOCOL
        or index.get("observation") != locked_r12_full_episode_observation()
    ):
        raise ValueError("R12-R4 cache equivalence index identity differs")
    encoder = R12SpatialObservationEncoder(
        # Cache construction and deployed Gate20 inference both encode all
        # present fixed views in one bounded batch (at most five).  CUDA GEMM
        # reduction order is batch-shape dependent, so the equivalence audit
        # must use the identical micro-batch contract before requiring exact
        # fp16 equality.
        index["observation"], args.vision_artifact, inference_batch_size=5
    ).to(device).eval()
    rows = []
    for task in TASKS:
        candidates = [
            row
            for row in index["episodes"]
            if row["task"] == task and row["split"] == "validation"
        ]
        row = sorted(candidates, key=lambda item: item["episode_index"])[0]
        timestep = int(row["steps"]) // 2
        with h5py.File(row["path"], "r") as cache:
            metadata = json.loads(str(cache.attrs["metadata_json"]))
            stored = torch.from_numpy(
                np.asarray(cache["spatial_tokens"][timestep])
            ).to(device)
            view_mask = torch.from_numpy(
                np.asarray(cache["spatial_view_mask"][timestep])
            ).to(device).bool()
        images = torch.zeros((5, 3, 480, 640), dtype=torch.uint8)
        with h5py.File(metadata["source_hdf5"], "r") as source:
            image_group = source["data/observation/images"]
            names = ["global"] + [
                f"agent_{index}"
                for index in range(int(view_mask.sum().item()) - 1)
            ]
            for view_index, name in enumerate(names):
                image = np.asarray(image_group[name][timestep])
                if tuple(image.shape) != (480, 640, 3):
                    raise ValueError("cache audit source RGB is not native resolution")
                images[view_index] = torch.from_numpy(image.transpose(2, 0, 1))
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            recomputed, recomputed_mask = encoder(
                images[None].to(device), view_mask[None]
            )
        recomputed = recomputed[0].to(torch.float16)
        difference = (recomputed - stored.to(torch.float16)).abs()
        rows.append(
            {
                "task": task,
                "feature_shard": row["path"],
                "source_hdf5": metadata["source_hdf5"],
                "timestep": timestep,
                "native_input_shape": [480, 640],
                "encoder_patch_grid": [30, 40],
                "cached_grid": [6, 8],
                "mask_exact": bool(torch.equal(view_mask, recomputed_mask[0])),
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
            }
        )
    passed = all(row["mask_exact"] and row["max_abs"] == 0.0 for row in rows)
    result = {
        "schema_version": 1,
        "round": "R12-R4",
        "passed": passed,
        "rule": "one validation canary per task; native 480x640 online DINO recomputation cast to fp16 must equal cached post-encoder 6x8 tokens exactly",
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
