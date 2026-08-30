from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .common import ARMS, TASKS, atomic_json
from .precompute import (
    CACHE_SCHEMA,
    ENCODER_PRECISION,
    RGB_PREPROCESSING,
    STORED_DTYPE,
    load_encoder,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-task", type=int, default=4)
    # Match the batch shape used by precompute.py.  NoPoSplat has tiny
    # batch-shape-dependent FP32 reduction differences, so comparing a
    # mixed batch against a cache generated in per-trajectory batches can
    # trip an otherwise healthy parity gate.
    parser.add_argument("--batch-size", type=int, default=120)
    args = parser.parse_args()

    data_root, cache_root = Path(args.data_root), Path(args.cache_root)
    metadata = json.loads((cache_root / "metadata.json").read_text())
    expected = {
        "schema": CACHE_SCHEMA,
        "rgb_preprocessing": RGB_PREPROCESSING,
        "encoder_precision": ENCODER_PRECISION,
        "stored_dtype": STORED_DTYPE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"cache metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")

    cached, sample_ids = [], []
    samples = {}
    for task in TASKS:
        task_meta = json.loads((cache_root / task / "metadata.json").read_text())
        row = task_meta["rows"][0]
        source_path = data_root / task / "motionplanning" / row["shard"]
        with h5py.File(source_path, "r") as source, h5py.File(cache_root / task / f"{task}.h5", "r") as cache:
            trajectory = source[row["trajectory"]]
            length = int(row["length"])
            positions = np.linspace(0, length - 1, args.samples_per_task, dtype=np.int64)
            for index, position in enumerate(positions):
                arm = index % ARMS[task]
                offset = int(row["offsets"][str(arm)])
                cached.append(np.asarray(cache[f"gaussian_arm{arm}"][offset + position], np.float32))
                position = int(position)
                samples[(task, arm, position)] = np.asarray(
                    trajectory[f"obs/sensor_data/head_camera_agent{arm}/rgb"][position], np.uint8
                )
                sample_ids.append({"task": task, "trajectory": row["trajectory"], "arm": arm, "position": position})

    device = torch.device("cuda:0")
    encoder = load_encoder(Path(args.weight), device)
    # Reproduce precompute.py's per-task/per-arm trajectory batches, then
    # retain only the sampled positions. This makes the comparison sensitive
    # to real preprocessing/cache errors rather than harmless batch numerics.
    online_by_key = {}
    for task in TASKS:
        task_meta = json.loads((cache_root / task / "metadata.json").read_text())
        row = task_meta["rows"][0]
        source_path = data_root / task / "motionplanning" / row["shard"]
        with h5py.File(source_path, "r") as source:
            trajectory = source[row["trajectory"]]
            length = int(row["length"])
            for arm in range(ARMS[task]):
                images = np.asarray(
                    trajectory[f"obs/sensor_data/head_camera_agent{arm}/rgb"][:length], np.uint8
                )
                for begin in range(0, length, args.batch_size):
                    batch = images[begin : begin + args.batch_size]
                    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255.0)
                    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).mul(2).sub(1).to(device)
                    with torch.inference_mode():
                        y = encoder({"image": x[:, None]})[:, 0].float()
                    y = F.interpolate(y, size=tuple(metadata["gaussian_hw"]), mode="bilinear", align_corners=False).cpu().numpy()
                    for position in range(begin, min(begin + args.batch_size, length)):
                        key = (task, arm, position)
                        if key in samples:
                            online_by_key[key] = y[position - begin]
    online = np.stack([online_by_key[(x["task"], x["arm"], x["position"])] for x in sample_ids])
    cached_array = np.stack(cached)
    delta = cached_array - online
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    max_abs = float(np.max(np.abs(delta)))
    correlation = float(np.corrcoef(cached_array.reshape(-1), online.reshape(-1))[0, 1])
    range_delta = float(max(abs(float(cached_array.min() - online.min())), abs(float(cached_array.max() - online.max()))))
    passed = bool(mae <= 1e-4 and rmse <= 2e-4 and max_abs <= 5e-3 and correlation >= 0.999999 and range_delta <= 5e-3)
    result = {
        "schema": "mars-control.gaudp.cache-parity.v1",
        "status": "complete" if passed else "failed",
        "passed": passed,
        "samples": len(sample_ids),
        "sample_ids": sample_ids,
        "rgb_preprocessing": RGB_PREPROCESSING,
        "encoder_precision": ENCODER_PRECISION,
        "stored_dtype": STORED_DTYPE,
        "cache_range": [float(cached_array.min()), float(cached_array.max())],
        "online_range": [float(online.min()), float(online.max())],
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "correlation": correlation,
        "range_delta": range_delta,
    }
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError("FP32 Gaussian cache/online parity gate failed")


if __name__ == "__main__":
    main()
