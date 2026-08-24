#!/usr/bin/env python3
"""Validate the complete arm-local GauDP cache before formal training."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
AGENTS = {
    "lift_barrier": 2,
    "camera_alignment": 3,
    "long_pipeline_delivery": 4,
    "take_photo": 4,
    "pass_shoe": 2,
    "place_food": 2,
}
CACHE_ROOT = Path("/workspace/bwa_gau_dp_data/cache")
OUTPUT = Path("/workspace/bwa_gau_dp_runs/audit/cache_contract.json")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    rows = {}
    local_samples = 0
    for task in TASKS:
        marker = json.loads((CACHE_ROOT / f"{task}.complete.json").read_text())
        if marker.get("status") != "complete" or marker.get("episodes") != 150:
            raise RuntimeError(f"{task}: incomplete cache marker")
        frames = int(marker["frames"])
        agents = AGENTS[task]
        expected_keys = {"episode_ends"}
        expected_keys.update(f"head_cam_{agent}" for agent in range(agents))
        expected_keys.update(f"gaussian_{agent}" for agent in range(agents))
        with h5py.File(CACHE_ROOT / f"{task}.h5", "r", swmr=True) as cache:
            if set(cache) != expected_keys:
                raise RuntimeError(f"{task}: unexpected cache keys {sorted(set(cache) - expected_keys)}")
            ends = np.asarray(cache["episode_ends"][:], np.int64)
            if ends.shape != (150,) or not np.all(np.diff(ends) > 0) or int(ends[-1]) != frames:
                raise RuntimeError(f"{task}: invalid episode boundaries")
            probe = np.unique(np.linspace(0, frames - 1, min(32, frames), dtype=np.int64))
            for agent in range(agents):
                image = cache[f"head_cam_{agent}"]
                gaussian = cache[f"gaussian_{agent}"]
                if image.shape != (frames, 3, 120, 160) or image.dtype != np.dtype("uint8"):
                    raise RuntimeError(f"{task}/agent_{agent}: invalid RGB cache {image.shape} {image.dtype}")
                if gaussian.shape != (frames, 13, 120, 160) or gaussian.dtype != np.dtype("float16"):
                    raise RuntimeError(f"{task}/agent_{agent}: invalid Gaussian cache {gaussian.shape} {gaussian.dtype}")
                if not np.isfinite(gaussian[probe]).all():
                    raise RuntimeError(f"{task}/agent_{agent}: non-finite Gaussian probe")
        local_samples += frames * agents
        rows[task] = {"episodes": 150, "frames": frames, "agents": agents, "local_samples": frames * agents}

    sys.path.insert(0, "/workspace/repos/Policy-Lightning")
    from bwa.robofactory_gaudp_dataset import RoboFactoryGauDPDataset

    dataset = RoboFactoryGauDPDataset()
    if len(dataset) != 795727 or local_samples != 795727:
        raise RuntimeError(f"dataset/cache sample mismatch: {len(dataset)} / {local_samples}")
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        obs = sample["obs"]
        if obs["head_cam_0"].shape != (3, 3, 120, 160):
            raise RuntimeError(f"sample {index}: invalid RGB sequence")
        if obs["gaussian_0"].shape != (3, 13, 120, 160):
            raise RuntimeError(f"sample {index}: invalid Gaussian sequence")
        if obs["state"].shape != (3, 9) or sample["action"].shape != (8, 8):
            raise RuntimeError(f"sample {index}: invalid local state/action sequence")

    atomic_json(
        OUTPUT,
        {
            "schema": "bwa.gaudp.cache_contract.v1",
            "status": "complete",
            "episodes": 900,
            "local_agent_streams": 2550,
            "local_samples": local_samples,
            "tasks": rows,
            "cache_fields": ["local_rgb", "local_single_view_gaussian"],
            "raw_fields_read_by_training": ["local_qpos9", "local_commanded_action8"],
            "forbidden_fields": ["peer_rgb", "global_rgb", "global_state", "joint_action"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
