from __future__ import annotations

import json, os
from pathlib import Path

import h5py
import numpy as np

from .common import ACTION_HIGH, ACTION_LOW, TASKS, atomic_json
from .dataset import MarsDPDataset
from .modeling import ACTION_STEPS, HORIZON, OBS_STEPS, _limits


def main():
    data = Path(os.environ.get("MARS_DP_DATA_ROOT", "/workspace/datasets/mars_control"))
    run = Path(os.environ.get("MARS_DP_RUN_ROOT", "/workspace/runs/mars_dp_v2"))
    assert (OBS_STEPS, HORIZON, ACTION_STEPS) == (3, 8, 8)
    ds = MarsDPDataset(data, run / "preflight_normalization.json")
    low, high = np.asarray(ACTION_LOW, np.float32), np.asarray(ACTION_HIGH, np.float32)
    assert np.all(np.asarray(ds.stats["a_min"]) >= low - 1e-6)
    assert np.all(np.asarray(ds.stats["a_max"]) <= high + 1e-6)

    # Pick a non-padded window and prove obs/action share the official timeline:
    # obs=[t-2,t-1,t], action=[t-2,...,t+5].
    idx = next(i for i, row in enumerate(ds.entries) if row[4] == 5)
    path, trajectory, arm, n, current, _ = ds.entries[idx]
    sample = ds[idx]
    positions = np.arange(current - 2, current + 6)
    with h5py.File(path, "r") as handle:
        group = handle[trajectory]
        expected_q = np.asarray(group[f"obs/agent/panda-{arm}/qpos"])[positions[:3]]
        expected_a = np.clip(np.asarray(group[f"actions/panda-{arm}"])[positions], low, high)
    assert sample["head_cam"].shape == (3, 3, 240, 320)
    assert sample["agent_pos"].shape == (3, 9)
    assert sample["action"].shape == (8, 8)
    np.testing.assert_allclose(sample["agent_pos"].numpy(), expected_q, atol=0, rtol=0)
    np.testing.assert_allclose(sample["action"].numpy(), expected_a, atol=1e-6, rtol=0)

    codec = _limits(ds.stats["a_min"], ds.stats["a_max"])
    decoded = codec.unnormalize(codec.normalize(sample["action"])).numpy()
    codec_error = float(np.max(np.abs(decoded - sample["action"].numpy())))
    assert codec_error < 1e-5
    report = {
        "schema": "mars-control.dp.preflight.v2",
        "status": "complete",
        "temporal_contract": "obs[t-2:t], action[t-2:t+5], execute prediction.action == action_pred[:,2:]",
        "policy_shape": {"obs_steps": 3, "horizon": 8, "action_steps": 8, "executable_steps": 6},
        "action_targets_clipped_before_stats": True,
        "normalizer_roundtrip_max_abs_error": codec_error,
        "episodes": ds.stats["episodes"],
        "local_streams": ds.stats["local_streams"],
    }
    atomic_json(run / "preflight_v2.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
