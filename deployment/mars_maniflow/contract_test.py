"""Fail-fast checks for ManiFlow's temporal and action-range contracts."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .common import ACTION_HIGH, ACTION_LOW, POLICY_CONTRACT, atomic_json, load_frozen_config
from .dataset import MarsManiFlowDataset
from .modeling import ACTION_STEPS, HORIZON, OBS_STEPS, model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frozen = load_frozen_config()
    dataset = MarsManiFlowDataset(args.data_root, args.stats)
    low, high = np.asarray(ACTION_LOW, np.float32), np.asarray(ACTION_HIGH, np.float32)
    action_stats = dataset.stats["action"]
    assert action_stats["clipped_before_stats"] is True
    np.testing.assert_allclose(action_stats["clip_low"], low, atol=0, rtol=0)
    np.testing.assert_allclose(action_stats["clip_high"], high, atol=0, rtol=0)
    assert np.all(np.asarray(action_stats["min"]) >= low - 1e-6)
    assert np.all(np.asarray(action_stats["max"]) <= high + 1e-6)
    assert (OBS_STEPS, HORIZON, ACTION_STEPS) == (2, 16, 15)

    # Prove an interior dataset window is obs[t-1:t] with action[t-1:t+14].
    index = next(i for i, row in enumerate(dataset.entries) if row[-1] == 5)
    _tid, path, trajectory, arm, _n, current = dataset.entries[index]
    sample = dataset[index]
    with h5py.File(path, "r") as handle:
        group = handle[trajectory]
        expected_q = np.asarray(group[f"obs/agent/panda-{arm}/qpos"][current - 1:current + 1], np.float32)
        expected_a = np.clip(np.asarray(group[f"actions/panda-{arm}"][current - 1:current + 15], np.float32), low, high)
    qmin, qmax = dataset.qmin, dataset.qmax
    amin, amax = dataset.amin, dataset.amax
    decoded_q = (sample["obs"]["agent_pos"].numpy() + 1) * 0.5 * (qmax - qmin + 1e-6) + qmin
    decoded_a = (sample["action"].numpy() + 1) * 0.5 * (amax - amin + 1e-6) + amin
    np.testing.assert_allclose(decoded_q, expected_q, atol=2e-5, rtol=0)
    np.testing.assert_allclose(decoded_a, expected_a, atol=2e-5, rtol=0)

    report = {
        "schema": "mars-control.maniflow.contract-test.v2",
        "status": "complete",
        "policy_contract": POLICY_CONTRACT,
        "temporal_contract": model_config()["temporal_contract"],
        "action_clip": model_config()["action_clip"],
        "frozen_schema": frozen["schema"],
        "dataset_size": len(dataset),
    }
    atomic_json(Path(args.output), report)


if __name__ == "__main__":
    main()
