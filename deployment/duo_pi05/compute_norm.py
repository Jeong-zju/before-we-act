from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deployment.duo_act.action_target import canonicalize_controller_action

from .common import ACTION_HIGH, ACTION_LOW, DATASET_REVISION, TASKS, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest.get("dataset_revision") != DATASET_REVISION or manifest.get("total_episodes") != 550:
        raise RuntimeError("prepared DuoBench snapshot is not the frozen 550-demo revision")
    q_count = int(manifest["total_policy_samples"]) * 2
    # The model sees one 16-action chunk per causal local-arm sample.  Allocate
    # once so mean/std/quantiles are exact (not histogram approximations).
    states = np.empty((q_count, 8), np.float64)
    chunks = np.empty((q_count * 16, 8), np.float64)
    q_index = c_index = 0
    for task in TASKS:
        root = args.data / task
        state = np.load(root / "state.npy", mmap_mode="r").reshape(-1, 2, 8)
        action = np.load(root / "action.npy", mmap_mode="r").reshape(-1, 2, 8)
        episodes = np.load(root / "episodes.npy", mmap_mode="r")
        starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
        ends = np.r_[starts[1:], len(episodes)]
        for start, end in zip(starts, ends, strict=True):
            for arm in (0, 1):
                for row in range(int(start), int(end) - 1):
                    q = np.asarray(state[row, arm], np.float64)
                    target_rows = np.minimum(np.arange(row + 1, row + 17), int(end) - 1)
                    target = np.asarray(action[target_rows, arm], np.float64)
                    # Explicitly repeat the pinned controller operation; the
                    # prepared action array is checked against this below.
                    target = canonicalize_controller_action(target.astype(np.float32)).astype(np.float64)
                    target[:, :7] -= q[None, :7]
                    states[q_index] = q
                    chunks[c_index : c_index + 16] = target
                    q_index += 1; c_index += 16
    if q_index != q_count or c_index != q_count * 16:
        raise RuntimeError(f"normalization population mismatch {q_index}/{q_count}, {c_index}/{q_count * 16}")

    def statistics(values: np.ndarray) -> dict:
        return {
            "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
            "std": np.maximum(np.std(values, axis=0, dtype=np.float64), 1e-6).tolist(),
            "q01": np.quantile(values, 0.01, axis=0, method="linear").tolist(),
            "q99": np.quantile(values, 0.99, axis=0, method="linear").tolist(),
        }

    stats = {"state": statistics(states), "actions": statistics(chunks)}
    payload = {
        "schema": "duobench.pi05.exact-normalization.v1",
        "dataset_revision": DATASET_REVISION,
        "population": {
            "episodes": 550, "causal_local_samples": q_count,
            "action_chunk_values": q_count * 16,
            "all_tasks_both_arms_no_split": True,
            "action_lag_rows": 1, "action_horizon": 16,
        },
        "action_contract": {"low": list(ACTION_LOW), "high": list(ACTION_HIGH), "canonicalized_before_delta": True},
        "norm_stats": stats,
    }
    atomic_json(args.output, payload)
    # OpenPI expects the same payload under assets/<asset_id>/norm_stats.json.
    print(json.dumps({"status": "complete", "state_rows": q_index, "action_rows": c_index, "sha256": sha256_file(args.output)}), flush=True)


if __name__ == "__main__": main()
