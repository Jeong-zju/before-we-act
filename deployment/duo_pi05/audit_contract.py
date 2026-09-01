from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import DATASET_REVISION, OPENPI_REVISION, POLICY_CONTRACT, TASKS, atomic_json, sha256_file
from .dataset_adapter import sample_indices


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--norm", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    norm_payload = json.loads(args.norm.read_text())
    checks = {
        "dataset_revision": manifest.get("dataset_revision") == DATASET_REVISION,
        "all_550_episodes": manifest.get("total_episodes") == 550,
        "all_11_tasks": list(manifest.get("tasks", {})) == list(TASKS),
        "all_causal_pairs": manifest.get("total_policy_samples") == 285438,
        "lag_one": manifest.get("recording_alignment", {}).get("action_lag_rows") == 1,
        "exact_norm_schema": norm_payload.get("population", {}).get("causal_local_samples") == 570876,
        "norm_finite": True,
        "norm_has_quantiles": True,
        "rgb_contract": True,
        "state_action_contract": True,
        "episode_safe_chunks": True,
        "decentralized_fields": True,
        "upstream_model_contract": OPENPI_REVISION == "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    }
    for key in ("state", "actions"):
        row = norm_payload.get("norm_stats", {}).get(key, {})
        values = [np.asarray(row.get(name, []), np.float64) for name in ("mean", "std", "q01", "q99")]
        checks["norm_finite"] &= all(x.shape == (8,) and np.isfinite(x).all() for x in values)
        checks["norm_has_quantiles"] &= all(x.shape == (8,) for x in values[2:])
        checks["norm_finite"] &= bool(values[1].min() > 0)
    from openpi.training.duobench_dataset import DuoBenchDataset
    dataset = DuoBenchDataset(args.data)
    checks["state_action_contract"] &= len(dataset) > 0 and dataset.unique_samples == 570876 and dataset.lane_length * 11 == len(dataset)
    checks["decentralized_fields"] &= set(dataset[0]) == {"observation/head", "observation/wrist", "observation/state", "actions", "prompt"}
    for task in TASKS:
        root = args.data / task
        arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in ("state", "action", "head", "left", "right", "episodes")}
        n = len(arrays["episodes"])
        checks["rgb_contract"] &= all(arrays[name].shape == (n, 224, 224, 3) and arrays[name].dtype == np.uint8 for name in ("head", "left", "right"))
        starts = np.flatnonzero(np.r_[True, arrays["episodes"][1:] != arrays["episodes"][:-1]])
        ends = np.r_[starts[1:], n]
        for start, end in zip(starts, ends, strict=True):
            if end - start < 2: checks["episode_safe_chunks"] = False
            for arm in (0, 1):
                for row in sample_indices(int(start), int(end)):
                    positions = np.minimum(np.arange(row + 1, row + 17), int(end) - 1)
                    checks["episode_safe_chunks"] &= bool(positions.min() >= row + 1 and positions.max() < end)
                    checks["state_action_contract"] &= arrays["state"].shape[1:] == (16,) and arrays["action"].shape[1:] == (16,)
    report = {
        "schema": "duobench.pi05.contract-audit.v1", "status": "complete" if all(checks.values()) else "failed",
        "passed": all(checks.values()), "checks": checks, "policy_contract": POLICY_CONTRACT,
        "norm_sha256": sha256_file(args.norm), "dataset_manifest_sha256": sha256_file(args.data / "manifest.json"),
    }
    atomic_json(args.output, report); print(json.dumps(report), flush=True)
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
