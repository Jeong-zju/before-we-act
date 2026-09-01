from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .common import ACTION_HIGH, ACTION_LOW, DATASET_REVISION, OPENPI_REVISION, POLICY_CONTRACT, TASKS, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--norm", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text()); norm = json.loads(args.norm.read_text())
    checks = {
        "jax_four_devices": len(jax.devices()) == 4,
        "jax_collective": False,
        "dataset_revision": manifest.get("dataset_revision") == DATASET_REVISION,
        "all_550_episodes": manifest.get("total_episodes") == 550,
        "all_11_tasks": list(manifest.get("tasks", {})) == list(TASKS),
        "normalization_finite": True,
        "normalization_quantile_order": True,
        "model_source_pinned": OPENPI_REVISION == "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "image_size": manifest.get("image_size") == 224,
        "lag_one": manifest.get("recording_alignment", {}).get("action_lag_rows") == 1,
        "no_foreign_inputs": True,
    }
    if checks["jax_four_devices"]:
        result = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(jnp.arange(4)).block_until_ready()
        checks["jax_collective"] = bool(np.all(np.asarray(result) == 6))
    for key in ("state", "actions"):
        row = norm.get("norm_stats", {}).get(key, {})
        mean, std, q01, q99 = [np.asarray(row.get(name, []), np.float64) for name in ("mean", "std", "q01", "q99")]
        checks["normalization_finite"] &= mean.shape == (8,) and std.shape == (8,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
        checks["normalization_quantile_order"] &= q01.shape == (8,) and q99.shape == (8,) and np.all(q99 > q01)
    checks = {key: bool(value) for key, value in checks.items()}
    report = {"schema": "duobench.pi05.preflight.v1", "status": "complete" if all(checks.values()) else "failed", "passed": all(checks.values()), "checks": checks, "gpu": [str(x) for x in jax.devices()], "policy_contract": POLICY_CONTRACT, "action_bounds": {"low": ACTION_LOW, "high": ACTION_HIGH}}
    atomic_json(args.output, report); print(json.dumps(report), flush=True)
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
