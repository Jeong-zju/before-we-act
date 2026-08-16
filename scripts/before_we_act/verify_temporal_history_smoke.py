#!/usr/bin/env python3
"""Verify Step-2 four-update fresh/resume equivalence and frozen input boundaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    if reference.get("update") != 4 or resumed.get("update") != 4:
        raise ValueError("F1 checkpoints must both end at update 4")
    if reference["sample_cursor"] != resumed["sample_cursor"]:
        raise AssertionError("resume changed the deterministic sample cursor")
    if reference["provenance"] != resumed["provenance"]:
        # Stage/output paths are not provenance fields; any difference here is
        # a real training-identity drift.
        raise AssertionError("fresh/resumed provenance differs")
    if set(reference["model"]) != set(resumed["model"]):
        raise AssertionError("fresh/resumed model keys differ")
    maximum = 0.0
    differing = 0
    for key in reference["model"]:
        left = reference["model"][key]
        right = resumed["model"][key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"fresh/resumed tensor contract differs: {key}")
        if not torch.equal(left, right):
            differing += 1
            if left.is_floating_point():
                maximum = max(maximum, float((left - right).abs().max()))
            else:
                maximum = float("inf")
    if differing and maximum > 1e-7:
        raise AssertionError(
            f"resume is not numerically reproducible: tensors={differing}, max={maximum}"
        )
    config = resumed["config"]
    excluded = str(config.get("excluded_inputs", ""))
    checks = {
        "cursor_exact": True,
        "resume_model_max_abs_le_1e_7": maximum <= 1e-7,
        "resume_differing_tensor_count": differing,
        "original_episode_count": resumed["provenance"].get(
            "original_640x480_episodes"
        )
        == 720,
        "social_inputs_disabled": resumed["provenance"].get("social_inputs")
        is False,
        "w10_weights_not_loaded": resumed["provenance"].get("w10_weights_loaded")
        is False,
        "forbidden_inputs_declared": all(
            value in excluded
            for value in ("episode", "frame", "agent ID", "future", "B/P/T")
        ),
        "loss_finite": all(
            torch.isfinite(torch.tensor(float(resumed["last_metrics"][key]))).item()
            for key in ("loss", "action", "kl", "grad_norm")
        ),
    }
    if not all(value is True or isinstance(value, int) for value in checks.values()):
        raise AssertionError(f"F1 boundary checks failed: {checks}")
    if not all(value for key, value in checks.items() if not key.endswith("count")):
        raise AssertionError(f"F1 boundary checks failed: {checks}")
    receipt = {
        "format_version": "before-we-act.step2.f1/1",
        "status": "PASSED",
        "reference_checkpoint": str(args.reference.resolve()),
        "resumed_checkpoint": str(args.resumed.resolve()),
        "update": 4,
        "checks": checks,
        "resume_model_max_abs": maximum,
        "sample_cursor": resumed["sample_cursor"],
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    atomic_json(args.output, receipt)
    print("TEMPORAL_SMOKE_PASSED")


if __name__ == "__main__":
    main()
