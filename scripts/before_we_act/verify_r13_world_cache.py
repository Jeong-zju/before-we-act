#!/usr/bin/env python3
"""Verify the immutable shared R13 cache and emit a JSON receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from before_we_act.data.world_windows import CachedWorldWindows, LEGAL_INPUT_KEYS, TARGET_KEYS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--belief-sha256", required=True)
    parser.add_argument("--action-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.cache).resolve(strict=True)
    train = CachedWorldWindows(path, "train")
    validation = CachedWorldWindows(path, "validation")
    metadata = train.metadata
    checks = {
        "train_rows_4096": len(train) == 4096,
        "validation_rows_1024": len(validation) == 1024,
        "future_targets_are_model_inputs_false": metadata.get("future_targets_are_model_inputs") is False,
        "belief_checkpoint_pinned": metadata.get("belief_checkpoint_sha256") == args.belief_sha256,
        "action_checkpoint_pinned": metadata.get("action_checkpoint_sha256") == args.action_sha256,
        "horizons_exact": metadata.get("prediction_horizons") == [1, 5, 15],
        "all_five_tasks_train": set(train.data["task_index"].tolist()) == set(range(5)),
        "all_five_tasks_validation": set(validation.data["task_index"].tolist()) == set(range(5)),
        "legal_and_target_keys_disjoint": not set(LEGAL_INPUT_KEYS) & set(TARGET_KEYS),
    }
    result = {
        "schema_version": 1,
        "round": "R13",
        "cache": str(path),
        "cache_sha256": sha256(path),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "future_targets_are_model_inputs": metadata.get("future_targets_are_model_inputs"),
        "metadata": metadata,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cache_sha256": result["cache_sha256"], "passed": result["passed"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
