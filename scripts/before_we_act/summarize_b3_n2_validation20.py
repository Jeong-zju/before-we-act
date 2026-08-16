#!/usr/bin/env python3
"""Summarize the owner-authorized N2 Validation20 diagnostic fairly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


SIX_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


EXPECTED_BASELINES = {
    "w10": {
        "checkpoint_sha256": (
            "e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574"
        ),
        "successes": 88,
    },
    "b0h": {
        "checkpoint_sha256": (
            "a3aa1d25ff67820ee9c354f87e0e6bff2b2d83a60662fbf88b05e2b9c5c73743"
        ),
        "successes": 95,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conclusion", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--w10-summary", type=Path, required=True)
    parser.add_argument("--b0h-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_validation20_candidate(training: Mapping) -> dict:
    """Freeze the cross-seed choice without looking at closed-loop outcomes."""
    seed, row = min(
        training.items(),
        key=lambda item: (
            float(item[1]["selected_validation"]["macro"]["b_core"]),
            int(item[0]),
        ),
    )
    return {
        "selection_rule": (
            "lowest selected-checkpoint validation B-core action MSE across the "
            "three frozen seeds; integer seed breaks exact ties"
        ),
        "seed": int(seed),
        "selected_update": int(row["selected_update"]),
        "selection_metric": "selected_validation.macro.b_core",
        "selection_value": float(
            row["selected_validation"]["macro"]["b_core"]
        ),
        "deployment_checkpoint": row["deployment_checkpoint"],
        "deployment_checkpoint_sha256": row["deployment_checkpoint_sha256"],
        "closed_loop_results_used_for_selection": False,
    }


def load_baseline(path: Path, seed_root: Path, name: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = EXPECTED_BASELINES[name]
    if int(value.get("episodes", -1)) != 120:
        raise RuntimeError(f"{name} baseline does not contain 120 episodes")
    if int(value.get("successes", -1)) != expected["successes"]:
        raise RuntimeError(f"{name} baseline success count drifted")
    if value.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
        raise RuntimeError(f"{name} baseline checkpoint hash drifted")
    tasks = value.get("tasks", {})
    if set(tasks) != set(SIX_TASKS):
        raise RuntimeError(f"{name} baseline task set drifted")
    if any(int(tasks[task].get("episodes", -1)) != 20 for task in SIX_TASKS):
        raise RuntimeError(f"{name} baseline per-task episode count drifted")
    receipts = {}
    for task in SIX_TASKS:
        result_path = path.parent / f"{task}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        seed_file = seed_root / f"{task}.json"
        if result.get("seed_protocol", {}).get("sha256") != sha256_file(seed_file):
            raise RuntimeError(f"{name} baseline seed receipt drifted for {task}")
        expected_seeds = json.loads(seed_file.read_text(encoding="utf-8"))["seeds"][:20]
        actual_seeds = [int(row["seed"]) for row in result.get("rows", [])]
        if actual_seeds != [int(seed) for seed in expected_seeds]:
            raise RuntimeError(f"{name} baseline seed order drifted for {task}")
        if result.get("checkpoint") != value.get("checkpoint"):
            raise RuntimeError(f"{name} per-task checkpoint path drifted for {task}")
        if int(result.get("successes", -1)) != int(tasks[task]["successes"]):
            raise RuntimeError(f"{name} per-task success count drifted for {task}")
        receipts[task] = sha256_file(result_path)
    value["validation_receipts"] = receipts
    return value


def load_candidate(
    validation_root: Path,
    seed_root: Path,
    expected_checkpoint_sha256: str,
) -> dict:
    rows = {}
    for task in SIX_TASKS:
        path = validation_root / f"{task}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("mode") != "n2":
            raise RuntimeError(f"N2 Validation20 mode drifted for {task}")
        if int(value.get("episodes", -1)) != 20 or len(value.get("rows", [])) != 20:
            raise RuntimeError(f"N2 Validation20 is incomplete for {task}")
        if value.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise RuntimeError(f"N2 checkpoint hash drifted for {task}")
        seed_file = seed_root / f"{task}.json"
        if value.get("seed_protocol", {}).get("sha256") != sha256_file(seed_file):
            raise RuntimeError(f"N2 seed receipt drifted for {task}")
        expected_seeds = json.loads(seed_file.read_text(encoding="utf-8"))["seeds"][:20]
        actual_seeds = [int(row["seed"]) for row in value["rows"]]
        if actual_seeds != [int(seed) for seed in expected_seeds]:
            raise RuntimeError(f"N2 seed order drifted for {task}")
        rows[task] = value
    successes = sum(int(rows[task]["successes"]) for task in SIX_TASKS)
    steps = sum(int(rows[task]["steps"]) for task in SIX_TASKS)
    inactivity = sum(
        int(rows[task]["paired_inactivity_steps"]) for task in SIX_TASKS
    )
    return {
        "episodes": 120,
        "successes": successes,
        "success_rate": successes / 120,
        "steps": steps,
        "paired_inactivity_steps": inactivity,
        "paired_inactivity_rate": inactivity / max(steps, 1),
        "tasks": {
            task: {
                "episodes": 20,
                "successes": int(rows[task]["successes"]),
                "success_rate": int(rows[task]["successes"]) / 20,
            }
            for task in SIX_TASKS
        },
        "receipts": {
            task: sha256_file(validation_root / f"{task}.json")
            for task in SIX_TASKS
        },
    }


def qualification(candidate: Mapping) -> dict:
    tasks = candidate["tasks"]
    protected = (
        "lift_barrier",
        "long_pipeline_delivery",
        "take_photo",
        "pass_shoe",
    )
    checks = {
        "total_ge_80": int(candidate["successes"]) >= 80,
        "protected_sum_ge_72": sum(
            int(tasks[task]["successes"]) for task in protected
        )
        >= 72,
        "protected_each_ge_16": all(
            int(tasks[task]["successes"]) >= 16 for task in protected
        ),
        "camera_ge_6": int(tasks["camera_alignment"]["successes"]) >= 6,
        "camera_plus_food_ge_8": (
            int(tasks["camera_alignment"]["successes"])
            + int(tasks["place_food"]["successes"])
            >= 8
        ),
    }
    return {
        "checks": checks,
        "passes_numeric_n4_gate_if_it_were_formal": all(checks.values()),
        "matches_or_beats_w10": int(candidate["successes"]) >= 88,
        "matches_or_beats_b0h": int(candidate["successes"]) >= 95,
        "formal_pass": False,
    }


def compare(candidate: Mapping, baseline: Mapping) -> dict:
    return {
        "success_delta": int(candidate["successes"]) - int(baseline["successes"]),
        "success_rate_delta_pp": 100.0
        * (
            float(candidate["successes"]) / 120.0
            - float(baseline["successes"]) / 120.0
        ),
        "task_success_delta": {
            task: int(candidate["tasks"][task]["successes"])
            - int(baseline["tasks"][task]["successes"])
            for task in SIX_TASKS
        },
    }


def summarize(
    conclusion: Mapping,
    validation_root: Path,
    seed_root: Path,
    w10: Mapping,
    b0h: Mapping,
) -> dict:
    selected = conclusion.get("validation20_candidate")
    allowed_statuses = {
        "POSITIVE_SIGNAL",
        "OWNER_AUTHORIZED_CLOSED_LOOP_AFTER_PRIMARY_PLATEAU",
    }
    if conclusion.get("status") not in allowed_statuses:
        raise RuntimeError(
            "N2 Validation20 requires a positive frozen Validation5 or owner exception"
        )
    if not isinstance(selected, dict):
        raise RuntimeError("N2 conclusion lacks the pre-closed-loop seed selection")
    if selected.get("closed_loop_results_used_for_selection") is not False:
        raise RuntimeError("N2 seed selection used closed-loop results")
    candidate = load_candidate(
        validation_root,
        seed_root,
        str(selected["deployment_checkpoint_sha256"]),
    )
    return {
        "format_version": "before-we-act.b3-n2-validation20-diagnostic/1",
        "stage": "B3-N2-ARCHITECTURE",
        "status": "COMPLETED_OWNER_AUTHORIZED_VALIDATION20_DIAGNOSTIC",
        "completed_at_utc": utc_now(),
        "selection": selected,
        "n2": candidate,
        "w10": {
            "checkpoint_sha256": w10["checkpoint_sha256"],
            "episodes": 120,
            "successes": int(w10["successes"]),
            "tasks": w10["tasks"],
            "validation_receipts": w10.get("validation_receipts"),
        },
        "b0h": {
            "checkpoint_sha256": b0h["checkpoint_sha256"],
            "episodes": 120,
            "successes": int(b0h["successes"]),
            "tasks": b0h["tasks"],
            "validation_receipts": b0h.get("validation_receipts"),
        },
        "comparison": {
            "versus_w10": compare(candidate, w10),
            "versus_b0h": compare(candidate, b0h),
        },
        "qualification_diagnostic": qualification(candidate),
        "formal_pass": False,
        "claim_limits": [
            "This is an owner-authorized N2 diagnostic, not the N4 formal stage.",
            "If the owner exception was used, the immutable training-sufficiency conclusion remains inconclusive.",
            "The N2 seed was selected only by frozen offline action MSE before closed-loop results were read.",
            "Validation20 may compare closed-loop success with W10 and B0-H, but cannot replace N3 attribution, N4 retraining, or Confirmation50.",
            "Paired inactivity is reported for N2 only because the historical W10/B0-H summaries did not freeze the same proxy.",
        ],
    }


def main() -> None:
    args = parse_args()
    conclusion = json.loads(args.conclusion.read_text(encoding="utf-8"))
    w10 = load_baseline(args.w10_summary, args.seed_root, "w10")
    b0h = load_baseline(args.b0h_summary, args.seed_root, "b0h")
    payload = summarize(
        conclusion, args.validation_root, args.seed_root, w10, b0h
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
