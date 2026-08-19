#!/usr/bin/env python3
"""Summarize the frozen B-core versus matched direct-reactive screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar_two_sided(bcore_only: int, direct_only: int) -> float:
    discordant = bcore_only + direct_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(bcore_only, direct_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def screen_decision(net_successes: int) -> str:
    if net_successes >= 5:
        return "ADVANCE_TO_MECHANISM_CHECK"
    if net_successes >= 1:
        return "SMALL_POSITIVE_REQUIRE_MECHANISM_CHECK"
    return "STOP_TEAM_SPECIFIC_MAIN_CLAIM"


def load_rows(path: Path, expected_mode: str | None = None) -> dict[int, dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if expected_mode is not None and value.get("mode") != expected_mode:
        raise RuntimeError(f"unexpected mode in {path}: {value.get('mode')}")
    rows = value.get("rows", [])
    if len(rows) != 20 or len({int(row["seed"]) for row in rows}) != 20:
        raise RuntimeError(f"{path} is not a complete 20-episode result")
    return {int(row["seed"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bcore-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = contract["immutable_inputs"]
    if sha256_file(Path(expected["bcore_checkpoint"]["path"])) != expected["bcore_checkpoint"]["sha256"]:
        raise RuntimeError("frozen B-core checkpoint hash drifted")
    if sha256_file(Path(expected["training_checkpoint"]["path"])) != expected["training_checkpoint"]["sha256"]:
        raise RuntimeError("frozen training checkpoint hash drifted")
    if sha256_file(Path(expected["b0h_checkpoint"]["path"])) != expected["b0h_checkpoint"]["sha256"]:
        raise RuntimeError("frozen B0-H checkpoint hash drifted")

    totals = {
        "episodes": 0,
        "bcore_successes": 0,
        "direct_successes": 0,
        "bcore_only_successes": 0,
        "direct_only_successes": 0,
        "both_success": 0,
        "both_failure": 0,
    }
    task_rows = []
    all_finite = True
    for task in TASKS:
        bcore = load_rows(args.bcore_root / f"{task}.json", "n2")
        direct = load_rows(args.direct_root / f"{task}.json", "direct_reactive")
        if set(bcore) != set(direct):
            raise RuntimeError(f"paired seeds differ for {task}")
        seed_hash = expected["validation_seeds"][task]
        direct_value = json.loads((args.direct_root / f"{task}.json").read_text())
        if "seed_protocol" not in direct_value:
            raise RuntimeError(f"missing seed receipt for {task}")
        if direct_value["seed_protocol"]["sha256"] != seed_hash:
            raise RuntimeError(f"seed hash drifted for {task}")
        counts = {
            "task": task,
            "episodes": 20,
            "bcore_successes": 0,
            "direct_successes": 0,
            "bcore_only_successes": 0,
            "direct_only_successes": 0,
            "both_success": 0,
            "both_failure": 0,
        }
        for seed in sorted(bcore):
            b_success = bool(bcore[seed]["success"])
            d_success = bool(direct[seed]["success"])
            counts["bcore_successes"] += int(b_success)
            counts["direct_successes"] += int(d_success)
            counts["bcore_only_successes"] += int(b_success and not d_success)
            counts["direct_only_successes"] += int(d_success and not b_success)
            counts["both_success"] += int(b_success and d_success)
            counts["both_failure"] += int(not b_success and not d_success)
            all_finite &= bool(direct[seed].get("finite_actions", False))
        counts["net_bcore_successes"] = (
            counts["bcore_successes"] - counts["direct_successes"]
        )
        task_rows.append(counts)
        for key in totals:
            if key != "episodes":
                totals[key] += counts[key]
        totals["episodes"] += 20

    totals["net_bcore_successes"] = totals["bcore_successes"] - totals["direct_successes"]
    totals["bcore_success_rate"] = totals["bcore_successes"] / totals["episodes"]
    totals["direct_success_rate"] = totals["direct_successes"] / totals["episodes"]
    totals["percentage_point_difference"] = 100.0 * (
        totals["bcore_success_rate"] - totals["direct_success_rate"]
    )
    totals["mcnemar_exact_two_sided_p"] = exact_mcnemar_two_sided(
        totals["bcore_only_successes"], totals["direct_only_successes"]
    )
    integrity = {
        "complete_120_paired_episodes": totals["episodes"] == 120,
        "all_direct_actions_finite": all_finite,
        "bcore_historical_total_matches_111": totals["bcore_successes"] == 111,
    }
    if not all(integrity.values()):
        decision = "BLOCKED_BY_INTEGRITY_FAILURE"
    else:
        decision = screen_decision(totals["net_bcore_successes"])
    result = {
        "format_version": "before-we-act.b3-n3-minimal-attribution-summary/1",
        "stage": contract["stage"],
        "contract_sha256": sha256_file(args.contract),
        "status": decision,
        "integrity": integrity,
        "screen_rule_frozen_before_results": contract["decision_rules"]["validation20_screen"],
        "tasks": task_rows,
        "overall": totals,
        "interpretation_boundary": [
            "This screen compares the full B-core with its jointly trained matched legal-history direct control.",
            "It does not by itself assign a human-readable meaning to individual belief factors.",
            "The 120-episode screen decides whether more attribution work is worth running; it is not Confirmation50.",
        ],
        "result_sha256": {
            "bcore": {task: sha256_file(args.bcore_root / f"{task}.json") for task in TASKS},
            "direct": {task: sha256_file(args.direct_root / f"{task}.json") for task in TASKS},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
