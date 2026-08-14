#!/usr/bin/env python3
"""Expand and audit the deterministic SSC-V7 seed contract without writing files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def derive(namespace: str, purpose: str, task: str, index: int, retry: int) -> int:
    message = f"{namespace}|{purpose}|{task}|{index}|{retry}".encode("utf-8")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_646 + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--w10-seed-root", required=True)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("stage_id") != "SSC-V7-M1":
        raise SystemExit("seed contract is not the SSC-V7-M1 revision")
    schedule = contract["w10_measurement_rollout_schedule"]
    rollout_count = int(contract["measurement_purposes_per_task"]["w10_measurement_rollout"])
    if int(schedule["maximum_prefix_per_task"]) != rollout_count:
        raise SystemExit("W10 rollout reserve does not match the scheduled maximum")
    if int(schedule["initial_prefix_per_task"]) > rollout_count:
        raise SystemExit("W10 initial rollout prefix exceeds the reserve")
    if "measurement_statistics" not in contract["common_training_purposes"]:
        raise SystemExit("measurement_statistics seed purpose is missing")
    seed_root = Path(args.w10_seed_root)
    historical: set[int] = set()
    for task in contract["tasks"]:
        payload = json.loads((seed_root / f"{task}.json").read_text(encoding="utf-8"))
        historical.update(int(seed) for seed in payload["seeds"])

    used = set(historical)
    expanded: dict[str, object] = {"per_task": {}, "common_training": {}}
    for task in contract["tasks"]:
        task_payload: dict[str, list[int]] = {}
        purposes = list(contract["measurement_purposes_per_task"].items())
        purposes += list(contract["shared_candidate_evaluation_purposes_per_task"].items())
        for purpose, count in purposes:
            values = []
            for index in range(int(count)):
                retry = 0
                while True:
                    value = derive(contract["namespace"], purpose, task, index, retry)
                    if value not in used:
                        break
                    retry += 1
                used.add(value)
                values.append(value)
            task_payload[purpose] = values
        expanded["per_task"][task] = task_payload

    for purpose in contract["common_training_purposes"]:
        retry = 0
        while True:
            value = derive(contract["namespace"], purpose, "common", 0, retry)
            if value not in used:
                break
            retry += 1
        used.add(value)
        expanded["common_training"][purpose] = value

    encoded = json.dumps(expanded, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    generated = len(used) - len(historical)
    expected = (
        len(contract["tasks"])
        * (
            sum(contract["measurement_purposes_per_task"].values())
            + sum(contract["shared_candidate_evaluation_purposes_per_task"].values())
        )
        + len(contract["common_training_purposes"])
    )
    if generated != expected:
        raise SystemExit(f"seed count mismatch: generated={generated}, expected={expected}")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "generated_unique_seeds": generated,
                "historical_w10_seeds_excluded": len(historical),
                "expanded_seed_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
