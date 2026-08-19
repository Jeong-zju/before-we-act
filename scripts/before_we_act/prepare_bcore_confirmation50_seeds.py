#!/usr/bin/env python3
"""Create the frozen, disjoint seed manifests for B-core Confirmation50."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def deterministic_seeds(
    namespace: str, task: str, count: int, excluded: set[int]
) -> list[int]:
    """Derive positive 31-bit seeds without consuming mutable RNG state."""
    values: list[int] = []
    used = set(excluded)
    counter = 0
    while len(values) < count:
        payload = f"{namespace}/{task}/{counter}".encode("utf-8")
        candidate = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        candidate &= 0x7FFFFFFF
        counter += 1
        if candidate == 0 or candidate in used:
            continue
        used.add(candidate)
        values.append(candidate)
    return values


def build_manifest(contract: dict, task: str, excluded: set[int]) -> dict:
    seed_spec = contract["seed_protocol"]
    seeds = deterministic_seeds(
        str(seed_spec["namespace"]),
        task,
        int(seed_spec["episodes_per_task"]),
        excluded,
    )
    return {
        "count": len(seeds),
        "format_version": "before-we-act.b3-n3-confirmation50-seeds/1",
        "namespace": seed_spec["namespace"],
        "selection_method": (
            "SHA-256 namespace/task/counter 的前31位正整数；排除 Validation20 "
            "种子以及本任务内重复值；生成规则和摘要均在闭环结果前冻结"
        ),
        "stage": contract["stage"],
        "task": task,
        "seeds": seeds,
    }


def encoded_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    seed_spec = contract["seed_protocol"]
    validation_root = Path(seed_spec["excluded_validation20_root"])
    expected_validation_hashes = seed_spec["excluded_validation20_sha256"]
    expected_confirmation_hashes = seed_spec["confirmation50_sha256"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    receipt = {"stage": contract["stage"], "tasks": {}}
    for task in TASKS:
        validation_path = validation_root / f"{task}.json"
        if sha256_file(validation_path) != expected_validation_hashes[task]:
            raise RuntimeError(f"Validation20 seed manifest drifted: {task}")
        prior = json.loads(validation_path.read_text(encoding="utf-8"))
        excluded = {int(value) for value in prior["seeds"]}
        manifest = build_manifest(contract, task, excluded)
        payload = encoded_manifest(manifest)
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_confirmation_hashes[task]:
            raise RuntimeError(f"frozen Confirmation50 seed digest mismatched: {task}")
        if excluded.intersection(int(value) for value in manifest["seeds"]):
            raise RuntimeError(f"Confirmation50 overlaps Validation20: {task}")
        output = args.output_root / f"{task}.json"
        output.write_bytes(payload)
        receipt["tasks"][task] = {
            "count": len(manifest["seeds"]),
            "sha256": actual_hash,
            "validation20_overlap": 0,
        }

    receipt_path = args.output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
