#!/usr/bin/env python3
"""Freeze the R1-3 same-state teammate-intervention pilot contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from before_we_act.temporal_history_data import SIX_TASKS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--action-grounded-contract", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("R1-3 contract is already frozen")
    r1 = json.loads(args.action_grounded_contract.read_text(encoding="utf-8"))
    if r1.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF":
        raise RuntimeError("wrong parent R1 contract")
    selections: list[dict] = []
    for task in SIX_TASKS:
        manifest_path = args.dataset_root / task / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        eligible = sorted(
            (
                row
                for row in manifest["episodes"]
                if row.get("split") == "train"
                and bool(row.get("success"))
                and int(row.get("steps", 0)) >= 64
            ),
            key=lambda row: str(row["hdf5_sha256"]),
        )
        if len(eligible) < 10:
            raise RuntimeError(f"R1-3 lacks ten eligible recoverable episodes for {task}")
        for state_index, row in enumerate(eligible[:10]):
            steps = int(row["steps"])
            selections.append(
                {
                    "task": task,
                    "state_index": state_index,
                    "episode_index": int(row["episode_index"]),
                    "seed": int(row["seed"]),
                    "hdf5_path": str((args.dataset_root / task / row["hdf5_path"]).resolve()),
                    "hdf5_sha256": str(row["hdf5_sha256"]),
                    "recorded_steps": steps,
                    "anchor_frame": steps // 2,
                    "timing_mode": "early_plus_4" if state_index % 2 == 0 else "late_minus_4",
                }
            )
    contract = {
        "format_version": "before-we-act.b3-n1-r1-pilot-contract/1",
        "stage": "R1-3-COUNTERFACTUAL-PILOT",
        "status": "FROZEN_BEFORE_COLLECTION",
        "created_at_utc": utc_now(),
        "parent_r1_contract": str(args.action_grounded_contract.resolve()),
        "parent_r1_contract_sha256": sha256_file(args.action_grounded_contract),
        "collector": str(args.collector.resolve()),
        "collector_sha256": sha256_file(args.collector),
        "selection_rule": (
            "per task, manifest-successful train episodes with >=64 steps, sort by HDF5 "
            "SHA256, take first 10; recoverable anchor=floor(recorded_steps/2)"
        ),
        "states": selections,
        "states_sha256": hashlib.sha256(
            json.dumps(selections, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "design": {
            "tasks": 6,
            "states_per_task": 10,
            "modes": ["normal", "delay_freeze", "timing_early_or_late", "wrong_role"],
            "repeats": 3,
            "rollouts": 720,
            "rollout_horizon": 32,
            "timing_offset_steps": 4,
            "ego_agent": "panda-0",
            "intervened_teammate": "panda-1",
            "other_agents": "follow the recorded expert action",
            "only_teammate_changes": True,
            "freeze_action": "hold teammate 7D arm qpos at anchor and retain recorded gripper command",
            "wrong_role_action": "feed panda-0 recorded command to panda-1; panda-0 remains recorded",
        },
        "labels": {
            "corrective_ego_action_available": False,
            "use": "paired outcome/value only",
            "reason": "the scripted solver cannot be resumed from an arbitrary restored snapshot",
            "fabricated_action_labels_forbidden": True,
            "primary": "normal cumulative dense reward minus intervention cumulative dense reward",
            "secondary": [
                "terminal success",
                "final object displacement",
                "contact/custody pattern",
                "drop/collision risk",
            ],
        },
        "gate": {
            "unit": "recoverable-state block; average three repeats and three intervention modes",
            "bootstrap_draws": 10_000,
            "bootstrap_seed": 20260815,
            "task_positive": "paired state-block reward-delta 95% CI excludes zero",
            "pass": "at least 4/6 tasks positive and all same-mode restores repeat exactly",
            "power_analysis": "use pilot state-block variance once to freeze later sample size; no adaptive additions",
        },
    }
    atomic_json(args.output, contract)
    print(
        json.dumps(
            {
                "status": contract["status"],
                "states": len(selections),
                "rollouts": 720,
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
