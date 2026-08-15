#!/usr/bin/env python3
"""Freeze the independent 3-N1-R1 contract and grouped scenario split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from before_we_act.b3_n1_data import load_n1_metadata
from before_we_act.b3_n1_r1 import (
    R1_CONDITIONS,
    R1_DATA_SEED,
    R1_EARLIEST_PLATFORM,
    R1_EVAL_EVERY,
    R1_LR_DROP,
    R1_MAX_UPDATES,
    R1_MIN_UPDATES,
    R1_SEEDS,
    canonical_sha256,
)
from before_we_act.step2_temporal_data import SIX_TASKS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n1-cache", type=Path, required=True)
    parser.add_argument("--n1-run", type=Path, required=True)
    parser.add_argument("--step2-run", type=Path, required=True)
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


def descriptor(array: np.ndarray, offset: int, length: int) -> np.ndarray:
    frames = np.asarray(array[offset : offset + min(8, length)], dtype=np.float32)
    value = frames.mean(0).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-8)


def balanced_nearest_groups(rows: list[dict], values: np.ndarray) -> list[list[int]]:
    """Make 20 deterministic visual-neighbor groups of exactly six episodes."""

    remaining = set(range(len(rows)))
    groups: list[list[int]] = []
    order = sorted(range(len(rows)), key=lambda index: rows[index]["hdf5_sha256"])
    for anchor in order:
        if anchor not in remaining:
            continue
        candidates = sorted(
            (index for index in remaining if index != anchor),
            key=lambda index: (
                -float(values[anchor] @ values[index]),
                rows[index]["hdf5_sha256"],
            ),
        )
        group = [anchor, *candidates[:5]]
        if len(group) != 6:
            raise RuntimeError("R1 grouping exhausted before a full visual-neighbor group")
        remaining.difference_update(group)
        groups.append(group)
    if remaining or len(groups) != 20:
        raise RuntimeError(f"R1 grouping differs: groups={len(groups)} remaining={remaining}")
    return groups


def group_quality(groups: list[list[int]], values: np.ndarray) -> dict:
    within: list[float] = []
    across: list[float] = []
    group_of = {index: group for group, rows in enumerate(groups) for index in rows}
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            target = within if group_of[left] == group_of[right] else across
            target.append(float(values[left] @ values[right]))
    return {
        "within_cosine_mean": float(np.mean(within)),
        "across_cosine_mean": float(np.mean(across)),
        "within_pairs": len(within),
        "across_pairs": len(across),
    }


def build_split(cache: Path) -> dict:
    metadata, episodes = load_n1_metadata(cache)
    result: list[dict] = []
    diagnostics: dict[str, dict] = {}
    for task in SIX_TASKS:
        task_rows = [episode for episode in episodes if episode.task == task]
        if len(task_rows) != 120:
            raise ValueError(f"R1 expected 120 episodes for {task}")
        array = np.load(cache / f"{task}_visual.npy", mmap_mode="r")
        values = np.stack(
            [descriptor(array, episode.offset, episode.length) for episode in task_rows]
        )
        groups = balanced_nearest_groups(
            [episode.__dict__ for episode in task_rows], values
        )
        ordered_groups = sorted(
            range(len(groups)),
            key=lambda group: hashlib.sha256(
                (
                    f"{R1_DATA_SEED}:{task}:group:{group}:"
                    + ":".join(
                        sorted(task_rows[index].hdf5_sha256 for index in groups[group])
                    )
                ).encode()
            ).hexdigest(),
        )
        assignment = {
            group: ("train" if rank < 16 else "validation" if rank < 18 else "test")
            for rank, group in enumerate(ordered_groups)
        }
        for group, members in enumerate(groups):
            group_hash = canonical_sha256(
                sorted(task_rows[index].hdf5_sha256 for index in members)
            )
            for index in members:
                episode = task_rows[index]
                result.append(
                    {
                        "task": task,
                        "episode_key": episode.episode_key,
                        "hdf5_sha256": episode.hdf5_sha256,
                        "scenario_group": f"{task}-{group:02d}-{group_hash[:12]}",
                        "split": assignment[group],
                    }
                )
        diagnostics[task] = {
            **group_quality(groups, values),
            "groups": 20,
            "episodes_per_group": 6,
            "train_groups": 16,
            "validation_groups": 2,
            "test_groups": 2,
        }
    ordered = sorted(result, key=lambda row: (row["task"], row["hdf5_sha256"]))
    return {
        "format_version": "before-we-act.b3-n1-r1-scenario-split/1",
        "created_at_utc": utc_now(),
        "algorithm": (
            "per task: mean first min(8,T) frozen-DINO global/agent0/agent1 vectors; "
            "L2 normalize; SHA256-ordered greedy anchors; five nearest remaining cosine "
            "neighbors per six-episode group; SHA256-seeded 16/2/2 group assignment"
        ),
        "data_seed": R1_DATA_SEED,
        "source_n1_metadata_sha256": sha256_file(cache / "metadata.json"),
        "episodes": ordered,
        "episodes_sha256": canonical_sha256(ordered),
        "diagnostics": diagnostics,
    }


def selected_n1_checkpoints(n1_run: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for seed in R1_SEEDS:
        root = n1_run / "representation" / f"seed_{seed}"
        status_path = root / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") not in {"PLATFORM_REACHED", "SATURATED_BY_OVERFIT"}:
            raise RuntimeError(f"N1 representation seed {seed} is not frozen/complete")
        update = int(status["selected_update"])
        checkpoint = root / f"checkpoint_{update:06d}.pt"
        result[str(seed)] = {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "selected_update": update,
            "status_sha256": sha256_file(status_path),
        }
    return result


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    contract_dir = output / "contract"
    split_path = contract_dir / "scenario_split.json"
    contract_path = contract_dir / "r1_contract.json"
    if split_path.exists() or contract_path.exists():
        raise FileExistsError("R1-0 refuses to overwrite a frozen contract or split")

    old_conclusion = args.n1_run / "n1_conclusion.json"
    old_contract = args.n1_run / "contract" / "n1_contract.json"
    b0h = args.step2_run / "hidden_residual" / "formal" / "checkpoint_120000.pt"
    step2_contract = args.step2_run / "contract" / "step2_contract.json"
    for path in (old_conclusion, old_contract, b0h, step2_contract):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_status = json.loads(old_conclusion.read_text(encoding="utf-8")).get("status")
    if old_status != "INCONCLUSIVE_TRAINING_NOT_CONVERGED":
        raise RuntimeError(f"old N1 conclusion differs: {old_status}")

    split = build_split(args.n1_cache)
    atomic_json(split_path, split)
    n1_checkpoints = selected_n1_checkpoints(args.n1_run)
    contract: dict[str, object] = {
        "format_version": "before-we-act.b3-n1-r1-contract/1",
        "stage_id": "B3-N1-R1-ACTION-GROUNDED-BELIEF",
        "status": "FROZEN_BEFORE_F0_F1",
        "created_at_utc": utc_now(),
        "question": (
            "Does teammate-conditioned belief improve ego action prediction beyond the "
            "true frozen B0-H hidden, survive scenario-group holdout and causal forks, "
            "and remain inferable from legal 16-step history?"
        ),
        "old_n1_read_only": {
            "root": str(args.n1_run.resolve()),
            "contract_sha256": sha256_file(old_contract),
            "conclusion_sha256": sha256_file(old_conclusion),
            "status": old_status,
            "representation_checkpoints": n1_checkpoints,
        },
        "b0h": {
            "checkpoint": str(b0h.resolve()),
            "checkpoint_sha256": sha256_file(b0h),
            "step2_contract": str(step2_contract.resolve()),
            "step2_contract_sha256": sha256_file(step2_contract),
            "frozen": True,
            "hidden": "B0HPolicy._encode_history history_summary from its trained hidden-residual checkpoint",
            "visual_feature_source": (
                "the same frozen-DINO float16 history cache bound by Step-2; current slot is "
                "passed through the same B0-H history encoder without the N1 encoder"
            ),
        },
        "scenario_split": {
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
            "groups_per_task": 20,
            "episodes_per_group": 6,
            "train_validation_test_episodes_per_task": [96, 12, 12],
            "test_policy": "sealed until checkpoint selection and final R1-1 classification",
        },
        "runtime_inputs": [
            "16-step global/ego-local frozen-DINO history",
            "ego qpos",
            "already-commanded ego action history",
            "canonical task text",
            "validity/reset masks",
        ],
        "forbidden_runtime_inputs": [
            "future ego/teammate action",
            "current or future teammate state",
            "success/reward/final outcome",
            "episode/frame identity",
            "simulator truth",
            "ARB/B/P/T sidecars",
        ],
        "r1_1": {
            "seeds": list(R1_SEEDS),
            "data_seed": R1_DATA_SEED,
            "old_n1_capacity": 16,
            "conditions": list(R1_CONDITIONS),
            "main_comparison": "h_b vs h",
            "architecture": "H is query; 8-head cross-attention reads all 16 B tokens; no token mean pooling",
            "effective_batch": 48,
            "samples_per_task": 8,
            "minimum_updates": R1_MIN_UPDATES,
            "earliest_platform": R1_EARLIEST_PLATFORM,
            "maximum_updates": R1_MAX_UPDATES,
            "validation_every": R1_EVAL_EVERY,
            "learning_rate_drop_update": R1_LR_DROP,
            "learning_rate": 3e-4,
            "post_drop_multiplier": 0.1,
            "platform": (
                "each independently trainable condition has <1% relative validation "
                "improvement in each of the last three 5k intervals after the LR drop"
            ),
            "positive": (
                "h_b beats h in every seed; cross-seed per-task median is positive in >=4/6; "
                "h_b has lower macro MSE than both h_b_shuffle and h_matched_capacity in every "
                "seed on validation and sealed test; validation and "
                "sealed test scenario-group directions do not reverse"
            ),
            "report": "absolute MSE, relative delta, six-task profile, episode-block bootstrap 95% CI",
        },
        "r1_2": {
            "conditional": "run only if R1-1 does not pass",
            "oracle_inputs_training_or_audit_only": [
                "current teammate qpos",
                "previous teammate qpos",
                "teammate qpos changes at t+4/8/16/32",
                "actual teammate action distribution over the next 16 steps",
            ],
            "positive": "oracle beats H in every seed and cross-seed task median is positive in >=4/6",
        },
        "r1_3": {
            "rollouts": 720,
            "design": "6 tasks x 10 recoverable replay states x 4 teammate modes x 3 repeats",
            "modes": ["normal", "delay_freeze", "timing_early_or_late", "wrong_role"],
            "ego_counterfactual_label": (
                "no fabricated corrective action; if no solver can resume from an arbitrary "
                "snapshot, branches supervise only paired outcome/value"
            ),
            "positive": "paired outcome/value changes in >=4/6 tasks and task-level paired bootstrap 95% CI excludes zero",
            "power": "pilot variance freezes any later collection size once; no result-chasing additions",
        },
        "r1_4_r1_5": {
            "teacher_privileged_only": [
                "complete joint state",
                "actual teammate action",
                "shared state change",
                "counterfactual branch value",
            ],
            "student_runtime": "only the frozen legal 16-step runtime whitelist above",
            "student_stages": [
                "posterior alignment",
                "teammate distribution and shared-change prediction",
                "frozen-belief B0-H residual",
                "low-LR final belief-layer action correction",
            ],
            "b0h_always_frozen": True,
            "maximum_updates_each_trainable_group": R1_MAX_UPDATES,
            "matched_direct_residual_required": True,
            "belief_off_exact_fallback_required": True,
        },
        "classification": {
            "pass": "POSITIVE_ACTION_RELEVANT_BELIEF_SIGNAL",
            "not_converged": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "attribution_conflict": "INCONCLUSIVE_ATTRIBUTION",
            "oracle_negative": "DATA_OR_TASK_HAS_NO_IDENTIFIABLE_TEAMMATE_ACTION_VALUE",
            "pilot_negative": "NO_EXPLICIT_TEAMMATE_AWARE_CORRECTION_NEEDED",
            "teacher_positive_student_negative": "ACTION_VALUE_EXISTS_BUT_LEGAL_HISTORY_NOT_OBSERVABLE",
            "direct_matches_belief": "CAPACITY_VALUE_WITHOUT_NECESSARY_EXPLICIT_BELIEF",
        },
        "n2_authorization": "forbidden unless the final R1 receipt is POSITIVE_ACTION_RELEVANT_BELIEF_SIGNAL",
    }
    atomic_json(contract_path, contract)
    atomic_json(
        contract_dir / "r1_0_receipt.json",
        {
            "format_version": "before-we-act.b3-n1-r1-r1-0-receipt/1",
            "status": "PASSED",
            "completed_at_utc": utc_now(),
            "contract_sha256": sha256_file(contract_path),
            "scenario_split_sha256": sha256_file(split_path),
            "old_n1_preserved": True,
            "episodes": 720,
            "split_counts_per_task": {"train": 96, "validation": 12, "test": 12},
        },
    )
    print(
        json.dumps(
            {
                "status": "PASSED",
                "contract": str(contract_path),
                "contract_sha256": sha256_file(contract_path),
                "scenario_split_sha256": sha256_file(split_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
