#!/usr/bin/env python3
"""Freeze the 720-episode Step-2 contract and emit human-readable F0 checks."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from before_we_act.step2_temporal_data import (  # noqa: E402
    ACTION_HORIZON,
    EFFECTIVE_BATCH,
    ExactSixTaskDistributedBatchSampler,
    HISTORY_STEPS,
    SIX_TASKS,
    TeamTemporalDataset,
    TeamTemporalRequest,
    canonical_sha256,
    episode_receipt,
    load_step2_episodes,
    sha256_file,
)


FORMAL_UPDATES = 120_000
DISCOVERY_UPDATES = 5_000
SEED = 20260814


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


def git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(ROOT), *arguments), text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("contract", "f0", "cursor"))
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-source", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--measurement-label-receipt", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    return parser.parse_args()


def load_stats(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("stats", payload)
    result = {
        key: np.asarray(source[key], dtype=np.float32)
        for key in ("q_mean", "q_std", "a_mean", "a_std")
    }
    expected = {"q_mean": (9,), "q_std": (9,), "a_mean": (8,), "a_std": (8,)}
    for key, shape in expected.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError(f"invalid normalization field {key}")
    if np.any(result["q_std"] <= 0) or np.any(result["a_std"] <= 0):
        raise ValueError("normalization standard deviation is not positive")
    return result


def contract(args: argparse.Namespace) -> None:
    if git_value("rev-parse", "HEAD") != args.base_commit:
        raise RuntimeError("Step-2 worktree is not at the frozen base commit")
    episodes = load_step2_episodes(args.manifests)
    dataset = episode_receipt(episodes)
    stats = load_stats(args.normalization_source)
    if not args.measurement_label_receipt.is_file():
        raise FileNotFoundError(args.measurement_label_receipt)
    if not args.dino_model.is_dir():
        raise FileNotFoundError(args.dino_model)
    foundation_receipt_path = args.dino_model / "foundation_receipt.json"
    if not foundation_receipt_path.is_file():
        raise FileNotFoundError(foundation_receipt_path)
    foundation_receipt = json.loads(
        foundation_receipt_path.read_text(encoding="utf-8")
    )
    if (
        foundation_receipt.get("status") != "PASSED"
        or foundation_receipt.get("non_vision_policy_tensors_loaded") is not False
        or foundation_receipt.get("candidate_initialization_from_w10_policy") is not False
    ):
        raise RuntimeError("DINO foundation-only recovery boundary failed")
    payload = {
        "format_version": "before-we-act.step2.contract/1",
        "status": "FROZEN_BEFORE_F0_F1",
        "stage": "P1_STEP2_B0H",
        "base_commit": args.base_commit,
        "repository_branch": git_value("branch", "--show-current"),
        "dataset": dataset,
        "sample": {
            "name": "TeamTemporalSample",
            "history_observation_indices": "t-15..t, zero-left-pad with history_mask",
            "history_action_indices": "t-16..t-1, zero-left-pad with action_history_mask",
            "action_target_indices": "t..t+99, repeat-last pad with action_mask",
            "episode_reset": "true iff t == 0; model has no cross-sample memory",
            "model_inputs": sorted(TeamTemporalDataset.MODEL_INPUT_FIELDS),
            "training_targets": sorted(TeamTemporalDataset.TARGET_FIELDS),
            "audit_only": sorted(TeamTemporalDataset.AUDIT_ONLY_FIELDS),
            "forbidden_model_inputs": [
                "episode_index",
                "time_index",
                "agent_slot/fixed robot ID",
                "future action",
                "future observation",
                "oracle B/P/T",
                "simulator truth",
                "success",
            ],
        },
        "social_supervision_boundary": {
            "original_720_sidecars_available": False,
            "b0h_reads_social_targets": False,
            "missing_target_representation": "explicit false supervision mask",
            "measurement_label_schema_reference": str(
                args.measurement_label_receipt.resolve()
            ),
            "measurement_label_schema_reference_sha256": sha256_file(
                args.measurement_label_receipt
            ),
            "measurement_compact_rgb_used_for_training": False,
            "b_core_formal_label_coverage": "NOT_AUTHORIZED_BY_THIS_STEP",
        },
        "normalization": {
            "source": str(args.normalization_source.resolve()),
            "source_sha256": sha256_file(args.normalization_source),
            "reuse_scope": "data-derived qpos/action statistics only; no W10 weights",
            "values_sha256": canonical_sha256(
                {key: value.tolist() for key, value in stats.items()}
            ),
        },
        "visual_history_cache": {
            "root": str(args.visual_cache.resolve()),
            "source": "original 640x480 RGB",
            "encoder": "frozen DINOv3 ViT-B/16",
            "representation": "mean of raw 30x40 patch tokens, float16, width 768",
            "current_frame_rule": "cache slot is replaced by the exact current forward feature",
            "dino_model": str(args.dino_model.resolve()),
            "foundation_receipt": str(foundation_receipt_path.resolve()),
            "foundation_receipt_sha256": sha256_file(foundation_receipt_path),
            "non_vision_policy_tensors_loaded": False,
        },
        "models": {
            "history_only": {
                "budget": DISCOVERY_UPDATES,
                "closed_loop": "Validation5 diagnostic",
                "social_input": False,
                "direct_residual": False,
            },
            "hidden_residual": {
                "budget": FORMAL_UPDATES,
                "closed_loop": "Validation20 acceptance",
                "social_input": False,
                "direct_residual": "zero-init; decoded hidden + history hidden only",
            },
            "common": {
                "effective_batch": EFFECTIVE_BATCH,
                "samples_per_task_per_update": 8,
                "action_horizon": ACTION_HORIZON,
                "history_steps": HISTORY_STEPS,
                "seed": SEED,
                "initialization": "independent from the same base seed; no W10 checkpoint",
                "optimizer": "AdamW; body lr 2e-4, router lr 3e-4",
                "precision": "bfloat16 autocast",
            },
        },
        "validation20_acceptance": {
            "total_success_min": 80,
            "lift_long_photo_shoe_sum_min": 72,
            "lift_long_photo_shoe_each_min": 16,
            "camera_min": 6,
            "camera_plus_food_min": 8,
            "w10_raw_success_match": 88,
        },
        "stopping_rules": [
            "stop on non-finite loss/gradient",
            "stop on CUDA OOM or stale worker heartbeat",
            "do not tune history length, sample cursor, model variant, seed, or budget from results",
        ],
        "created_at_utc": utc_now(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "step2_contract.json", payload)
    atomic_json(args.output / "dataset_receipt.json", dataset)
    torch.save({"stats": stats}, args.output / "normalization.pt")
    print("STEP2_CONTRACT_FROZEN")


def f0(args: argparse.Namespace) -> None:
    contract_path = args.output / "step2_contract.json"
    cache_receipt = args.visual_cache / "cache_receipt.json"
    if not contract_path.is_file() or not cache_receipt.is_file():
        raise FileNotFoundError("contract and visual cache receipt must precede F0")
    contract_value = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract_value.get("status") != "FROZEN_BEFORE_F0_F1":
        raise RuntimeError("Step-2 contract is not frozen")
    cache_value = json.loads(cache_receipt.read_text(encoding="utf-8"))
    if cache_value.get("status") != "PASSED" or cache_value.get("episodes") != 720:
        raise RuntimeError("visual history cache is incomplete")
    episodes = load_step2_episodes(args.manifests)
    stats = load_stats(args.output / "normalization.pt")
    dataset = TeamTemporalDataset(episodes, stats, args.visual_cache, cache_limit=8)
    rows = []
    checks = Counter()
    for task in SIX_TASKS:
        episode_list_index = next(
            index for index, episode in enumerate(episodes) if episode.task == task
        )
        episode = episodes[episode_list_index]
        points = (0, episode.length // 2, episode.length - 1)
        for position_name, time_index in zip(("begin", "middle", "end"), points):
            identity = (
                f"{episode.manifest_sha256}:{episode.hdf5_sha256}:"
                f"{episode.episode_index}:{task}:0:{time_index}"
            )
            request = TeamTemporalRequest(
                episode_list_index,
                0,
                time_index,
                hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                task,
            )
            sample = dataset[request]
            obs_start = max(0, time_index - 15)
            action_start = max(0, time_index - 16)
            target_end = min(time_index + 100, episode.length)
            expected_history = time_index - obs_start + 1
            expected_past_actions = time_index - action_start
            expected_target = target_end - time_index
            local_checks = {
                "original_global_640x480": tuple(sample["global_rgb"].shape)
                == (3, 480, 640),
                "original_local_640x480": tuple(sample["local_rgb"].shape)
                == (3, 480, 640),
                "history_mask_exact": int(sample["history_mask"].sum())
                == expected_history,
                "action_history_mask_exact": int(
                    sample["action_history_mask"].sum()
                )
                == expected_past_actions,
                "target_mask_exact": int(sample["action_mask"].sum())
                == expected_target,
                "current_history_slot_valid": bool(sample["history_mask"][-1]),
                "episode_reset_exact": bool(sample["episode_reset"])
                == (time_index == 0),
                "social_supervision_absent": not bool(
                    sample["social_supervision_mask"]
                ),
                "finite_numeric_inputs": all(
                    torch.isfinite(sample[key].float()).all().item()
                    for key in (
                        "history_visual_raw",
                        "history_qpos",
                        "history_action",
                        "action",
                    )
                ),
                "field_partitions_disjoint": not (
                    TeamTemporalDataset.MODEL_INPUT_FIELDS
                    & TeamTemporalDataset.TARGET_FIELDS
                    or TeamTemporalDataset.MODEL_INPUT_FIELDS
                    & TeamTemporalDataset.AUDIT_ONLY_FIELDS
                ),
            }
            if not all(local_checks.values()):
                raise AssertionError(f"F0 failed for {task}/{position_name}: {local_checks}")
            checks.update({key: int(value) for key, value in local_checks.items()})
            rows.append(
                {
                    "task": task,
                    "position": position_name,
                    "episode_index": episode.episode_index,
                    "episode_seed": episode.seed,
                    "agent_slot": 0,
                    "time_index": time_index,
                    "history_observation_indices": [obs_start, time_index],
                    "history_observation_valid_steps": expected_history,
                    "history_action_indices": (
                        [action_start, time_index - 1] if time_index else []
                    ),
                    "history_action_valid_steps": expected_past_actions,
                    "action_target_indices": [time_index, target_end - 1],
                    "action_target_valid_steps": expected_target,
                    "sidecar_target": None,
                    "social_supervision_mask": False,
                    "sample_key": sample["sample_key"],
                    "hdf5_sha256": sample["hdf5_sha256"],
                    "checks": local_checks,
                }
            )
    permutation_checks = []
    for task in SIX_TASKS:
        episode_list_index = next(
            index
            for index, episode in enumerate(episodes)
            if episode.task == task and len(episode.arms) >= 2
        )
        episode = episodes[episode_list_index]
        time_index = episode.length // 2
        samples = []
        for arm in (0, 1):
            identity = f"{episode.hdf5_sha256}:{task}:{arm}:{time_index}"
            samples.append(
                dataset[
                    TeamTemporalRequest(
                        episode_list_index,
                        arm,
                        time_index,
                        hashlib.sha256(identity.encode()).hexdigest(),
                        task,
                    )
                ]
            )
        same_global = torch.equal(samples[0]["global_rgb"], samples[1]["global_rgb"])
        own_fields_move = (
            samples[0]["agent_slot"].item() == 0
            and samples[1]["agent_slot"].item() == 1
            and samples[0]["action"].shape == samples[1]["action"].shape
            and samples[0]["history_qpos"].shape == samples[1]["history_qpos"].shape
        )
        if not same_global or not own_fields_move:
            raise AssertionError(f"agent-slot F0 failed for {task}")
        permutation_checks.append(
            {
                "task": task,
                "episode_index": episode.episode_index,
                "time_index": time_index,
                "global_view_unchanged": same_global,
                "ego_local_qpos_action_identity_moves_together": own_fields_move,
                "social_target": "absent for both slots",
            }
        )
    receipt = {
        "format_version": "before-we-act.step2.f0/1",
        "status": "PASSED",
        "human_readable_samples": rows,
        "agent_permutation_checks": permutation_checks,
        "sample_count": len(rows),
        "tasks": list(SIX_TASKS),
        "positions_per_task": ["begin", "middle", "end"],
        "aggregate_check_passes": dict(checks),
        "future_leakage": False,
        "episode_crossing": False,
        "model_receives_audit_identity": False,
        "model_receives_social_target": False,
        "completed_at_utc": utc_now(),
    }
    atomic_json(args.output / "f0_receipt.json", receipt)
    print("STEP2_F0_PASSED")


def cursor(args: argparse.Namespace) -> None:
    episodes = load_step2_episodes(args.manifests)
    first = ExactSixTaskDistributedBatchSampler(
        episodes, updates=FORMAL_UPDATES, seed=SEED
    )
    second = ExactSixTaskDistributedBatchSampler(
        episodes, updates=FORMAL_UPDATES, seed=SEED
    )
    rows = []
    for update in (1, 2, 3, 4, 5_000, 120_000):
        a = first.requests_for_update(update)
        b = second.requests_for_update(update)
        keys_a = [item.sample_key for item in a]
        keys_b = [item.sample_key for item in b]
        if keys_a != keys_b:
            raise AssertionError(f"sample cursor is not reproducible at update {update}")
        counts = Counter(item.task for item in a)
        if counts != Counter({task: 8 for task in SIX_TASKS}):
            raise AssertionError(f"sample balance failed at update {update}")
        rows.append(
            {
                "update": update,
                "sample_keys_sha256": canonical_sha256(keys_a),
                "per_task": dict(counts),
            }
        )
    receipt = {
        "format_version": "before-we-act.step2.cursor_replay/1",
        "status": "PASSED",
        "seed": SEED,
        "effective_batch": EFFECTIVE_BATCH,
        "checks": rows,
        "completed_at_utc": utc_now(),
    }
    atomic_json(args.output / "cursor_replay_receipt.json", receipt)
    print("STEP2_CURSOR_REPLAY_PASSED")


def main() -> None:
    args = parse_args()
    if args.command == "contract":
        contract(args)
    elif args.command == "f0":
        f0(args)
    else:
        cursor(args)


if __name__ == "__main__":
    main()
