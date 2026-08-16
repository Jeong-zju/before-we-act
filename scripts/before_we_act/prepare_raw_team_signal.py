#!/usr/bin/env python3
"""Freeze the Step 3-N1 contract and build its read-only memory-mapped cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch

from before_we_act.raw_team_signal_data import CAPACITY_CANDIDATES, FUTURE_OFFSETS
from before_we_act.temporal_history_data import SIX_TASKS, load_temporal_episodes, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--temporal-contract", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def load_stats(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("stats", payload)
    return {key: np.asarray(source[key], dtype=np.float32) for key in ("q_mean", "q_std", "a_mean", "a_std")}


def main() -> None:
    args = parse_args()
    contract = json.loads(args.temporal_contract.read_text())
    if contract.get("status") != "FROZEN_BEFORE_F0_F1" or contract["dataset"]["episodes"] != 720:
        raise RuntimeError("3-N1 requires the frozen 720-episode Step-2 contract")
    cache_receipt = json.loads((args.visual_cache / "cache_receipt.json").read_text())
    if cache_receipt.get("status") != "PASSED" or cache_receipt.get("episodes") != 720:
        raise RuntimeError("3-N1 requires the complete frozen-DINO cache")
    episodes = load_temporal_episodes(args.manifests)
    stats = load_stats(args.normalization)
    root = args.cache_output
    metadata_path = root / "metadata.json"
    if args.force_rebuild and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    records = []
    validation = []
    global_index = 0
    visual_sum = np.zeros((3, 768), dtype=np.float64)
    visual_square = np.zeros((3, 768), dtype=np.float64)
    visual_count = np.zeros((3, 1), dtype=np.float64)
    for task_index, task in enumerate(SIX_TASKS):
        task_episodes = sorted(
            [episode for episode in episodes if episode.task == task],
            key=lambda episode: episode.hdf5_sha256,
        )
        if len(task_episodes) != 120:
            raise ValueError(f"expected 120 episodes for {task}")
        validation_hashes = {episode.hdf5_sha256 for episode in task_episodes[:20]}
        total = sum(episode.length for episode in task_episodes)
        visual_out = np.lib.format.open_memmap(root / f"{task}_visual.npy", mode="w+", dtype=np.float16, shape=(total, 3, 768))
        qpos_out = np.lib.format.open_memmap(root / f"{task}_qpos.npy", mode="w+", dtype=np.float32, shape=(total, 2, 9))
        action_out = np.lib.format.open_memmap(root / f"{task}_action.npy", mode="w+", dtype=np.float32, shape=(total, 2, 8))
        offset = 0
        for local_index, episode in enumerate(task_episodes):
            visual_path = args.visual_cache / task / f"{episode.hdf5_sha256}.npz"
            with np.load(visual_path, allow_pickle=False) as source:
                visual = np.stack(
                    (
                        source["view_global"],
                        source["view_agent_0"],
                        source["view_agent_1"],
                    ),
                    axis=1,
                )
            if visual.shape != (episode.length, 3, 768):
                raise ValueError(f"visual cache shape drift: {visual_path}")
            with h5py.File(episode.path, "r") as handle:
                data = handle["data"]
                qpos = np.stack(
                    [np.asarray(data["observation"]["agents"][f"panda_{arm}"]["qpos"][: episode.length], dtype=np.float32) for arm in (0, 1)],
                    axis=1,
                )
                action = np.stack(
                    [np.asarray(data["action"]["agents"][f"panda_{arm}"]["commanded"][: episode.length], dtype=np.float32) for arm in (0, 1)],
                    axis=1,
                )
            end = offset + episode.length
            visual_out[offset:end] = visual
            qpos_out[offset:end] = qpos
            action_out[offset:end] = action
            split = "validation" if episode.hdf5_sha256 in validation_hashes else "train"
            episode_key = hashlib.sha256(f"{task}:{episode.hdf5_sha256}".encode()).hexdigest()
            records.append(
                {
                    "task": task,
                    "task_index": task_index,
                    "local_index": local_index,
                    "offset": offset,
                    "length": episode.length,
                    "split": split,
                    "episode_key": episode_key,
                    "hdf5_sha256": episode.hdf5_sha256,
                }
            )
            if split == "validation":
                times = np.unique(np.linspace(0, episode.length - 1, num=min(16, episode.length), dtype=np.int64))
                for arm in (0, 1):
                    for time_index in times.tolist():
                        validation.append(
                            {
                                "episode_index": global_index,
                                "arm": arm,
                                "time_index": time_index,
                                "sample_key": f"{episode_key}:{arm}:{time_index}",
                                "task": task,
                            }
                        )
            else:
                values = visual.astype(np.float64)
                visual_sum += values.sum(0)
                visual_square += np.square(values).sum(0)
                visual_count += episode.length
            offset = end
            global_index += 1
        visual_out.flush(); qpos_out.flush(); action_out.flush()
    visual_mean = visual_sum / visual_count
    visual_var = visual_square / visual_count - np.square(visual_mean)
    visual_std = np.sqrt(np.maximum(visual_var, 1e-8))
    torch.save(
        {
            "visual_mean": visual_mean.astype(np.float32),
            "visual_std": visual_std.astype(np.float32),
            **stats,
        },
        root / "target_stats.pt",
    )
    metadata = {
        "format_version": "before-we-act.b3-n1-cache/1",
        "created_at_utc": utc_now(),
        "source_step2_contract": str(args.temporal_contract.resolve()),
        "source_step2_contract_sha256": sha256_file(args.temporal_contract),
        "source_visual_cache_receipt_sha256": sha256_file(args.visual_cache / "cache_receipt.json"),
        "split_rule": "within each task, sort by HDF5 SHA256; first 20 validation, remaining 100 train",
        "episodes": records,
        "validation_requests": validation,
        "validation_rows": len(validation),
        "future_offsets_steps": list(FUTURE_OFFSETS),
    }
    atomic_json(metadata_path, metadata)
    n1_contract = {
        "format_version": "before-we-act.b3-n1-contract/1",
        "status": "FROZEN_BEFORE_F0_F1",
        "created_at_utc": utc_now(),
        "question": "Do raw synchronized multiview/time/teammate-state targets yield action-relevant signal beyond ordinary legal history hidden?",
        "source": {
            "step2_contract": str(args.temporal_contract.resolve()),
            "step2_contract_sha256": sha256_file(args.temporal_contract),
            "cache_metadata": str(metadata_path.resolve()),
            "cache_metadata_sha256": sha256_file(metadata_path),
            "episodes": 720,
            "train_episodes_per_task": 100,
            "validation_episodes_per_task": 20,
        },
        "runtime_inputs": ["16-step global/ego-local frozen-DINO history", "ego qpos", "executed ego action", "canonical task identity", "masks"],
        "training_targets_only": ["current teammate qpos", "teammate qpos change at t+4/8/16/32", "global and teammate-view DINO at t+4/8/16/32"],
        "forbidden_runtime_inputs": ["future observations/actions", "teammate future action", "episode/frame/fixed robot ID", "success", "remaining goals", "simulator truth", "ARB/B/P/T sidecar"],
        "seeds": [20260815, 20260816, 20260817],
        "data_seed": 20260815,
        "effective_batch": 48,
        "capacity_candidates": list(CAPACITY_CANDIDATES),
        "capacity_selection": "smallest capacity within 1% of the best cross-seed median for both raw-target and action-probe validation score",
        "training": {
            "minimum_updates_each_stage": 25000,
            "earliest_platform_decision": 35000,
            "maximum_updates_each_stage": 120000,
            "validation_every": 5000,
            "learning_rate_drop_update": 20000,
            "platform": "last three post-drop validation intervals each improve the primary score by less than 1%, with no key anchor still improving by >=1%",
            "overfit": "three consecutive post-minimum validation degradations select the checkpoint before degradation",
        },
        "controls": ["persistence", "zero", "phase-matched shuffled raw targets", "matched hidden-only action probe", "time-only action probe", "row-shuffle action probe", "phase-matched belief shuffle at validation"],
        "classification": {
            "positive_raw": "real-target macro beats persistence, zero, and shuffled-target model in every seed; median direction positive in >=4/6 tasks and every future anchor",
            "positive_action": "belief probe beats hidden-only in every seed and median direction positive in >=4/6 tasks; time/row/phase-shuffle do not reproduce it",
            "modelable_no_action_value": "raw criterion passes but action criterion fails",
            "no_signal": "real targets do not stably beat controls after training sufficiency",
            "non_collapse": "mean token feature standard deviation must exceed 0.05 in every seed",
        },
        "n2_capacity_mapping_if_positive": {
            "4": {"n_belief_tokens": 4, "n_evidence_queries": 2, "event_capacity": 2, "temporal_layers": 1},
            "8": {"n_belief_tokens": 8, "n_evidence_queries": 4, "event_capacity": 4, "temporal_layers": 2},
            "16": {"n_belief_tokens": 16, "n_evidence_queries": 8, "event_capacity": 8, "temporal_layers": 2},
        },
    }
    atomic_json(args.run_root / "contract" / "n1_contract.json", n1_contract)
    atomic_json(args.run_root / "contract" / "cache_receipt.json", {"status": "PASSED", "metadata_sha256": sha256_file(metadata_path), "validation_rows": len(validation), "completed_at_utc": utc_now()})
    print(json.dumps({"status": "PASSED", "episodes": len(records), "validation_rows": len(validation)}))


if __name__ == "__main__":
    main()
