#!/usr/bin/env python3
"""Augment the frozen R11 legal-input cache with future joint actions only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch

from before_we_act.data.raw_team_windows import TASKS, manifest_receipt


EXPECTED_PARENT_SHA256 = "061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_examples(manifests: dict, split: str, count: int, seed: int):
    """Reproduce the exact R11 selection order without changing that cache."""
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    pools = {}
    for task in TASKS:
        rows = [row for row in manifests[task]["episodes"] if row["split"] == split]
        if not rows:
            raise ValueError(f"no {split} episodes for {task}")
        pools[task] = rows
    examples = []
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        episode = rng.choice(pools[task])
        steps = int(episode["steps"])
        current = rng.randint(2, steps - 2)
        examples.append((task, episode, current))
    rng.shuffle(examples)
    return examples


def read_joint_actions(data_root: Path, item, stats: dict, horizon: int = 100):
    task, episode, current = item
    path = data_root / task / episode["hdf5_path"]
    actions = np.zeros((horizon, 4, 8), dtype=np.float32)
    step_mask = np.zeros(horizon, dtype=np.bool_)
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        agents = sorted(data["observation/agents"].keys())
        end = min(current + horizon, int(episode["steps"]))
        valid = end - current
        step_mask[:valid] = True
        for agent_index, agent in enumerate(agents):
            command = np.asarray(
                data[f"action/agents/{agent}/commanded"][current:end],
                dtype=np.float32,
            )
            if not len(command):
                raise ValueError(f"empty action suffix: {path}:{current}")
            actions[:valid, agent_index] = command
            actions[valid:, agent_index] = command[-1]
    actions = (actions - stats["a_mean"][None, None]) / stats["a_std"][None, None]
    return torch.from_numpy(actions), torch.from_numpy(step_mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r11-cache", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--heartbeat", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if payload.get("round") != "R12":
            raise ValueError("existing action cache identity differs")
        print(json.dumps({"reused": str(output), "sha256": sha256(output)}))
        return
    state, heartbeat = Path(args.state), Path(args.heartbeat)
    atomic_json(state, {"state": "PREPARING", "stage": "action_targets", "updated_at": now()})
    atomic_json(heartbeat, {"producer": "prepare_r12_action_cache", "updated_at": now()})
    parent_path = Path(args.parent_checkpoint).resolve(strict=True)
    if sha256(parent_path) != EXPECTED_PARENT_SHA256:
        raise ValueError("W10 normalization source checkpoint hash differs")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in parent["stats"].items()}
    r11_path = Path(args.r11_cache).resolve(strict=True)
    r11 = torch.load(r11_path, map_location="cpu", weights_only=False)
    metadata = r11["metadata"]
    if metadata["seed"] != 20260805 or metadata["history_steps"] != 3:
        raise ValueError("R11 cache selection protocol differs")
    data_root = Path(args.data_root).resolve(strict=True)
    manifests = {
        task: json.loads((data_root / task / "training_manifest.json").read_text(encoding="utf-8"))
        for task in TASKS
    }
    result_splits = {}
    last_beat = time.monotonic()
    for split, count_key in (("train", "train_windows"), ("validation", "validation_windows")):
        source = r11[split]
        examples = choose_examples(manifests, split, int(metadata[count_key]), int(metadata["seed"]))
        action_rows, mask_rows = [], []
        for index, item in enumerate(examples, 1):
            action, mask = read_joint_actions(data_root, item, stats)
            action_rows.append(action)
            mask_rows.append(mask)
            if time.monotonic() - last_beat >= 20:
                atomic_json(
                    heartbeat,
                    {"producer": "prepare_r12_action_cache", "split": split, "row": index, "total": len(examples), "updated_at": now()},
                )
                last_beat = time.monotonic()
        result_splits[split] = {
            key: value for key, value in source.items()
            if key in {"visual", "view_mask", "qpos", "actions", "agent_mask", "task_index"}
        }
        result_splits[split]["joint_actions"] = torch.stack(action_rows)
        result_splits[split]["action_step_mask"] = torch.stack(mask_rows)
    payload = {
        "schema_version": 1,
        "round": "R12",
        "metadata": {
            "created_at": now(),
            "r11_cache": str(r11_path),
            "r11_cache_sha256": sha256(r11_path),
            "parent_normalization_checkpoint": str(parent_path),
            "parent_normalization_checkpoint_sha256": EXPECTED_PARENT_SHA256,
            "data_root": str(data_root),
            "manifest_sha256": manifest_receipt(data_root),
            "tasks": TASKS,
            "seed": 20260805,
            "history_steps": 3,
            "horizon": 100,
            "max_agents": 4,
            "action_dim": 8,
            "train_windows": int(metadata["train_windows"]),
            "validation_windows": int(metadata["validation_windows"]),
            "legal_inputs": metadata["legal_inputs"],
            "forbidden_inputs": metadata["forbidden_inputs"],
        },
        "stats": {key: torch.from_numpy(value) for key, value in stats.items()},
        **result_splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    receipt = {"state": "PASSED", "stage": "complete", "output": str(output), "sha256": sha256(output), "updated_at": now()}
    atomic_json(state, receipt)
    atomic_json(heartbeat, {"producer": "prepare_r12_action_cache", "updated_at": now(), "complete": True})
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
