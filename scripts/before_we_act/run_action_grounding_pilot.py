#!/usr/bin/env python3
"""Collect and classify the 720 R1-3 same-state teammate interventions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import numpy as np

import audit_ssc_v7_m2 as m2

from before_we_act.temporal_history_data import SIX_TASKS, sha256_file
from robofactory_rpc import scalar_bool, split_robofactory_action


MODES = ("normal", "delay_freeze", "timing_early_or_late", "wrong_role")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, default=Path("/workspace/RoboFactory"))
    parser.add_argument("--output-root", type=Path, required=True)
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


def clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, list):
        return [clone(item) for item in value]
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value).reshape(-1)


def scalar_reward(value: Any) -> float:
    array = vector(value)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError("R1-3 reward must be one finite scalar")
    return float(array[0])


def snapshot_delta(first: Mapping, second: Mapping) -> dict:
    displacement = {
        name: float(
            np.linalg.norm(
                np.asarray(second["object_positions"][name])
                - np.asarray(first["object_positions"][name])
            )
        )
        for name in first["object_positions"]
    }
    return {
        "object_displacement": displacement,
        "max_object_displacement": max(displacement.values(), default=0.0),
        "contact_changed": first["contact"] != second["contact"],
        "grasp_changed": first["grasp"] != second["grasp"],
        "drop_risk_final": second["drop_risk"],
        "robot_collision_final": bool(second["robot_collision"]),
    }


def branch_action(
    actions: np.ndarray,
    frame: int,
    anchor: int,
    mode: str,
    state_index: int,
    teammate_hold: np.ndarray,
) -> np.ndarray:
    result = np.asarray(actions[frame], dtype=np.float32).copy()
    teammate = slice(8, 16)
    if mode == "normal":
        return result
    if mode == "delay_freeze":
        hold = teammate_hold.copy()
        hold[-1] = result[teammate][-1]
        result[teammate] = hold
        return result
    if mode == "timing_early_or_late":
        source = (
            min(len(actions) - 1, frame + 4)
            if state_index % 2 == 0
            else max(anchor, frame - 4)
        )
        result[teammate] = actions[source, teammate]
        return result
    if mode == "wrong_role":
        result[teammate] = actions[frame, :8]
        return result
    raise ValueError(mode)


def collect_state(
    environment: Any,
    task: str,
    state: Mapping,
    horizon: int,
) -> list[dict]:
    path = Path(state["hdf5_path"])
    if sha256_file(path) != state["hdf5_sha256"]:
        raise RuntimeError(f"R1-3 HDF5 hash differs: {path}")
    with h5py.File(path, "r") as stream:
        metadata = json.loads(str(stream.attrs["episode_metadata_json"]))
        agent_order = tuple(str(value) for value in metadata["agent_order"])
        actions = np.asarray(
            stream["data/action/commanded"][: int(state["recorded_steps"])],
            dtype=np.float32,
        )
    if len(agent_order) != int(m2.TASKS[task]["agents"]):
        raise RuntimeError("R1-3 agent order differs")
    observation, _ = environment.reset(seed=int(state["seed"]))
    anchor = int(state["anchor_frame"])
    for frame in range(anchor):
        observation, _, terminated, truncated, _ = environment.step(
            split_robofactory_action(actions[frame], agent_order=agent_order)
        )
        if scalar_bool(terminated, name="terminated") or scalar_bool(
            truncated, name="truncated"
        ):
            raise RuntimeError(f"R1-3 selected anchor is not recoverable: {task}/{state['state_index']}")
    base_env = getattr(environment, "base_env", environment.unwrapped)
    anchor_state = clone(base_env.get_state_dict())
    elapsed = getattr(environment, "_elapsed_steps", None)
    anchor_snapshot = m2.extract_snapshot(base_env, task, False)
    teammate_name = agent_order[1]
    teammate_qpos = vector(observation["agent"][teammate_name]["qpos"])
    teammate_hold = np.concatenate((teammate_qpos[:7], actions[anchor, 15:16])).astype(
        np.float32
    )
    rows: list[dict] = []
    for mode in MODES:
        for repeat in range(3):
            base_env.set_state_dict(clone(anchor_state))
            if elapsed is not None:
                environment._elapsed_steps = elapsed
            cumulative_reward = 0.0
            success = False
            terminal = False
            executed = 0
            last_info: Mapping[str, Any] = {"success": np.asarray([False])}
            for frame in range(anchor, min(len(actions), anchor + horizon)):
                joint = branch_action(
                    actions,
                    frame,
                    anchor,
                    mode,
                    int(state["state_index"]),
                    teammate_hold,
                )
                _, reward, terminated, truncated, info = environment.step(
                    split_robofactory_action(joint, agent_order=agent_order)
                )
                cumulative_reward += scalar_reward(reward)
                executed += 1
                last_info = info
                terminal = scalar_bool(terminated, name="terminated") or scalar_bool(
                    truncated, name="truncated"
                )
                success = scalar_bool(info["success"], name="info.success")
                if terminal:
                    break
            final_snapshot = m2.extract_snapshot(base_env, task, success)
            rows.append(
                {
                    "task": task,
                    "state_index": int(state["state_index"]),
                    "episode_index": int(state["episode_index"]),
                    "hdf5_sha256": state["hdf5_sha256"],
                    "anchor_frame": anchor,
                    "mode": mode,
                    "repeat": repeat,
                    "timing_mode": state["timing_mode"],
                    "steps": executed,
                    "cumulative_dense_reward": cumulative_reward,
                    "terminal": terminal,
                    "success": success,
                    "shared_state_change": snapshot_delta(anchor_snapshot, final_snapshot),
                    "only_teammate_command_modified": mode != "normal",
                    "ego_command_source": "recorded expert unchanged",
                }
            )
    return rows


def classify(rows: list[dict], contract: Mapping) -> dict:
    per_task: dict[str, dict] = {}
    exact_repeats = True
    positive_tasks = 0
    state_effects_all: dict[str, list[float]] = {}
    for task in SIX_TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        state_effects: list[float] = []
        for state_index in range(10):
            state_rows = [row for row in task_rows if row["state_index"] == state_index]
            by_mode = {
                mode: [row for row in state_rows if row["mode"] == mode] for mode in MODES
            }
            signatures = {}
            for mode, mode_rows in by_mode.items():
                signatures[mode] = {
                    (
                        round(float(row["cumulative_dense_reward"]), 8),
                        bool(row["success"]),
                        bool(row["terminal"]),
                        round(float(row["shared_state_change"]["max_object_displacement"]), 8),
                    )
                    for row in mode_rows
                }
                exact_repeats &= len(signatures[mode]) == 1
            normal = float(np.mean([row["cumulative_dense_reward"] for row in by_mode["normal"]]))
            intervention = float(
                np.mean(
                    [
                        row["cumulative_dense_reward"]
                        for mode in MODES[1:]
                        for row in by_mode[mode]
                    ]
                )
            )
            state_effects.append(normal - intervention)
        values = np.asarray(state_effects, dtype=np.float64)
        rng = np.random.default_rng(int(contract["gate"]["bootstrap_seed"]) + SIX_TASKS.index(task))
        draws = values[
            rng.integers(
                0,
                len(values),
                size=(int(contract["gate"]["bootstrap_draws"]), len(values)),
            )
        ].mean(1)
        low, high = np.quantile(draws, [0.025, 0.975])
        positive = bool(low > 0 or high < 0)
        positive_tasks += int(positive)
        state_effects_all[task] = state_effects
        per_task[task] = {
            "paired_reward_delta_normal_minus_intervention": float(values.mean()),
            "ci95": [float(low), float(high)],
            "state_effects": state_effects,
            "positive_nonzero": positive,
        }
    passed = positive_tasks >= 4 and exact_repeats
    pooled = np.concatenate(
        [np.asarray(value, dtype=np.float64) for value in state_effects_all.values()]
    )
    required = int(
        np.ceil(
            (1.959963984540054 * float(pooled.std(ddof=1)) / max(abs(float(pooled.mean())), 1e-12))
            ** 2
        )
    )
    return {
        "status": "PASSED_R1_3_COUNTERFACTUAL_PILOT" if passed else "FAILED_R1_3_COUNTERFACTUAL_PILOT",
        "passed": passed,
        "positive_tasks": positive_tasks,
        "required_positive_tasks": 4,
        "same_mode_restore_repeats_exact": exact_repeats,
        "per_task": per_task,
        "power_analysis": {
            "method": "normal approximation on 60 state-block paired reward effects; 95% half-width below observed absolute mean",
            "pooled_state_blocks": int(len(pooled)),
            "observed_mean": float(pooled.mean()),
            "observed_std": float(pooled.std(ddof=1)),
            "estimated_state_blocks_for_same_precision": required,
            "later_collection_size_if_needed": max(60, required),
        },
    }


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_COLLECTION":
        raise RuntimeError("R1-3 pilot contract is not frozen")
    if sha256_file(Path(contract["collector"])) != contract["collector_sha256"]:
        raise RuntimeError("R1-3 collector hash differs from frozen contract")
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    for task in SIX_TASKS:
        environment = m2.make_env(task, args.robofactory_root)
        try:
            for state in contract["states"]:
                if state["task"] != task:
                    continue
                print(f"[R1-3] task={task} state={state['state_index']}/9", flush=True)
                rows.extend(
                    collect_state(
                        environment,
                        task,
                        state,
                        int(contract["design"]["rollout_horizon"]),
                    )
                )
        finally:
            environment.close()
    if len(rows) != 720:
        raise RuntimeError(f"R1-3 rollout count differs: {len(rows)}")
    jsonl = args.output_root / "rollouts.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    conclusion = classify(rows, contract)
    conclusion.update(
        {
            "format_version": "before-we-act.b3-n1-r1-pilot-conclusion/1",
            "stage": "R1-3-COUNTERFACTUAL-PILOT",
            "completed_at_utc": utc_now(),
            "contract_sha256": sha256_file(args.contract),
            "rollouts": len(rows),
            "rollouts_sha256": sha256_file(jsonl),
            "corrective_action_labels_used": False,
            "n2_authorized": False,
        }
    )
    conclusion["human_summary"] = (
        "同一个物理状态下，只改队友后续行为就会稳定改变团队回报；这批任务里确实存在需要理解队友的因果信号。"
        if conclusion["passed"]
        else "同一个物理状态下改队友行为，并没有在至少四个任务里稳定改变团队回报；当前任务不足以证明显式 team belief 是必要的。"
    )
    atomic_json(args.output_root / "conclusion.json", conclusion)
    print(json.dumps({key: conclusion[key] for key in ("status", "positive_tasks", "rollouts")}, sort_keys=True))


if __name__ == "__main__":
    main()
