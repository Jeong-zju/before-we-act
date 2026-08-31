"""Replay successful raw Duo trajectories through the joint-action interface.

This is an interface diagnostic, not a data repair.  It restores only the
initial recorded MuJoCo state, converts the remaining recorded Cartesian
targets with the same RCS Pin converter used to publish DuoBench, and then
runs the resulting joint commands open loop through the formal 30 Hz joint
environment.  Successful replay is direct evidence for action ordering,
joint semantics, gripper semantics, and environment stepping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import rcs
from rcs._core.common import GripperType, RobotType
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo
from rcs.sim.replayer import load_distinct_uuids, load_trajectory

from .action_target import canonicalize_controller_action
from .dataset import TASKS


def make_no_camera_env(task: str):
    module = __import__(f"duobench.tasks.{task}", fromlist=["*"])
    config_class = getattr(
        module, "".join(part.title() for part in task.split("_")) + "EnvConfig"
    )
    config = config_class().config()
    config.headless = True
    config.control_mode = ControlMode.JOINTS
    config.relative_to = RelativeTo.NONE
    config.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30)
    config.wrapper_cfg.binary_gripper = True
    config.camera_cfgs = None
    config.camera_adds = None
    return gym.make(f"duobench/{task}", cfg=config)


def convert_step(
    step, ik: rcs.common.Pin
) -> dict[str, dict[str, np.ndarray]] | None:
    result = {}
    for key in ("left", "right"):
        target = np.asarray(step.info[key]["absolute_action"], np.float64)
        seed = np.asarray(step.observation[key]["joints"], np.float64)
        pose = rcs.common.Pose(
            translation=target[:3], quaternion=target[3:7]
        )
        joints = ik.inverse(
            pose,
            seed,
            tcp_offset=rcs.GRIPPER_TCP_OFFSETS[GripperType("Robotiq2F85")],
        )
        if joints is None:
            # This is the official converter contract: if either local IK
            # fails, the entire synchronized dual-arm row is omitted.
            return None
        gripper = np.asarray(step.action[key]["gripper"], np.float32).reshape(1)
        local = canonicalize_controller_action(
            np.concatenate((np.asarray(joints, np.float32), gripper))
        )
        result[key] = {
            "joints": local[:7].astype(np.float32),
            "gripper": local[7:].astype(np.float32),
        }
    return result


def flattened_action(action: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    return np.concatenate(
        [
            np.concatenate((action[key]["joints"], action[key]["gripper"]))
            for key in ("left", "right")
        ]
    )


def replay(task: str, source: Path) -> dict:
    uuids = load_distinct_uuids(source)
    if len(uuids) != 1:
        raise RuntimeError(f"{task}: expected one diagnostic trajectory, got {len(uuids)}")
    steps = load_trajectory(source, uuids[0])
    if len(steps) < 2 or not any(step.success for step in steps):
        raise RuntimeError(f"{task}: source diagnostic is not a successful trajectory")

    robot = rcs.ROBOTS[RobotType.FR3]
    ik = rcs.common.Pin(robot.mjcf_model_path, robot.attachment_site)
    converted = []
    ik_dropped = 0
    duplicate_dropped = 0
    previous = None
    for step in steps:
        action = convert_step(step, ik)
        if action is None:
            ik_dropped += 1
            continue
        vector = flattened_action(action)
        if previous is not None and np.allclose(
            vector, previous, atol=1e-4, rtol=0.0
        ):
            duplicate_dropped += 1
            previous = vector
            continue
        previous = vector
        converted.append((step, action))
    if len(converted) < 2:
        raise RuntimeError(f"{task}: converter retained fewer than two rows")
    env = make_no_camera_env(task)
    try:
        env.reset()
        sim = env.get_wrapper_attr("sim")
        sim.set_state(converted[0][0].sim_state, converted[0][0].sim_state_schema)
        joint_errors = []
        max_progress = 0.0
        final_progress = 0.0
        success = False
        executed = 0
        # Row i is the post-action state for action i.  Restore row 0 once,
        # then execute rows 1..N without any further privileged state resets.
        for recorded_step, action in converted[1:]:
            observation, reward, terminated, truncated, info = env.step(action)
            expected = np.stack(
                [
                    np.asarray(recorded_step.observation[key]["joints"], np.float64)
                    for key in ("left", "right")
                ]
            )
            actual = np.stack(
                [
                    np.asarray(observation[key]["joints"], np.float64)
                    for key in ("left", "right")
                ]
            )
            joint_errors.append(np.abs(actual - expected))
            executed += 1
            final_progress = float(reward)
            max_progress = max(max_progress, final_progress)
            success = bool(info.get("success", False))
            if success or bool(np.asarray(terminated).all()) or bool(
                np.asarray(truncated).all()
            ):
                break
    finally:
        env.close()

    error = np.asarray(joint_errors, np.float64)
    return {
        "task": task,
        "source": str(source),
        "source_steps": len(steps),
        "converter_retained_steps": len(converted),
        "converter_ik_dropped_steps": ik_dropped,
        "converter_duplicate_dropped_steps": duplicate_dropped,
        "executed_steps": executed,
        "success": success,
        "max_stage_progress": max_progress,
        "final_stage_progress": final_progress,
        "joint_abs_error_mean": float(error.mean()),
        "joint_abs_error_p95": float(np.quantile(error, 0.95)),
        "joint_abs_error_max": float(error.max()),
        "contract": (
            "initial_recorded_state_then_official_pin_joint_targets_"
            "absolute_joint_30hz_binary_gripper"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    rows = []
    for task in args.tasks:
        candidates = sorted((args.source_root / task).glob("*.parquet"))
        if len(candidates) != 1:
            raise RuntimeError(f"{task}: expected one source parquet, got {candidates}")
        row = replay(task, candidates[0])
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "schema": "duobench-act-joint-interface-replay-v1",
        "passed": all(row["success"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
