from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms.v2 import functional as TVF
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo

from deployment.duo_act.action_target import canonicalize_controller_action

from .common import (
    EVALUATOR_REVISION,
    EXECUTION_STEPS,
    IMAGE_SIZE,
    OBS_STEPS,
    POLICY_CONTRACT,
    TASKS,
    atomic_json,
    sha256_file,
)
from .modeling import load_policy


STATE_GRIPPER_BINARIZE_THRESHOLD = 0.9


def evaluator_revision_for(replan_steps: int, inference_steps: int, weights: str) -> str:
    """Give each closed-loop variant an isolated, recoverable identity."""
    if (
        int(replan_steps) == EXECUTION_STEPS
        and int(inference_steps) == 20
        and weights == "ema"
    ):
        return EVALUATOR_REVISION
    return (
        f"duobench-dp-ablation-r1-replan{int(replan_steps)}"
        f"-inf{int(inference_steps)}-{weights}-v1"
    )


def policy_state(observation: dict, key: str) -> np.ndarray:
    joints = np.asarray(observation[key]["joints"], np.float32)
    gripper = np.asarray(observation[key]["gripper"], np.float32).reshape(-1)
    if joints.shape != (7,) or gripper.shape != (1,):
        raise ValueError(f"unexpected {key} proprioception {joints.shape}/{gripper.shape}")
    return np.concatenate(
        (joints, np.asarray([float(gripper[0] > STATE_GRIPPER_BINARIZE_THRESHOLD)], np.float32))
    )


def policy_image(observation: dict, arm: int) -> torch.Tensor:
    head = np.asarray(observation["frames"]["head"]["rgb"]["data"], np.uint8)
    wrist_key = "left_wrist" if arm == 0 else "right_wrist"
    wrist = np.asarray(observation["frames"][wrist_key]["rgb"]["data"], np.uint8)
    if head.ndim != 3 or wrist.ndim != 3 or head.shape[-1] != 3 or wrist.shape[-1] != 3:
        raise ValueError(f"unexpected runtime RGB {head.shape}/{wrist.shape}")
    views = torch.from_numpy(np.stack((head, wrist)).copy()).permute(0, 3, 1, 2)
    views = TVF.resize(views, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    return torch.cat((views[0], views[1]), dim=2)


def make_env(task: str):
    module = __import__(f"duobench.tasks.{task}", fromlist=["*"])
    config_class = getattr(module, "".join(part.title() for part in task.split("_")) + "EnvConfig")
    config = config_class().config()
    config.headless = True
    config.control_mode = ControlMode.JOINTS
    config.relative_to = RelativeTo.NONE
    config.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30)
    config.wrapper_cfg.binary_gripper = True
    return gym.make(f"duobench/{task}", cfg=config)


def _done(value) -> bool:
    return bool(np.asarray(value).all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20260820)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--weights", choices=("ema", "online"), default="ema")
    parser.add_argument("--replan-steps", type=int, default=EXECUTION_STEPS)
    parser.add_argument("--revision")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.replan_steps <= EXECUTION_STEPS:
        raise ValueError(f"replan-steps must be in [1, {EXECUTION_STEPS}]")
    if args.inference_steps <= 0:
        raise ValueError("inference-steps must be positive")
    evaluator_revision = args.revision or evaluator_revision_for(
        args.replan_steps, args.inference_steps, args.weights
    )
    device = torch.device("cuda:0")
    policy, payload = load_policy(args.checkpoint, device, args.inference_steps, args.weights)
    task_conditioning = bool(payload.get("config", {}).get("task_conditioning", False))
    manifest = json.loads((args.data / "manifest.json").read_text())
    max_steps = args.max_steps or int(manifest["tasks"][args.task]["validation_max_steps"])
    task_id = TASKS.index(args.task)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix(".jsonl")
    recovered = {}
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                row = json.loads(line)
                if (
                    row.get("evaluator_revision") == evaluator_revision
                    and row.get("checkpoint_sha256") == checkpoint_sha256
                ):
                    recovered[int(row["seed"])] = row
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
    env = make_env(args.task)
    rows = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed_start + task_id * 1000 + episode_index
            if seed in recovered:
                rows.append(recovered[seed])
                continue
            observation, _info = env.reset(seed=seed)
            histories = [deque(maxlen=OBS_STEPS), deque(maxlen=OBS_STEPS)]
            for arm, key in enumerate(("left", "right")):
                histories[arm].append((policy_image(observation, arm), policy_state(observation, key)))
            trace = hashlib.sha256()
            success = False
            final_progress = max_progress = 0.0
            controls = replans = 0
            emitted_gripper_transitions = 0
            emitted_gripper_open_steps = 0
            emitted_joint_min = np.full(7, np.inf, dtype=np.float32)
            emitted_joint_max = np.full(7, -np.inf, dtype=np.float32)
            emitted_joint_delta_max = 0.0
            previous_emitted_joint = [None, None]
            previous_emitted_gripper = [None, None]
            inference_times = []
            started = time.perf_counter()
            terminated = truncated = False
            while controls < max_steps and not (success or _done(terminated) or _done(truncated)):
                images, states = [], []
                for arm in (0, 1):
                    history = list(histories[arm])
                    while len(history) < OBS_STEPS:
                        history.insert(0, history[0])
                    images.append(torch.stack([row[0] for row in history]))
                    local_states = np.stack([row[1] for row in history])
                    if task_conditioning:
                        task_onehot = np.zeros((OBS_STEPS, len(TASKS)), dtype=np.float32)
                        task_onehot[:, task_id] = 1.0
                        local_states = np.concatenate((local_states, task_onehot), axis=-1)
                    states.append(local_states)
                sample_seed = seed + replans
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                inference_started = time.perf_counter()
                with torch.inference_mode():
                    prediction = policy.predict_action(
                        {
                            "head_wrist": torch.stack(images).to(device).float().div_(255.0),
                            "agent_pos": torch.from_numpy(np.stack(states)).to(device),
                        }
                    )["action"][:, : args.replan_steps]
                inference_times.append(time.perf_counter() - inference_started)
                chunk = prediction.float().cpu().numpy()
                replans += 1
                for offset in range(min(args.replan_steps, max_steps - controls)):
                    action = {}
                    for arm, key in enumerate(("left", "right")):
                        local = canonicalize_controller_action(chunk[arm, offset]).astype(np.float32)
                        emitted_joint_min = np.minimum(emitted_joint_min, local[:7])
                        emitted_joint_max = np.maximum(emitted_joint_max, local[:7])
                        emitted_gripper_open_steps += int(local[7] >= 0.5)
                        if previous_emitted_gripper[arm] is not None:
                            emitted_gripper_transitions += int(
                                local[7] != previous_emitted_gripper[arm]
                            )
                        if previous_emitted_joint[arm] is not None:
                            emitted_joint_delta_max = max(
                                emitted_joint_delta_max,
                                float(np.max(np.abs(local[:7] - previous_emitted_joint[arm]))),
                            )
                        previous_emitted_joint[arm] = local[:7].copy()
                        previous_emitted_gripper[arm] = float(local[7])
                        action[key] = {
                            "joints": local[:7],
                            "gripper": np.asarray([local[7]], np.float32),
                        }
                        trace.update(local.tobytes())
                    observation, reward, terminated, truncated, info = env.step(action)
                    controls += 1
                    final_progress = float(reward)
                    max_progress = max(max_progress, final_progress)
                    success = bool(info.get("success", False))
                    for arm, key in enumerate(("left", "right")):
                        histories[arm].append((policy_image(observation, arm), policy_state(observation, key)))
                    if success or _done(terminated) or _done(truncated):
                        break
            row = {
                "task": args.task,
                "seed": seed,
                "success": success,
                "steps": controls,
                "max_steps": max_steps,
                "replans": replans,
                "replan_steps": args.replan_steps,
                "final_stage_progress": final_progress,
                "max_stage_progress": max_progress,
                "emitted_gripper_transitions": emitted_gripper_transitions,
                "emitted_gripper_open_fraction": float(
                    emitted_gripper_open_steps / max(controls * 2, 1)
                ),
                "emitted_joint_min": emitted_joint_min.astype(float).tolist(),
                "emitted_joint_max": emitted_joint_max.astype(float).tolist(),
                "emitted_joint_delta_max": float(emitted_joint_delta_max),
                "mean_inference_seconds": float(np.mean(inference_times)) if inference_times else None,
                "p95_inference_seconds": float(np.quantile(inference_times, 0.95)) if inference_times else None,
                "action_trace_sha256": trace.hexdigest(),
                "wall_seconds": time.perf_counter() - started,
                "checkpoint_sha256": checkpoint_sha256,
                "evaluator_revision": evaluator_revision,
            }
            rows.append(row)
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    result = {
        "schema": "duobench.dp.validation-task.v1",
        "status": "complete",
        "task": args.task,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "rows": rows,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_revision": evaluator_revision,
        "policy_contract": payload["policy_contract"],
        "task_conditioning": task_conditioning,
        "weights": args.weights,
        "inference_steps": args.inference_steps,
        "replan_interval": args.replan_steps,
        "state_gripper_encoding": "physical_width_gt_0.9_to_binary",
        "action_encoding": "controller_equivalent_absolute_joint7_binary_gripper1",
        "rgb_preprocessing": "resize_views_independently_uint8_bilinear_antialias_then_concat_div255_imagenet",
        "max_steps": max_steps,
        "smoke": args.smoke,
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
