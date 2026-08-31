from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms.v2 import functional as TVF
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo

from .dataset import TASKS
from .model import ACT
from .action_target import canonicalize_controller_action


# JointDatasetConverter._maybe_binarize_gripper uses a strict > 0.9 test for
# both observation.state and action.  RCS' runtime observation is the
# continuous physical gripper width even when binary_gripper=True, so it must
# be converted explicitly before applying the dataset normalization.
STATE_GRIPPER_BINARIZE_THRESHOLD = 0.9


def policy_state(observation, key: str) -> np.ndarray:
    joints = np.asarray(observation[key]["joints"], np.float32)
    physical_width = np.asarray(
        observation[key]["gripper"], np.float32
    ).reshape(-1)
    if joints.shape != (7,) or physical_width.shape != (1,):
        raise ValueError(
            f"unexpected {key} proprioception shapes: "
            f"joints={joints.shape}, gripper={physical_width.shape}"
        )
    binary_gripper = np.asarray(
        [float(physical_width[0] > STATE_GRIPPER_BINARIZE_THRESHOLD)],
        np.float32,
    )
    return np.concatenate((joints, binary_gripper))


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


def policy_image(observation, arm: int):
    head = np.asarray(observation["frames"]["head"]["rgb"]["data"], np.uint8)
    wrist_name = "left_wrist" if arm == 0 else "right_wrist"
    wrist = np.asarray(observation["frames"][wrist_name]["rgb"]["data"], np.uint8)
    # RCS renders 1280x720 in headless mode; training uses the official
    # 224x224 LeRobot streams. Resize each view independently, then preserve
    # the same side-by-side [head|local-wrist] contract.
    views = torch.from_numpy(np.stack((head, wrist)).copy()).permute(0, 3, 1, 2)
    # This is the same uint8 batched torchvision resize used by the official
    # RCS -> LeRobot converter (bilinear with antialiasing enabled).
    views = TVF.resize(views, (224, 224), antialias=True)
    return torch.cat((views[0], views[1]), dim=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20260820)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--mode",
        choices=("first", "open30", "ensemble"),
        default="ensemble",
        help="ACT chunk execution strategy (default: temporal ensemble)",
    )
    args = parser.parse_args()
    device = torch.device("cuda:0")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ACT(**{key: value for key, value in saved["model_config"].items() if key != "vision_backbone"}).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    norm = saved["normalization"]
    q_mean = np.asarray(norm["qpos_mean"], np.float32)
    q_std = np.asarray(norm["qpos_std"], np.float32)
    a_mean = np.asarray(norm["action_mean"], np.float32)
    a_std = np.asarray(norm["action_std"], np.float32)
    manifest = json.loads((args.data / "manifest.json").read_text())
    max_steps = args.max_steps or int(manifest["tasks"][args.task]["validation_max_steps"])
    task_id = TASKS.index(args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resume_path = args.output.with_suffix(".jsonl")
    recovered = {}
    if resume_path.is_file():
        for line in resume_path.read_text().splitlines():
            try:
                row = json.loads(line)
                recovered[int(row["seed"])] = row
            except Exception:
                pass
    env = make_env(args.task)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed_start + task_id * 1000 + episode
        if seed in recovered:
            rows.append(recovered[seed])
            continue
        observation, info = env.reset(seed=seed)
        chunks = [[], []]
        chunk_starts = [[], []]
        trace = hashlib.sha256()
        success = False
        max_stage_progress = 0.0
        final_stage_progress = 0.0
        started = time.perf_counter()
        for step in range(max_steps):
            open_loop = args.mode == "open30"
            refresh = not open_loop or step % 30 == 0 or not chunks[0]
            if refresh:
                images, states = [], []
                for arm, key in enumerate(("left", "right")):
                    raw = policy_state(observation, key)
                    images.append(policy_image(observation, arm))
                    states.append((raw - q_mean) / q_std)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, _, _ = model(
                        torch.stack(images).to(device, non_blocking=True).float().div_(255),
                        torch.from_numpy(np.asarray(states)).to(device),
                        torch.full((2,), task_id, dtype=torch.long, device=device),
                    )
                prediction = prediction.float().cpu().numpy() * a_std + a_mean
                for arm in range(2):
                    chunks[arm].append(prediction[arm])
                    chunk_starts[arm].append(step)
            action = {}
            for arm, key in enumerate(("left", "right")):
                if args.mode == "first":
                    local = chunks[arm][-1][0]
                elif args.mode == "open30":
                    offset = step - chunk_starts[arm][-1]
                    current = chunks[arm][-1]
                    local = current[offset] if offset < len(current) else current[-1]
                else:
                    candidates = [
                        chunk[step - start]
                        for start, chunk in zip(chunk_starts[arm], chunks[arm])
                        if 0 <= step - start < len(chunk)
                    ]
                    weights = np.exp(-0.01 * np.arange(len(candidates) - 1, -1, -1))
                    weights /= weights.sum()
                    local = np.sum(np.asarray(candidates) * weights[:, None], axis=0)
                # Absolute joint targets use the converter/RCS XML range.  The
                # Gym Box is a conservative API bound and is narrower than
                # recorded expert targets on several tasks; clipping here
                # would silently change the learned action semantics.
                local = canonicalize_controller_action(local)
                action[key] = {"joints": local[:7].astype(np.float32), "gripper": np.asarray([local[7]], np.float32)}
                trace.update(local.astype(np.float32).tobytes())
            observation, reward, terminated, truncated, info = env.step(action)
            final_stage_progress = float(reward)
            max_stage_progress = max(max_stage_progress, final_stage_progress)
            success = bool(info.get("success", False))
            if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        row = {
            "task": args.task, "seed": seed, "success": success, "steps": step + 1,
            "max_steps": max_steps, "final_stage_progress": final_stage_progress,
            "max_stage_progress": max_stage_progress, "action_trace_sha256": trace.hexdigest(),
            "wall_seconds": time.perf_counter() - started, "execution_mode": args.mode,
        }
        rows.append(row)
        with resume_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    env.close()
    result = {
        "status": "complete", "schema": "duobench-act-validation20-task-v1", "task": args.task,
        "episodes": len(rows), "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])), "rows": rows,
        "policy_contract": saved["policy_contract"], "action_encoding": saved["action_encoding"],
        "state_gripper_encoding": "physical_width_gt_0.9_to_binary",
        "execution_mode": args.mode,
        "temporal_ensemble_decay": 0.01 if args.mode == "ensemble" else None,
        "open_loop_replan_steps": 30 if args.mode == "open30" else None,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
