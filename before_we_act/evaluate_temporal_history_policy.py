"""Closed-loop evaluator for temporal-history policies."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401
from two_three_task_manifest import TASKS, get_task

from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from before_we_act.temporal_history_data import (
    HISTORY_STEPS,
    TASK_TEXT,
    TASK_TEXT_BYTES,
    task_text_tensor,
)


def reset_reproducibly(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def load_model(checkpoint: str, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    policy_variant = str(config.get("policy_variant", ""))
    if not policy_variant.startswith("b0h_"):
        raise ValueError(f"wrong checkpoint variant: {policy_variant}")
    variant = policy_variant.removeprefix("b0h_")
    model = TemporalHistoryPolicy(
        config.get("state_dim", 9),
        config.get("action_dim", 8),
        variant=variant,
        horizon=config.get("horizon", 100),
        d_model=config.get("d_model", 384),
        enc_layers=config.get("enc_layers", 4),
        dec_layers=config.get("dec_layers", 7),
        roles=config.get("roles", 4),
        role_rank=config.get("role_rank", 32),
        history_layers=config.get("history_layers", 2),
        dino_model=config["dino_model"],
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in saved["stats"].items()
    }
    return model, stats, config


def prepare_current(observation, arms, stats, device):
    sensors = observation["sensor_data"]
    global_image = np.asarray(sensors["head_camera_global"]["rgb"])
    global_image = global_image[0] if global_image.ndim == 4 else global_image
    global_images, local_images, qposes = [], [], []
    for arm in arms:
        local_key = f"head_camera_agent{arm}"
        local = np.asarray(
            sensors[local_key]["rgb"] if local_key in sensors else global_image
        )
        qpos = np.asarray(observation["agent"][f"panda-{arm}"]["qpos"])
        global_images.append(global_image)
        local_images.append(local[0] if local.ndim == 4 else local)
        qposes.append(qpos[0] if qpos.ndim == 2 else qpos)
    global_rgb = torch.as_tensor(np.stack(global_images)).permute(0, 3, 1, 2)
    local_rgb = torch.as_tensor(np.stack(local_images)).permute(0, 3, 1, 2)
    if tuple(global_rgb.shape[-2:]) != (480, 640) or tuple(local_rgb.shape[-2:]) != (
        480,
        640,
    ):
        raise ValueError("B0-H validation requires original 640x480 observations")
    global_rgb = global_rgb.float().div_(255).to(device)
    local_rgb = local_rgb.float().div_(255).to(device)
    qpos = torch.as_tensor(np.stack(qposes)).float().to(device)
    qpos = (qpos - stats["q_mean"]) / stats["q_std"]
    return global_rgb, local_rgb, qpos


class EpisodeHistory:
    """Per-arm, resettable deployment history matching TeamTemporalSample."""

    def __init__(self, arms):
        self.arms = tuple(arms)
        self.visual = {arm: deque(maxlen=HISTORY_STEPS - 1) for arm in self.arms}
        self.qpos = {arm: deque(maxlen=HISTORY_STEPS - 1) for arm in self.arms}
        self.actions = {arm: deque(maxlen=HISTORY_STEPS) for arm in self.arms}

    def batch(self, current_qpos: torch.Tensor, task: str, device: torch.device):
        count = len(self.arms)
        visual = torch.zeros(count, HISTORY_STEPS, 2, 768, device=device)
        qpos = torch.zeros(count, HISTORY_STEPS, 9, device=device)
        action = torch.zeros(count, HISTORY_STEPS, 8, device=device)
        history_mask = torch.zeros(
            count, HISTORY_STEPS, dtype=torch.bool, device=device
        )
        action_mask = torch.zeros(
            count, HISTORY_STEPS, dtype=torch.bool, device=device
        )
        for index, arm in enumerate(self.arms):
            visual_values = list(self.visual[arm])
            qpos_values = list(self.qpos[arm])
            if len(visual_values) != len(qpos_values):
                raise RuntimeError("visual/qpos deployment history drift")
            if visual_values:
                first = HISTORY_STEPS - 1 - len(visual_values)
                visual[index, first:-1] = torch.stack(visual_values).to(device)
                qpos[index, first:-1] = torch.stack(qpos_values).to(device)
                history_mask[index, first:-1] = True
            qpos[index, -1] = current_qpos[index]
            history_mask[index, -1] = True
            action_values = list(self.actions[arm])
            if action_values:
                first = HISTORY_STEPS - len(action_values)
                action[index, first:] = torch.stack(action_values).to(device)
                action_mask[index, first:] = True
        text, text_mask = task_text_tensor(TASK_TEXT[task])
        return {
            "history_visual_raw": visual,
            "history_qpos": qpos,
            "history_action": action,
            "history_mask": history_mask,
            "action_history_mask": action_mask,
            "task_bytes": text.unsqueeze(0).expand(count, TASK_TEXT_BYTES).to(device),
            "task_text_mask": text_mask.unsqueeze(0)
            .expand(count, TASK_TEXT_BYTES)
            .to(device),
            "episode_reset": torch.tensor(
                [not self.visual[arm] and not self.actions[arm] for arm in self.arms],
                dtype=torch.bool,
                device=device,
            ),
        }

    def append_observation(
        self, current_visual: torch.Tensor, current_qpos: torch.Tensor
    ) -> None:
        for index, arm in enumerate(self.arms):
            self.visual[arm].append(current_visual[index].detach().float().cpu())
            self.qpos[arm].append(current_qpos[index].detach().float().cpu())

    def append_action(self, normalized: Mapping[int, torch.Tensor]) -> None:
        for arm in self.arms:
            self.actions[arm].append(normalized[arm].detach().float().cpu())


class TemporalChunkEnsembler:
    def __init__(self, arms, decay: float = 0.01):
        self.arms = tuple(arms)
        self.decay = float(decay)
        self.histories = [[] for _ in self.arms]

    def append_and_select(self, step: int, chunks: np.ndarray) -> dict[str, np.ndarray]:
        action = {}
        for local_index, arm in enumerate(self.arms):
            self.histories[local_index].append((step, chunks[local_index]))
            self.histories[local_index] = [
                item
                for item in self.histories[local_index]
                if step - item[0] < len(item[1])
            ]
            candidates = np.asarray(
                [
                    chunk[step - start]
                    for start, chunk in self.histories[local_index]
                ]
            )
            weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1))
            weights /= weights.sum()
            action[f"panda-{arm}"] = np.sum(candidates * weights[:, None], axis=0)
        return action


@torch.no_grad()
def predict_all(model, stats, observation, arms, history, task, device):
    global_rgb, local_rgb, qpos = prepare_current(observation, arms, stats, device)
    temporal = history.batch(qpos, task, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        chunks, _mu, _logvar, current_visual = model(
            global_rgb,
            local_rgb,
            **temporal,
            return_current_visual=True,
        )
    history.append_observation(current_visual, qpos)
    denormalized = chunks * stats["a_std"] + stats["a_mean"]
    return denormalized.float().cpu().numpy()


def evaluate(checkpoint, task_name, seeds, device_name, max_steps):
    torch.set_num_threads(12)
    device = torch.device(device_name)
    model, stats, config = load_model(checkpoint, device)
    specification = get_task(task_name)
    arms = specification["agents"]
    env = gym.make(
        specification["env_id"],
        config=f"/workspace/RoboFactory/{specification['config']}",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sim_backend="cpu",
        sensor_configs=dict(shader_pack="default", width=640, height=480),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    rows = []
    for seed in seeds:
        observation, _ = reset_reproducibly(env, seed)
        ensembler = TemporalChunkEnsembler(arms)
        history = EpisodeHistory(arms)
        success = False
        for step in range(max_steps):
            chunks = predict_all(
                model, stats, observation, arms, history, task_name, device
            )
            action = ensembler.append_and_select(step, chunks)
            normalized = {
                arm: (
                    torch.as_tensor(action[f"panda-{arm}"], device=device)
                    - stats["a_mean"]
                )
                / stats["a_std"]
                for arm in arms
            }
            history.append_action(normalized)
            observation, _, terminated, truncated, info = env.step(action)
            success = bool(np.asarray(info.get("success", False)).all())
            if bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        row = {"seed": seed, "success": success, "steps": step + 1}
        rows.append(row)
        print(json.dumps({"task": task_name, **row}), flush=True)
    env.close()
    return rows, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    args = parser.parse_args()
    raw = Path(args.seed_file).read_bytes()
    manifest = json.loads(raw)
    all_seeds = [int(seed) for seed in manifest["seeds"]]
    if not 1 <= args.episodes <= len(all_seeds):
        raise ValueError("requested validation episode count is outside seed file")
    requested = all_seeds[: args.episodes]
    recovered = []
    if args.resume_log and Path(args.resume_log).is_file():
        allowed = set(requested)
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("task") == args.task
                and row.get("seed") in allowed
                and isinstance(row.get("success"), bool)
                and isinstance(row.get("steps"), int)
            ):
                recovered.append(
                    {key: row[key] for key in ("seed", "success", "steps")}
                )
        recovered = list({row["seed"]: row for row in recovered}.values())
    completed = {row["seed"] for row in recovered}
    remaining = [seed for seed in requested if seed not in completed]
    print(
        json.dumps(
            {"task": args.task, "recovered": len(recovered), "remaining": len(remaining)}
        ),
        flush=True,
    )
    if remaining:
        rows, config = evaluate(
            args.checkpoint, args.task, remaining, args.device, args.max_steps
        )
    else:
        rows = []
        config = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )["config"]
    rows = recovered + rows
    rows.sort(key=lambda row: requested.index(row["seed"]))
    successes = sum(row["success"] for row in rows)
    result = {
        "task": args.task,
        "env_id": get_task(args.task)["env_id"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "rows": rows,
        "camera": "original fixed global+matching-agent RGB at 640x480",
        "policy_input": config.get("policy_input"),
        "policy_variant": config.get("policy_variant"),
        "seed_protocol": {
            "source": args.seed_file,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "selection_method": manifest.get("selection_method"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result | {"rows": "saved"}), flush=True)


if __name__ == "__main__":
    main()
