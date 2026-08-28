from __future__ import annotations

import argparse, hashlib, json, os, time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import TASK_BY_NAME, local_observation, make_env
from .model import CAREPolicy, LegacyCAREPolicy, ModelConfig


def scalar(value: Any) -> bool:
    return bool(np.asarray(value).all())


def load_policy(path: Path, device: torch.device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    encoding = saved.get("action_encoding") or saved.get("normalization", {}).get("action_encoding")
    model = (CAREPolicy(ModelConfig(**saved["model_config"])) if encoding == "joint_residual_gripper_absolute" else LegacyCAREPolicy(ModelConfig(**saved["model_config"]))).to(device)
    model.load_state_dict(saved["model"], strict=True); model.eval()
    model.action_encoding = encoding or "absolute"
    norm = saved["normalization"]
    return model, np.asarray(norm["qpos_mean"], np.float32), np.asarray(norm["qpos_std"], np.float32), np.asarray(norm["action_mean"], np.float32), np.asarray(norm["action_std"], np.float32)


def preprocess(observation, arms, q_mean, q_std, size, device):
    images, qposes = [], []
    for arm in arms:
        image, qpos = local_observation(observation, arm)
        images.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255))
        qposes.append(torch.from_numpy((qpos.reshape(-1) - q_mean) / q_std))
    images = torch.stack(images).to(device)
    images = torch.nn.functional.interpolate(images, (size, size), mode="bilinear", align_corners=False)
    return images, torch.stack(qposes).to(device)


def policy_inputs(observation, arms, histories, previous_actions, q_mean, q_std, a_mean, a_std, size, device):
    images, qposes = preprocess(observation, arms, q_mean, q_std, size, device)
    raw_qposes = []
    for row, arm in enumerate(arms):
        _image, raw_qpos = local_observation(observation, arm)
        raw_qpos = raw_qpos.reshape(-1)
        raw_qposes.append(raw_qpos)
        token = np.zeros(17, dtype=np.float32)
        token[:9] = (raw_qpos - q_mean) / q_std
        if previous_actions[row] is not None:
            token[9:] = (previous_actions[row] - a_mean) / a_std
        histories[row].append(token)
    history = np.zeros((len(arms), histories[0].maxlen, 17), dtype=np.float32)
    mask = np.zeros((len(arms), histories[0].maxlen), dtype=np.float32)
    for row, values in enumerate(histories):
        values = np.asarray(values, dtype=np.float32)
        history[row, -len(values):] = values
        mask[row, -len(values):] = 1
    return images, qposes, np.asarray(raw_qposes), torch.from_numpy(history).to(device), torch.from_numpy(mask).to(device)


def decode_actions(chunks, raw_qposes, a_mean, a_std):
    rows = chunks.float().cpu().numpy() * a_std + a_mean
    rows[..., :7] += raw_qposes[:, None, :7]
    return rows


@torch.no_grad()
def run_episode(model, task, root, seed, device, stats, max_steps):
    q_mean, q_std, a_mean, a_std = stats; env = make_env(task, root)
    observation, _ = env.reset(seed=int(seed)); arms = tuple(range(task.arms)); trace = hashlib.sha256(); inference = []; success = False
    task_ids = torch.full((task.arms,), list(TASK_BY_NAME).index(task.name), dtype=torch.long, device=device)
    histories = [deque(maxlen=model.config.history) for _ in arms]
    previous_actions = [None for _ in arms]
    predictions: list[tuple[int, np.ndarray]] = []
    ensemble_decay = 0.01
    try:
        for step in range(max_steps):
            images, qposes, raw_qposes, history, history_mask = policy_inputs(
                observation, arms, histories, previous_actions, q_mean, q_std, a_mean, a_std,
                model.config.image_size, device,
            )
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16): chunks, selected = model.act(images, qposes, task_ids, history, history_mask)
            encoded_chunk = chunks.float().cpu().numpy() * a_std + a_mean
            predictions.append((step, encoded_chunk.copy()))
            predictions = [(born, chunk) for born, chunk in predictions if step - born < model.config.horizon]
            weights = np.asarray([np.exp(-ensemble_decay * (step - born)) for born, _ in predictions], np.float32)
            weights /= weights.sum()
            history_rows = sum(chunk[:, step - born] * weights[i] for i, (born, chunk) in enumerate(predictions))
            if model.action_encoding == "joint_residual_gripper_absolute":
                action_rows = history_rows.copy()
                action_rows[:, :7] += raw_qposes[:, :7]
            else:
                # The v1 checkpoint predicts absolute pd_joint_pos targets.
                # Keep every recent chunk and use the action for the current
                # timestamp from each chunk (ACT-style temporal ensembling).
                action_rows = history_rows.copy()
            inference.append(time.perf_counter() - started)
            action = {}
            for arm, row in zip(arms, action_rows):
                key = f"panda-{arm}"; low = np.asarray(env.action_space.spaces[key].low); high = np.asarray(env.action_space.spaces[key].high)
                row = np.clip(row, low, high).astype(np.float32); trace.update(row.tobytes()); action[key] = row
                previous_actions[arm] = history_rows[arm]
            observation, _reward, terminated, truncated, info = env.step(action)
            success = scalar(info.get("success", False))
            if success or scalar(terminated) or scalar(truncated): break
    finally: env.close()
    return {"task": task.name, "seed": int(seed), "success": success, "steps": step + 1, "mean_inference_seconds": float(np.mean(inference)), "p95_inference_seconds": float(np.quantile(inference, .95)), "action_trace_sha256": trace.hexdigest()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--task", choices=list(TASK_BY_NAME), required=True); p.add_argument("--robofactory-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--episodes", type=int, default=20); p.add_argument("--seed-start", type=int, default=20260824); p.add_argument("--max-steps", type=int); p.add_argument("--device", default="cuda:0")
    args = p.parse_args(); task = TASK_BY_NAME[args.task]; device = torch.device(args.device); model, *stats = load_policy(args.checkpoint, device)
    recovered = {}
    if args.output.with_suffix(".jsonl").is_file():
        for line in args.output.with_suffix(".jsonl").read_text().splitlines():
            try: row = json.loads(line); recovered[int(row["seed"])] = row
            except Exception: pass
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        row = recovered.get(seed) or run_episode(model, task, args.robofactory_root, seed, device, tuple(stats), args.max_steps or task.max_steps)
        rows.append(row)
        if seed not in recovered:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.with_suffix(".jsonl").open("a") as stream: stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    result = {"status": "complete", "task": task.name, "episodes": len(rows), "successes": sum(int(x["success"]) for x in rows), "success_rate": float(np.mean([x["success"] for x in rows])), "rows": rows, "policy_runtime": "temporal_ensemble_v2", "action_encoding": getattr(model, "action_encoding", "absolute")}
    tmp = args.output.with_suffix(".tmp"); tmp.write_text(json.dumps(result, indent=2) + "\n"); os.replace(tmp, args.output)


if __name__ == "__main__": main()
