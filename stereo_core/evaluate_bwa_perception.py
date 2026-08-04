"""Closed-loop R10 evaluator with legal temporal context and paired interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401
try:
    from .bwa_contracts import CoreDeploymentContext
    from .bwa_perception import build_perception_extension
    from .evaluate_no_wrist_pair import (
        TemporalChunkEnsembler,
        denormalize_action_chunks,
        prepare_no_wrist_batch,
        reset_reproducibly,
    )
    from .no_wrist_pair_model import NoWristPAIRRoute
    from .two_three_task_manifest import TASKS, get_task
except ImportError:
    from bwa_contracts import CoreDeploymentContext
    from bwa_perception import build_perception_extension
    from evaluate_no_wrist_pair import (
        TemporalChunkEnsembler,
        denormalize_action_chunks,
        prepare_no_wrist_batch,
        reset_reproducibly,
    )
    from no_wrist_pair_model import NoWristPAIRRoute
    from two_three_task_manifest import TASKS, get_task


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(parent_checkpoint: Path, extension_checkpoint: Path, device):
    parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    candidate = torch.load(extension_checkpoint, map_location="cpu", weights_only=False)
    config = candidate["config"]
    if file_sha256(parent_checkpoint) != candidate["parent_sha256"]:
        raise ValueError("candidate checkpoint parent hash mismatch")
    parent_config = parent["config"]
    model = NoWristPAIRRoute(
        parent_config.get("state_dim", 9),
        parent_config.get("action_dim", 8),
        horizon=parent_config.get("horizon", 100),
        d_model=parent_config.get("d_model", 384),
        enc_layers=parent_config.get("enc_layers", 4),
        dec_layers=parent_config.get("dec_layers", 7),
        roles=parent_config.get("roles", 4),
        role_rank=parent_config.get("role_rank", 32),
        dino_model=parent_config["dino_model"],
    ).to(device)
    model.load_state_dict(parent["model"], strict=True)
    extension = build_perception_extension(config["bridge"]).to(device)
    extension.load_state_dict(candidate["extension"], strict=True)
    model.register_perception_extension(extension)
    model.eval()
    stats = {key: torch.as_tensor(value, device=device) for key, value in parent["stats"].items()}
    return model, extension, stats, config


class DeploymentHistory:
    def __init__(self, arms, steps: int, device):
        self.arms = tuple(arms)
        self.steps = int(steps)
        self.device = device
        self.views = [[] for _ in arms]
        self.qpos = [[] for _ in arms]
        self.actions = [[] for _ in arms]

    @staticmethod
    def _pad(rows, steps, example):
        rows = rows[-steps:]
        valid = [False] * (steps - len(rows)) + [True] * len(rows)
        padded = [torch.zeros_like(example) for _ in range(steps - len(rows))] + rows
        return torch.stack(padded), torch.tensor(valid, device=example.device)

    def context(self, qpos, current_views, metadata):
        if self.steps == 0:
            return CoreDeploymentContext(fixed_camera_metadata=metadata)
        pooled = torch.stack(
            (current_views.local_tokens.mean(1), current_views.global_tokens.mean(1)), dim=1
        )
        view_rows, qpos_rows, action_rows, masks = [], [], [], []
        for index in range(len(self.arms)):
            view, mask = self._pad(self.views[index], self.steps, pooled[index])
            position, _ = self._pad(self.qpos[index], self.steps, qpos[index])
            action_example = torch.zeros(8, device=qpos.device, dtype=qpos.dtype)
            action, _ = self._pad(self.actions[index], self.steps, action_example)
            view_rows.append(view)
            qpos_rows.append(position)
            action_rows.append(action)
            masks.append(mask)
        return CoreDeploymentContext(
            view_token_history=torch.stack(view_rows),
            qpos_history=torch.stack(qpos_rows),
            executed_action_history=torch.stack(action_rows),
            history_mask=torch.stack(masks),
            fixed_camera_metadata=metadata,
        )

    def append(self, views, qpos, normalized_actions):
        pooled = torch.stack(
            (views.local_tokens.mean(1), views.global_tokens.mean(1)), dim=1
        ).detach()
        for index in range(len(self.arms)):
            self.views[index].append(pooled[index])
            self.qpos[index].append(qpos[index].detach())
            self.actions[index].append(normalized_actions[index].detach())


@torch.no_grad()
def predict(model, extension, stats, observation, arms, history, device, intervention):
    global_rgb, local_rgb, qpos = prepare_no_wrist_batch(observation, arms, stats, device)
    metadata = {
        "global_view_mask": torch.ones(len(arms), device=device, dtype=torch.bool),
        "local_view_mask": torch.ones(len(arms), device=device, dtype=torch.bool),
        "diagnostic_intervention": intervention,
        "calibration_sha256": "configured-fixed-camera-calibration",
    }
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        views = model.encode_view_tokens(global_rgb, local_rgb)
        state_vec = model.state(qpos)
        context = model.encode_context(
            global_rgb,
            local_rgb,
            qpos,
            deployment_context=history.context(qpos, views, metadata),
            _views=views,
            _state_vec=state_vec,
        )
        chunks = model.decode_with_gates(context, context.sparse_routes)
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    chunks = denormalize_action_chunks(chunks, stats).float()
    return chunks, qpos, views, latency_ms


def evaluate(
    parent_checkpoint: Path,
    extension_checkpoint: Path,
    task_name: str,
    seeds: list[int],
    device_name: str,
    max_steps: int,
    intervention: str,
):
    torch.set_num_threads(12)
    device = torch.device(device_name)
    model, extension, stats, config = load_candidate(parent_checkpoint, extension_checkpoint, device)
    expected = str(config["intervention"]["name"])
    if intervention not in {"normal", expected}:
        raise ValueError(f"intervention must be normal or {expected}")
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
    rows, latencies = [], []
    history_steps = int(config["bridge"].get("history_steps", 0))
    for seed in seeds:
        observation, _ = reset_reproducibly(env, seed)
        history = DeploymentHistory(arms, history_steps, device)
        ensembler = TemporalChunkEnsembler(arms)
        success = False
        for step in range(max_steps):
            chunks, qpos, views, latency = predict(
                model, extension, stats, observation, arms, history, device, intervention
            )
            latencies.append(latency)
            chunk_array = chunks.cpu().numpy()
            action = ensembler.append_and_select(step, chunk_array)
            normalized = torch.stack(
                [
                    (torch.as_tensor(action[f"panda-{arm}"], device=device) - stats["a_mean"])
                    / stats["a_std"]
                    for arm in arms
                ]
            )
            history.append(views, qpos, normalized)
            observation, _, terminated, truncated, info = env.step(action)
            success = bool(np.asarray(info.get("success", False)).all())
            if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        row = {"seed": seed, "success": success, "steps": step + 1}
        rows.append(row)
        print(json.dumps({"task": task_name, "condition": intervention, **row}), flush=True)
    env.close()
    return rows, latencies, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--extension-checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--intervention", default="normal")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    args = parser.parse_args()

    seed_path = Path(args.seed_file).resolve(strict=True)
    raw = seed_path.read_bytes()
    seed_manifest = json.loads(raw)
    all_seeds = [int(seed) for seed in seed_manifest["seeds"]]
    if not 1 <= args.episodes <= len(all_seeds):
        raise ValueError("invalid episode count")
    requested = all_seeds[: args.episodes]
    recovered = []
    if args.resume_log and Path(args.resume_log).is_file():
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("task") == args.task
                and row.get("condition") == args.intervention
                and row.get("seed") in requested
                and isinstance(row.get("success"), bool)
            ):
                recovered.append({key: row[key] for key in ("seed", "success", "steps")})
    recovered = list({row["seed"]: row for row in recovered}.values())
    complete = {row["seed"] for row in recovered}
    remaining = [seed for seed in requested if seed not in complete]
    if remaining:
        rows, latencies, config = evaluate(
            Path(args.parent_checkpoint).resolve(strict=True),
            Path(args.extension_checkpoint).resolve(strict=True),
            args.task,
            remaining,
            args.device,
            args.max_steps,
            args.intervention,
        )
    else:
        rows, latencies = [], []
        checkpoint = torch.load(args.extension_checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
    rows = recovered + rows
    rows.sort(key=lambda row: requested.index(row["seed"]))
    successes = sum(row["success"] for row in rows)
    result = {
        "schema_version": 1,
        "candidate_id": config["candidate_id"],
        "task": args.task,
        "condition": args.intervention,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "rows": rows,
        "latency_ms": {
            "samples": len(latencies),
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
        },
        "policy_inputs": "current fixed global/local RGB, own qpos, legal past context only",
        "privileged_inputs": False,
        "seed_protocol": {"source": str(seed_path), "sha256": hashlib.sha256(raw).hexdigest()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
