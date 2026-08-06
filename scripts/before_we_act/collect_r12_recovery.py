#!/usr/bin/env python3
"""Collect training-only student states labeled by the frozen W10 teacher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401

from before_we_act.action_generator.base import JointActionGenerator, load_r12_config
from before_we_act.benchmark import get_task
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.evaluate_action_generator import TeamHistory, TemporalChunkEnsembler
from before_we_act.spatial_observation import R12SpatialObservationEncoder
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config
from stereo_core.evaluate_no_wrist_pair import load_model as load_teacher
from stereo_core.evaluate_no_wrist_pair import predict_all as predict_teacher


PROTOCOL = "r12r2_student_on_policy_w10_teacher_recovery_shard_v1"
TASK_NAMES = tuple(TASKS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_reproducibly(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def load_student_models(args, device):
    config = load_r12_config(args.student_config)
    if config.candidate_id != args.candidate:
        raise ValueError("student config/candidate differs")
    saved = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    if saved.get("candidate_id") != args.candidate:
        raise ValueError("student checkpoint/candidate differs")
    generator = JointActionGenerator(config).to(device)
    incompatible = generator.load_state_dict(saved["model"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("spatial_") for key in incompatible.missing_keys
    ):
        raise ValueError(f"student checkpoint differs outside spatial adapter: {incompatible}")
    if float(generator.spatial_gate.detach()) != 0.0:
        raise ValueError("R12-R2 rollout student must have a closed spatial gate")
    generator.eval()
    belief_config = load_r11_config(args.belief_config)
    belief_saved = torch.load(args.belief_checkpoint, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device).eval()
    belief.load_state_dict(belief_saved["model"], strict=True)
    spatial = R12SpatialObservationEncoder(
        config.observation, args.vision_artifact, inference_batch_size=5
    ).to(device).eval()
    stats = {key: torch.as_tensor(value, device=device) for key, value in saved["stats"].items()}
    return config, generator, belief, spatial, stats


def stack_rows(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("p0", "p2"), required=True)
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--vision-artifact", required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--sample-every", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        if payload.get("protocol_variant") != PROTOCOL or payload.get("candidate") != args.candidate:
            raise ValueError("existing recovery shard identity differs")
        print(json.dumps({"reused": str(args.output), "rows": len(payload["train"]["task_index"])}))
        return
    if args.max_steps < 1 or args.sample_every < 1:
        raise ValueError("recovery rollout limits must be positive")
    device = torch.device(args.device)
    config, student, belief, spatial, stats = load_student_models(args, device)
    teacher, teacher_stats, _teacher_config = load_teacher(args.teacher_checkpoint, device)
    for key in ("a_mean", "a_std", "q_mean", "q_std"):
        torch.testing.assert_close(stats[key], teacher_stats[key], atol=0, rtol=0)
    seeds = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    if seeds.get("protocol") != "training_only_recovery_seeds_v1":
        raise ValueError("recovery seed manifest identity differs")
    rows, episodes = [], []
    last_heartbeat = time.monotonic()
    for task in TASK_NAMES:
        specification = get_task(task)
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
        for seed in map(int, seeds["tasks"][task]["seeds"]):
            observation, _ = reset_reproducibly(env, seed)
            history = TeamHistory(arms)
            ensemble = TemporalChunkEnsembler(arms)
            previous_action = None
            success = False
            collected = 0
            for step in range(args.max_steps):
                batch = history.batch(observation, previous_action, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    belief_state = belief(batch)["belief"]
                    zero_spatial = torch.zeros(
                        (1, 5, 16, 768), device=device, dtype=belief_state.tokens.dtype
                    )
                    noise = torch.randn(
                        (1, 100, 32),
                        generator=torch.Generator(device=device).manual_seed(seed * 1_000_003 + step),
                        device=device,
                    )
                    proposals = student.sample(
                        belief_state,
                        spatial_tokens=zero_spatial,
                        spatial_view_mask=batch["spatial_view_mask"],
                        noise=noise,
                    )
                normalized = proposals.actions[0, 0, : len(arms)]
                chunks = (
                    normalized * stats["a_std"][None, None] + stats["a_mean"][None, None]
                ).float().cpu().numpy()
                action = ensemble.append_and_select(step, chunks)
                if step % args.sample_every == 0:
                    teacher_chunks = predict_teacher(
                        teacher, teacher_stats, observation, arms, device
                    )
                    teacher_normalized = (
                        torch.from_numpy(teacher_chunks).to(device)
                        - stats["a_mean"][None, None]
                    ) / stats["a_std"][None, None]
                    target = torch.zeros((100, 4, 8), dtype=torch.float32)
                    target[:, : len(arms)] = teacher_normalized.permute(1, 0, 2).float().cpu()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        spatial_tokens, spatial_mask = spatial(
                            batch["raw_fixed_rgb"], batch["spatial_view_mask"]
                        )
                    rows.append(
                        {
                            "visual": batch["visual"][0].float().cpu(),
                            "view_mask": batch["view_mask"][0].float().cpu(),
                            "qpos": batch["qpos"][0].float().cpu(),
                            "actions": batch["actions"][0].float().cpu(),
                            "agent_mask": batch["agent_mask"][0].cpu(),
                            "task_index": torch.tensor(TASK_NAMES.index(task), dtype=torch.long),
                            "joint_actions": target,
                            "action_step_mask": torch.ones(100, dtype=torch.bool),
                            "spatial_tokens": spatial_tokens[0].to(device="cpu", dtype=torch.float16),
                            "spatial_view_mask": spatial_mask[0].cpu(),
                            "rollout_seed": torch.tensor(seed, dtype=torch.long),
                            "rollout_step": torch.tensor(step, dtype=torch.long),
                            "source_policy": torch.tensor(int(args.candidate[1:]), dtype=torch.long),
                        }
                    )
                    collected += 1
                previous_action = {key: value.copy() for key, value in action.items()}
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(np.asarray(info.get("success", False)).all())
                if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                    break
                if time.monotonic() - last_heartbeat >= 20:
                    atomic_json(
                        args.heartbeat,
                        {
                            "producer": "collect_r12_recovery",
                            "candidate": args.candidate,
                            "task": task,
                            "seed": seed,
                            "step": step,
                            "rows": len(rows),
                            "updated_at": now(),
                        },
                    )
                    last_heartbeat = time.monotonic()
            episodes.append(
                {"task": task, "seed": seed, "steps": step + 1, "success": success, "rows": collected}
            )
        env.close()
    if not rows:
        raise RuntimeError("recovery collector produced no rows")
    payload = {
        "schema_version": 1,
        "round": "R12-R3",
        "protocol_variant": PROTOCOL,
        "candidate": args.candidate,
        "metadata": {
            "created_at": now(),
            "student_checkpoint": str(args.student_checkpoint.resolve()),
            "student_checkpoint_sha256": sha256(args.student_checkpoint),
            "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
            "teacher_checkpoint_sha256": sha256(args.teacher_checkpoint),
            "seed_manifest": str(args.seed_manifest.resolve()),
            "seed_manifest_sha256": sha256(args.seed_manifest),
            "max_steps": args.max_steps,
            "sample_every": args.sample_every,
            "episodes": episodes,
            "legal_student_state": "current/history fixed RGB, qpos, lagged executed student action",
            "teacher_usage": "offline training label only; absent from deployment runtime",
        },
        "train": stack_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    receipt = {
        "state": "PASSED",
        "stage": "recovery_collection_complete",
        "candidate": args.candidate,
        "rows": len(rows),
        "output": str(args.output),
        "sha256": sha256(args.output),
        "updated_at": now(),
    }
    atomic_json(args.state, receipt)
    atomic_json(args.heartbeat, {"complete": True, "rows": len(rows), "updated_at": now()})
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
