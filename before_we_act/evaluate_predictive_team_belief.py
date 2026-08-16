"""Closed-loop evaluator for predictive team belief and its temporal control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401
from two_three_task_manifest import TASKS, get_task

from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.evaluate_temporal_history_policy import (
    EpisodeHistory,
    TemporalChunkEnsembler,
    load_model as load_temporal_history,
    prepare_current,
    reset_reproducibly,
)
from before_we_act.team_belief.predictive_core import TeamBeliefConfig


def load_team_belief(checkpoint: str, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "before-we-act.b3-n2-deployment-checkpoint/1":
        raise ValueError("wrong N2 deployment checkpoint format")
    config = saved["config"]
    belief_values = dict(config["n2_config"])
    for key in ("future_offsets_steps", "future_offsets_seconds"):
        if key in belief_values:
            belief_values[key] = tuple(belief_values[key])
    team_belief_config = TeamBeliefConfig(**belief_values)
    model = PredictiveTeamBeliefPolicy(
        team_belief_config,
        state_dim=int(config.get("state_dim", 9)),
        action_dim=int(config.get("action_dim", 8)),
        horizon=int(config.get("horizon", 100)),
        d_model=int(config.get("d_model", 384)),
        enc_layers=int(config.get("enc_layers", 4)),
        dec_layers=int(config.get("dec_layers", 7)),
        roles=int(config.get("roles", 4)),
        role_rank=int(config.get("role_rank", 32)),
        history_layers=int(config.get("history_layers", 2)),
        dino_model=str(config["dino_model"]),
        include_teacher=False,
    ).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in saved["stats"].items()
    }
    return model, stats, config


@torch.no_grad()
def predict_team_belief(model, stats, observation, arms, history, task, device):
    global_rgb, local_rgb, qpos = prepare_current(observation, arms, stats, device)
    temporal = history.batch(qpos, task, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(global_rgb, local_rgb, **temporal)
    history.append_observation(output.current_visual_raw, qpos)
    denormalized = output.prediction * stats["a_std"] + stats["a_mean"]
    diagnostics = {
        "gate": float(output.residual_gate.float().mean().cpu()),
        "reliability": float(output.belief.reliability.float().mean().cpu()),
        "sigma": float(output.belief.sigma.float().mean().cpu()),
        "events": int(output.belief.event_mask.sum().cpu()),
    }
    return denormalized.float().cpu().numpy(), qpos, diagnostics


@torch.no_grad()
def predict_temporal_history(model, stats, observation, arms, history, task, device):
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
    return denormalized.float().cpu().numpy(), qpos, {}


def evaluate(checkpoint, task_name, seeds, device_name, max_steps, mode):
    torch.set_num_threads(12)
    device = torch.device(device_name)
    if mode == "n2":
        model, stats, config = load_team_belief(checkpoint, device)
        predictor = predict_team_belief
    else:
        model, stats, config = load_temporal_history(checkpoint, device)
        predictor = predict_temporal_history
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
        inactivity = 0
        diagnostic_rows = []
        for step in range(max_steps):
            chunks, normalized_qpos, diagnostics = predictor(
                model, stats, observation, arms, history, task_name, device
            )
            action = ensembler.append_and_select(step, chunks)
            current_qpos = normalized_qpos * stats["q_std"] + stats["q_mean"]
            joint_changes = [
                float(
                    np.linalg.norm(
                        np.asarray(action[f"panda-{arm}"])[:7]
                        - current_qpos[index, :7].float().cpu().numpy()
                    )
                )
                for index, arm in enumerate(arms)
            ]
            inactivity += int(all(value < 0.02 for value in joint_changes))
            if diagnostics:
                diagnostic_rows.append(diagnostics)
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
            if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        row = {
            "seed": seed,
            "success": success,
            "steps": step + 1,
            "paired_inactivity_steps": inactivity,
        }
        if diagnostic_rows:
            row["belief_diagnostics"] = {
                key: float(np.mean([value[key] for value in diagnostic_rows]))
                for key in ("gate", "reliability", "sigma", "events")
            }
        rows.append(row)
        print(json.dumps({"task": task_name, "mode": mode, **row}), flush=True)
    env.close()
    return rows, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("n2", "b0h"), required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    args = parser.parse_args()
    raw = Path(args.seed_file).read_bytes()
    manifest = json.loads(raw)
    requested = [int(seed) for seed in manifest["seeds"][: args.episodes]]
    recovered = []
    if args.resume_log and Path(args.resume_log).is_file():
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task") == args.task and row.get("mode") == args.mode and row.get("seed") in requested:
                recovered.append({
                    key: row[key]
                    for key in ("seed", "success", "steps", "paired_inactivity_steps")
                    if key in row
                } | ({"belief_diagnostics": row["belief_diagnostics"]} if "belief_diagnostics" in row else {}))
    recovered = list({row["seed"]: row for row in recovered}.values())
    remaining = [seed for seed in requested if seed not in {row["seed"] for row in recovered}]
    if remaining:
        rows, config = evaluate(
            args.checkpoint, args.task, remaining, args.device, args.max_steps, args.mode
        )
    else:
        rows = []
        config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    rows = recovered + rows
    rows.sort(key=lambda row: requested.index(row["seed"]))
    result = {
        "task": args.task,
        "mode": args.mode,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "paired_inactivity_steps": sum(int(row["paired_inactivity_steps"]) for row in rows),
        "steps": sum(int(row["steps"]) for row in rows),
        "rows": rows,
        "policy_variant": config.get("policy_variant"),
        "seed_protocol": {
            "source": args.seed_file,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "selection_method": manifest.get("selection_method"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result | {"rows": "saved"}), flush=True)


if __name__ == "__main__":
    main()
