"""Closed-loop evaluator for B-core's frozen matched direct-reactive control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401
from two_three_task_manifest import TASKS, get_task

from before_we_act.direct_reactive_policy import DirectReactiveDeploymentPolicy
from before_we_act.deployment_safety import ResidualSafetyConfig
from before_we_act.evaluate_predictive_team_belief import action_inactive
from before_we_act.evaluate_temporal_history_policy import (
    EpisodeHistory,
    TemporalChunkEnsembler,
    prepare_current,
    reset_reproducibly,
)
from before_we_act.team_belief.predictive_core import TeamBeliefConfig


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_direct_reactive(
    b0h_checkpoint: str, training_checkpoint: str, device: torch.device
):
    base = torch.load(b0h_checkpoint, map_location="cpu", weights_only=False)
    training = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
    if training.get("format_version") != "before-we-act.b3-n2-training-checkpoint/1":
        raise ValueError("wrong B-core training checkpoint format")
    base_config = base["config"]
    if not str(base_config.get("policy_variant", "")).startswith("b0h_"):
        raise ValueError("direct control requires a B0-H checkpoint")
    team_values = dict(training["config"])
    for key in ("future_offsets_steps", "future_offsets_seconds"):
        if key in team_values:
            team_values[key] = tuple(team_values[key])
    team_config = TeamBeliefConfig(**team_values)
    model = DirectReactiveDeploymentPolicy(
        team_config,
        state_dim=int(base_config.get("state_dim", 9)),
        action_dim=int(base_config.get("action_dim", 8)),
        horizon=int(base_config.get("horizon", 100)),
        d_model=int(base_config.get("d_model", 384)),
        enc_layers=int(base_config.get("enc_layers", 4)),
        dec_layers=int(base_config.get("dec_layers", 7)),
        roles=int(base_config.get("roles", 4)),
        role_rank=int(base_config.get("role_rank", 32)),
        history_layers=int(base_config.get("history_layers", 2)),
        dino_model=str(base_config["dino_model"]),
    ).to(device)
    model.load_frozen_sources(base["model"], training["model"])
    model.eval()
    stats = {
        key: torch.as_tensor(value, device=device) for key, value in base["stats"].items()
    }
    config = {
        **base_config,
        "policy_variant": "b3_n3_matched_direct_reactive",
        "source_training_update": int(training["update"]),
        "source_training_checkpoint_sha256": sha256_file(training_checkpoint),
        "source_b0h_checkpoint_sha256": sha256_file(b0h_checkpoint),
    }
    return model, stats, config


@torch.no_grad()
def predict_direct(model, stats, observation, arms, history, task, device):
    global_rgb, local_rgb, qpos = prepare_current(observation, arms, stats, device)
    temporal = history.batch(qpos, task, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(global_rgb, local_rgb, **temporal)
    history.append_observation(output.current_visual_raw, qpos)
    prediction = output.prediction * stats["a_std"] + stats["a_mean"]
    base = output.base_prediction * stats["a_std"] + stats["a_mean"]
    diagnostics = {
        "direct_gate": float(output.direct_gate.float().mean().cpu()),
        "direct_residual_norm": float(
            output.direct_residual.float().norm(dim=-1).mean().cpu()
        ),
    }
    return prediction.float().cpu().numpy(), base.float().cpu().numpy(), qpos, diagnostics


def evaluate(
    b0h_checkpoint: str,
    training_checkpoint: str,
    task_name: str,
    seeds: list[int],
    device_name: str,
    max_steps: int,
):
    torch.set_num_threads(12)
    device = torch.device(device_name)
    model, stats, config = load_direct_reactive(
        b0h_checkpoint, training_checkpoint, device
    )
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
    inactivity_threshold = ResidualSafetyConfig().progress_inactivity_l2
    for seed in seeds:
        observation, _ = reset_reproducibly(env, seed)
        candidate_ensembler = TemporalChunkEnsembler(arms)
        base_ensembler = TemporalChunkEnsembler(arms)
        history = EpisodeHistory(arms)
        success = False
        inactivity = 0
        diagnostic_rows = []
        finite_actions = True
        for step in range(max_steps):
            chunks, base_chunks, normalized_qpos, diagnostics = predict_direct(
                model, stats, observation, arms, history, task_name, device
            )
            candidate_action = candidate_ensembler.append_and_select(step, chunks)
            base_action = base_ensembler.append_and_select(step, base_chunks)
            finite_actions = finite_actions and all(
                bool(np.isfinite(value).all()) for value in candidate_action.values()
            )
            if not finite_actions:
                raise FloatingPointError("direct-reactive policy produced a non-finite action")
            current_qpos = normalized_qpos * stats["q_std"] + stats["q_mean"]
            inactive, _ = action_inactive(
                candidate_action,
                current_qpos,
                arms,
                threshold=inactivity_threshold,
            )
            inactivity += int(inactive)
            diagnostic_rows.append(diagnostics)
            normalized = {
                arm: (
                    torch.as_tensor(candidate_action[f"panda-{arm}"], device=device)
                    - stats["a_mean"]
                )
                / stats["a_std"]
                for arm in arms
            }
            history.append_action(normalized)
            observation, _, terminated, truncated, info = env.step(candidate_action)
            success = bool(np.asarray(info.get("success", False)).all())
            if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        rows.append(
            {
                "seed": seed,
                "success": success,
                "steps": step + 1,
                "paired_inactivity_steps": inactivity,
                "finite_actions": finite_actions,
                "diagnostics": {
                    key: float(np.mean([row[key] for row in diagnostic_rows]))
                    for key in ("direct_gate", "direct_residual_norm")
                },
            }
        )
        print(json.dumps({"task": task_name, "mode": "direct_reactive", **rows[-1]}), flush=True)
    env.close()
    return rows, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0h-checkpoint", required=True)
    parser.add_argument("--training-checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    args = parser.parse_args()

    seed_bytes = Path(args.seed_file).read_bytes()
    manifest = json.loads(seed_bytes)
    available = [int(seed) for seed in manifest["seeds"]]
    if not 1 <= args.episodes <= len(available):
        raise ValueError("requested episode count is outside the frozen seed file")
    requested = available[: args.episodes]
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
                and row.get("mode") == "direct_reactive"
                and row.get("seed") in allowed
                and isinstance(row.get("success"), bool)
            ):
                recovered.append(
                    {
                        key: row[key]
                        for key in (
                            "seed",
                            "success",
                            "steps",
                            "paired_inactivity_steps",
                            "finite_actions",
                            "diagnostics",
                        )
                    }
                )
    recovered = list({int(row["seed"]): row for row in recovered}.values())
    complete = {int(row["seed"]) for row in recovered}
    remaining = [seed for seed in requested if seed not in complete]
    if remaining:
        rows, config = evaluate(
            args.b0h_checkpoint,
            args.training_checkpoint,
            args.task,
            remaining,
            args.device,
            args.max_steps,
        )
    else:
        rows = []
        training = torch.load(
            args.training_checkpoint, map_location="cpu", weights_only=False
        )
        config = {
            "policy_variant": "b3_n3_matched_direct_reactive",
            "source_training_update": int(training["update"]),
        }
    rows = recovered + rows
    rows.sort(key=lambda row: requested.index(int(row["seed"])))
    result = {
        "format_version": "before-we-act.b3-n3-direct-validation/1",
        "task": args.task,
        "mode": "direct_reactive",
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "rows": rows,
        "b0h_checkpoint": str(Path(args.b0h_checkpoint).resolve()),
        "b0h_checkpoint_sha256": sha256_file(args.b0h_checkpoint),
        "training_checkpoint": str(Path(args.training_checkpoint).resolve()),
        "training_checkpoint_sha256": sha256_file(args.training_checkpoint),
        "source_training_update": int(config["source_training_update"]),
        "policy_variant": config["policy_variant"],
        "seed_protocol": {
            "source": str(Path(args.seed_file).resolve()),
            "sha256": hashlib.sha256(seed_bytes).hexdigest(),
            "selection_method": manifest.get("selection_method"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result | {"rows": "saved"}), flush=True)


if __name__ == "__main__":
    main()
