"""Trace a strict wrist-only LPD rollout for first-deviation analysis.

The policy receives exactly the same local observation as formal evaluation.
The shoe position, TCP distances, and external render are analysis-only
privileged evidence; they are never passed to the policy.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import imageio.v2 as imageio
import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401
from evaluate_stereo_act import load, predict_all, reset_reproducibly


def as_np(value):
    value = np.asarray(value)
    return value[0] if value.ndim > 1 and value.shape[0] == 1 else value


def pipeline_stage(shoe_y: float) -> tuple[int, str]:
    """Stages follow the canonical LPD planner's 3 -> 2 -> 1 -> 0 relay."""
    if shoe_y > 0.60:
        return 0, "robot3_pick_and_handoff_to_2"
    if shoe_y > 0.00:
        return 1, "robot2_pick_and_handoff_to_1"
    if shoe_y > -0.60:
        return 2, "robot1_pick_and_handoff_to_0"
    if shoe_y > -1.00:
        return 3, "robot0_pick_and_deliver"
    return 4, "goal_region"


def depth_preview(depth):
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if not len(finite):
        return np.zeros((*depth.shape, 3), np.uint8)
    if np.median(finite) > 10:
        depth = depth * 0.001
    scaled = np.clip((depth - 0.05) / 1.95, 0, 1)
    # Neutral grayscale preserves metric near/far ordering without pretending
    # the map is RGB.
    return np.repeat((255 * (1 - scaled))[..., None].astype(np.uint8), 3, axis=2)


def resize(image, height, width):
    yy = np.linspace(0, image.shape[0] - 1, height).astype(int)
    xx = np.linspace(0, image.shape[1] - 1, width).astype(int)
    return image[np.ix_(yy, xx)]


def capture_panel(env, obs, label: str):
    external = as_np(env.render())
    # Four 160-pixel wrist tiles define the common panel width.
    external = resize(external, 180, 640)
    rgb_tiles, depth_tiles = [], []
    for arm in range(4):
        sensor = obs["sensor_data"][f"head_camera_agent{arm}"]
        rgb_tiles.append(resize(as_np(sensor["rgb"]), 120, 160))
        depth_tiles.append(resize(depth_preview(as_np(sensor["depth"])), 120, 160))
    return np.concatenate((external, np.concatenate(rgb_tiles, axis=1), np.concatenate(depth_tiles, axis=1)), axis=0)


def snapshot(env, obs, step, actions):
    raw = env.unwrapped
    shoe = as_np(raw.shoe.pose.p).astype(float)
    goal = as_np(raw.goal_region.pose.p).astype(float)
    tcps = [as_np(agent.tcp.pose.p).astype(float) for agent in raw.agent.agents]
    qposes = [as_np(obs["agent"][f"panda-{arm}"]["qpos"]).astype(float) for arm in range(4)]
    stage, stage_name = pipeline_stage(float(shoe[1]))
    return {
        "step": int(step),
        "stage": stage,
        "stage_name": stage_name,
        "shoe_xyz": shoe.round(6).tolist(),
        "distance_to_goal_xy": float(np.linalg.norm(shoe[:2] - goal[:2])),
        "tcp_shoe_distance": [float(np.linalg.norm(tcp - shoe)) for tcp in tcps],
        "gripper_qpos": [float(q[-2:].mean()) for q in qposes],
        "executed_action_norm": [float(np.linalg.norm(actions[f"panda-{arm}"])) for arm in range(4)],
        "executed_gripper_target": [float(np.asarray(actions[f"panda-{arm}"])[-2:].mean()) for arm in range(4)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--capture-stride", type=int, default=30)
    args = parser.parse_args()
    out = Path(args.output); frames = out / "frames"; frames.mkdir(parents=True, exist_ok=True)
    # Must be set before the task environment is constructed.  The policy's
    # 640x480 contract is intentionally enforced by the model itself.
    os.environ["ROBOFACTORY_WRIST_WIDTH"] = "640"
    os.environ["ROBOFACTORY_WRIST_HEIGHT"] = "480"
    import wrist_camera_patch  # noqa: F401
    device = torch.device(args.device)
    model, stats, _ = load(args.checkpoint, device)
    model.eval()
    env = gym.make(
        "LongPipelineDelivery-rf",
        config="/workspace/RoboFactory/robofactory/configs/table/long_pipeline_delivery.yaml",
        obs_mode="rgbd", control_mode="pd_joint_pos", render_mode="sensors",
        reward_mode="dense", sim_backend="cpu", sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    obs, _ = reset_reproducibly(env, args.seed)
    histories = [[] for _ in range(4)]
    trace, transitions = [], []
    previous_stage = None
    success = False
    try:
        for step in range(args.max_steps):
            chunks = predict_all(model, stats, obs, (0, 1, 2, 3), device)
            actions = {}
            for arm in range(4):
                histories[arm].append(chunks[arm])
                candidates = [chunk[step - start] for start, chunk in enumerate(histories[arm]) if step - start < len(chunk)]
                weights = np.exp(-0.01 * np.arange(len(candidates) - 1, -1, -1)); weights /= weights.sum()
                actions[f"panda-{arm}"] = np.sum(np.asarray(candidates) * weights[:, None], axis=0)
            record = snapshot(env, obs, step, actions)
            if previous_stage is None or record["stage"] != previous_stage:
                transitions.append(record.copy())
                previous_stage = record["stage"]
            if args.capture_stride > 0 and step % args.capture_stride == 0:
                imageio.imwrite(frames / f"step_{step:04d}.png", capture_panel(env, obs, f"step {step}"))
            trace.append(record)
            obs, _, terminated, truncated, info = env.step(actions)
            success = bool(np.asarray(info.get("success", False)).all())
            if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        final = snapshot(env, obs, step + 1, actions)
        if args.capture_stride > 0 and step % args.capture_stride:
            imageio.imwrite(frames / f"step_{step + 1:04d}.png", capture_panel(env, obs, f"step {step + 1}"))
    finally:
        env.close()
    payload = {
        "protocol": "formal policy input is wrist-only RGB-D plus own qpos; trace fields/external frames are analysis-only",
        "checkpoint": str(Path(args.checkpoint).resolve()), "seed": args.seed,
        "success": success, "steps": step + 1, "stage_transitions": transitions,
        "final": final, "trace": trace,
    }
    (out / "trace.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: payload[key] for key in ("seed", "success", "steps", "stage_transitions", "final")}, indent=2))


if __name__ == "__main__":
    main()
