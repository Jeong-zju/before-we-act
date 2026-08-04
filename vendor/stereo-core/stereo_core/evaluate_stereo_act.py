"""Held-out evaluation for strict local-wrist RGB-D Stereo-ACT-cross_relbias."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401
from train_stereo_act import StereoACT
from two_three_task_manifest import TASKS, get_task


def reset_reproducibly(env, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def _cpu_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def update_lpd_diagnostic(env, arms, step, diagnostic):
    """Read-only rollout audit; none of these privileged values reach the policy."""
    raw = env.unwrapped
    shoe_xyz = _cpu_numpy(raw.shoe.pose.p).reshape(-1, 3)[0]
    goal_xyz = _cpu_numpy(raw.goal_region.pose.p).reshape(-1, 3)[0]
    diagnostic["shoe_min_y"] = min(diagnostic["shoe_min_y"], float(shoe_xyz[1]))
    diagnostic["shoe_max_z"] = max(diagnostic["shoe_max_z"], float(shoe_xyz[2]))
    diagnostic["min_goal_xy_distance"] = min(
        diagnostic["min_goal_xy_distance"], float(np.linalg.norm(shoe_xyz[:2] - goal_xyz[:2]))
    )
    diagnostic["final_shoe_xyz"] = [float(v) for v in shoe_xyz]
    diagnostic["final_goal_xy_distance"] = float(np.linalg.norm(shoe_xyz[:2] - goal_xyz[:2]))
    for arm in arms:
        key = str(arm); robot = raw.agent.agents[arm]
        tcp_xyz = _cpu_numpy(robot.tcp.pose.p).reshape(-1, 3)[0]
        diagnostic["min_tcp_shoe_distance"][key] = min(
            diagnostic["min_tcp_shoe_distance"][key], float(np.linalg.norm(tcp_xyz - shoe_xyz))
        )
        grasped = bool(_cpu_numpy(robot.is_grasping(raw.shoe)).reshape(-1)[0])
        if grasped:
            diagnostic["ever_grasped"][key] = True
            if diagnostic["first_grasp_step"][key] is None:
                diagnostic["first_grasp_step"][key] = int(step + 1)
            diagnostic["last_grasp_step"][key] = int(step + 1)


def finalize_lpd_diagnostic(diagnostic, success):
    # LPD's expert chain is robot 3 -> 2 -> 1 -> 0 -> goal.
    failure_robot, failure_stage = None, None
    if not success:
        for arm in (3, 2, 1, 0):
            if not diagnostic["ever_grasped"][str(arm)]:
                failure_robot = arm
                failure_stage = f"robot_{arm}_never_established_grasp"
                break
        if failure_robot is None:
            failure_robot, failure_stage = 0, "robot_0_final_delivery_or_release"
    diagnostic["failure_robot"] = failure_robot
    diagnostic["failure_stage"] = failure_stage
    for key, value in diagnostic["min_tcp_shoe_distance"].items():
        diagnostic["min_tcp_shoe_distance"][key] = float(value)
    return diagnostic


def load(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg, raw_stats = saved["config"], saved["stats"]
    # The first 10k preflight checkpoints predate explicit evaluator metadata;
    # infer only architecture defaults, never an observation input or label.
    common = dict(horizon=cfg.get("horizon", 100), d_model=cfg.get("d_model", 384),
                  enc_layers=cfg.get("enc_layers", 4), dec_layers=cfg.get("dec_layers", 7),
                  dino_model=cfg.get("dino_model", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
                  defm_model=cfg.get("defm_model", "defm_vit_s14"))
    # A missing policy_variant denotes the original Stereo-ACT checkpoint.
    # Only historical checkpoints explicitly tagged mode=sync may infer the
    # Sync-ARCA variant; never default a vanilla Stereo checkpoint to ARCA.
    variant = cfg.get("policy_variant") or ("stereo_sync_arca" if cfg.get("mode") == "sync" else "stereo")
    state_dim, action_dim = cfg.get("state_dim", len(raw_stats["q_mean"])), cfg.get("action_dim", len(raw_stats["a_mean"]))
    if variant == "stereo_ffn_moe":
        from stereo_decoder_variants import StereoFFNMoE
        model = StereoFFNMoE(state_dim, action_dim, experts=cfg.get("experts", 4), **common)
    elif variant == "stereo_arca":
        from stereo_decoder_variants import StereoARCA
        model = StereoARCA(state_dim, action_dim, roles=cfg.get("experts", 4), role_rank=cfg.get("role_rank", 32), **common)
    elif variant in ("stereo_history_arca", "stereo_pair_belief"):
        from history_pair_model import StereoHistoryARCA
        model = StereoHistoryARCA(state_dim, action_dim, roles=cfg.get("experts", 4),
                                  role_rank=cfg.get("role_rank", 32),
                                  history=cfg.get("history", 32),
                                  history_stride=cfg.get("history_stride", 4),
                                  role_dim=cfg.get("teacher_role_dim", 64), **common)
    elif variant == "stereo_pair_adapter":
        from stereo_decoder_variants import StereoPAIRAdapter
        model = StereoPAIRAdapter(state_dim, action_dim, roles=cfg.get("experts", 4),
                                  role_rank=cfg.get("role_rank", 32), **common)
    elif variant == "stereo_pair_route":
        from pair_route_model import StereoPAIRRoute
        model = StereoPAIRRoute(state_dim, action_dim, roles=cfg.get("experts", 4),
                                role_rank=cfg.get("role_rank", 32), **common)
    elif variant == "stereo_predictability_pair":
        from pair_predictive_model import StereoPredictabilityPAIR
        model = StereoPredictabilityPAIR(
            state_dim,
            action_dim,
            roles=cfg.get("experts", 4),
            role_rank=cfg.get("role_rank", 32),
            event_dim=cfg.get("event_dim", 128),
            **common,
        )
    elif variant == "stereo_pair_residual":
        from pair_residual_model import StereoPAIRResidual
        model = StereoPAIRResidual(
            state_dim,
            action_dim,
            roles=cfg.get("experts", 4),
            role_rank=cfg.get("role_rank", 32),
            event_dim=cfg.get("event_dim", 96),
            residual_rank=cfg.get("residual_rank", 32),
            joint_training=cfg.get("joint_training", False),
            **common,
        )
    elif variant == "stereo_sync_arca":
        from stereo_decoder_variants import StereoSyncARCA
        model = StereoSyncARCA(state_dim, action_dim, roles=cfg.get("experts", 4),
                               role_rank=cfg.get("role_rank", 32), phases=cfg.get("clusters", 8), **common)
    elif variant in ("stereo_msa_sticky", "stereo_msa_interaction_graph"):
        from stereo_decoder_variants import StereoMSA
        model = StereoMSA(state_dim, action_dim, roles=cfg.get("experts", 4),
                          role_rank=cfg.get("role_rank", 32), **common)
    else:
        model = StereoACT(state_dim, action_dim, **common)
    model = model.to(device); model.load_state_dict(saved["model"]); model.eval()
    return model, {key: torch.as_tensor(value, device=device) for key, value in raw_stats.items()}, cfg


@torch.no_grad()
def predict_all(model, stats, obs, arms, device):
    rgb, depth, qposes = [], [], []
    for arm in arms:
        sensor = obs["sensor_data"][f"head_camera_agent{arm}"]
        image, metric_depth = np.asarray(sensor["rgb"]), np.asarray(sensor["depth"])
        qpos = np.asarray(obs["agent"][f"panda-{arm}"]["qpos"])
        rgb.append(image[0] if image.ndim == 4 else image)
        depth.append(metric_depth[0] if metric_depth.ndim == 4 else metric_depth)
        qposes.append(qpos[0] if qpos.ndim == 2 else qpos)
    rgb = torch.as_tensor(np.stack(rgb)).permute(0, 3, 1, 2).float().div_(255).to(device)
    depth = torch.as_tensor(np.stack(depth)).permute(0, 3, 1, 2).to(device)
    qpos = (torch.as_tensor(np.stack(qposes)).float().to(device) - stats["q_mean"]) / stats["q_std"]
    chunks = model(rgb, depth, qpos)[0]
    return (chunks * stats["a_std"] + stats["a_mean"]).float().cpu().numpy()


def evaluate_slice(checkpoint, task_name, seeds, device_name, max_steps):
    torch.set_num_threads(12); device = torch.device(device_name)
    model, stats, cfg = load(checkpoint, device)
    os.environ["ROBOFACTORY_WRIST_WIDTH"], os.environ["ROBOFACTORY_WRIST_HEIGHT"] = "640", "480"
    import wrist_camera_patch  # noqa: F401
    spec, arms = get_task(task_name), get_task(task_name)["agents"]
    env = gym.make(spec["env_id"], config=f"/workspace/RoboFactory/{spec['config']}", obs_mode="rgbd",
                   control_mode="pd_joint_pos", render_mode="sensors", reward_mode="dense", sim_backend="cpu",
                   sensor_configs=dict(shader_pack="default"), human_render_camera_configs=dict(shader_pack="default"),
                   viewer_camera_configs=dict(shader_pack="default"))
    rows = []
    for seed in seeds:
        obs, _ = reset_reproducibly(env, seed); histories = [[] for _ in arms]; success = False
        lpd_diagnostic = None
        if task_name == "long_pipeline_delivery":
            lpd_diagnostic = {
                "ever_grasped": {str(arm): False for arm in arms},
                "first_grasp_step": {str(arm): None for arm in arms},
                "last_grasp_step": {str(arm): None for arm in arms},
                "min_tcp_shoe_distance": {str(arm): float("inf") for arm in arms},
                "shoe_min_y": float("inf"), "shoe_max_z": float("-inf"),
                "min_goal_xy_distance": float("inf"), "final_shoe_xyz": None,
                "final_goal_xy_distance": None,
            }
        if hasattr(model, "reset_history"):
            model.reset_history()
        for step in range(max_steps):
            chunks = predict_all(model, stats, obs, arms, device); actions = {}
            for local, arm in enumerate(arms):
                histories[local].append(chunks[local])
                candidates = [chunk[step - start] for start, chunk in enumerate(histories[local]) if step - start < len(chunk)]
                weights = np.exp(-0.01 * np.arange(len(candidates) - 1, -1, -1)); weights /= weights.sum()
                actions[f"panda-{arm}"] = np.sum(np.asarray(candidates) * weights[:, None], axis=0)
            obs, _, terminated, truncated, info = env.step(actions)
            if lpd_diagnostic is not None:
                update_lpd_diagnostic(env, arms, step, lpd_diagnostic)
            success = bool(np.asarray(info.get("success", False)).all())
            if bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        row = {"seed": seed, "success": success, "steps": step + 1}
        if lpd_diagnostic is not None:
            row["lpd_diagnostic"] = finalize_lpd_diagnostic(lpd_diagnostic, success)
        rows.append(row)
        print(json.dumps({"task": task_name, **rows[-1]}), flush=True)
    env.close(); return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed-file", required=True); parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--devices", default="0"); parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default=None,
                        help="Recover completed per-seed JSON rows from an interrupted evaluator stdout log.")
    args = parser.parse_args(); raw = Path(args.seed_file).read_bytes(); seed_manifest = json.loads(raw)
    seeds = [int(seed) for seed in seed_manifest["seeds"]]
    if len(seeds) != args.episodes or len(set(seeds)) != len(seeds):
        raise ValueError("seed file must contain exactly the requested unique seeds")
    recovered = []
    if args.resume_log:
        allowed = set(seeds)
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (row.get("task") == args.task and isinstance(row.get("seed"), int)
                    and row["seed"] in allowed and isinstance(row.get("success"), bool)
                    and isinstance(row.get("steps"), int)):
                recovered.append({key: row[key] for key in ("seed", "success", "steps")})
        by_seed = {row["seed"]: row for row in recovered}
        if len(by_seed) != len(recovered):
            raise ValueError("resume log contains duplicate completed seeds; refuse ambiguous aggregation")
        recovered = [by_seed[seed] for seed in seeds if seed in by_seed]
        seeds = [seed for seed in seeds if seed not in by_seed]
        print(json.dumps({"task": args.task, "recovered_rows": len(recovered),
                          "remaining_rows": len(seeds)}), flush=True)
    device_ids = [int(item) for item in args.devices.split(",") if item]
    workers = min(max(args.workers, 1), len(seeds))
    if not seeds:
        rows = recovered
    elif workers == 1:
        rows = evaluate_slice(args.checkpoint, args.task, seeds, f"cuda:{device_ids[0]}", args.max_steps)
    else:
        parts = [seeds[index::workers] for index in range(workers)]
        payload = [(args.checkpoint, args.task, part, f"cuda:{device_ids[index % len(device_ids)]}", args.max_steps)
                   for index, part in enumerate(parts)]
        mp.set_start_method("spawn", force=True)
        with mp.Pool(workers) as pool:
            rows = [row for part in pool.starmap(evaluate_slice, payload) for row in part]
    rows = recovered + rows
    if len(rows) != args.episodes or len({row["seed"] for row in rows}) != args.episodes:
        raise RuntimeError("resume aggregation did not reconstruct exactly the frozen seed set")
    rows.sort(key=lambda row: row["seed"])
    successes = sum(row["success"] for row in rows)
    cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    model_contract = {key: cfg.get(key) for key in (
        "vision_backbone", "dino_model", "defm_model", "horizon", "enc_layers", "dec_layers", "d_model",
        "camera_width", "camera_height", "patch_grid", "fusion_layers", "depth_storage_unit", "arms",
    )}
    result = {"task": args.task, "env_id": get_task(args.task)["env_id"],
              "checkpoint": str(Path(args.checkpoint).resolve()), "model_contract": model_contract,
              "episodes": len(rows), "successes": successes,
              "success_rate": successes / len(rows), "rows": rows,
              "camera": "strictly local single wrist RGB-D on matching panda_hand; no global/peer/right-camera input",
              "seed_protocol": {"method": seed_manifest["selection_method"], "source": args.seed_file,
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "training_seed_overlap": seed_manifest["training_seed_overlap"]}}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result | {"rows": "saved"}), flush=True)


if __name__ == "__main__":
    main()
