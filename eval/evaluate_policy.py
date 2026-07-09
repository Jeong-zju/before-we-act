from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from policies.closed_loop import PolicyConfig, ClosedLoopPolicy, make_env, rollout_episode


def write_video(path: Path, frames, fps: int = 20):
    if not frames:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))

    for frame in frames:
        img = np.asarray(frame)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        # OpenCV expects BGR, most env renderers return RGB.
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def aggregate(df: pd.DataFrame):
    out = {}
    for k in [
        "success",
        "failure",
        "episode_steps",
        "return",
        "collision_count",
        "mean_force",
        "max_force",
        "mean_robot_distance",
        "min_robot_distance",
        "comm_count",
        "comm_rate",
        "timeout",
    ]:
        if k in df.columns:
            out[f"{k}_mean"] = float(df[k].mean())
            out[f"{k}_std"] = float(df[k].std(ddof=0))
    out["num_episodes"] = int(len(df))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/policy_rollouts/selective_comm")
    parser.add_argument("--mode", type=str, default="selective_comm", choices=["scripted", "no_comm", "always_comm", "selective_comm"])
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_video", type=int, default=1)
    parser.add_argument("--video_episodes", type=int, default=5)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--wam_ckpt", type=str, default="artifacts/wam/wam.pt")
    parser.add_argument("--slot_ckpt", type=str, default="artifacts/slot_encoder/slot_encoder.pt")
    parser.add_argument("--plan_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--intention_ckpt", type=str, default="artifacts/intention/intention.pt")

    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--k_exec", type=int, default=2)
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--ego_id", type=int, default=0)
    parser.add_argument("--active_codes", type=str, default="2,3,6,24,32,44,51")
    parser.add_argument("--residual_noise_std", type=float, default=0.5)
    parser.add_argument("--action_clip", type=float, default=1.0)

    parser.add_argument("--alpha_goal", type=float, default=1.0)
    parser.add_argument("--alpha_safety", type=float, default=2.0)
    parser.add_argument("--alpha_collab", type=float, default=1.0)
    parser.add_argument("--alpha_unc", type=float, default=0.5)
    parser.add_argument("--alpha_ctrl", type=float, default=0.05)
    parser.add_argument("--lambda_bits", type=float, default=2e-4)
    parser.add_argument("--lambda_delay", type=float, default=0.1)
    parser.add_argument("--lambda_redundancy", type=float, default=0.2)
    parser.add_argument("--message_uncertainty_floor", type=float, default=0.10)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    active_codes = tuple(int(x) for x in args.active_codes.split(",") if x.strip())

    cfg = PolicyConfig(
        history=args.history,
        horizon=args.horizon,
        k_exec=args.k_exec,
        num_candidates=args.num_candidates,
        ego_id=args.ego_id,
        active_codes=active_codes,
        residual_noise_std=args.residual_noise_std,
        action_clip=args.action_clip,
        alpha_goal=args.alpha_goal,
        alpha_safety=args.alpha_safety,
        alpha_collab=args.alpha_collab,
        alpha_unc=args.alpha_unc,
        alpha_ctrl=args.alpha_ctrl,
        lambda_bits=args.lambda_bits,
        lambda_delay=args.lambda_delay,
        lambda_redundancy=args.lambda_redundancy,
        message_uncertainty_floor=args.message_uncertainty_floor,
        amp_dtype=args.amp_dtype,
    )

    model_paths = {
        "wam": args.wam_ckpt,
        "slot": args.slot_ckpt,
        "plan": args.plan_ckpt,
        "intention": args.intention_ckpt,
    }

    episode_rows = []
    decision_rows = []

    for ep in tqdm(range(args.num_episodes), desc=f"rollout {args.mode}"):
        env = make_env(seed=args.seed + ep)
        policy = ClosedLoopPolicy(model_paths=model_paths, cfg=cfg, mode=args.mode, device=device)

        do_video = bool(args.render_video) and ep < args.video_episodes
        metrics, decisions, frames = rollout_episode(
            env,
            policy,
            max_steps=args.max_steps,
            render=do_video,
            width=args.width,
            height=args.height,
        )

        metrics["episode"] = ep
        metrics["mode"] = args.mode
        metrics["seed"] = args.seed + ep
        episode_rows.append(metrics)

        for d in decisions:
            d = dict(d)
            d["episode"] = ep
            d["mode"] = args.mode
            decision_rows.append(d)

        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

        if do_video:
            ok = write_video(video_dir / f"{args.mode}_episode_{ep:03d}.mp4", frames, fps=args.video_fps)
            metrics["video_written"] = float(ok)

    ep_df = pd.DataFrame(episode_rows)
    dec_df = pd.DataFrame(decision_rows)
    ep_df.to_csv(out_dir / "episode_metrics.csv", index=False)
    dec_df.to_csv(out_dir / "decision_log.csv", index=False)

    summary = aggregate(ep_df)
    summary["mode"] = args.mode
    summary["device"] = str(device)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
