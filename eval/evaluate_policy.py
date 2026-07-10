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


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _overlay_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _overlay_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _overlay_decision(overlay_data, frame_idx: int, frame_count: int):
    if not overlay_data:
        return None
    decisions = overlay_data.get("decisions", []) if isinstance(overlay_data, dict) else overlay_data
    if decisions is None or len(decisions) == 0:
        return None

    if isinstance(overlay_data, dict):
        k_exec = overlay_data.get("k_exec")
    else:
        k_exec = None

    if k_exec is not None and _overlay_int(k_exec, 0) > 0:
        decision_idx = frame_idx // _overlay_int(k_exec, 1)
    elif frame_count > 1 and len(decisions) > 1:
        decision_idx = int(round(frame_idx * (len(decisions) - 1) / (frame_count - 1)))
    else:
        decision_idx = 0
    decision_idx = int(np.clip(decision_idx, 0, len(decisions) - 1))
    row = dict(decisions[decision_idx])
    row["_decision_idx"] = decision_idx
    return row


def _draw_text_with_shadow(img, text: str, org, font_scale: float, color, thickness: int = 1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_video_overlay(img, frame_idx: int, frame_count: int, overlay_data):
    if not overlay_data:
        return img

    row = _overlay_decision(overlay_data, frame_idx, frame_count)
    if row is None:
        return img

    if isinstance(overlay_data, dict):
        mode = row.get("mode") or overlay_data.get("mode", "")
    else:
        mode = row.get("mode", "")
    scenario = overlay_data.get("scenario", "") if isinstance(overlay_data, dict) else ""
    episode = overlay_data.get("episode", row.get("episode", "")) if isinstance(overlay_data, dict) else row.get("episode", "")
    decision_step = _overlay_int(row.get("t", row.get("_decision_idx", 0)))
    env_step = frame_idx
    trigger_0 = _overlay_int(row.get("trigger_robot_0", row.get("trigger", 0)))
    trigger_1 = _overlay_int(row.get("trigger_robot_1", 0))
    comm_count_step = _overlay_int(row.get("comm_count_step", trigger_0 + trigger_1))
    if comm_count_step <= 0 and _overlay_int(row.get("trigger", 0)) > 0:
        comm_count_step = 1

    lines = [
        f"mode: {mode}",
        f"scenario: {scenario}",
        f"episode: {episode}",
        f"env_step: {env_step}  decision_step: {decision_step}",
        f"trigger_robot_0: {trigger_0}  trigger_robot_1: {trigger_1}",
        f"comm_count_step: {comm_count_step}",
        (
            "robot_0 G_no/G_comm/delta_G: "
            f"{_overlay_float(row.get('G_no_comm_robot_0')):.2f}/"
            f"{_overlay_float(row.get('G_comm_robot_0')):.2f}/"
            f"{_overlay_float(row.get('delta_G_robot_0')):.2f}"
        ),
        (
            "robot_1 G_no/G_comm/delta_G: "
            f"{_overlay_float(row.get('G_no_comm_robot_1')):.2f}/"
            f"{_overlay_float(row.get('G_comm_robot_1')):.2f}/"
            f"{_overlay_float(row.get('delta_G_robot_1')):.2f}"
        ),
        (
            "uncertainty_robot_0/1: "
            f"{_overlay_float(row.get('uncertainty_robot_0')):.2f}/"
            f"{_overlay_float(row.get('uncertainty_robot_1')):.2f}"
        ),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    line_gap = 20
    pad = 8
    max_width = 0
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_w)

    h, w = img.shape[:2]
    panel_w = min(w - 8, max_width + 2 * pad)
    panel_h = min(h - 8, len(lines) * line_gap + 2 * pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (4, 4), (4 + panel_w, 4 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)

    y = 4 + pad + 12
    for line in lines:
        _draw_text_with_shadow(img, line, (4 + pad, y), font_scale, (255, 255, 255), thickness)
        y += line_gap

    status = "COMM" if comm_count_step > 0 else "NO COMM"
    status_color = (0, 255, 0) if comm_count_step > 0 else (255, 220, 0)
    status_scale = 1.05
    status_thickness = 3
    (status_w, status_h), status_base = cv2.getTextSize(status, font, status_scale, status_thickness)
    sx = max(8, w - status_w - 18)
    sy = 28 + status_h
    cv2.rectangle(
        img,
        (sx - 10, sy - status_h - 10),
        (sx + status_w + 10, sy + status_base + 10),
        (0, 0, 0),
        -1,
    )
    _draw_text_with_shadow(img, status, (sx, sy), status_scale, status_color, status_thickness)
    return img


def _write_video_once(path: Path, frames, fps: int, overlay_data=None) -> bool:
    if not frames:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    codec = "mp4v" if path.suffix.lower() == ".mp4" else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        writer.release()
        return False

    frame_count = len(frames)
    for frame_idx, frame in enumerate(frames):
        img = np.asarray(frame)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3].copy()
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        img = _draw_video_overlay(img, frame_idx, frame_count, overlay_data)
        # OpenCV expects BGR, most env renderers return RGB.
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def write_video(path: Path, frames, fps: int = 20, overlay_data=None):
    ok = _write_video_once(path, frames, fps=fps, overlay_data=overlay_data)
    if ok or path.suffix.lower() == ".avi":
        return ok
    return _write_video_once(path.with_suffix(".avi"), frames, fps=fps, overlay_data=overlay_data)


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
        "communication_required_rate",
        "occlusion_rate",
        "force_violation_rate",
        "mean_object_goal_distance",
        "final_object_goal_distance",
        "progress_mean",
        "progress_final",
        "comm_count",
        "comm_rate",
        "comm_count_robot_0",
        "comm_count_robot_1",
        "comm_rate_robot_0",
        "comm_rate_robot_1",
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
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--occlusion_prob", type=float, default=None)
    parser.add_argument("--teammate_delay_steps", type=int, default=None)
    parser.add_argument("--asymmetric_obstacle", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--narrow_width_scale", type=float, default=None)
    parser.add_argument("--blocked_passage_prob", type=float, default=None)
    parser.add_argument("--false_belief_prob", type=float, default=None)
    parser.add_argument("--belief_code_error_prob", type=float, default=0.0)
    parser.add_argument("--belief_residual_noise_std", type=float, default=0.0)
    parser.add_argument("--belief_uncertainty_boost", type=float, default=0.0)
    parser.add_argument("--scripted_mix", type=float, default=0.0)
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
        belief_code_error_prob=args.belief_code_error_prob,
        belief_residual_noise_std=args.belief_residual_noise_std,
        belief_uncertainty_boost=args.belief_uncertainty_boost,
        scripted_mix=args.scripted_mix,
        amp_dtype=args.amp_dtype,
    )

    scenario_kwargs = {}
    for key in [
        "occlusion_prob",
        "teammate_delay_steps",
        "asymmetric_obstacle",
        "narrow_width_scale",
        "blocked_passage_prob",
        "false_belief_prob",
    ]:
        value = getattr(args, key)
        if value is not None:
            scenario_kwargs[key] = value

    model_paths = {
        "wam": args.wam_ckpt,
        "slot": args.slot_ckpt,
        "plan": args.plan_ckpt,
        "intention": args.intention_ckpt,
    }

    episode_rows = []
    decision_rows = []

    for ep in tqdm(range(args.num_episodes), desc=f"rollout {args.mode}"):
        env = make_env(seed=args.seed + ep, scenario=args.scenario, **scenario_kwargs)
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
        metrics["scenario"] = args.scenario
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
            overlay_data = {
                "decisions": decisions,
                "mode": args.mode,
                "scenario": args.scenario,
                "episode": ep,
                "k_exec": args.k_exec,
            }
            ok = write_video(
                video_dir / f"{args.mode}_episode_{ep:03d}.mp4",
                frames,
                fps=args.video_fps,
                overlay_data=overlay_data,
            )
            metrics["video_written"] = float(ok)

    ep_df = pd.DataFrame(episode_rows)
    dec_df = pd.DataFrame(decision_rows)
    ep_df.to_csv(out_dir / "episode_metrics.csv", index=False)
    dec_df.to_csv(out_dir / "decision_log.csv", index=False)

    summary = aggregate(ep_df)
    summary["mode"] = args.mode
    summary["scenario"] = args.scenario
    summary["device"] = str(device)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
