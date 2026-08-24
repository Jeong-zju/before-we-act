#!/usr/bin/env python3
"""Run ACT checkpoints in the RoboFactory simulator and write Validation20 evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import gymnasium as gym
import numpy as np
import torch
import yaml
import robofactory.tasks  # noqa: F401 - registers the six Gym environments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stereo_core.train_act import ACT

TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)

CARE_TASK_MAX_STEPS = {
    "lift_barrier": 500,
    "camera_alignment": 1500,
    "long_pipeline_delivery": 1500,
    "take_photo": 1500,
    "pass_shoe": 500,
    "place_food": 500,
}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _action_codec(stats_root: Path, task: str, arm: int):
    payload = json.loads((stats_root / task / "training_manifest.json").read_text())
    cfg = payload["action"]["codec"]["config"]
    low = np.asarray(cfg["low"], np.float32)[arm * 8:(arm + 1) * 8]
    high = np.asarray(cfg["high"], np.float32)[arm * 8:(arm + 1) * 8]
    return low, high


def _success(info) -> bool:
    value = info.get("success", False) if isinstance(info, dict) else False
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    return bool(value)


def _make_env(config: Path, sim_backend: str):
    with config.open() as handle:
        env_id = yaml.safe_load(handle)["task_name"] + "-rf"
    return gym.make(
        env_id,
        config=str(config),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=sim_backend,
        sensor_configs={"shader_pack": "default", "width": 320, "height": 240},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
    )


@torch.no_grad()
def _predict(model, obs, arm: int, stats, codec, device):
    image = obs["sensor_data"][f"head_camera_agent{arm}"]["rgb"]
    image = image.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
    qpos = obs["agent"][f"panda-{arm}"]["qpos"][:, :9].to(device=device)
    qpos = (qpos - torch.as_tensor(stats["q_mean"], device=device)) / torch.as_tensor(stats["q_std"], device=device)
    pred, _, _ = model(image, qpos)
    # ACT predicts standardized physical joint targets. Undo standardization
    # once, then apply the task/arm affine action codec once.
    action = pred[0].float().cpu().numpy()
    encoded = action * stats["a_std"] + stats["a_mean"]
    low, high = codec
    encoded = np.clip(encoded, -1.0, 1.0)
    return (low + 0.5 * (encoded + 1.0) * (high - low)).astype(np.float32)


class TemporalChunkEnsembler:
    """Aggregate every still-valid ACT prediction at the current timestep."""

    def __init__(self, agent_count: int, decay: float):
        self.decay = float(decay)
        self.histories = [[] for _ in range(agent_count)]

    def append_and_select(self, step: int, chunks: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
        actions = {}
        for arm, history in enumerate(self.histories):
            chunk = chunks[arm]
            history.append((step, chunk))
            history[:] = [(start, value) for start, value in history if step - start < len(value)]
            candidates = np.asarray([value[step - start] for start, value in history])
            weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1))
            weights /= weights.sum()
            actions[f"panda-{arm}"] = np.sum(candidates * weights[:, None], axis=0).astype(np.float32)
        return actions


def _validate_formal_contract(args, config: dict) -> None:
    if not args.formal_six_task:
        return
    if args.episodes != 20 or args.max_steps_profile != "care":
        raise ValueError("formal Validation20 requires 20 episodes and the CARE task horizon profile")
    episode_end = args.episode_end if args.episode_end is not None else args.episodes
    if not 0 <= args.episode_start < episode_end <= args.episodes:
        raise ValueError("formal Validation20 episode shard must lie inside [0, 20)")
    if args.seed != 20260820 or args.sim_backend != "cpu" or abs(args.temporal_ensemble_decay - 0.01) > 1e-12:
        raise ValueError("formal six-task Validation20 requires seed=20260820, cpu simulation, decay=0.01")
    expected = {"state_dim": 9, "action_dim": 8, "horizon": 100, "d_model": 384, "enc_layers": 4, "dec_layers": 7}
    for key, value in expected.items():
        if int(config.get(key, -1)) != value:
            raise ValueError(f"formal ACT checkpoint mismatch for {key}: {config.get(key)!r} != {value}")
    if config.get("vision_backbone", "resnet18") != "resnet18":
        raise ValueError("formal ACT Validation20 requires the resnet18 checkpoint")


def evaluate(args) -> dict:
    torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    _validate_formal_contract(args, config)
    model = ACT(
        int(config["state_dim"]), int(config["action_dim"]), int(config["horizon"]),
        int(config["d_model"]), int(config["enc_layers"]), int(config["dec_layers"]),
        vision_backbone=config.get("vision_backbone", "resnet18"),
        dino_model=config.get("dino_model", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = {key: np.asarray(value, np.float32) for key, value in checkpoint["stats"].items()}
    stats_root = Path(args.stats_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    episode_end = args.episode_end if args.episode_end is not None else args.episodes
    for task in args.task or TASKS:
        max_steps = (
            CARE_TASK_MAX_STEPS[task]
            if args.max_steps_profile == "care"
            else args.max_steps
        )
        config_path = Path(args.config_root) / f"{task}.yaml"
        # All current RoboFactory six-task configs use two local Panda policies;
        # reading the config keeps this evaluator valid if that changes.
        with config_path.open() as handle:
            agent_count = len(yaml.safe_load(handle)["agents"])
        result_path = output / f"{task}.json"
        previous = {}
        if result_path.is_file():
            try:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        episode_map = {
            int(row["episode"]): row
            for row in previous.get("episodes_detail", [])
            if not row.get("error")
        }
        for episode in range(args.episode_start, episode_end):
            if episode in episode_map:
                continue
            env = None
            row = {"episode": episode, "seed": args.seed + episode, "success": False, "steps": 0}
            try:
                env = _make_env(config_path, args.sim_backend)
                obs, _ = env.reset(seed=args.seed + episode)
                ensembler = TemporalChunkEnsembler(agent_count, args.temporal_ensemble_decay)
                for step in range(max_steps):
                    chunks = {
                        arm: _predict(
                            model, obs, arm, stats, _action_codec(stats_root, task, arm), args.device
                        )
                        for arm in range(agent_count)
                    }
                    action = ensembler.append_and_select(step, chunks)
                    obs, _, terminated, truncated, info = env.step(action)
                    row["success"] = _success(info)
                    row["steps"] = step + 1
                    if row["success"] or bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                        break
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None:
                    env.close()
                torch.cuda.empty_cache()
            episode_map[episode] = row
            episodes = [episode_map[key] for key in sorted(episode_map)]
            successes = sum(bool(item["success"]) for item in episodes)
            summary = {"baseline": "act", "task": task, "successes": successes, "episodes": len(episodes),
                       "target_episodes": args.episodes, "success_rate": successes / len(episodes),
                       "evaluator_protocol": "act_temporal_chunk_ensemble_v3_affine_codec",
                       "max_steps": max_steps, "max_steps_profile": args.max_steps_profile,
                       "episode_range": [args.episode_start, episode_end],
                       "temporal_ensemble_decay": args.temporal_ensemble_decay,
                       "shader_pack": "default", "camera_size": [320, 240],
                       "episodes_detail": episodes, "updated_at": datetime.now(timezone.utc).isoformat()}
            _atomic_json(result_path, summary)
        summaries[task] = json.loads(result_path.read_text(encoding="utf-8"))
    errors = [row for task in summaries.values() for row in task["episodes_detail"] if row.get("error")]
    report = {"baseline": "act", "status": "failed" if errors else "complete", "episodes_per_task": args.episodes, "tasks": summaries,
              "evaluator_protocol": "act_temporal_chunk_ensemble_v3_affine_codec", "shader_pack": "default",
              "max_steps_profile": args.max_steps_profile,
              "max_steps_by_task": {task: CARE_TASK_MAX_STEPS[task] for task in summaries} if args.max_steps_profile == "care" else {task: args.max_steps for task in summaries},
              "macro_success_rate": float(np.mean([row["success_rate"] for row in summaries.values()]))}
    _atomic_json(output / "summary.json", report)
    if errors:
        raise RuntimeError(f"{len(errors)} closed-loop episodes failed; see per-task JSON")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats-root", required=True)
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-steps-profile", choices=("uniform", "care"), default="uniform")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--sim-backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--formal-six-task", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
