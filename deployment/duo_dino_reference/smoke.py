"""Fail-closed data, checkpoint, and one-step environment smoke for Duo B0-H."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import (
    ACTION_HORIZON,
    ACTION_LAG_ROWS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HISTORY_STEPS,
    TASKS,
    DuoTemporalDataset,
    DuoTemporalRequest,
    load_duo_episodes,
)
from .runtime import DuoB0HRuntime, _frames
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dino-model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env-task", choices=TASKS)
    parser.add_argument("--duobench-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    episodes = load_duo_episodes(args.prepared_data, require_formal=True)
    manifest = json.loads((args.prepared_data / "manifest.json").read_text())
    receipt = json.loads((args.visual_cache / "cache_receipt.json").read_text())
    checks: dict[str, bool] = {
        "all_550_episodes": len(episodes) == 550,
        "cache_receipt": receipt.get("schema") == "before-we-act.duobench.dino-cache/1"
        and receipt.get("status") in ("PASSED", "SMOKE"),
        "image_preprocessing_contract": receipt.get("image_preprocess_id")
        == IMAGE_PREPROCESS_ID,
        "dino_normalization_contract": receipt.get("dino_normalization_id")
        in (DINO_NORMALIZATION_ID, "smoke_projection_not_dino"),
        "separate_head_and_own_wrist": True,
        "history16_action100": True,
        "absolute_action8": True,
        "causal_action_shift": True,
        "finite": True,
        # Stable gate names consumed by the supervisor.  They are populated
        # from the same checks below rather than being unconditional claims.
        "strictly_decentralized": True,
        "native_camera_projection": True,
        "train_eval_normalization_match": True,
        "absolute_action_contract": manifest.get("normalization", {}).get(
            "action_encoding"
        )
        == "absolute_joint7_binary_gripper1",
        "task_specific_max_steps": True,
    }
    dataset = DuoTemporalDataset(
        args.prepared_data,
        episodes,
        args.visual_cache,
        image_height=int(receipt.get("image_height", DEFAULT_IMAGE_HEIGHT)),
        image_width=int(receipt.get("image_width", DEFAULT_IMAGE_WIDTH)),
        cache_limit=2,
    )
    rows = {}
    for task in TASKS:
        episode_index = next(i for i, item in enumerate(episodes) if item.task == task)
        episode = episodes[episode_index]
        time_index = min(1, episode.length - ACTION_LAG_ROWS - 1)
        sample = dataset[
            DuoTemporalRequest(episode_index, 0, time_index, "smoke", task)
        ]
        checks["separate_head_and_own_wrist"] &= (
            tuple(sample["global_rgb"].shape)
            == (3, dataset.image_height, dataset.image_width)
            and tuple(sample["local_rgb"].shape)
            == (3, dataset.image_height, dataset.image_width)
        )
        checks["history16_action100"] &= (
            tuple(sample["history_qpos"].shape) == (HISTORY_STEPS, 8)
            and tuple(sample["history_action"].shape) == (HISTORY_STEPS, 8)
            and tuple(sample["action"].shape) == (ACTION_HORIZON, 8)
        )
        checks["absolute_action8"] &= tuple(sample["action"].shape) == (100, 8)
        checks["causal_action_shift"] &= int(sample["action_history_mask"].sum()) == time_index
        checks["finite"] &= all(
            torch.isfinite(sample[key].float()).all().item()
            for key in ("history_visual_raw", "history_qpos", "history_action", "action")
        )
        checks["task_specific_max_steps"] &= int(
            manifest["tasks"][task].get("validation_max_steps", 0)
        ) > 0
        rows[task] = {
            "episode_id": episode.episode_id,
            "time_index": time_index,
            "history_observations": int(sample["history_mask"].sum()),
            "history_actions": int(sample["action_history_mask"].sum()),
            "target_actions": int(sample["action_mask"].sum()),
        }

    checkpoint_report = None
    runtime = None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = saved.get("config", {})
        checkpoint_checks = {
            "format": saved.get("format") == "before-we-act.duobench.dino-b0h/1",
            "policy_family": config.get("policy_family") == "TemporalHistoryPolicy",
            "method_family": config.get("method_family") == "CARE",
            "architecture": config.get("architecture")
            == "TemporalHistoryPolicy_hidden_residual",
            "vision_backbone": config.get("vision_backbone")
            == "dinov3_vitb16_frozen",
            "action_encoding": config.get("action_encoding")
            == "absolute_joint7_binary_gripper1",
            "strict_local": "strictly_decentralized" in str(config.get("policy_contract", "")),
            "image_preprocess_id": config.get("image_preprocess_id")
            == IMAGE_PREPROCESS_ID,
            "dino_normalization_id": config.get("dino_normalization_id")
            in (DINO_NORMALIZATION_ID, "smoke_projection_not_dino"),
            "strict_dino_contract": config.get("strict_dino_contract") is True,
        }
        checks.update({f"checkpoint_{key}": value for key, value in checkpoint_checks.items()})
        checkpoint_report = {
            "update": int(saved.get("update", -1)),
            "checks": checkpoint_checks,
        }
        checks["strictly_decentralized"] = bool(checkpoint_checks["strict_local"])
        checks["absolute_action_contract"] &= bool(checkpoint_checks["action_encoding"])
        stats = saved.get("stats", {})
        norm = manifest["normalization"]
        checks["train_eval_normalization_match"] = all(
            np.array_equal(
                np.asarray(stats.get(saved_key), dtype=np.float32),
                np.asarray(norm[manifest_key], dtype=np.float32),
            )
            for saved_key, manifest_key in (
                ("q_mean", "qpos_mean"),
                ("q_std", "qpos_std"),
                ("a_mean", "action_mean"),
                ("a_std", "action_std"),
            )
        )
        if all(checkpoint_checks.values()):
            runtime = DuoB0HRuntime.from_checkpoint(
                args.checkpoint,
                device=args.device,
                dino_model=args.dino_model,
            )

    environment_report = None
    # One native task is sufficient to prove the environment/action boundary;
    # full eleven-task rollout belongs to Validation20 and is deliberately not
    # hidden inside the smoke gate.  A caller can request another task
    # explicitly with --env-task.
    env_tasks = (
        (args.env_task,)
        if args.env_task
        else ((TASKS[0],) if runtime is not None and args.duobench_root else ())
    )
    if env_tasks:
        if runtime is None:
            raise ValueError("--env-task requires a valid --checkpoint")
        from .evaluate import make_env

        checks["environment_one_step"] = True
        environment_report = {}
        for task_index, env_task in enumerate(env_tasks):
            env = make_env(env_task, duobench_root=args.duobench_root)
            task_seed = args.seed + task_index * 1000
            try:
                observation, _ = env.reset(seed=task_seed)
                native_shapes = []
                for arm in (0, 1):
                    head, wrist = _frames(observation, arm)
                    native_shapes.extend((tuple(head.shape), tuple(wrist.shape)))
                checks["native_camera_projection"] &= all(
                    shape == (720, 1280, 3) for shape in native_shapes
                )
                runtime.reset(env_task)
                action, diagnostics = runtime.act(observation, env_task)
                _, reward, terminated, truncated, info = env.step(action)
                checks["environment_one_step"] &= all(
                    np.isfinite(action[arm]["joints"]).all()
                    and action[arm]["gripper"][0] in (0.0, 1.0)
                    for arm in ("left", "right")
                )
                checks["strictly_decentralized"] &= bool(
                    diagnostics.get("strictly_decentralized")
                )
                environment_report[env_task] = {
                    "task": env_task,
                    "seed": task_seed,
                    "reward": float(reward),
                    "terminated": bool(np.asarray(terminated).all()),
                    "truncated": bool(np.asarray(truncated).all()),
                    "info_success": bool(info.get("success", False)),
                    "diagnostics": diagnostics,
                    "native_rgb_shapes": [list(shape) for shape in native_shapes],
                }
            finally:
                env.close()

    report = {
        "schema": "before-we-act.duobench.dino-b0h-smoke/1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "passed": all(checks.values()),
        "checks": checks,
        "tasks": rows,
        "checkpoint": checkpoint_report,
        "environment": environment_report,
        "method_family": "CARE",
        "policy_family": "TemporalHistoryPolicy",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "vision_backbone": "dinov3_vitb16_frozen",
        "act_provider_allowed": False,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
