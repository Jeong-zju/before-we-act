"""Eleven-task Validation20 evaluator for DuoBench B-core."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
)
from .bcore_runtime import DuoBcoreRuntime, validate_bcore_payload
from .data import TASKS, load_manifest
from .evaluate import make_env, _bool, _task_success
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


TASK_SCHEMA = "before-we-act.duobench.bcore-validation20-task/1"
SUMMARY_SCHEMA = "before-we-act.duobench.bcore-validation20/1"


def _json_default(value: Any) -> Any:
    """Serialize NumPy/PyTorch scalar wrappers returned by task info."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _progress(info: Mapping[str, Any], reward: Any) -> float:
    if isinstance(info, Mapping):
        try:
            stage = float(np.asarray(info.get("stage", 0)).reshape(-1)[0])
            maximum = max(float(np.asarray(info.get("max_stage", 1)).reshape(-1)[0]), 1.0)
            return float(np.clip(stage / maximum, 0.0, 1.0))
        except Exception:
            pass
    try:
        return float(np.asarray(reward).reshape(-1)[0])
    except Exception:
        return 0.0


@torch.inference_mode()
def run_episode(
    runtime: DuoBcoreRuntime,
    env: gym.Env,
    task: str,
    seed: int,
    *,
    max_steps: int,
    belief_enabled: bool = True,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    observation, reset_info = env.reset(seed=int(seed))
    runtime.reset(task)
    trace = hashlib.sha256()
    success = False
    terminated = truncated = False
    progress: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    info: Mapping[str, Any] = reset_info if isinstance(reset_info, Mapping) else {}
    for step in range(int(max_steps)):
        # Match the prepared targets at the controller boundary.  The narrow
        # RCS/Gym API Box is not the MuJoCo position-actuator ctrlrange and may
        # not alter an otherwise valid controller command.
        action, diagnostic = runtime.act(
            observation, task, belief_enabled=belief_enabled
        )
        diagnostics.append(diagnostic)
        for arm in ("left", "right"):
            trace.update(np.asarray(action[arm]["joints"], dtype=np.float32).tobytes())
            trace.update(np.asarray(action[arm]["gripper"], dtype=np.float32).tobytes())
        observation, reward, terminated, truncated, info = env.step(action)
        progress.append(_progress(info if isinstance(info, Mapping) else {}, reward))
        success = _task_success(info if isinstance(info, dict) else {}, terminated)
        if success or _bool(terminated) or _bool(truncated):
            break
    elapsed = time.perf_counter() - started
    return {
        "task": task,
        "seed": int(seed),
        "success": bool(success),
        "steps": int(step + 1),
        "max_steps": int(max_steps),
        "final_stage_progress": float(progress[-1]) if progress else 0.0,
        "max_stage_progress": float(max(progress)) if progress else 0.0,
        "action_trace_sha256": trace.hexdigest(),
        "wall_seconds": float(elapsed),
        "reset_info": dict(reset_info) if isinstance(reset_info, Mapping) else {},
        "strictly_decentralized": True,
        "per_robot_independent_inputs": True,
        "belief_enabled": bool(belief_enabled),
        "policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "action_encoding": "absolute_joint7_binary_gripper1",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "rcs_api_limits_used_for_canonicalization": False,
        "belief_diagnostics": {
            key: float(np.mean([row[key] for row in diagnostics]))
            for key in (
                "residual_gate_mean",
                "residual_norm_mean",
                "belief_reliability_mean",
                "belief_sigma_mean",
            )
        }
        if diagnostics
        else {},
    }


def evaluate_task(
    checkpoint: Path,
    prepared_data: Path,
    task: str,
    output: Path,
    *,
    episodes: int = 20,
    seed_start: int = 20260830,
    max_steps: int | None = None,
    device: str = "cuda:0",
    dino_model: str | None = None,
    duobench_root: str | Path | None = None,
    ensemble_decay: float = 0.01,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(task)
    manifest = load_manifest(prepared_data, require_formal=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_bcore_payload(saved)
    runtime = DuoBcoreRuntime.from_checkpoint(
        checkpoint,
        device=torch.device(device),
        dino_model=dino_model,
        ensemble_decay=ensemble_decay,
    )
    expected_limit = int(manifest["tasks"][task]["validation_max_steps"])
    limit = int(max_steps or expected_limit)
    if limit <= 0:
        raise ValueError(f"invalid validation horizon for {task}: {limit}")
    if limit != expected_limit:
        raise ValueError(
            f"formal validation horizon for {task} is {expected_limit}, got {limit}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".jsonl")
    recovered: dict[int, dict[str, Any]] = {}
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
                recovered[int(row["seed"])] = row
            except Exception:
                continue
    env = make_env(task, duobench_root=duobench_root)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(int(episodes)):
            seed = int(seed_start + index)
            row = recovered.get(seed)
            if row is None:
                row = run_episode(runtime, env, task, seed, max_steps=limit)
                with log_path.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
                print(json.dumps(row, default=_json_default), flush=True)
            rows.append(row)
    finally:
        env.close()
    result = {
        "schema": TASK_SCHEMA,
        "status": "complete",
        "task": task,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows]))
        if rows
        else 0.0,
        "rows": rows,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_update": int(saved.get("update", -1)),
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy",
        "benchmark_adapter": "DuoBench",
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "rcs_api_limits_used_for_canonicalization": False,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "strictly_decentralized": True,
        "strict_local": True,
        "per_robot_independent_inputs": True,
        "act_provider_allowed": False,
        "task_specific_max_steps": int(limit),
        "future_offsets_steps": list(runtime.model.team_belief_config.future_offsets_steps),
        "future_offsets_seconds": list(runtime.model.team_belief_config.future_offsets_seconds),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20260830)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dino-model")
    parser.add_argument("--duobench-root", type=Path)
    parser.add_argument("--ensemble-decay", type=float, default=0.01)
    args = parser.parse_args()
    evaluate_task(
        args.checkpoint,
        args.prepared_data,
        args.task,
        args.output,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        device=args.device,
        dino_model=args.dino_model,
        duobench_root=args.duobench_root,
        ensemble_decay=args.ensemble_decay,
    )


if __name__ == "__main__":
    main()


__all__ = ["evaluate_task", "run_episode"]
