"""Validation20 evaluator for the DuoBench DINO B0-H reference policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .data import TASKS, load_manifest
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID
from .runtime import DuoB0HRuntime


def make_env(task: str, *, duobench_root: str | Path | None = None) -> gym.Env:
    """Construct an unmodified native-resolution DuoBench environment."""

    if duobench_root is not None:
        root = str(Path(duobench_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    module = __import__(f"duobench.tasks.{task}", fromlist=["*"])
    config_name = "".join(part.title() for part in task.split("_")) + "EnvConfig"
    config = getattr(module, config_name)().config()
    # Keep the benchmark's native 1280x720 camera projection.  We resize the
    # returned RGB tensors in runtime.py, exactly as training does.
    try:
        from rcs._core.sim import SimConfig
        from rcs.envs.base import ControlMode, RelativeTo

        config.headless = True
        config.control_mode = ControlMode.JOINTS
        config.relative_to = RelativeTo.NONE
        config.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30)
        config.wrapper_cfg.binary_gripper = True
    except (ImportError, AttributeError):
        # A unit-test fake environment may expose a compatible config without
        # the optional RCS control enums; leaving it untouched is safer.
        pass
    return gym.make(f"duobench/{task}", cfg=config)


def _bool(value: Any) -> bool:
    try:
        return bool(np.asarray(value).all())
    except Exception:
        return bool(value)


def _task_success(info: dict[str, Any], terminated: Any) -> bool:
    if "success" in info:
        return _bool(info["success"])
    if "stage" in info and "max_stage" in info:
        try:
            return int(info["stage"]) == int(info["max_stage"])
        except (TypeError, ValueError):
            pass
    return _bool(terminated)


def _progress(info: dict[str, Any], reward: Any) -> float:
    """Return a comparable task-progress scalar for paired diagnostics.

    DuoBench exposes a discrete ``stage``/``max_stage`` pair in ``info``;
    use that when available so selector-off and CARE are compared on the same
    task scale.  A numeric reward remains a safe fallback for lightweight
    wrappers and unit-test environments.
    """

    if "stage" in info and "max_stage" in info:
        try:
            denominator = max(float(info["max_stage"]), 1.0)
            return float(np.clip(float(info["stage"]) / denominator, 0.0, 1.0))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    try:
        value = float(np.asarray(reward).mean())
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _space_map(env: gym.Env) -> Any:
    space = getattr(env, "action_space", None)
    if space is None:
        return None
    # runtime accepts a mapping keyed by left/right. Gym Dict spaces already
    # satisfy this interface; wrappers may expose ``spaces`` one level down.
    return getattr(space, "spaces", space)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays emitted by native DuoBench wrappers.

    Some task wrappers expose ``numpy.bool_`` in ``reset_info``.  The values
    are semantically ordinary JSON booleans, but the stdlib encoder does not
    handle NumPy scalar types.  Keep this conversion at the serialization
    boundary so rollout semantics and success labels remain untouched.
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@torch.inference_mode()
def run_episode(
    runtime: DuoB0HRuntime,
    env: gym.Env,
    task: str,
    seed: int,
    *,
    max_steps: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    observation, reset_info = env.reset(seed=seed)
    runtime.reset(task)
    trace = hashlib.sha256()
    progress: list[float] = []
    flags: dict[str, bool] = {}
    started = time.perf_counter()
    success = False
    terminated = truncated = False
    diagnostics: list[dict[str, Any]] = []
    for step in range(max_steps):
        # The converted demonstrations contain absolute targets outside Gym's
        # conservative API Box on several tasks.  Match training exactly and
        # do not pre-clip joints; RCS's absolute controller applies its native
        # XML/controller range.  Only the gripper is binarized in runtime.
        action, diag = runtime.act(observation, task)
        diagnostics.append(diag)
        for arm in ("left", "right"):
            trace.update(np.asarray(action[arm]["joints"], dtype=np.float32).tobytes())
            trace.update(np.asarray(action[arm]["gripper"], dtype=np.float32).tobytes())
        observation, reward, terminated, truncated, info = env.step(action)
        progress.append(_progress(info if isinstance(info, dict) else {}, reward))
        if isinstance(info, dict):
            for key, value in info.items():
                try:
                    flags[str(key)] = bool(flags.get(str(key), False) or _bool(value))
                except Exception:
                    continue
        success = _task_success(info if isinstance(info, dict) else {}, terminated)
        if success or _bool(terminated) or _bool(truncated):
            break
    elapsed = time.perf_counter() - started
    max_stage = None
    final_stage = None
    if isinstance(info, dict):
        try:
            final_stage = int(info.get("stage"))
            max_stage = int(info.get("max_stage"))
        except (TypeError, ValueError):
            pass
    return {
        "task": task,
        "seed": int(seed),
        "success": bool(success),
        "steps": int(step + 1),
        "max_steps": int(max_steps),
        "final_stage": final_stage,
        "max_stage": max_stage,
        "final_stage_progress": float(progress[-1]) if progress else 0.0,
        "max_stage_progress": float(max(progress)) if progress else 0.0,
        "action_trace_sha256": trace.hexdigest(),
        "wall_seconds": float(elapsed),
        "reset_info": reset_info if isinstance(reset_info, dict) else {},
        "ever_info_flags": flags,
        "strictly_decentralized": True,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "method_family": "CARE",
        "policy_family": "TemporalHistoryPolicy",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "vision_backbone": "dinov3_vitb16_frozen",
        "act_provider_allowed": False,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "rcs_api_limits_used_for_canonicalization": False,
    }


def require_matched_chunk_weighting(
    action_loss_decay: float, ensemble_decay: float, *, tolerance: float = 2.0
) -> None:
    """Refuse to score a policy through an ensembler its training disagrees with.

    Training weights chunk position ``t`` by ``exp(-t / action_loss_decay)``;
    the deployed ensembler weights a chunk read at offset ``age`` by
    ``exp(-ensemble_decay * age)``. The two curves therefore have scales
    ``action_loss_decay`` and ``1 / ensemble_decay``, and when they disagree the
    executed action is drawn largely from positions training barely supervised.

    A run at scale 16 against an ensembler at scale 100 produced 1.36% success
    while its logged action loss -- the same weighted mean -- looked healthy, so
    nothing surfaced the mismatch until a closed-loop sweep had been paid for.
    A decay of zero is uniform supervision and constrains nothing.
    """

    decay = float(action_loss_decay or 0.0)
    if decay <= 0.0:
        return
    if ensemble_decay <= 0.0:
        raise ValueError("ensemble decay must be positive to compare weightings")
    ensemble_scale = 1.0 / float(ensemble_decay)
    ratio = max(decay, ensemble_scale) / min(decay, ensemble_scale)
    if ratio > tolerance:
        raise ValueError(
            "training and evaluation weight the action chunk on different "
            f"scales: training exp(-t/{decay:g}) against ensembling "
            f"exp(-age/{ensemble_scale:g}), a factor of {ratio:.1f}. The "
            "executed action would come mostly from chunk positions training "
            "barely supervised. Retrain with --action-loss-decay 0, or set "
            f"--ensemble-decay {1.0 / decay:g} to match."
        )


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
    require_matched_chunk_weighting(
        saved.get("config", {}).get("action_loss_decay", 0.0), ensemble_decay
    )
    runtime = DuoB0HRuntime.from_checkpoint(
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
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    env = make_env(task, duobench_root=duobench_root)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(episodes):
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
        "schema": "before-we-act.duobench.dino-b0h-validation20-task/1",
        "status": "complete",
        "task": task,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])) if rows else 0.0,
        "rows": rows,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_update": int(saved.get("update", -1)),
        "method_family": "CARE",
        "policy_family": "TemporalHistoryPolicy",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "vision_backbone": "dinov3_vitb16_frozen",
        "act_provider_allowed": False,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "rcs_api_limits_used_for_canonicalization": False,
        "policy_contract": saved.get("config", {}).get("policy_contract"),
        "action_encoding": "absolute_joint7_binary_gripper1",
        "strictly_decentralized": True,
        "native_camera_contract": "native_1280x720_render_each_view_resized_224x224",
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
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
