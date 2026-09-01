"""Closed-loop B0-H probe/Validation20 adapter for native BiCoord."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np

from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    HISTORY_STEPS,
    TASKS,
    VALIDATION_EPISODES,
    VALIDATION_MAX_STEPS,
    SMOKE_INTERFACE_STEPS,
    validate_native_gripper_vector,
)
from .runtime import B0HRuntime
from .stage_common import (
    artifact,
    assert_common_paths,
    atomic_json,
    common_parser,
    publish_result,
    read_json,
    require_stage_result,
    sha256_file,
)


def _checkpoint_from_dependencies(args: argparse.Namespace) -> Path:
    # Probe/Validation20 are formal gates.  A smoke checkpoint must never
    # enter their candidate set, even when both stage results coexist.
    stage = (
        "b0h_smoke_train"
        if args.operation == "smoke-closed-loop"
        else "b0h_formal"
    )
    dep = require_stage_result(
        args.run,
        stage,
        config_sha256=args.config_sha256,
    )
    root = (args.run / "artifacts" / stage).resolve()
    candidates: list[Path] = []
    for row in dep.get("artifacts", []):
        if not isinstance(row, Mapping) or row.get("kind") not in {
            "checkpoint",
            "b0h_checkpoint",
            "training_checkpoint",
        }:
            continue
        path = Path(str(row.get("path", "")))
        if not path.is_absolute():
            path = args.run / path
        try:
            path = path.expanduser().resolve(strict=True)
        except FileNotFoundError:
            continue
        if (
            root in path.parents
            and path.is_file()
            and sha256_file(path) == row.get("sha256")
        ):
            candidates.append(path)
    verified = list(dict.fromkeys(candidates))

    # An explicit checkpoint field can disambiguate multiple hashed evidence
    # files, but it cannot introduce an un-hashed or out-of-layout path.
    declared: list[Path] = []
    for key in ("checkpoint", "final_checkpoint"):
        raw = dep.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = args.run / path
        try:
            path = path.expanduser().resolve(strict=True)
        except FileNotFoundError:
            continue
        if path in verified:
            declared.append(path)
    declared = list(dict.fromkeys(declared))
    selected = declared if declared else verified
    if len(selected) != 1:
        raise RuntimeError(
            f"{stage} must publish exactly one hash-verified B0-H checkpoint "
            f"below {root}; found {selected}"
        )
    return selected[0]


def _stage_name(operation: str) -> str:
    """Map an evaluator operation to its immutable supervisor stage name."""

    return {
        "smoke-closed-loop": "b0h_smoke_closed_loop",
        "probe": "b0h_probe",
        "validation20": "b0h_validation20",
    }.get(str(operation), "b0h_evaluate")


def _progress_paths(run: Path, operation: str, task: str) -> tuple[Path, Path]:
    """Return progress files isolated from every other policy evaluator."""

    root = run / "progress" / _stage_name(operation)
    return root / f"{task}.jsonl", root / f"{task}.receipt.json"


def _reset_progress(progress: Path, receipt: Path) -> None:
    """Remove an uncommitted task trace before replaying its worker.

    Evaluator workers do not support resuming an episode from a partial trace.
    Clearing both evidence files makes a supervisor retry fail closed instead
    of appending new rows to a prior attempt's JSONL (or exposing a stale
    receipt while the replacement worker is still running).
    """

    progress.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)


def _bench_env(benchmark_root: Path, task: str, seed: int):
    """Construct the official RoboTwin task without changing its source."""
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    env = None
    try:
        import importlib
        import yaml

        module = importlib.import_module(f"envs.{task}")
        cls = getattr(module, task)
        env = cls()
        config_path = benchmark_root / "task_config" / "demo_clean.yml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("demo_clean.yml is not a mapping")
        config.update(
            {
                "task_name": task,
                "task_config": "demo_clean",
                "seed": int(seed),
                "now_ep_num": 0,
                "is_test": True,
                "eval_mode": True,
                "render_freq": 0,
                "save_data": False,
                "save_path": str(benchmark_root / "data"),
                "need_plan": False,
                "dual_arm": True,
                "data_type": {
                    "rgb": True,
                    "qpos": True,
                    "endpose": False,
                    "depth": False,
                    "pointcloud": False,
                    "third_view": False,
                },
            }
        )
        embodiment = config.get("embodiment", ["aloha-agilex"])
        emb_cfg_path = benchmark_root / "task_config" / "_embodiment_config.yml"
        emb_cfg = yaml.safe_load(emb_cfg_path.read_text(encoding="utf-8"))
        if not isinstance(emb_cfg, dict) or not embodiment:
            raise ValueError("BiCoord embodiment config is missing")
        if len(embodiment) != 1:
            raise ValueError("the frozen BiCoord CARE run requires one shared embodiment")
        robot_rel = emb_cfg[embodiment[0]]["file_path"]
        robot_root = (benchmark_root / robot_rel).resolve() if not os.path.isabs(str(robot_rel)) else Path(robot_rel)
        config["left_robot_file"] = str(robot_root)
        config["right_robot_file"] = str(robot_root)
        robot_config = yaml.safe_load((robot_root / "config.yml").read_text(encoding="utf-8"))
        config["left_embodiment_config"] = robot_config
        config["right_embodiment_config"] = robot_config
        config["dual_arm_embodied"] = True
        env.setup_demo(**config)
        # The pinned task source expects contact point two on ``003_plate``.
        # Apply the separately audited, contact-only metadata overlay in
        # memory; no benchmark task/model/normalization source is changed.
        from .asset_runtime import apply_configured_task_overlay

        env._bicoord_asset_overlay = apply_configured_task_overlay(env, task)
        return env
    except Exception:
        try:
            if env is not None:
                env.close_env()
        except Exception:
            pass
        raise


def _progress_value(env: Any) -> float:
    try:
        value = float(getattr(env, "stage_eval_score"))
        return float(np.clip(value, 0.0, 1.0))
    except Exception:
        return 0.0


def _asset_overlay_evidence(env: Any, task: str) -> dict[str, Any]:
    """Copy the immutable runtime asset proof into an episode receipt.

    ``_bench_env`` applies the audited compatibility overlay immediately after
    constructing the official task.  Keeping that proof on the returned
    episode row (which is itself embedded in the hashed progress receipt)
    lets the supervisor verify that validation used the same run-local overlay
    as seed discovery.  Do not synthesize a PASS when an adapter forgot to
    publish its evidence; the resulting row is intentionally incomplete and
    is rejected by the supervisor's affected-task gate.
    """

    observed = getattr(env, "_bicoord_asset_overlay", None)
    if isinstance(observed, Mapping):
        return {"asset_overlay": copy.deepcopy(dict(observed))}
    return {}


def _run_episode(
    runtime: B0HRuntime,
    benchmark_root: Path,
    task: str,
    seed: int,
    max_steps: int,
    progress_path: Path,
) -> dict[str, Any]:
    env = _bench_env(benchmark_root, task, seed)
    asset_overlay = _asset_overlay_evidence(env, task)
    runtime.reset()
    observation = env.get_obs()
    trace = hashlib.sha256(); success = False; steps = 0
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    final_progress = 0.0
    prediction_oob = 0
    plan_oob = 0
    with progress_path.open("a", encoding="utf-8") as stream:
        try:
            for step in range(int(max_steps)):
                chunks = runtime.act(observation, task)
                diagnostics = dict(runtime.last_prediction_diagnostics)
                prediction_oob += int(diagnostics.get("prediction_gripper_oob_count", 0))
                plan_oob += int(diagnostics.get("ensemble_plan_gripper_oob_count", 0))
                current = np.concatenate((chunks[0][0], chunks[1][0]), axis=0).astype(np.float32)
                if current.shape != (ACTION_DIM * 2,) or not np.isfinite(current).all():
                    raise RuntimeError(f"invalid B0-H action at {task}/{seed}/{step}: {current.shape}")
                validate_native_gripper_vector(
                    current.reshape(2, ACTION_DIM),
                    context="B0-H native controller command",
                )
                trace.update(current.tobytes())
                env.take_action(current)
                observation = env.get_obs()
                steps = step + 1
                try:
                    success = bool(getattr(env, "eval_success", False) or env.check_success())
                except Exception:
                    success = bool(getattr(env, "eval_success", False))
                final_progress = _progress_value(env)
                row = {
                    "task": task,
                    "seed": int(seed),
                    "episode": int(seed),
                    "step": steps,
                    "max_steps": int(max_steps),
                    "progress": _progress_value(env),
                    "success": bool(success),
                    "prediction_gripper_oob_count": int(diagnostics.get("prediction_gripper_oob_count", 0)),
                    "ensemble_plan_gripper_oob_count": int(diagnostics.get("ensemble_plan_gripper_oob_count", 0)),
                    "executed_gripper_oob_count": 0,
                    "action_clipped": False,
                    "policy_output_clipping": False,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                if steps == 1 or steps % 25 == 0 or success:
                    stream.flush()
                if success:
                    break
        finally:
            try:
                env.close_env()
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass
    return {
        "task": task,
        "seed": int(seed),
        "success": bool(success),
        "steps": int(steps),
        "max_steps": int(max_steps),
        "progress": final_progress,
        "action_trace_sha256": trace.hexdigest(),
        "strictly_decentralized": True,
        "per_arm_independent_inputs": True,
        "prediction_gripper_oob_count": prediction_oob,
        "ensemble_plan_gripper_oob_count": plan_oob,
        "executed_gripper_oob_count": 0,
        "policy_output_clipping": False,
        "action_clipping": False,
        "state_clipping": False,
        "gripper_reparameterization": False,
        **asset_overlay,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args)
    task = getattr(args, "task", None)
    if task is None:
        raise ValueError("task is required for the supervisor task queue")
    if task not in TASKS:
        raise ValueError(task)
    episodes = int(getattr(args, "episodes", 1))
    expected_episodes = (
        VALIDATION_EPISODES if args.operation == "validation20" else 1
    )
    if episodes != expected_episodes:
        raise ValueError(
            f"{args.operation} requires {expected_episodes} episodes, got {episodes}"
        )
    requested_max_steps = int(getattr(args, "max_steps", 0) or 0)
    expected_max_steps = (
        SMOKE_INTERFACE_STEPS
        if args.operation == "smoke-closed-loop"
        else VALIDATION_MAX_STEPS[task]
    )
    max_steps = requested_max_steps or expected_max_steps
    if max_steps != expected_max_steps:
        raise ValueError(f"{task}: expected max_steps={expected_max_steps} for {args.operation}, got {max_steps}")
    checkpoint = _checkpoint_from_dependencies(args)
    runtime = B0HRuntime.from_checkpoint(checkpoint, dino_model=args.dino_model, device=os.environ.get("BICOORD_EVAL_DEVICE", "cuda:0"))
    seed_value = getattr(args, "seed_start", None)
    seed_start = int(seed_value if seed_value is not None else 20260920 + TASKS.index(task) * 100)
    stage_name = _stage_name(args.operation)
    progress, progress_receipt = _progress_paths(args.run, args.operation, task)
    # This evaluator has no episode-level resume contract: a supervisor retry
    # replays the complete task worker.  Start every operation (including the
    # formal probe/Validation20 paths) from an empty JSONL so rows from an
    # interrupted attempt cannot be re-hashed into the replacement receipt.
    _reset_progress(progress, progress_receipt)
    rows: list[dict[str, Any]] = []
    for index in range(episodes):
        rows.append(
            _run_episode(
                runtime,
                args.benchmark_repo,
                task,
                seed_start + index,
                max_steps,
                progress,
            )
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    progress_sha256 = sha256_file(progress)
    atomic_json(progress_receipt, {
        "schema": "before-we-act.bicoord.validation-progress/1",
        "status": "PASSED",
        "stage": stage_name,
        "stage_operation": args.operation,
        "task": task,
        "episodes": len(rows),
        "completed": len(rows),
        "max_steps": max_steps,
        "rows_path": str(progress.resolve()),
        "rows_sha256": progress_sha256,
        "rows": rows,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "rollout_steps": [int(row["steps"]) for row in rows],
        "smoke_interface_steps": (
            SMOKE_INTERFACE_STEPS
            if args.operation == "smoke-closed-loop"
            else None
        ),
        "policy_output_clipping": False,
        "action_clipping": False,
        "state_clipping": False,
        "gripper_reparameterization": False,
        "executed_gripper_oob_count": sum(
            int(row.get("executed_gripper_oob_count", 0)) for row in rows
        ),
        "prediction_gripper_oob_count": sum(
            int(row.get("prediction_gripper_oob_count", 0)) for row in rows
        ),
        "ensemble_plan_gripper_oob_count": sum(
            int(row.get("ensemble_plan_gripper_oob_count", 0)) for row in rows
        ),
    })
    result = publish_result(
        args,
        stage=stage_name,
        include_model_contract=True,
        artifacts=[artifact(progress, kind="validation_progress"), artifact(progress_receipt, kind="progress_receipt")],
        task=task,
        episodes=len(rows),
        completed=len(rows),
        successes=sum(int(row["success"]) for row in rows),
        max_steps=max_steps,
        rollout_steps=[int(row["steps"]) for row in rows],
        progress_receipt=str(progress_receipt.resolve()),
        progress_receipt_sha256=sha256_file(progress_receipt),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        action_encoding=ACTION_ENCODING,
        history_steps=HISTORY_STEPS,
        action_horizon=ACTION_HORIZON,
        strict_local_inputs=True,
        smoke_interface_steps=SMOKE_INTERFACE_STEPS if args.operation == "smoke-closed-loop" else None,
        policy_output_clipping=False,
        action_clipping=False,
        state_clipping=False,
        gripper_reparameterization=False,
        executed_gripper_oob_count=sum(int(row.get("executed_gripper_oob_count", 0)) for row in rows),
        prediction_gripper_oob_count=sum(int(row.get("prediction_gripper_oob_count", 0)) for row in rows),
        ensemble_plan_gripper_oob_count=sum(int(row.get("ensemble_plan_gripper_oob_count", 0)) for row in rows),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("smoke-closed-loop", "probe", "validation20"))
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--record-progress", action="store_true")
    parser.add_argument("--seed-start", type=int)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
