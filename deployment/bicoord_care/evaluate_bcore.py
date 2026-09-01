"""Closed-loop B-core/TUNE evaluator with paired local-input enforcement."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from .config import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_ENCODING,
    SMOKE_INTERFACE_STEPS,
    TASKS,
    VALIDATION_EPISODES,
    VALIDATION_MAX_STEPS,
    validate_native_gripper_vector,
)
from .stage_common import artifact, assert_common_paths, atomic_json, common_parser, publish_result, require_stage_result, sha256_file
from .seed_discovery import SEED_MANIFEST_SCHEMA


def _is_smoke_operation(args: argparse.Namespace) -> bool:
    return str(getattr(args, "operation", "")) in {"smoke-closed-loop", "smoke", "smoke-paired"}


def _stage_name(operation: str) -> str:
    return {
        "smoke-closed-loop": "bcore_smoke_closed_loop",
        "validation20": "bcore_validation20",
    }.get(str(operation), "bcore_evaluate")


def _progress_paths(run: Path, operation: str, task: str) -> tuple[Path, Path]:
    root = run / "progress" / _stage_name(operation)
    return root / f"{task}.jsonl", root / f"{task}.receipt.json"


def _reset_progress(progress: Path, receipt: Path) -> None:
    """Discard partial/stale task evidence before a complete worker replay."""

    progress.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)


def _checkpoint(args: argparse.Namespace) -> Path:
    smoke = _is_smoke_operation(args)
    stage = "bcore_smoke_train" if smoke else "bcore_select"
    dep = require_stage_result(args.run, stage, config_sha256=args.config_sha256)
    root = (args.run / "artifacts" / stage).expanduser().resolve()
    candidates: list[Path] = []
    for row in dep.get("artifacts", []):
        if not isinstance(row, Mapping):
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
            and row.get("kind")
            in {"checkpoint", "deployment_checkpoint", "bcore_checkpoint"}
            and sha256_file(path) == row.get("sha256")
        ):
            candidates.append(path)
    verified = list(dict.fromkeys(candidates))
    declared: list[Path] = []
    for key in ("checkpoint", "deployment_checkpoint", "selected_checkpoint"):
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
        # A declared field may disambiguate hashed artifacts, but can never
        # introduce an unhashed checkpoint or escape the stage namespace.
        if path in verified:
            declared.append(path)
    declared = list(dict.fromkeys(declared))
    values = declared if declared else verified
    if len(values) != 1:
        raise RuntimeError(
            f"{stage} must publish exactly one hash-verified B-core deployment "
            f"checkpoint below {root}; found {values}"
        )
    return values[0]


def _runtime(checkpoint: Path, args: argparse.Namespace):
    # bcore_runtime is supplied by the benchmark adapter.  Import lazily so a
    # data-only audit does not pull CUDA/SAPIEN into its process.
    try:
        from .bcore_runtime import BicoordBcoreRuntime  # preferred spelling
    except ImportError:
        try:
            from .bcore_runtime import BiCoordBcoreRuntime as BicoordBcoreRuntime
        except ImportError as error:
            raise RuntimeError("bicoord_care.bcore_runtime is required for physical B-core validation") from error
    return BicoordBcoreRuntime.from_checkpoint(checkpoint, device=os.environ.get("BICOORD_EVAL_DEVICE", "cuda:0"), dino_model=str(args.dino_model))


def _official_seeds(args: argparse.Namespace, task: str, *, count: int) -> list[int]:
    """Load and hash-check the expert-valid seed manifest; never synthesize it."""

    smoke = _is_smoke_operation(args)
    stage = "seed_discovery_smoke" if smoke else "seed_discovery"
    expected_count = 1 if smoke else VALIDATION_EPISODES
    dependency = require_stage_result(args.run, stage, config_sha256=args.config_sha256)
    path = Path(str(dependency.get("seed_manifest", "")))
    digest = dependency.get("seed_manifest_sha256")
    if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
        raise RuntimeError("seed discovery manifest is missing or changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != SEED_MANIFEST_SCHEMA
        or value.get("policy_independent") is not True
        or value.get("status") != "PASSED"
        or value.get("stage") != stage
        or value.get("seed_role") != "validation"
        or value.get("seed_bucket") != 0
        or value.get("episodes_per_task") != expected_count
        or dependency.get("stage") != stage
        or dependency.get("seed_role") != "validation"
        or dependency.get("seed_bucket") != 0
        or dependency.get("episodes_per_task") != expected_count
    ):
        raise RuntimeError("seed manifest is not the frozen policy-independent contract")
    if count != expected_count:
        raise RuntimeError(f"{stage} requires exactly {expected_count} seeds")
    rows = value.get("valid_seeds", {}).get(task)
    if not isinstance(rows, list) or len(rows) < count or len(set(rows)) != len(rows):
        raise RuntimeError(f"expert-valid seed coverage is incomplete for {task}")
    return [int(seed) for seed in rows[:count]]


def _make_env(root: Path, task: str, seed: int):
    root = root.expanduser().resolve(strict=True)
    root_text = str(root)
    sys.path[:] = [root_text] + [entry for entry in sys.path if entry != root_text]
    cached = sys.modules.get("envs")
    cached_file = getattr(cached, "__file__", None)
    if cached_file is not None and root not in Path(cached_file).expanduser().resolve().parents:
        for name in list(sys.modules):
            if name == "envs" or name.startswith("envs."):
                del sys.modules[name]
    import importlib, yaml
    module = importlib.import_module(f"envs.{task}")
    module_file = getattr(module, "__file__", None)
    if module_file is not None and root not in Path(module_file).expanduser().resolve().parents:
        raise RuntimeError(f"BiCoord task import escaped benchmark root: {module_file}")
    cls = getattr(module, task); env = cls()
    cfg = yaml.safe_load((root / "task_config" / "demo_clean.yml").read_text())
    cfg.update({"task_name": task, "task_config": "demo_clean", "seed": int(seed), "now_ep_num": 0, "is_test": True, "eval_mode": True, "render_freq": 0, "save_data": False, "save_path": str(root / "data"), "need_plan": False, "dual_arm": True, "data_type": {"rgb": True, "qpos": True, "endpose": False, "depth": False, "pointcloud": False, "third_view": False}})
    emb = cfg.get("embodiment", ["aloha-agilex"]); emb_cfg = yaml.safe_load((root / "task_config" / "_embodiment_config.yml").read_text()); robot = Path(emb_cfg[emb[0]]["file_path"]); robot = robot if robot.is_absolute() else root / robot
    robot_conf = yaml.safe_load((robot / "config.yml").read_text()); cfg.update({"left_robot_file": str(robot), "right_robot_file": str(robot), "left_embodiment_config": robot_conf, "right_embodiment_config": robot_conf, "dual_arm_embodied": True})
    try:
        env.setup_demo(**cfg)
        from .asset_runtime import apply_configured_task_overlay

        env._bicoord_asset_overlay = apply_configured_task_overlay(env, task)
        return env
    except BaseException:
        try:
            env.close_env()
        except Exception:
            try: env.close()
            except Exception: pass
        raise


def _asset_overlay_evidence(env: Any) -> dict[str, Any]:
    """Return a detached runtime-overlay proof for a hashed episode row."""

    observed = getattr(env, "_bicoord_asset_overlay", None)
    if isinstance(observed, Mapping):
        return {"asset_overlay": copy.deepcopy(dict(observed))}
    return {}


def _run_episode(runtime: Any, root: Path, task: str, seed: int, limit: int, progress: Path, belief_enabled: bool = True) -> dict[str, Any]:
    env = _make_env(root, task, seed); asset_overlay = _asset_overlay_evidence(env); runtime.reset(task); observation = env.get_obs(); digest = hashlib.sha256(); success = False; steps = 0; progress_value = 0.0
    prediction_oob = 0; plan_oob = 0
    progress.parent.mkdir(parents=True, exist_ok=True)
    try:
        with progress.open("a") as stream:
            for step in range(limit):
                action, diagnostic = runtime.act(observation, task, belief_enabled=belief_enabled)
                prediction_oob += int(diagnostic.get("reference_chunk_gripper_oob_count", 0))
                plan_oob += int(diagnostic.get("reference_plan_gripper_oob_count", 0))
                # Runtime must return one controller command per arm, not a
                # single concatenated vector or peer-conditioned tensor.
                if not isinstance(action, Mapping) or set(action) != {0, 1}:
                    raise ValueError(f"B-core runtime action keys differ: {action!r}")
                rows = []
                for arm in (0, 1):
                    value = action[arm]
                    if isinstance(value, Mapping):
                        joints = np.asarray(value.get("joints"), np.float32).reshape(-1); grip = np.asarray(value.get("gripper"), np.float32).reshape(-1)
                        value = np.r_[joints, grip]
                    value = np.asarray(value, np.float32).reshape(-1)
                    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all(): raise ValueError("B-core local action must be finite [7]")
                    rows.append(value); digest.update(value.tobytes())
                command = np.concatenate(rows)
                validate_native_gripper_vector(
                    np.stack(rows), context="B-core native controller command"
                )
                env.take_action(command); observation = env.get_obs(); steps = step + 1
                try: success = bool(getattr(env, "eval_success", False) or env.check_success())
                except Exception: success = bool(getattr(env, "eval_success", False))
                progress_value = float(np.clip(getattr(env, "stage_eval_score", 0.0), 0.0, 1.0))
                stream.write(json.dumps({"task": task, "seed": seed, "step": steps, "max_steps": limit, "progress": progress_value, "success": success, "belief_enabled": belief_enabled, "diagnostic": diagnostic, "action_clipped": False, "policy_output_clipping": False, "executed_gripper_oob_count": 0, "prediction_gripper_oob_count": int(diagnostic.get("reference_chunk_gripper_oob_count", 0)), "ensemble_plan_gripper_oob_count": int(diagnostic.get("reference_plan_gripper_oob_count", 0))}, default=str, sort_keys=True) + "\n"); stream.flush()
                if success: break
    finally:
        try: env.close_env()
        except Exception:
            try: env.close()
            except Exception: pass
    return {"task": task, "seed": int(seed), "success": success, "steps": steps, "max_steps": limit, "progress": progress_value, "action_trace_sha256": digest.hexdigest(), "strictly_decentralized": True, "per_arm_independent_inputs": True, "belief_enabled": belief_enabled, "policy_output_clipping": False, "executed_gripper_oob_count": 0, "prediction_gripper_oob_count": prediction_oob, "ensemble_plan_gripper_oob_count": plan_oob, **asset_overlay}


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args)
    task = getattr(args, "task", None)
    if task not in TASKS: raise ValueError(f"invalid B-core task: {task}")
    episodes = int(getattr(args, "episodes", 1)); expected = VALIDATION_EPISODES if args.operation == "validation20" else 1
    if episodes != expected: raise ValueError(f"{args.operation} requires {expected} episodes")
    expected_limit = SMOKE_INTERFACE_STEPS if args.operation == "smoke-closed-loop" else VALIDATION_MAX_STEPS[task]
    limit = int(getattr(args, "max_steps", 0) or expected_limit)
    if limit != expected_limit: raise ValueError(f"{task}: max steps for {args.operation} must be {expected_limit}")
    checkpoint = _checkpoint(args); runtime = _runtime(checkpoint, args)
    if args.operation == "validation20":
        seeds = _official_seeds(args, task, count=episodes)
    else:
        seed_start = int(getattr(args, "seed_start", None) or (20260930 + TASKS.index(task) * 100))
        seeds = [seed_start + i for i in range(episodes)]
    stage = _stage_name(args.operation)
    progress, receipt = _progress_paths(args.run, args.operation, task)
    # B-core task workers replay all requested episodes after failure.  The
    # JSONL is therefore attempt-local evidence even though its stable path is
    # reused; clear it for formal Validation20 as well as for smoke.
    _reset_progress(progress, receipt)
    rows = [_run_episode(runtime, args.benchmark_repo, task, seed, limit, progress) for seed in seeds]
    rollout_steps = [int(row["steps"]) for row in rows]
    progress_sha256 = sha256_file(progress)
    prediction_oob = sum(int(row.get("prediction_gripper_oob_count", 0)) for row in rows)
    plan_oob = sum(int(row.get("ensemble_plan_gripper_oob_count", 0)) for row in rows)
    atomic_json(receipt, {"schema": "before-we-act.bicoord.validation-progress/1", "status": "PASSED", "stage": stage, "task": task, "episodes": episodes, "completed": episodes, "max_steps": limit, "rollout_steps": rollout_steps, "rows": rows, "rows_path": str(progress.resolve()), "rows_sha256": progress_sha256, "seeds": seeds, "seed_source": "expert_seed_manifest" if args.operation == "validation20" else "smoke_probe", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "selector": "bcore", "smoke_interface_steps": SMOKE_INTERFACE_STEPS if args.operation == "smoke-closed-loop" else None, "policy_output_clipping": False, "action_clipping": False, "state_clipping": False, "gripper_reparameterization": False, "executed_gripper_oob_count": 0, "prediction_gripper_oob_count": prediction_oob, "ensemble_plan_gripper_oob_count": plan_oob})
    return publish_result(args, stage=stage, include_model_contract=True, artifacts=[artifact(progress, kind="validation_progress"), artifact(receipt, kind="progress_receipt")], task=task, episodes=episodes, completed=episodes, successes=sum(int(r["success"]) for r in rows), max_steps=limit, rollout_steps=rollout_steps, seeds=seeds, seed_source="expert_seed_manifest" if args.operation == "validation20" else "smoke_probe", progress_receipt=str(receipt.resolve()), progress_receipt_sha256=sha256_file(receipt), checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), action_encoding=ACTION_ENCODING, action_horizon=ACTION_HORIZON, paired=False, selector_off_control=False, smoke_interface_steps=SMOKE_INTERFACE_STEPS if args.operation == "smoke-closed-loop" else None, policy_output_clipping=False, action_clipping=False, state_clipping=False, gripper_reparameterization=False, executed_gripper_oob_count=0, prediction_gripper_oob_count=prediction_oob, ensemble_plan_gripper_oob_count=plan_oob)


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("smoke-closed-loop", "validation20")); parser.add_argument("--task", choices=TASKS, required=True); parser.add_argument("--episodes", type=int, default=1); parser.add_argument("--max-steps", type=int); parser.add_argument("--record-progress", action="store_true"); parser.add_argument("--seed-start", type=int); args = parser.parse_args(argv); run(args); return 0


if __name__ == "__main__": raise SystemExit(main())
