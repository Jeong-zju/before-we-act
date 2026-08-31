"""Freeze the official policy-independent BiCoord expert-valid seed lists.

The upstream BiCoord evaluator does not evaluate a policy on arbitrary seeds:
it first runs the task's built-in expert trajectory and retains only seeds for
which the expert reaches the goal.  CARE/selector-off paired validation must
use exactly that same seed list.  This stage performs that filter once, before
any learned closed-loop result can influence it, and publishes an immutable
manifest consumed by all later evaluators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np

from .config import TASKS, VALIDATION_EPISODES, VALIDATION_MAX_STEPS
from .stage_common import (
    artifact,
    assert_common_paths,
    atomic_json,
    common_parser,
    publish_result,
    require_stage_result,
    sha256_file,
)


SEED_MANIFEST_SCHEMA = "before-we-act.bicoord.expert-seed-manifest/1"
OFFICIAL_SEED_MULTIPLIER = 100_000
DEFAULT_SEED_BUCKET = 0
DEFAULT_MAX_ATTEMPTS = 5_000


def _make_env(root: Path, task: str, seed: int) -> Any:
    """Construct a fresh official task environment using benchmark defaults."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib
    import yaml

    module = importlib.import_module(f"envs.{task}")
    cls = getattr(module, task)
    env = cls()
    config_path = root / "task_config" / "demo_clean.yml"
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
            "save_path": str(root / "data"),
            "need_plan": True,
            "dual_arm": True,
            "data_type": {
                "rgb": False,
                "qpos": True,
                "endpose": False,
                "depth": False,
                "pointcloud": False,
                "third_view": False,
            },
        }
    )
    embodiment = config.get("embodiment", ["aloha-agilex"])
    if not isinstance(embodiment, (list, tuple)) or len(embodiment) != 1:
        raise ValueError("the frozen BiCoord run requires one shared embodiment")
    emb_cfg_path = root / "task_config" / "_embodiment_config.yml"
    emb_cfg = yaml.safe_load(emb_cfg_path.read_text(encoding="utf-8"))
    robot_rel = emb_cfg[embodiment[0]]["file_path"]
    robot_root = (
        (root / robot_rel).resolve()
        if not os.path.isabs(str(robot_rel))
        else Path(robot_rel).resolve()
    )
    robot_config = yaml.safe_load((robot_root / "config.yml").read_text(encoding="utf-8"))
    config.update(
        {
            "left_robot_file": str(robot_root),
            "right_robot_file": str(robot_root),
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "dual_arm_embodied": True,
        }
    )
    try:
        env.setup_demo(**config)
    except BaseException:
        try:
            env.close_env()
        except Exception:
            pass
        raise
    return env


def _close_env(env: Any) -> None:
    try:
        env.close_env()
    except Exception:
        try:
            env.close()
        except Exception:
            pass


def _expert_valid(env: Any) -> tuple[bool, dict[str, Any]]:
    """Run only the benchmark expert; no learned policy is imported here."""

    info: Mapping[str, Any] | None = None
    try:
        result = env.play_once()
        if isinstance(result, Mapping):
            info = result
        plan_ok = bool(getattr(env, "plan_success", False))
        try:
            success = bool(env.check_success())
        except Exception:
            success = bool(getattr(env, "eval_success", False))
        return bool(plan_ok and success), {
            "plan_success": plan_ok,
            "expert_success": success,
            "stage_eval_score": float(getattr(env, "stage_eval_score", 0.0)),
            "expert_info_present": info is not None,
        }
    except BaseException as error:
        return False, {
            "plan_success": bool(getattr(env, "plan_success", False)),
            "expert_success": False,
            "stage_eval_score": float(getattr(env, "stage_eval_score", 0.0)),
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }


def discover(
    benchmark_repo: Path,
    *,
    episodes: int,
    seed_bucket: int = DEFAULT_SEED_BUCKET,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    task: str | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("expert seed discovery requires a positive episode count")
    if max_attempts < episodes:
        raise ValueError("max_attempts must cover the requested seed count")
    selected_tasks = TASKS if task is None else (task,)
    if task is not None and task not in TASKS:
        raise ValueError(f"unknown BiCoord task: {task}")
    valid: dict[str, list[int]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}
    for task_name in selected_tasks:
        start = OFFICIAL_SEED_MULTIPLIER * (1 + int(seed_bucket))
        rows: list[dict[str, Any]] = []
        seeds: list[int] = []
        candidate = start
        while len(seeds) < episodes and len(rows) < max_attempts:
            row: dict[str, Any] = {"seed": int(candidate)}
            env = None
            try:
                env = _make_env(benchmark_repo, task_name, candidate)
                ok, evidence = _expert_valid(env)
                row.update(evidence, valid=bool(ok))
                if ok:
                    seeds.append(int(candidate))
            except BaseException as error:
                row.update(
                    {
                        "valid": False,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                )
            finally:
                if env is not None:
                    _close_env(env)
            rows.append(row)
            candidate += 1
        if len(seeds) != episodes:
            raise RuntimeError(
                f"expert seed discovery could not find {episodes} valid seeds for "
                f"{task_name} after {len(rows)} attempts"
            )
        valid[task_name] = seeds
        attempts[task_name] = rows
    # A formal manifest always covers every task.  Smoke callers may request a
    # single task, but the supervisor's smoke stage asks for one per task.
    if task is None and set(valid) != set(TASKS):
        raise RuntimeError("expert seed manifest has incomplete task coverage")
    return {
        "schema": SEED_MANIFEST_SCHEMA,
        "status": "PASSED",
        "policy_independent": True,
        "selection_policy": "official_expert_play_once_then_plan_success_and_check_success",
        "seed_protocol": "100000*(1+seed_bucket), increment by one",
        "seed_bucket": int(seed_bucket),
        "seed_multiplier": OFFICIAL_SEED_MULTIPLIER,
        "episodes_per_task": int(episodes),
        "tasks": list(selected_tasks),
        "max_steps": {name: VALIDATION_MAX_STEPS[name] for name in selected_tasks},
        "valid_seeds": valid,
        "attempts": attempts,
        "learned_policy_used": False,
        "closed_loop_policy_results_used": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args)
    smoke = args.operation == "smoke-discover" or bool(getattr(args, "smoke", False))
    # Smoke runs follow the real smoke B-core interface and must not require
    # the formal B-core selection receipt (which is intentionally later in the
    # supervisor DAG).  Formal discovery remains downstream of offline select.
    require_stage_result(
        args.run,
        "bcore_smoke_closed_loop" if smoke else "bcore_select",
        config_sha256=args.config_sha256,
    )
    requested = getattr(args, "episodes", None)
    episodes = int(requested if requested is not None else (1 if smoke else VALIDATION_EPISODES))
    if smoke and episodes != 1:
        raise ValueError("seed-discovery smoke requires one expert-valid seed")
    if not smoke and episodes != VALIDATION_EPISODES:
        raise ValueError(f"formal seed discovery requires {VALIDATION_EPISODES} seeds")
    seed_bucket = int(getattr(args, "seed_bucket", DEFAULT_SEED_BUCKET))
    max_attempts = int(getattr(args, "max_attempts", DEFAULT_MAX_ATTEMPTS))
    manifest = discover(
        args.benchmark_repo,
        episodes=episodes,
        seed_bucket=seed_bucket,
        max_attempts=max_attempts,
        task=getattr(args, "task", None),
    )
    root = args.run / "artifacts" / ("seed_discovery_smoke" if smoke else "seed_discovery")
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.task}" if getattr(args, "task", None) else ""
    manifest_path = root / f"seed_manifest{suffix}.json"
    atomic_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    report = root / f"status{suffix}.json"
    atomic_json(
        report,
        {
            "schema": SEED_MANIFEST_SCHEMA,
            "status": "PASSED",
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": digest,
            "episodes_per_task": episodes,
            "task_counts": {
                name: len(values) for name, values in manifest["valid_seeds"].items()
            },
            "policy_independent": True,
            "learned_policy_used": False,
        },
    )
    if getattr(args, "task", None):
        stage = "seed_discovery_worker"
    else:
        stage = "seed_discovery_smoke" if smoke else "seed_discovery"
    fields: dict[str, Any] = {
        "episodes_per_task": episodes,
        "tasks": list(manifest["tasks"]),
        "valid_seeds": manifest["valid_seeds"],
        "seed_manifest": str(manifest_path.resolve()),
        "seed_manifest_sha256": digest,
        "policy_independent": True,
        "learned_policy_used": False,
        "closed_loop_policy_results_used": False,
    }
    if getattr(args, "task", None):
        fields.update({"task": args.task, "episodes": episodes, "completed": episodes})
    return publish_result(
        args,
        stage=stage,
        artifacts=[
            artifact(manifest_path, kind="expert_seed_manifest"),
            artifact(report, kind="expert_seed_status"),
        ],
        **fields,
    )


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("smoke-discover", "discover-seeds"))
    parser.add_argument("--seed-bucket", type=int, default=DEFAULT_SEED_BUCKET)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_SEED_BUCKET",
    "OFFICIAL_SEED_MULTIPLIER",
    "SEED_MANIFEST_SCHEMA",
    "discover",
    "main",
    "run",
]
