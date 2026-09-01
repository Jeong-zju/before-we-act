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
import re
import sys
import traceback
from typing import Any, Mapping

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
SEED_PROGRESS_SCHEMA = "before-we-act.bicoord.expert-seed-progress/1"
SEED_ATTEMPT_SCHEMA = "before-we-act.bicoord.expert-seed-attempt/1"
OFFICIAL_SEED_MULTIPLIER = 100_000
DEFAULT_SEED_BUCKET = 0
DEFAULT_MAX_ATTEMPTS = 5_000
STRUCTURAL_ERROR_STREAK_LIMIT = 3


class RepeatedStructuralSeedError(RuntimeError):
    """Raised when seed variation cannot repair a stable benchmark failure."""


_MEMORY_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_VOLATILE_NUMBER = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z_])"
)
_EXPECTED_PLANNER_EXCEPTION = re.compile(
    r"(?:planner|planning|curobo|motion\s*(?:gen|planning)|"
    r"no\s+(?:valid|feasible|collision[- ]?free)\s+"
    r"(?:solution|path|trajectory)|(?:ik|trajectory|path)\s+fail)",
    re.IGNORECASE,
)


def _stable_text(value: object) -> str:
    """Remove values that legitimately vary from seed to seed.

    Exception identity must not depend on the candidate seed, a pointer value,
    or a sampled floating-point pose.  Names such as ``003_plate`` remain
    intact because their digits are part of an identifier, while standalone
    numeric values are replaced by one token.
    """

    text = _MEMORY_ADDRESS.sub("<address>", str(value))
    text = _VOLATILE_NUMBER.sub("<number>", text)
    return " ".join(text.split())


def _is_structural_exception(error: BaseException, env: Any = None) -> bool:
    """Classify failures that are invariant under candidate-seed changes.

    BiCoord's collector treats ``UnStableError`` as an ordinary seed rejection,
    and the motion planner can report a no-path exception on an otherwise valid
    scene.  Those outcomes must remain searchable.  Setup/asset/API errors are
    structural by default and therefore participate in the streak gate.
    """

    error_class = type(error)
    if error_class.__name__ == "UnStableError" or error_class.__qualname__.endswith(
        ".UnStableError"
    ):
        return False
    plan_success = getattr(env, "plan_success", None)
    if _EXPECTED_PLANNER_EXCEPTION.search(str(error)):
        # Configuration/asset/API failures can mention a planner while still
        # being structural.  Keep those fail-closed; only an unqualified
        # no-path/planner result is treated as an expected seed rejection.
        lowered = str(error).lower()
        structural_markers = (
            "config",
            "missing",
            "not found",
            "required",
            " is none",
            "yml",
            "attribute",
            "contact",
            "index",
            "shape",
            "cuda out of memory",
        )
        if not any(marker in lowered for marker in structural_markers):
            return False
        if plan_success is False and any(
            marker in lowered for marker in ("no solution", "no path", "planning failed")
        ):
            return False
    return True


def _is_expected_seed_rejection(error: BaseException, env: Any = None) -> bool:
    """Whether an exception is an ordinary candidate-seed rejection."""

    error_class = type(error)
    if error_class.__name__ == "UnStableError" or error_class.__qualname__.endswith(
        ".UnStableError"
    ):
        return True
    return not _is_structural_exception(error, env)


def _exception_evidence(
    error: BaseException, *, structural_error: bool | None = None, env: Any = None
) -> dict[str, Any]:
    """Return auditable evidence and a repeat-stable exception signature."""

    error_class = type(error)
    qualified_type = f"{error_class.__module__}.{error_class.__qualname__}"
    frames = []
    for frame in traceback.extract_tb(error.__traceback__)[-12:]:
        # Absolute checkout roots differ between workers.  A short source
        # suffix plus function/source identity is stable across those roots.
        source_parts = Path(frame.filename).as_posix().split("/")
        frames.append(
            {
                "file": "/".join(source_parts[-4:]),
                "function": frame.name,
                "source": _stable_text(frame.line or ""),
            }
        )
    message_template = _stable_text(error)
    basis = {
        "error_type": qualified_type,
        "message_template": message_template,
        "trace": frames,
    }
    signature = hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "structural_error": (
            _is_structural_exception(error, env)
            if structural_error is None
            else bool(structural_error)
        ),
        "expected_seed_rejection": _is_expected_seed_rejection(error, env),
        "error_type": error_class.__name__,
        "error_qualified_type": qualified_type,
        "error": str(error)[:500],
        "error_message_template": message_template,
        "error_trace": frames,
        "error_signature": signature,
    }


def _exception_count_rows(
    counts: Counter[tuple[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "error_type": error_type,
            "error_signature": signature,
            "count": int(count),
        }
        for (error_type, signature), count in sorted(counts.items())
    ]


def _progress_path(progress_dir: Path, task: str) -> Path:
    return progress_dir / f"progress_{task}.json"


def _seed_progress_path(progress_dir: Path, task: str, seed: int) -> Path:
    return progress_dir / "attempts" / task / f"seed_{int(seed)}.json"


def _write_seed_progress(
    path: Path,
    *,
    task: str,
    seed: int,
    attempt_index: int,
    row: Mapping[str, Any],
    status: str,
    episodes: int,
    max_attempts: int,
    valid_seed_count: int,
    exception_type_counts: Counter[str],
    exception_counts: Counter[tuple[str, str]],
    structural_exception_type_counts: Counter[str],
    structural_exception_counts: Counter[tuple[str, str]],
    consecutive_key: tuple[str, str] | None,
    consecutive_count: int,
) -> None:
    value: dict[str, Any] = {
        "schema": SEED_ATTEMPT_SCHEMA,
        "status": status,
        "task": task,
        "seed": int(seed),
        "attempt_index": int(attempt_index),
        "episodes_requested": int(episodes),
        "max_attempts": int(max_attempts),
        "valid_seed_count": int(valid_seed_count),
        "row": dict(row),
        "exception_type_counts": dict(sorted(exception_type_counts.items())),
        "exception_counts": _exception_count_rows(exception_counts),
        "structural_exception_type_counts": dict(
            sorted(structural_exception_type_counts.items())
        ),
        "structural_exception_counts": _exception_count_rows(
            structural_exception_counts
        ),
        "consecutive_structural_error": (
            {
                "error_type": consecutive_key[0],
                "error_signature": consecutive_key[1],
                "count": int(consecutive_count),
            }
            if consecutive_key is not None and consecutive_count
            else None
        ),
        "structural_error_streak_limit": STRUCTURAL_ERROR_STREAK_LIMIT,
        "policy_independent": True,
        "learned_policy_used": False,
        "closed_loop_policy_results_used": False,
    }
    atomic_json(path, value)


def _write_progress(
    path: Path,
    *,
    status: str,
    task: str,
    seed_bucket: int,
    episodes: int,
    max_attempts: int,
    rows: list[dict[str, Any]],
    seeds: list[int],
    exception_counts: Counter[tuple[str, str]],
    error_type_counts: Counter[str],
    structural_exception_counts: Counter[tuple[str, str]],
    structural_error_type_counts: Counter[str],
    consecutive_key: tuple[str, str] | None,
    consecutive_count: int,
    active_seed: int | None,
    next_seed: int | None,
    seed_receipt_dir: Path | None = None,
    seed_receipts_written: int = 0,
    last_seed_receipt: Path | None = None,
    failure: Mapping[str, Any] | None = None,
) -> None:
    streak = None
    if consecutive_key is not None and consecutive_count:
        streak = {
            "error_type": consecutive_key[0],
            "error_signature": consecutive_key[1],
            "count": int(consecutive_count),
        }
    value: dict[str, Any] = {
        "schema": SEED_PROGRESS_SCHEMA,
        "status": status,
        "task": task,
        "seed_bucket": int(seed_bucket),
        "seed_multiplier": OFFICIAL_SEED_MULTIPLIER,
        "episodes_requested": int(episodes),
        "max_attempts": int(max_attempts),
        "attempts_completed": len(rows),
        "valid_seeds": list(seeds),
        "active_seed": active_seed,
        "last_completed_seed": int(rows[-1]["seed"]) if rows else None,
        "next_seed": next_seed,
        # Keep the aggregate heartbeat bounded even when the official search
        # reaches 5,000 candidates.  Immutable per-seed receipts below retain
        # the complete row for every candidate; the final passed manifest also
        # stores the complete attempt list.
        "last_attempt": dict(rows[-1]) if rows else None,
        "recent_attempts": rows[-STRUCTURAL_ERROR_STREAK_LIMIT:],
        "attempts": (
            rows
            if len(rows) <= 64
            else rows[-64:]
        ),
        "attempts_truncated": len(rows) > 64,
        "seed_receipt_dir": (
            str(seed_receipt_dir.resolve()) if seed_receipt_dir is not None else None
        ),
        "seed_receipts_written": int(seed_receipts_written),
        "last_seed_receipt": (
            str(last_seed_receipt.resolve()) if last_seed_receipt is not None else None
        ),
        "exception_type_counts": dict(sorted(error_type_counts.items())),
        "exception_counts": _exception_count_rows(exception_counts),
        "structural_exception_type_counts": dict(
            sorted(structural_error_type_counts.items())
        ),
        "structural_exception_counts": _exception_count_rows(
            structural_exception_counts
        ),
        "consecutive_structural_error": streak,
        "structural_error_streak_limit": STRUCTURAL_ERROR_STREAK_LIMIT,
        "policy_independent": True,
        "learned_policy_used": False,
        "closed_loop_policy_results_used": False,
    }
    if failure is not None:
        value["failure"] = dict(failure)
    atomic_json(path, value)


def _asset_overlay_evidence(env: Any, task: str) -> dict[str, Any]:
    """Record the plate compatibility mapping on every plate-task attempt."""

    if task != "place_plate_and_cup":
        return {}
    configured = os.environ.get("BICOORD_PLATE_ASSET_OVERLAY")
    observed = getattr(env, "_bicoord_asset_overlay", None) if env is not None else None
    if isinstance(observed, Mapping):
        overlay = dict(observed)
    else:
        overlay = {
            "task": task,
            "applied": False,
            "reason": "environment_construction_failed_before_overlay_receipt",
        }
    # These three fields are deliberately present even on a failed setup so a
    # progress receipt never ambiguously omits the runtime asset state.
    overlay.setdefault("applied", False)
    overlay.setdefault("overlay", configured)
    overlay.setdefault("contact_points_pose_sha256", None)
    return {"asset_overlay": overlay}


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
        from .asset_runtime import apply_configured_task_overlay

        env._bicoord_asset_overlay = apply_configured_task_overlay(env, task)
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


def _expert_valid(
    env: Any, task: str | None = None
) -> tuple[bool, dict[str, Any]]:
    """Run only the benchmark expert; no learned policy is imported here."""

    info: Mapping[str, Any] | None = None
    try:
        result = env.play_once()
        if isinstance(result, Mapping):
            info = result
        plan_ok = bool(getattr(env, "plan_success", False))
        # Match the official collector's short-circuit exactly:
        # ``plan_success and check_success()``.  A clean planner miss is not an
        # exception and must continue to the next official candidate seed.
        success = bool(env.check_success()) if plan_ok else False
        evidence = {
            "structural_error": False,
            "plan_success": plan_ok,
            "expert_success": success,
            "stage_eval_score": float(getattr(env, "stage_eval_score", 0.0)),
            "expert_info_present": info is not None,
        }
        evidence.update(_asset_overlay_evidence(env, task or ""))
        return bool(plan_ok and success), evidence
    except Exception as error:
        evidence = _exception_evidence(error, env=env)
        evidence.update(
            {
                "plan_success": bool(getattr(env, "plan_success", False)),
                "expert_success": False,
                "stage_eval_score": float(getattr(env, "stage_eval_score", 0.0)),
            }
        )
        evidence.update(_asset_overlay_evidence(env, task or ""))
        return False, evidence


def discover(
    benchmark_repo: Path,
    *,
    episodes: int,
    seed_bucket: int = DEFAULT_SEED_BUCKET,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    task: str | None = None,
    progress_dir: Path | None = None,
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
    exception_counts: dict[str, list[dict[str, Any]]] = {}
    exception_type_counts: dict[str, dict[str, int]] = {}
    structural_exception_counts: dict[str, list[dict[str, Any]]] = {}
    structural_exception_type_counts: dict[str, dict[str, int]] = {}
    seed_receipts: dict[str, list[str]] = {}
    seed_receipts_sha256: dict[str, list[str]] = {}
    for task_name in selected_tasks:
        start = OFFICIAL_SEED_MULTIPLIER * (1 + int(seed_bucket))
        rows: list[dict[str, Any]] = []
        seeds: list[int] = []
        signature_counts: Counter[tuple[str, str]] = Counter()
        type_counts: Counter[str] = Counter()
        structural_signature_counts: Counter[tuple[str, str]] = Counter()
        structural_type_counts: Counter[str] = Counter()
        consecutive_key: tuple[str, str] | None = None
        consecutive_count = 0
        task_seed_receipts: list[str] = []
        candidate = start
        progress = (
            _progress_path(progress_dir, task_name)
            if progress_dir is not None
            else None
        )
        seed_receipt_dir = (
            progress_dir / "attempts" / task_name
            if progress_dir is not None
            else None
        )
        if progress is not None:
            # Replace a stale receipt before constructing the first simulator.
            # This is liveness evidence only; it never resumes or skips a seed.
            _write_progress(
                progress,
                status="RUNNING",
                task=task_name,
                seed_bucket=seed_bucket,
                episodes=episodes,
                max_attempts=max_attempts,
                rows=rows,
                seeds=seeds,
                exception_counts=signature_counts,
                error_type_counts=type_counts,
                structural_exception_counts=structural_signature_counts,
                structural_error_type_counts=structural_type_counts,
                consecutive_key=None,
                consecutive_count=0,
                active_seed=candidate,
                next_seed=candidate,
                seed_receipt_dir=seed_receipt_dir,
            )
        while len(seeds) < episodes and len(rows) < max_attempts:
            row: dict[str, Any] = {"seed": int(candidate)}
            env = None
            try:
                env = _make_env(benchmark_repo, task_name, candidate)
                ok, evidence = _expert_valid(env, task_name)
                row.update(evidence, valid=bool(ok))
                if ok:
                    seeds.append(int(candidate))
            except Exception as error:
                row.update(_exception_evidence(error, env=env), valid=False)
                row.setdefault("plan_success", False)
                row.setdefault("expert_success", False)
                row.setdefault("stage_eval_score", 0.0)
            finally:
                row.update(_asset_overlay_evidence(env, task_name))
                if env is not None:
                    _close_env(env)
            rows.append(row)
            candidate += 1

            if "error_signature" in row:
                error_type = str(row["error_type"])
                signature = str(row["error_signature"])
                key = (error_type, signature)
                signature_counts[key] += 1
                type_counts[error_type] += 1
            else:
                key = None
            if row.get("structural_error") is True and key is not None:
                structural_signature_counts[key] += 1
                structural_type_counts[key[0]] += 1
                if key == consecutive_key:
                    consecutive_count += 1
                else:
                    consecutive_key = key
                    consecutive_count = 1
            else:
                # Expected planner/goal failures prove that this is not a
                # consecutive structural-error run, even when the seed is not
                # selected.
                consecutive_key = None
                consecutive_count = 0

            repeated = (
                consecutive_key is not None
                and consecutive_count >= STRUCTURAL_ERROR_STREAK_LIMIT
            )
            exhausted = len(rows) >= max_attempts and len(seeds) < episodes
            completed = len(seeds) == episodes
            status = (
                "FAILED"
                if repeated or exhausted
                else ("PASSED" if completed else "RUNNING")
            )
            failure: dict[str, Any] | None = None
            if repeated:
                failure = {
                    "reason": "repeated_structural_exception",
                    "error_type": consecutive_key[0],
                    "error_signature": consecutive_key[1],
                    "consecutive_count": consecutive_count,
                    "seed": int(row["seed"]),
                }
            elif exhausted:
                failure = {
                    "reason": "insufficient_expert_valid_seeds",
                    "valid_seed_count": len(seeds),
                    "attempts_completed": len(rows),
                }
            seed_receipt: Path | None = None
            if seed_receipt_dir is not None:
                seed_receipt = _seed_progress_path(progress_dir, task_name, int(row["seed"]))
                _write_seed_progress(
                    seed_receipt,
                    task=task_name,
                    seed=int(row["seed"]),
                    attempt_index=len(rows),
                    row=row,
                    status=status,
                    episodes=episodes,
                    max_attempts=max_attempts,
                    valid_seed_count=len(seeds),
                    exception_type_counts=type_counts,
                    exception_counts=signature_counts,
                    structural_exception_type_counts=structural_type_counts,
                    structural_exception_counts=structural_signature_counts,
                    consecutive_key=consecutive_key,
                    consecutive_count=consecutive_count,
                )
                task_seed_receipts.append(str(seed_receipt.resolve()))
            if progress is not None:
                _write_progress(
                    progress,
                    status=status,
                    task=task_name,
                    seed_bucket=seed_bucket,
                    episodes=episodes,
                    max_attempts=max_attempts,
                    rows=rows,
                    seeds=seeds,
                    exception_counts=signature_counts,
                    error_type_counts=type_counts,
                    structural_exception_counts=structural_signature_counts,
                    structural_error_type_counts=structural_type_counts,
                    consecutive_key=consecutive_key,
                    consecutive_count=consecutive_count,
                    active_seed=candidate if status == "RUNNING" else None,
                    next_seed=candidate if status == "RUNNING" else None,
                    seed_receipt_dir=seed_receipt_dir,
                    seed_receipts_written=len(rows),
                    last_seed_receipt=seed_receipt,
                    failure=failure,
                )
            print(
                json.dumps(
                    {
                        "event": "expert_seed_attempt",
                        "task": task_name,
                        "attempt": len(rows),
                        **row,
                        "consecutive_structural_errors": consecutive_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if repeated:
                raise RepeatedStructuralSeedError(
                    "expert seed discovery failed closed after "
                    f"{consecutive_count} consecutive {consecutive_key[0]} exceptions "
                    f"with signature {consecutive_key[1]} for {task_name}; "
                    f"last seed {row['seed']}"
                )
            if exhausted:
                break
        if len(seeds) != episodes:
            raise RuntimeError(
                f"expert seed discovery could not find {episodes} valid seeds for "
                f"{task_name} after {len(rows)} attempts"
            )
        valid[task_name] = seeds
        attempts[task_name] = rows
        exception_counts[task_name] = _exception_count_rows(signature_counts)
        exception_type_counts[task_name] = dict(sorted(type_counts.items()))
        structural_exception_counts[task_name] = _exception_count_rows(
            structural_signature_counts
        )
        structural_exception_type_counts[task_name] = dict(
            sorted(structural_type_counts.items())
        )
        seed_receipts[task_name] = task_seed_receipts
        seed_receipts_sha256[task_name] = [
            sha256_file(path) for path in task_seed_receipts
        ]
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
        "exception_type_counts": exception_type_counts,
        "exception_counts": exception_counts,
        "structural_exception_type_counts": structural_exception_type_counts,
        "structural_exception_counts": structural_exception_counts,
        "structural_error_streak_limit": STRUCTURAL_ERROR_STREAK_LIMIT,
        "seed_receipts": seed_receipts,
        "seed_receipts_sha256": seed_receipts_sha256,
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
    root = args.run / "artifacts" / ("seed_discovery_smoke" if smoke else "seed_discovery")
    root.mkdir(parents=True, exist_ok=True)
    manifest = discover(
        args.benchmark_repo,
        episodes=episodes,
        seed_bucket=seed_bucket,
        max_attempts=max_attempts,
        task=getattr(args, "task", None),
        progress_dir=root,
    )
    progress_receipts = {
        task_name: str(_progress_path(root, task_name).resolve())
        for task_name in manifest["tasks"]
    }
    progress_receipt_sha256 = {
        task_name: sha256_file(path)
        for task_name, path in progress_receipts.items()
    }
    # Bind every task worker's incremental evidence into the immutable
    # manifest.  The supervisor's task-shard aggregate retains both these
    # fields and the independently hashed artifact rows below.
    manifest["progress_receipts"] = progress_receipts
    manifest["progress_receipt_sha256"] = progress_receipt_sha256
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
            "progress_receipts": progress_receipts,
            "progress_receipt_sha256": progress_receipt_sha256,
            "exception_type_counts": manifest["exception_type_counts"],
            "exception_counts": manifest["exception_counts"],
            "structural_exception_type_counts": manifest[
                "structural_exception_type_counts"
            ],
            "structural_exception_counts": manifest["structural_exception_counts"],
            "structural_error_streak_limit": STRUCTURAL_ERROR_STREAK_LIMIT,
            "seed_receipts": manifest["seed_receipts"],
            "seed_receipts_sha256": manifest["seed_receipts_sha256"],
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
        "progress_receipts": progress_receipts,
        "progress_receipts_sha256": progress_receipt_sha256,
        "exception_type_counts": manifest["exception_type_counts"],
        "exception_counts": manifest["exception_counts"],
        "structural_exception_type_counts": manifest[
            "structural_exception_type_counts"
        ],
        "structural_exception_counts": manifest["structural_exception_counts"],
        "structural_error_streak_limit": STRUCTURAL_ERROR_STREAK_LIMIT,
        "seed_receipts": manifest["seed_receipts"],
        "seed_receipts_sha256": manifest["seed_receipts_sha256"],
        "policy_independent": True,
        "learned_policy_used": False,
        "closed_loop_policy_results_used": False,
    }
    if getattr(args, "task", None):
        fields.update(
            {
                "task": args.task,
                "episodes": episodes,
                "completed": episodes,
                "progress_receipt": progress_receipts[args.task],
                "progress_receipt_sha256": progress_receipt_sha256[args.task],
            }
        )
    return publish_result(
        args,
        stage=stage,
        artifacts=[
            artifact(manifest_path, kind="expert_seed_manifest"),
            artifact(report, kind="expert_seed_status"),
            *(
                artifact(path, kind="expert_seed_progress")
                for path in progress_receipts.values()
            ),
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
