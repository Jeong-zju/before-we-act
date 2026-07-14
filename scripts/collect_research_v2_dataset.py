"""Collect a restartable Research-v2 matched-intervention dataset.

Each worker owns one deterministic episode and one output file.  The final
rename in :func:`save_research_v2_episode` is atomic, so a killed collector can
be resumed without accepting a half-written HDF5 file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Iterator, TypeVar

# Collection is process-parallel.  Giving every worker a second BLAS/OpenMP
# thread oversubscribes the CPU badly on the intended 32-thread workstation.
# Explicit user settings are respected.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collect import (  # noqa: E402
    RESEARCH_V2_BRANCH_ACTION_PROGRAM,
    collect_one_episode,
)
from data.local_observation import SensorSimulationConfig  # noqa: E402
from data.policies import ScriptedPolicy  # noqa: E402
from data.research_v2 import (  # noqa: E402
    RESEARCH_V2_DATA_CONTRACT,
    RESEARCH_V2_SCHEMA_VERSION,
    audit_research_v2_file,
    save_research_v2_episode,
)
from envs.two_robot_carry_env import (  # noqa: E402
    CarryEnvConfig,
    TwoRobotCarryNarrowPassageEnv,
)
from scripts.collect_fe_pc_wam_dataset import (  # noqa: E402
    BALANCED_SCENARIOS,
    CollectionConfig,
    SPLIT_SEED_OFFSETS,
)


PILOT_SEED_OFFSET = 3_000_000
COLLECTION_STATE_VERSION = "fe_pc_wam/research_v2_collection_state_v1"
COLLECTION_PROFILE = "research_v2_balanced_private_gates_v2"
DEFAULT_MIN_FREE_GIB = 20.0
DEFAULT_ESTIMATED_EPISODE_MIB = 2.0
EPISODE_NAME = re.compile(r"episode_(\d+)\.hdf5$")

RESEARCH_V2_MODES = (
    "scripted",
    "scripted",
    "scripted",
    "scripted",
    "scripted",
    "noisy",
    "noisy",
    "noisy",
    "noisy",
    "noisy",
    "recovery",
    "recovery",
    "recovery",
    "recovery",
    "exploratory",
    "exploratory",
    "exploratory",
    "near_miss",
    "near_miss",
    "near_miss",
)
# The pilot checks that the base expert/environment is healthy.  Deliberately
# failing exploratory and near-miss trajectories belong in the formal splits,
# but would make an 80% pilot success gate impossible by construction.
RESEARCH_V2_PILOT_MODES = (
    "scripted",
    "scripted",
    "scripted",
    "scripted",
    "recovery",
)
RESEARCH_V2_NOISE = {
    "scripted": 0.0,
    "noisy": 0.45,
    "recovery": 0.0,
    "exploratory": 0.60,
    "near_miss": 0.0,
}

# Keep the broad mixed-policy scenario coverage while reserving one eighth of
# formal episodes for the private-information task used by the communication
# claims.  V1's BALANCED_SCENARIOS deliberately excludes private_gates, so it
# must not be used directly by the Research-v2 collector.
RESEARCH_V2_SCENARIOS = (
    *BALANCED_SCENARIOS,
    "private_gates",
    "private_gates",
)
PRIVATE_EVENT_NAMES = ("decisive_private", "locally_inferable", "redundant")
MANEUVER_NAMES = ("left", "hold", "right")


@dataclass(frozen=True)
class EpisodeJob:
    root: str
    split: str
    index: int
    seed: int
    episode_len: int
    randomize: bool
    object_position_std: float
    object_yaw_std: float


@dataclass(frozen=True)
class ExistingEpisodeJob:
    path: str
    split: str
    index: int
    seed: int
    episode_len: int
    randomize: bool
    object_position_std: float
    object_yaw_std: float


def collect_research_v2(args: argparse.Namespace) -> dict[str, Any]:
    """Collect all requested splits while holding an exclusive dataset lock."""

    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    configuration = CollectionConfig(
        out_dir=str(root),
        train_episodes=int(args.train_episodes),
        val_episodes=int(args.val_episodes),
        test_episodes=int(args.test_episodes),
        pilot_episodes=int(args.pilot_episodes),
        seed=int(args.seed),
        episode_len=int(args.episode_len),
        randomize=bool(getattr(args, "randomize", 1)),
        object_position_std=float(getattr(args, "object_position_std", 0.025)),
        object_yaw_std=float(getattr(args, "object_yaw_std", 0.035)),
        profile="balanced",
        resume=bool(args.resume),
    )
    pilot_gate = float(args.pilot_success_gate)
    if not 0.0 <= pilot_gate <= 1.0:
        raise ValueError("pilot_success_gate must be in [0, 1]")
    requested_workers = int(getattr(args, "workers", 0))
    if requested_workers < 0:
        raise ValueError("workers must be non-negative (zero means auto)")
    workers = _resolve_workers(requested_workers)
    start_method = str(getattr(args, "start_method", _default_start_method()))
    if start_method not in multiprocessing.get_all_start_methods():
        raise ValueError(f"unsupported multiprocessing start method {start_method!r}")
    min_free_gib = float(getattr(args, "min_free_gb", DEFAULT_MIN_FREE_GIB))
    estimated_episode_mib = float(
        getattr(args, "estimated_episode_mib", DEFAULT_ESTIMATED_EPISODE_MIB)
    )
    if min_free_gib < 0.0 or estimated_episode_mib <= 0.0:
        raise ValueError("disk reserve must be non-negative and episode estimate positive")

    counts = {
        "pilot": int(args.pilot_episodes),
        "train": int(args.train_episodes),
        "val": int(args.val_episodes),
        "test": int(args.test_episodes),
    }
    smoke = bool(getattr(args, "smoke", False))
    code_hash = _collection_code_sha256()
    identity = _collection_identity(configuration, code_hash)
    started_at = datetime.now(timezone.utc).isoformat()

    with _exclusive_collection_lock(root):
        stale_temps = _handle_stale_temporaries(root, resume=configuration.resume)
        prior_state = _load_optional_json(root / "collection_state.json")
        prior_manifest = _load_optional_json(root / "dataset_manifest.json")
        _validate_resume_state(
            root,
            prior_state,
            identity,
            resume=configuration.resume,
        )
        _validate_requested_targets(root, counts)
        reuse_manifest = _can_reuse_completed_manifest(
            root,
            prior_state=prior_state,
            prior_manifest=prior_manifest,
            targets=counts,
            resume=configuration.resume,
        )
        # A final manifest is the training lineage anchor.  Do not leave an old
        # "complete" anchor visible while missing/new files are being written.
        # Conversely, an audit-only resume preserves it byte-for-byte below.
        if prior_manifest is not None and not reuse_manifest:
            (root / "dataset_manifest.json").unlink()
        if prior_state is not None:
            started_at = str(prior_state.get("started_at_utc", started_at))
        state: dict[str, Any] = {
            "state_version": COLLECTION_STATE_VERSION,
            "status": "collecting",
            "started_at_utc": started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "targets": counts,
            "completed_splits": {},
            "runtime": {
                "requested_workers": requested_workers,
                "resolved_workers": workers,
                "start_method": start_method,
                "min_free_gib": min_free_gib,
                "estimated_episode_mib": estimated_episode_mib,
                "stale_temporary_files_removed": stale_temps,
            },
        }
        _write_json(root / "collection_state.json", state)

        reports: dict[str, dict[str, Any]] = {}
        disk_preflight: dict[str, dict[str, float | int]] = {}
        for split, count in counts.items():
            if count <= 0:
                continue
            existing_count = len(_episode_paths(root / split))
            remaining = max(0, count - existing_count)
            disk_preflight[split] = _check_disk_space(
                root,
                remaining_episodes=remaining,
                workers=workers,
                min_free_gib=min_free_gib,
                estimated_episode_mib=estimated_episode_mib,
            )
            reports[split] = _collect_split(
                root,
                split,
                count,
                configuration,
                workers=workers,
                start_method=start_method,
            )
            _validate_formal_split_quality(split, reports[split], smoke=smoke)
            state["completed_splits"][split] = reports[split]
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(root / "collection_state.json", state)
            if split == "pilot" and reports[split]["success_rate"] < pilot_gate:
                state["status"] = "pilot_gate_failed"
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(root / "collection_state.json", state)
                raise RuntimeError(
                    "Research-v2 pilot success rate "
                    f"{reports[split]['success_rate']:.3f} is below gate {pilot_gate:.3f}"
                )

        completed_at = datetime.now(timezone.utc).isoformat()
        if reuse_manifest:
            assert prior_manifest is not None
            if prior_manifest.get("splits") != reports:
                raise ValueError(
                    "completed dataset no longer matches its frozen manifest; "
                    "collect into a new --out-dir"
                )
            state["status"] = "complete"
            state["completed_at_utc"] = completed_at
            state["updated_at_utc"] = completed_at
            _write_json(root / "collection_state.json", state)
            return prior_manifest

        manifest = {
            "schema_version": RESEARCH_V2_SCHEMA_VERSION,
            "data_contract": RESEARCH_V2_DATA_CONTRACT,
            "branch_action_program": RESEARCH_V2_BRANCH_ACTION_PROGRAM,
            "collection_state_version": COLLECTION_STATE_VERSION,
            "status": "complete",
            "created_at_utc": started_at,
            "completed_at_utc": completed_at,
            "collection_code_sha256": code_hash,
            "formal_scenario_mixture": list(RESEARCH_V2_SCENARIOS),
            "pilot_success_gate": pilot_gate,
            "config": asdict(configuration),
            "runtime": state["runtime"],
            "disk_preflight": disk_preflight,
            "splits": reports,
            "split_seed_offsets": {
                **dict(SPLIT_SEED_OFFSETS),
                "pilot": PILOT_SEED_OFFSET,
            },
            "seed_manifest": {
                split: {
                    "start": int(configuration.seed + _split_seed_offset(split)),
                    "stop_exclusive": int(
                        configuration.seed + _split_seed_offset(split) + count
                    ),
                }
                for split, count in counts.items()
                if count > 0
            },
            "split_seeds_frozen_before_collection": True,
            "branch_group_id_is_model_feature": False,
            "privileged_runtime_inputs": [],
            "legacy_compatible": False,
        }
        _write_json(root / "dataset_manifest.json", manifest)
        state["status"] = "complete"
        state["completed_at_utc"] = completed_at
        state["updated_at_utc"] = completed_at
        _write_json(root / "collection_state.json", state)
        return manifest


def _collect_split(
    root: Path,
    split: str,
    target: int,
    config: CollectionConfig,
    *,
    workers: int = 1,
    start_method: str | None = None,
) -> dict[str, Any]:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    indexed_paths = _indexed_episode_paths(split_dir)
    if indexed_paths and not config.resume:
        raise FileExistsError(
            f"{split_dir} already contains {len(indexed_paths)} episodes; use --resume"
        )
    out_of_range = sorted(index for index in indexed_paths if index >= target)
    if out_of_range:
        raise ValueError(
            f"{split_dir} contains episode indices outside requested target {target}: "
            f"{out_of_range[:8]}"
        )

    seed_base = config.seed + _split_seed_offset(split)
    existing_jobs = [
        ExistingEpisodeJob(
            path=str(path),
            split=split,
            index=index,
            seed=seed_base + index,
            episode_len=config.episode_len,
            randomize=config.randomize,
            object_position_std=config.object_position_std,
            object_yaw_std=config.object_yaw_std,
        )
        for index, path in sorted(indexed_paths.items())
    ]
    existing_reports = _execute_jobs(
        _inspect_existing_episode,
        existing_jobs,
        workers=workers,
        start_method=start_method or _default_start_method(),
        description=f"audit research-v2 {split}",
    )

    missing_indices = [index for index in range(target) if index not in indexed_paths]
    collection_jobs = [
        EpisodeJob(
            root=str(root),
            split=split,
            index=index,
            seed=seed_base + index,
            episode_len=config.episode_len,
            randomize=config.randomize,
            object_position_std=config.object_position_std,
            object_yaw_std=config.object_yaw_std,
        )
        for index in missing_indices
    ]
    new_reports = _execute_jobs(
        _collect_episode,
        collection_jobs,
        workers=workers,
        start_method=start_method or _default_start_method(),
        description=f"collect research-v2 {split}",
    )
    report = _summarize_episode_reports(
        split_dir,
        target,
        [*existing_reports, *new_reports],
    )
    _write_json(split_dir / "summary.json", report)
    return report


def _collect_episode(job: EpisodeJob) -> dict[str, Any]:
    """Worker entry point: collect, atomically save, audit, then report."""

    recipe = _episode_recipe(job.seed, pilot=job.split == "pilot")
    env = TwoRobotCarryNarrowPassageEnv(
        CarryEnvConfig(
            scenario=str(recipe["scenario"]),
            episode_len=job.episode_len,
            seed=job.seed,
        )
    )
    policy = ScriptedPolicy(
        noise_std=float(recipe["noise_std"]),
        seed=job.seed,
        mode=str(recipe["mode"]),
    )
    sensor_config = SensorSimulationConfig(
        control_dt=env.cfg.control_dt,
        object_dropout_prob=float(recipe["object_dropout_prob"]),
        object_position_std=job.object_position_std,
        object_yaw_std=job.object_yaw_std,
    )
    episode, _, spec = collect_one_episode(
        env,
        policy,
        job.seed,
        randomize=job.randomize,
        sensor_config=sensor_config,
        collect_matched_branches=True,
    )
    episode.metadata.update(
        {
            "split": job.split,
            "episode_index": job.index,
            "collection_profile": COLLECTION_PROFILE,
            "mode": recipe["mode"],
            "scenario": recipe["scenario"],
            "policy_noise_std": recipe["noise_std"],
            "object_dropout_prob": recipe["object_dropout_prob"],
            "collection_episode_limit": job.episode_len,
            "collection_randomize": job.randomize,
            "sensor_object_position_std": job.object_position_std,
            "sensor_object_yaw_std": job.object_yaw_std,
            "branch_action_program": RESEARCH_V2_BRANCH_ACTION_PROGRAM,
        }
    )
    path = Path(job.root) / job.split / f"episode_{job.index:06d}.hdf5"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing episode {path}")
    save_research_v2_episode(path, episode, spec, episode.research_v2_branch_groups)
    audit_research_v2_file(path)
    return _episode_report(path, expected=job)


def _inspect_existing_episode(job: ExistingEpisodeJob) -> dict[str, Any]:
    path = Path(job.path)
    audit_research_v2_file(path)
    return _episode_report(path, expected=job)


def _episode_report(
    path: Path,
    *,
    expected: EpisodeJob | ExistingEpisodeJob,
) -> dict[str, Any]:
    expected_recipe = _episode_recipe(expected.seed, pilot=expected.split == "pilot")
    with h5py.File(path, "r") as file:
        metadata = file["metadata"].attrs
        exact_expected = {
            "seed": expected.seed,
            "split": expected.split,
            "episode_index": expected.index,
            "collection_profile": COLLECTION_PROFILE,
            "scenario": expected_recipe["scenario"],
            "mode": expected_recipe["mode"],
            "collection_episode_limit": expected.episode_len,
            "collection_randomize": expected.randomize,
            "branch_action_program": RESEARCH_V2_BRANCH_ACTION_PROGRAM,
        }
        for name, value in exact_expected.items():
            stored = metadata.get(name, None)
            if stored != value:
                raise ValueError(
                    f"resume metadata mismatch in {path}: {name}={stored!r}, expected {value!r}"
                )
        float_expected = {
            "policy_noise_std": expected_recipe["noise_std"],
            "object_dropout_prob": expected_recipe["object_dropout_prob"],
            "sensor_object_position_std": expected.object_position_std,
            "sensor_object_yaw_std": expected.object_yaw_std,
        }
        for name, value in float_expected.items():
            stored = float(metadata.get(name, np.nan))
            if not np.isfinite(stored) or not np.isclose(stored, float(value), atol=1e-12):
                raise ValueError(
                    f"resume metadata mismatch in {path}: {name}={stored!r}, expected {value!r}"
                )
        truth = np.asarray(
            file["privileged/observations/private_event_truth"], dtype=np.int64
        )
        event_type = truth[:, 1]
        informed_agent = truth[:, 2]
        maneuver = np.clip(truth[:, 3] + 1, 0, 2)
        private_event = (0 <= event_type) & (event_type < len(PRIVATE_EVENT_NAMES))
        event_type_counts = np.bincount(
            event_type[private_event], minlength=len(PRIVATE_EVENT_NAMES)
        )
        informed_counts = np.bincount(
            informed_agent[private_event], minlength=2
        )
        maneuver_counts = np.bincount(
            maneuver[private_event], minlength=len(MANEUVER_NAMES)
        )
        active_context = np.asarray(
            file["privileged/observations/next_gate_context_agents"]
        )[..., 2]
        cue_valid = np.asarray(
            file["privileged/observations/private_event_valid_agents"]
        )
        return {
            "index": expected.index,
            "path": str(path.resolve()),
            "seed": expected.seed,
            "success": bool(metadata.get("success", False)),
            "failure_reason": str(metadata.get("failure_reason", "unknown")),
            "scenario": str(metadata.get("scenario", "unknown")),
            "mode": str(metadata.get("mode", "unknown")),
            "transitions": int(file.attrs["num_transitions"]),
            "branch_groups": int(file.attrs["branch_group_count"]),
            "bytes": int(path.stat().st_size),
            "private_event_type_counts": event_type_counts.astype(int).tolist(),
            "private_event_informed_agent_counts": informed_counts.astype(int).tolist(),
            "private_event_maneuver_counts": maneuver_counts.astype(int).tolist(),
            "private_event_active_observations": int(
                np.any(active_context > 0.5, axis=-1).sum()
            ),
            "private_event_cued_agent_observations": int((cue_valid > 0.5).sum()),
        }


JobT = TypeVar("JobT")


def _execute_jobs(
    function: Callable[[JobT], dict[str, Any]],
    jobs: list[JobT],
    *,
    workers: int,
    start_method: str,
    description: str,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    worker_count = min(max(1, workers), len(jobs))
    if worker_count == 1:
        return [function(job) for job in tqdm(jobs, desc=description)]

    context = multiprocessing.get_context(start_method)
    executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=context)
    futures = {executor.submit(function, job): job for job in jobs}
    results: list[dict[str, Any]] = []
    try:
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=description
        ):
            results.append(future.result())
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def _summarize_episode_reports(
    split_dir: Path,
    target: int,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    by_index = {int(report["index"]): report for report in reports}
    expected = set(range(target))
    if set(by_index) != expected or len(by_index) != len(reports):
        missing = sorted(expected - set(by_index))
        unexpected = sorted(set(by_index) - expected)
        raise RuntimeError(
            f"split summary is incomplete: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    ordered = [by_index[index] for index in range(target)]
    successes = sum(int(report["success"]) for report in ordered)
    scenarios = Counter(str(report["scenario"]) for report in ordered)
    modes = Counter(str(report["mode"]) for report in ordered)
    failures = Counter(
        str(report["failure_reason"])
        for report in ordered
        if not bool(report["success"])
    )
    total_bytes = sum(int(report["bytes"]) for report in ordered)
    event_type_counts = np.asarray(
        [report["private_event_type_counts"] for report in ordered], dtype=np.int64
    ).sum(axis=0)
    informed_counts = np.asarray(
        [report["private_event_informed_agent_counts"] for report in ordered],
        dtype=np.int64,
    ).sum(axis=0)
    maneuver_counts = np.asarray(
        [report["private_event_maneuver_counts"] for report in ordered], dtype=np.int64
    ).sum(axis=0)
    return {
        "path": str(split_dir.resolve()),
        "complete": True,
        "episodes": target,
        "successes": successes,
        "success_rate": successes / max(target, 1),
        "transitions": sum(int(report["transitions"]) for report in ordered),
        "branch_groups": sum(int(report["branch_groups"]) for report in ordered),
        "bytes": total_bytes,
        "size_gib": total_bytes / (1024**3),
        "scenario_counts": dict(sorted(scenarios.items())),
        "mode_counts": dict(sorted(modes.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
        "private_event_quality": {
            "private_gate_episodes": int(scenarios.get("private_gates", 0)),
            "event_type_observation_counts": {
                name: int(event_type_counts[index])
                for index, name in enumerate(PRIVATE_EVENT_NAMES)
            },
            "informed_agent_observation_counts": {
                str(index): int(informed_counts[index]) for index in range(2)
            },
            "maneuver_observation_counts": {
                name: int(maneuver_counts[index])
                for index, name in enumerate(MANEUVER_NAMES)
            },
            "active_observations": sum(
                int(report["private_event_active_observations"]) for report in ordered
            ),
            "cued_agent_observations": sum(
                int(report["private_event_cued_agent_observations"])
                for report in ordered
            ),
        },
    }


def _episode_recipe(seed: int, *, pilot: bool = False) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    scenario = str(rng.choice(RESEARCH_V2_SCENARIOS))
    modes = RESEARCH_V2_PILOT_MODES if pilot else RESEARCH_V2_MODES
    mode = str(rng.choice(modes))
    dropout = 0.40 if scenario in {"occlusion", "hard_comm"} else 0.10
    return {
        "scenario": scenario,
        "mode": mode,
        "noise_std": RESEARCH_V2_NOISE[mode],
        "object_dropout_prob": dropout,
    }


def _validate_formal_split_quality(
    split: str, report: dict[str, Any], *, smoke: bool
) -> None:
    """Reject formal splits that cannot support the private-event claims."""

    if smoke or split == "pilot" or int(report.get("episodes", 0)) <= 0:
        return
    quality = report.get("private_event_quality", {})
    episodes = int(report["episodes"])
    # The configured mixture is 12.5%; require at least 5% to tolerate finite
    # deterministic seed variation while still preventing accidental omission.
    minimum_private_episodes = max(1, (episodes + 19) // 20)
    private_episodes = int(quality.get("private_gate_episodes", 0))
    missing: list[str] = []
    if private_episodes < minimum_private_episodes:
        missing.append(
            f"private_gates episodes {private_episodes} < {minimum_private_episodes}"
        )
    for field in (
        "event_type_observation_counts",
        "informed_agent_observation_counts",
        "maneuver_observation_counts",
    ):
        counts = quality.get(field, {})
        absent = [str(name) for name, value in counts.items() if int(value) <= 0]
        if not counts or absent:
            missing.append(f"{field} missing {absent or 'all classes'}")
    if int(quality.get("active_observations", 0)) <= 0:
        missing.append("no active private-gate observations")
    if int(quality.get("cued_agent_observations", 0)) <= 0:
        missing.append("no delivered private-event cues")
    if missing:
        raise RuntimeError(
            f"Research-v2 {split} private-event quality gate failed: "
            + "; ".join(missing)
        )


def _indexed_episode_paths(split_dir: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in _episode_paths(split_dir):
        match = EPISODE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid Research-v2 episode filename: {path.name}")
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"duplicate Research-v2 episode index {index} in {split_dir}")
        indexed[index] = path
    return indexed


def _episode_paths(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("episode_*.hdf5"))


def _split_seed_offset(split: str) -> int:
    if split == "pilot":
        return PILOT_SEED_OFFSET
    try:
        return int(SPLIT_SEED_OFFSETS[split])
    except KeyError as exc:
        raise ValueError(f"unknown split {split!r}") from exc


def _resolve_workers(requested: int) -> int:
    if requested > 0:
        return requested
    try:
        logical_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        logical_cpus = os.cpu_count() or 1
    # 16 single-threaded MuJoCo workers fit comfortably in 64 GiB and map well
    # to a 16-core/32-thread workstation while leaving headroom for the OS.
    return min(16, max(1, logical_cpus // 2))


def _default_start_method() -> str:
    methods = multiprocessing.get_all_start_methods()
    return "fork" if "fork" in methods else methods[0]


def _check_disk_space(
    root: Path,
    *,
    remaining_episodes: int,
    workers: int,
    min_free_gib: float,
    estimated_episode_mib: float,
) -> dict[str, float | int]:
    usage = shutil.disk_usage(root)
    reserve_bytes = int(min_free_gib * 1024**3)
    # Account for all remaining final files plus one temporary file per worker.
    estimated_bytes = int(
        (remaining_episodes + min(workers, remaining_episodes))
        * estimated_episode_mib
        * 1024**2
    )
    required_free = reserve_bytes + estimated_bytes
    report: dict[str, float | int] = {
        "remaining_episodes": remaining_episodes,
        "free_gib": usage.free / (1024**3),
        "reserve_gib": min_free_gib,
        "estimated_required_gib": estimated_bytes / (1024**3),
    }
    if usage.free < required_free:
        raise OSError(
            f"insufficient disk space under {root}: free={report['free_gib']:.2f} GiB, "
            f"requested reserve={min_free_gib:.2f} GiB, "
            f"estimated collection={report['estimated_required_gib']:.2f} GiB; "
            "lower --min-free-gb only after verifying capacity"
        )
    return report


def _collection_identity(config: CollectionConfig, code_hash: str) -> dict[str, Any]:
    return {
        "schema_version": RESEARCH_V2_SCHEMA_VERSION,
        "data_contract": RESEARCH_V2_DATA_CONTRACT,
        "collection_profile": COLLECTION_PROFILE,
        "branch_action_program": RESEARCH_V2_BRANCH_ACTION_PROGRAM,
        "collection_code_sha256": code_hash,
        "seed": config.seed,
        "episode_len": config.episode_len,
        "randomize": config.randomize,
        "object_position_std": config.object_position_std,
        "object_yaw_std": config.object_yaw_std,
        "formal_modes": list(RESEARCH_V2_MODES),
        "pilot_modes": list(RESEARCH_V2_PILOT_MODES),
        "formal_scenarios": list(RESEARCH_V2_SCENARIOS),
    }


def _validate_resume_state(
    root: Path,
    prior_state: dict[str, Any] | None,
    identity: dict[str, Any],
    *,
    resume: bool,
) -> None:
    has_episodes = any(root.glob("*/episode_*.hdf5"))
    manifest_exists = (root / "dataset_manifest.json").is_file()
    if (has_episodes or prior_state is not None or manifest_exists) and not resume:
        raise FileExistsError(
            f"{root} already contains Research-v2 collection state; use --resume "
            "or choose a new --out-dir"
        )
    if prior_state is None:
        return
    if str(prior_state.get("state_version", "")) != COLLECTION_STATE_VERSION:
        raise ValueError("cannot resume an incompatible Research-v2 collection state")
    previous_identity = prior_state.get("identity")
    if previous_identity != identity:
        changed = sorted(
            key
            for key in set(previous_identity or {}) | set(identity)
            if (previous_identity or {}).get(key) != identity.get(key)
        )
        raise ValueError(
            "refusing to resume with changed collection semantics/code: "
            + ", ".join(changed)
        )


def _can_reuse_completed_manifest(
    root: Path,
    *,
    prior_state: dict[str, Any] | None,
    prior_manifest: dict[str, Any] | None,
    targets: dict[str, int],
    resume: bool,
) -> bool:
    if not resume or prior_state is None or prior_manifest is None:
        return False
    if prior_state.get("status") != "complete" or prior_state.get("targets") != targets:
        return False
    if prior_manifest.get("status") != "complete":
        return False
    for split, target in targets.items():
        if target <= 0:
            continue
        indexed = _indexed_episode_paths(root / split)
        if set(indexed) != set(range(target)):
            return False
    return True


def _validate_requested_targets(root: Path, targets: dict[str, int]) -> None:
    """Never silently orphan episodes when a resume target is reduced."""

    for split, target in targets.items():
        indexed = _indexed_episode_paths(root / split)
        out_of_range = sorted(index for index in indexed if index >= target)
        if out_of_range:
            raise ValueError(
                f"cannot reduce {split} target to {target}; existing episode indices "
                f"would be orphaned: {out_of_range[:8]}"
            )


def _handle_stale_temporaries(root: Path, *, resume: bool) -> int:
    stale = sorted(root.glob("*/episode_*.hdf5.v2tmp"))
    if stale and not resume:
        raise FileExistsError(
            f"found {len(stale)} interrupted Research-v2 temporary files; use --resume"
        )
    for path in stale:
        path.unlink()
    return len(stale)


@contextmanager
def _exclusive_collection_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".research_v2_collection.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another Research-v2 collector is already writing {root}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _collection_code_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scripts/collect_research_v2_dataset.py",
        "scripts/collect_fe_pc_wam_dataset.py",
        "data/collect.py",
        "data/local_observation.py",
        "data/policies.py",
        "data/research_v2.py",
        "data/schema.py",
        "envs/two_robot_carry_env.py",
    ):
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the formal Research-v2 matched-intervention dataset with "
            "deterministic process parallelism and safe resume"
        )
    )
    parser.add_argument("--out-dir", default="datasets/research_v2")
    parser.add_argument("--train-episodes", type=int, default=6400)
    parser.add_argument("--val-episodes", type=int, default=800)
    parser.add_argument("--test-episodes", type=int, default=800)
    parser.add_argument("--pilot-episodes", type=int, default=100)
    parser.add_argument("--pilot-success-gate", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--randomize", type=int, choices=(0, 1), default=1)
    parser.add_argument("--object-position-std", type=float, default=0.025)
    parser.add_argument("--object-yaw-std", type=float, default=0.035)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="episode worker processes; 0 auto-selects up to 16 (recommended)",
    )
    parser.add_argument(
        "--start-method",
        choices=multiprocessing.get_all_start_methods(),
        default=_default_start_method(),
        help="multiprocessing start method (fork is fastest on the target Linux host)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GIB,
        help="free disk space to preserve after estimated output (default: 20 GiB)",
    )
    parser.add_argument(
        "--estimated-episode-mib",
        type=float,
        default=DEFAULT_ESTIMATED_EPISODE_MIB,
        help="conservative disk preflight estimate per episode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="audit completed files, remove stale temporary files, and fill missing indices",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect 2/1/1 episodes with no pilot and at most 240 steps",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.train_episodes = 2
        args.val_episodes = 1
        args.test_episodes = 1
        args.pilot_episodes = 0
        args.episode_len = min(args.episode_len, 240)
    print(json.dumps(collect_research_v2(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
