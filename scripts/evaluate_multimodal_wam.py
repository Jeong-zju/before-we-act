"""Run paired visual-required closed-loop evaluation for Phase M1."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner  # noqa: E402
from envs.visual_required_env import (  # noqa: E402
    VISUAL_REQUIRED_TASKS,
    VISUAL_REQUIRED_TASK_TEXTS,
    VisualRequiredEnv,
    VisualRequiredEnvConfig,
)
from eval.m1_acceptance import (  # noqa: E402
    CLEAN,
    FREEZE_FIRST_RGB,
    OCCLUDE_CUE,
    PRIMARY_INTERVENTIONS,
    PRIMARY_VARIANT,
    REQUIRED_VARIANTS,
    SHUFFLE_RGB,
    SHUFFLE_STATE,
)
from eval.m1_statistics import M1EpisodeRecord, aggregate_episode_records  # noqa: E402
from eval.m1_vision_contract import (  # noqa: E402
    validate_loaded_checkpoint_vision,
    validate_training_summary_vision,
)
from models.wam import ActionChunkConfig  # noqa: E402
from policies.multimodal_joint_wam import (  # noqa: E402
    MultimodalJointWAMPolicy,
    MultimodalJointWAMPolicyConfig,
)
from train.m1_checkpointing import load_m1_checkpoint  # noqa: E402


INTERNAL_CHECKPOINT_VARIANTS = {
    "state_only": "state_only",
    "vision_only": "vision_only",
    "state_vision_no_future": "state_vision_no_future",
    "state_vision_future": "state_vision_future",
    "parameter_matched_mlp": "parameter_matched_mlp",
}
CANONICAL_CONFIG = ROOT / "configs/wam_multimodal/m1_latent_wam_dinov3.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CANONICAL_CONFIG,
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--physical-seeds", type=int)
    parser.add_argument("--variants", nargs="+", choices=REQUIRED_VARIANTS)
    parser.add_argument("--train-seeds", nargs="+", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=(CLEAN, *PRIMARY_INTERVENTIONS),
        help="diagnostic-only primary-policy intervention subset",
    )
    parser.add_argument("--skip-smoke-gate", action="store_true")
    parser.add_argument(
        "--disable-replan-warm-start",
        action="store_true",
        help="diagnostic-only: cold-start every execute-2 replan",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=24,
        help="total CPU intra-op thread budget shared across rollout workers",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "independent CPU rollout workers; defaults to min(3, train seeds) "
            "on CPU and 1 on accelerators"
        ),
    )
    return parser


class _InterventionPolicy:
    """Apply observation-only interventions before the learned policy."""

    def __init__(self, base: MultimodalJointWAMPolicy, condition: str) -> None:
        if condition not in {
            CLEAN,
            FREEZE_FIRST_RGB,
            OCCLUDE_CUE,
            SHUFFLE_STATE,
            SHUFFLE_RGB,
        }:
            raise ValueError(f"unsupported M1 intervention {condition!r}")
        self.base = base
        self.condition = condition
        self._first_rgb: np.ndarray | None = None
        self._state_history: list[np.ndarray] = []
        self.last_diagnostics: dict[str, Any] = {}
        self.latencies_ms: list[float] = []
        self.action_ages_ms: list[float] = []
        self.deadline_misses = 0
        self.replan_events = 0
        self.cold_replan_events = 0
        self.warm_replan_events = 0
        self.actions_finite_and_bounded = True

    def reset(self) -> None:
        self.base.reset()
        self._first_rgb = None
        self._state_history.clear()
        self.last_diagnostics = {}
        self.latencies_ms.clear()
        self.action_ages_ms.clear()
        self.deadline_misses = 0
        self.replan_events = 0
        self.cold_replan_events = 0
        self.warm_replan_events = 0
        self.actions_finite_and_bounded = True

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        transformed = _copy_observation(observation)
        if self.condition == FREEZE_FIRST_RGB:
            image = _fixed_rgb(transformed)
            if self._first_rgb is None:
                self._first_rgb = image.copy()
            transformed["images"]["fixed"] = self._first_rgb.copy()
        elif self.condition == OCCLUDE_CUE:
            transformed["images"]["fixed"] = _occlude_high_chroma(
                _fixed_rgb(transformed)
            )
        elif self.condition == SHUFFLE_STATE:
            if "proprioception" not in transformed:
                raise KeyError("state shuffle requires proprioception")
            state = np.asarray(transformed["proprioception"], dtype=np.float32).copy()
            self._state_history.append(state)
            # A deterministic 8-control-step temporal permutation.  The state
            # stream remains real held-out proprioception but is mismatched to
            # the current RGB/action decision by 400 ms once history exists.
            donor = self._state_history[max(0, len(self._state_history) - 9)]
            transformed["proprioception"] = donor.copy()
        action = np.asarray(self.base.act(transformed), dtype=np.float32)
        valid = bool(
            action.shape == (8,)
            and np.isfinite(action).all()
            and np.max(np.abs(action)) <= 1.0 + 1e-6
        )
        self.actions_finite_and_bounded &= valid
        diagnostics = dict(self.base.last_diagnostics)
        diagnostics["intervention"] = self.condition
        diagnostics["state_intervention"] = (
            "temporal_lag_8_control_steps"
            if self.condition == SHUFFLE_STATE
            else "none"
        )
        diagnostics["rgb_intervention"] = self.condition
        self.last_diagnostics = diagnostics
        if diagnostics.get("planned_mode") == "m1_latent_flow":
            warm_start_enabled = diagnostics.get("replan_warm_start_enabled")
            if (
                not isinstance(warm_start_enabled, bool)
                or warm_start_enabled is not self.base.config.replan_warm_start_enabled
            ):
                raise TypeError(
                    "M1 replan diagnostics must match the configured warm-start mode"
                )
            warm_start_used = diagnostics.get("warm_start_used")
            if not isinstance(warm_start_used, bool):
                raise TypeError(
                    "M1 replan diagnostics must report boolean warm_start_used"
                )
            self.replan_events += 1
            if warm_start_used:
                self.warm_replan_events += 1
            else:
                self.cold_replan_events += 1
        latency = diagnostics.get("latency_ms")
        age = diagnostics.get("action_age_ms")
        if isinstance(latency, (float, int)) and np.isfinite(float(latency)):
            self.latencies_ms.append(float(latency))
        if isinstance(age, (float, int)) and np.isfinite(float(age)):
            self.action_ages_ms.append(float(age))
        self.deadline_misses += int(diagnostics.get("deadline_exceeded") is True)
        return action


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve(strict=True)
    config = _load_yaml(config_path)
    variants = tuple(args.variants or REQUIRED_VARIANTS)
    train_seeds = tuple(
        args.train_seeds or [int(value) for value in config["training"]["seeds"]]
    )
    diagnostic_conditions = (
        None
        if args.conditions is None
        else tuple(str(value) for value in args.conditions)
    )
    if args.disable_replan_warm_start:
        config["action_chunk"]["replan_warm_start_enabled"] = False
    if diagnostic_conditions is not None and variants != (PRIMARY_VARIANT,):
        raise ValueError(
            "diagnostic condition subsets require only state_vision_future"
        )
    evaluation = config["evaluation"]
    physical_count = int(
        args.physical_seeds
        if args.physical_seeds is not None
        else evaluation["formal_physical_seeds_per_task"]
    )
    if physical_count <= 0:
        raise ValueError("physical-seeds must be positive")
    formal = bool(
        config_path == CANONICAL_CONFIG.resolve()
        and args.checkpoint_root is None
        and args.training_summary is None
        and args.output_dir is None
        and args.physical_seeds is None
        and args.variants is None
        and args.train_seeds is None
        and args.conditions is None
        and not args.skip_smoke_gate
        and not args.disable_replan_warm_start
        and variants == REQUIRED_VARIANTS
        and train_seeds == tuple(int(value) for value in config["training"]["seeds"])
        and physical_count == int(evaluation["formal_physical_seeds_per_task"])
    )
    if not formal and args.output_dir is None:
        raise ValueError("diagnostic evaluation requires a separate --output-dir")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    device = _device(args.device)
    workers = int(
        args.workers
        if args.workers is not None
        else (
            min(3, len(train_seeds), args.torch_threads) if device.type == "cpu" else 1
        )
    )
    if workers <= 0:
        raise ValueError("workers must be positive")
    if device.type != "cpu" and workers != 1:
        raise ValueError("multi-process M1 evaluation is supported only on CPU")
    if workers > args.torch_threads:
        raise ValueError("workers cannot exceed the total torch thread budget")
    workers = min(workers, len(variants) * len(train_seeds))
    threads_per_worker = max(1, args.torch_threads // workers)
    if workers == 1:
        _configure_torch_threads(args.torch_threads)
    checkpoint_root = (
        args.checkpoint_root or ROOT / str(config["training"]["checkpoint_root"])
    ).resolve()
    training_summary_path = (
        args.training_summary
        or ROOT / str(config["training"]["report_root"]) / "training_summary.json"
    ).resolve()
    output = (args.output_dir or ROOT / str(evaluation["output_directory"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "visual_required_episodes.jsonl"
    if formal and records_path.exists():
        raise FileExistsError(
            f"refusing to mix stale formal M1 records: {records_path}"
        )
    training_summary = _read_json(training_summary_path)
    _validate_training_summary_binding(
        training_summary,
        config,
        config_path=config_path,
        checkpoint_root=checkpoint_root,
        train_seeds=train_seeds,
    )
    if formal and (
        training_summary.get("formal_protocol") is not True
        or training_summary.get("passed") is not True
    ):
        raise RuntimeError(
            "formal M1 evaluation requires accepted formal training evidence"
        )
    expected_checkpoints = {
        variant: {
            seed: checkpoint_root
            / INTERNAL_CHECKPOINT_VARIANTS[variant]
            / f"seed_{seed}"
            for seed in train_seeds
        }
        for variant in variants
    }
    for variant, values in expected_checkpoints.items():
        for seed, path in values.items():
            if not path.is_dir():
                raise FileNotFoundError(
                    f"missing M1 checkpoint {variant}/seed-{seed}: {path}"
                )

    seed_start = int(evaluation["physical_seed_start"])
    evaluation_seeds = tuple(range(seed_start, seed_start + physical_count))
    smoke_count = min(int(evaluation["smoke_physical_seeds_per_task"]), physical_count)
    phases = (
        ("smoke", evaluation_seeds[:smoke_count]),
        ("formal_remainder", evaluation_seeds[smoke_count:]),
    )
    records: list[M1EpisodeRecord] = []
    runtime_latencies: list[float] = []
    runtime_ages: list[float] = []
    runtime_deadline_misses = 0
    runtime_replans = 0
    runtime_cold_replans = 0
    runtime_warm_replans = 0
    completed = 0
    started = time.perf_counter()
    for phase_name, phase_seeds in phases:
        if not phase_seeds:
            continue
        phase_records, phase_runtime = _evaluate_phase(
            config,
            checkpoints=expected_checkpoints,
            variants=variants,
            train_seeds=train_seeds,
            physical_seeds=phase_seeds,
            device=device,
            completed_start=completed,
            workers=workers,
            torch_threads=threads_per_worker,
            condition_override=diagnostic_conditions,
        )
        completed += len(phase_records)
        records.extend(phase_records)
        runtime_latencies.extend(phase_runtime["latencies_ms"])
        runtime_ages.extend(phase_runtime["action_ages_ms"])
        runtime_deadline_misses += int(phase_runtime["deadline_misses"])
        runtime_replans += int(phase_runtime["replan_events"])
        runtime_cold_replans += int(phase_runtime["cold_replan_events"])
        runtime_warm_replans += int(phase_runtime["warm_replan_events"])
        _write_jsonl(records_path, records)
        if phase_name == "smoke" and not args.skip_smoke_gate:
            smoke = _smoke_gate(records, train_seeds=train_seeds)
            _write_json(output / "smoke_gate.json", smoke)
            if not smoke["passed"]:
                report = _evaluation_report(
                    records,
                    formal_protocol=False,
                    phase="smoke_stopped",
                    expected_records=None,
                    evaluation_seeds=phase_seeds,
                    runtime_latencies=runtime_latencies,
                    runtime_ages=runtime_ages,
                    deadline_misses=runtime_deadline_misses,
                    replan_events=runtime_replans,
                    cold_replan_events=runtime_cold_replans,
                    warm_replan_events=runtime_warm_replans,
                    elapsed=time.perf_counter() - started,
                    config_path=config_path,
                    training_summary_path=training_summary_path,
                    episode_records_path=records_path,
                    control_hz=float(evaluation["control_hz"]),
                    visual_hz=float(evaluation["visual_refresh_hz"]),
                    execution_steps=int(config["action_chunk"]["execution_steps"]),
                    replan_warm_start_enabled=bool(
                        config["action_chunk"]["replan_warm_start_enabled"]
                    ),
                    training_warm_start_probability=float(
                        config["action_chunk"]["warm_start_probability"]
                    ),
                    workers=workers,
                    torch_threads_per_worker=(
                        args.torch_threads if workers == 1 else threads_per_worker
                    ),
                    direct_budget_ms=float(
                        config["acceptance"]["maximum_sensor_to_action_p95_ms"]
                    ),
                    decimated_budget_ms=float(
                        config["acceptance"]["maximum_decimated_action_age_ms"]
                    ),
                    device=device,
                )
                _write_json(output / "visual_required_evaluation.json", report)
                raise RuntimeError(f"M1 20-seed smoke gate failed: {smoke}")

    expected_records = _expected_episode_count(
        variants=variants,
        train_seeds=train_seeds,
        tasks=VISUAL_REQUIRED_TASKS,
        physical_seeds=evaluation_seeds,
        cue_variants=config["evaluation"]["cue_variants"],
        condition_override=diagnostic_conditions,
    )
    report = _evaluation_report(
        records,
        formal_protocol=formal,
        phase="formal" if formal else "diagnostic",
        expected_records=expected_records,
        evaluation_seeds=evaluation_seeds,
        runtime_latencies=runtime_latencies,
        runtime_ages=runtime_ages,
        deadline_misses=runtime_deadline_misses,
        replan_events=runtime_replans,
        cold_replan_events=runtime_cold_replans,
        warm_replan_events=runtime_warm_replans,
        elapsed=time.perf_counter() - started,
        config_path=config_path,
        training_summary_path=training_summary_path,
        episode_records_path=records_path,
        control_hz=float(evaluation["control_hz"]),
        visual_hz=float(evaluation["visual_refresh_hz"]),
        execution_steps=int(config["action_chunk"]["execution_steps"]),
        replan_warm_start_enabled=bool(
            config["action_chunk"]["replan_warm_start_enabled"]
        ),
        training_warm_start_probability=float(
            config["action_chunk"]["warm_start_probability"]
        ),
        workers=workers,
        torch_threads_per_worker=(
            args.torch_threads if workers == 1 else threads_per_worker
        ),
        direct_budget_ms=float(config["acceptance"]["maximum_sensor_to_action_p95_ms"]),
        decimated_budget_ms=float(
            config["acceptance"]["maximum_decimated_action_age_ms"]
        ),
        device=device,
    )
    report["passed"] = bool(
        formal
        and len(records) == expected_records
        and report["aggregation"]["passed"]
        and all(not item.privileged_observation_seen for item in records)
        and all(not item.fallback_used for item in records)
        and all(item.actions_finite_and_bounded for item in records)
        and config["action_chunk"]["replan_warm_start_enabled"] is False
        and float(config["action_chunk"]["warm_start_probability"]) == 0.0
        and runtime_replans > 0
        and runtime_cold_replans == runtime_replans
        and runtime_warm_replans == 0
        and all(
            item.warm_replan_events == 0
            and item.cold_replan_events == item.replan_events
            and item.replan_events == (item.steps + 1) // 2
            for item in records
            if item.model_variant == PRIMARY_VARIANT and item.intervention == CLEAN
        )
    )
    _write_json(output / "visual_required_evaluation.json", report)
    if formal and not report["passed"]:
        raise RuntimeError("formal M1 visual evaluation evidence is incomplete")
    return 0


def _evaluate_phase(
    config: Mapping[str, Any],
    *,
    checkpoints: Mapping[str, Mapping[int, Path]],
    variants: Sequence[str],
    train_seeds: Sequence[int],
    physical_seeds: Sequence[int],
    device: torch.device,
    completed_start: int,
    workers: int = 1,
    torch_threads: int = 1,
    condition_override: Sequence[str] | None = None,
) -> tuple[list[M1EpisodeRecord], dict[str, Any]]:
    if not physical_seeds:
        return [], {
            "latencies_ms": [],
            "action_ages_ms": [],
            "deadline_misses": 0,
            "replan_events": 0,
            "cold_replan_events": 0,
            "warm_replan_events": 0,
        }
    jobs = _evaluation_jobs(
        checkpoints=checkpoints,
        variants=variants,
        train_seeds=train_seeds,
        tasks=VISUAL_REQUIRED_TASKS,
        physical_seeds=physical_seeds,
        cue_variants=config["evaluation"]["cue_variants"],
        completed_start=completed_start,
        condition_override=condition_override,
    )
    total = _expected_episode_count(
        variants=variants,
        train_seeds=train_seeds,
        tasks=VISUAL_REQUIRED_TASKS,
        physical_seeds=physical_seeds,
        cue_variants=config["evaluation"]["cue_variants"],
        condition_override=condition_override,
    )
    if sum(int(job["records"]) for job in jobs) != total:
        raise RuntimeError(
            "M1 evaluation job partition does not cover the exact matrix"
        )

    results: dict[int, tuple[list[M1EpisodeRecord], dict[str, Any]]] = {}
    if workers == 1:
        for job in jobs:
            results[int(job["index"])] = _evaluate_checkpoint_job(
                config,
                job,
                physical_seeds=physical_seeds,
                device=str(device),
                phase_total=total,
                phase_completed_start=completed_start,
            )
    else:
        if device.type != "cpu":
            raise ValueError("parallel M1 evaluation requires a CPU device")
        worker_count = min(int(workers), len(jobs))
        if worker_count <= 0 or torch_threads <= 0:
            raise ValueError("workers and torch_threads must be positive")
        # Submit the three five-condition primary jobs first.  This balances
        # their 5x workload across workers; output is nevertheless reassembled
        # by the preregistered variant/seed index below, so scheduling cannot
        # change JSONL ordering or paired statistics.
        submission_order = sorted(
            jobs,
            key=lambda item: (-len(item["conditions"]), int(item["index"])),
        )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_configure_torch_threads,
            initargs=(int(torch_threads),),
        ) as executor:
            pending = {
                executor.submit(
                    _evaluate_checkpoint_job,
                    config,
                    job,
                    physical_seeds=tuple(int(value) for value in physical_seeds),
                    device="cpu",
                    phase_total=total,
                    phase_completed_start=completed_start,
                ): job
                for job in submission_order
            }
            for future in as_completed(pending):
                job = pending[future]
                results[int(job["index"])] = future.result()
                print(
                    "M1 visual evaluation completed job "
                    f"{len(results)}/{len(jobs)} "
                    f"({job['variant']}, seed={job['train_seed']})",
                    flush=True,
                )

    return _merge_evaluation_results(results, expected_jobs=len(jobs))


def _evaluate_checkpoint_job(
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    physical_seeds: Sequence[int],
    device: str,
    phase_total: int,
    phase_completed_start: int,
) -> tuple[list[M1EpisodeRecord], dict[str, Any]]:
    """Evaluate one checkpoint once across all of its registered conditions."""

    variant = str(job["variant"])
    train_seed = int(job["train_seed"])
    conditions = tuple(str(value) for value in job["conditions"])
    completed = int(job["completed_start"])
    result: list[M1EpisodeRecord] = []
    latencies: list[float] = []
    ages: list[float] = []
    misses = 0
    replan_events = 0
    cold_replan_events = 0
    warm_replan_events = 0
    model, flow, legacy_world, legacy_flow, metadata = load_m1_checkpoint(
        Path(str(job["checkpoint"])),
        device=device,
        expected_schema_version=str(config["data"]["schema_version"]),
    )
    if metadata["schema"].get("model_variant") != variant:
        raise ValueError("M1 checkpoint variant differs from evaluation matrix")
    if int(metadata["schema"].get("train_seed", -1)) != train_seed:
        raise ValueError("M1 checkpoint train seed differs from evaluation matrix")
    validate_loaded_checkpoint_vision(config, model, metadata)
    for condition in conditions:
        for task_id in VISUAL_REQUIRED_TASKS:
            render_mode = "opposite" if condition == SHUFFLE_RGB else "truth"
            env = VisualRequiredEnv(
                VisualRequiredEnvConfig(
                    task_id=task_id,
                    control_dt=1.0 / float(config["evaluation"]["control_hz"]),
                    episode_len=int(config["evaluation"]["max_steps"]),
                    image_width=int(config["model"]["vision_input_size"]),
                    image_height=int(config["model"]["vision_input_size"]),
                    render_cue_mode=render_mode,
                )
            )
            try:
                policy = _InterventionPolicy(
                    MultimodalJointWAMPolicy(
                        model,
                        flow,
                        legacy_world,
                        legacy_flow,
                        _policy_config(config),
                        device=device,
                    ),
                    condition,
                )
                consumes_vision = model.config.use_vision
                runner = SimulationRunner(
                    env,
                    policy,
                    RunnerConfig(
                        max_steps=int(config["evaluation"]["max_steps"]),
                        render=(
                            (
                                RenderRequest(
                                    "fixed",
                                    "fixed",
                                    width=int(config["model"]["vision_input_size"]),
                                    height=int(config["model"]["vision_input_size"]),
                                    fps=float(
                                        config["evaluation"]["visual_refresh_hz"]
                                    ),
                                ),
                            )
                            if consumes_vision
                            else ()
                        ),
                        policy_observation_keys=("proprioception",)
                        if model.config.use_state
                        else (),
                        expose_privileged_state_to_policy=False,
                        expose_rendered_images_to_policy=consumes_vision,
                        policy_image_streams=("fixed",) if consumes_vision else (),
                        expose_task_to_policy=True,
                        task_id=task_id,
                        task=VISUAL_REQUIRED_TASK_TEXTS[task_id],
                        policy_action_history=int(config["data"]["state_history"] - 1),
                    ),
                )
                _warm_policy_runtime(
                    env,
                    policy,
                    task_id=task_id,
                    consumes_state=model.config.use_state,
                    consumes_vision=consumes_vision,
                    seed=2 * int(physical_seeds[0]),
                    image_size=int(config["model"]["vision_input_size"]),
                )
                for physical_seed in physical_seeds:
                    for cue_id in config["evaluation"]["cue_variants"]:
                        episode_seed = 2 * int(physical_seed) + int(cue_id)
                        summary = runner.run_episode(
                            seed=episode_seed,
                            episode_index=completed,
                            randomize=True,
                        )
                        diagnostics = dict(policy.last_diagnostics)
                        info = summary.final_info
                        presented = tuple(
                            str(value)
                            for value in diagnostics.get(
                                "presented_observation_paths", ()
                            )
                        )
                        consumed = tuple(
                            str(value)
                            for value in diagnostics.get(
                                "consumed_observation_paths", ()
                            )
                        )
                        result.append(
                            M1EpisodeRecord(
                                task_id=task_id,
                                evaluation_seed=int(physical_seed),
                                cue_id=int(cue_id),
                                model_variant=variant,
                                train_seed=train_seed,
                                intervention=condition,
                                success=bool(info.get("success", False)),
                                steps=int(summary.steps),
                                total_reward=float(summary.total_reward),
                                action_source=str(diagnostics.get("action_source", "")),
                                presented_observation_paths=presented,
                                consumed_observation_paths=consumed,
                                privileged_observation_seen=bool(
                                    diagnostics.get("privileged_state_seen", False)
                                ),
                                fallback_used=bool(
                                    diagnostics.get("fallback_used", False)
                                ),
                                actions_finite_and_bounded=bool(
                                    policy.actions_finite_and_bounded
                                ),
                                replan_events=int(policy.replan_events),
                                cold_replan_events=int(policy.cold_replan_events),
                                warm_replan_events=int(policy.warm_replan_events),
                            )
                        )
                        # Deployment timing is a property of the preregistered
                        # primary policy on unmodified observations.  Mixing in
                        # state-only, cached ablations, or intervention paths
                        # could make the P95 look artificially faster or slower.
                        if variant == PRIMARY_VARIANT and condition == CLEAN:
                            latencies.extend(policy.latencies_ms)
                            ages.extend(policy.action_ages_ms)
                            misses += policy.deadline_misses
                            replan_events += policy.replan_events
                            cold_replan_events += policy.cold_replan_events
                            warm_replan_events += policy.warm_replan_events
                        completed += 1
                        if (completed - int(job["completed_start"])) % 50 == 0:
                            local = completed - phase_completed_start
                            print(
                                f"M1 visual evaluation {local}/{phase_total} "
                                f"({variant}, seed={train_seed}, {condition}, {task_id})",
                                flush=True,
                            )
            finally:
                env.close()
    return result, {
        "latencies_ms": latencies,
        "action_ages_ms": ages,
        "deadline_misses": misses,
        "replan_events": replan_events,
        "cold_replan_events": cold_replan_events,
        "warm_replan_events": warm_replan_events,
    }


def _evaluation_jobs(
    *,
    checkpoints: Mapping[str, Mapping[int, Path]],
    variants: Sequence[str],
    train_seeds: Sequence[int],
    tasks: Sequence[str],
    physical_seeds: Sequence[int],
    cue_variants: Sequence[int],
    completed_start: int,
    condition_override: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    completed = int(completed_start)
    for variant in variants:
        conditions = (
            tuple(str(value) for value in condition_override)
            if condition_override is not None
            else (CLEAN, *PRIMARY_INTERVENTIONS)
            if variant == PRIMARY_VARIANT
            else (CLEAN,)
        )
        for train_seed in train_seeds:
            count = (
                len(conditions) * len(tasks) * len(physical_seeds) * len(cue_variants)
            )
            jobs.append(
                {
                    "index": len(jobs),
                    "variant": str(variant),
                    "train_seed": int(train_seed),
                    "checkpoint": str(checkpoints[variant][int(train_seed)]),
                    "conditions": tuple(conditions),
                    "completed_start": completed,
                    "records": count,
                }
            )
            completed += count
    return jobs


def _merge_evaluation_results(
    results: Mapping[int, tuple[list[M1EpisodeRecord], Mapping[str, Any]]],
    *,
    expected_jobs: int,
) -> tuple[list[M1EpisodeRecord], dict[str, Any]]:
    """Merge concurrently completed jobs in deterministic matrix order."""

    expected = set(range(int(expected_jobs)))
    if set(results) != expected:
        raise RuntimeError("M1 evaluation jobs are missing or unexpectedly indexed")
    records: list[M1EpisodeRecord] = []
    latencies: list[float] = []
    ages: list[float] = []
    misses = 0
    replan_events = 0
    cold_replan_events = 0
    warm_replan_events = 0
    for index in range(int(expected_jobs)):
        job_records, runtime = results[index]
        records.extend(job_records)
        latencies.extend(float(value) for value in runtime["latencies_ms"])
        ages.extend(float(value) for value in runtime["action_ages_ms"])
        misses += int(runtime["deadline_misses"])
        replan_events += int(runtime["replan_events"])
        cold_replan_events += int(runtime["cold_replan_events"])
        warm_replan_events += int(runtime["warm_replan_events"])
    return records, {
        "latencies_ms": latencies,
        "action_ages_ms": ages,
        "deadline_misses": misses,
        "replan_events": replan_events,
        "cold_replan_events": cold_replan_events,
        "warm_replan_events": warm_replan_events,
    }


def _expected_episode_count(
    *,
    variants: Sequence[str],
    train_seeds: Sequence[int],
    tasks: Sequence[str],
    physical_seeds: Sequence[int],
    cue_variants: Sequence[int],
    condition_override: Sequence[str] | None = None,
) -> int:
    condition_count = (
        len(condition_override) * len(variants)
        if condition_override is not None
        else len(variants)
        + (len(PRIMARY_INTERVENTIONS) if PRIMARY_VARIANT in variants else 0)
    )
    return (
        condition_count
        * len(train_seeds)
        * len(tasks)
        * len(physical_seeds)
        * len(cue_variants)
    )


def _configure_torch_threads(thread_count: int) -> None:
    """Bound intra/inter-op pools once per process to avoid oversubscription."""

    count = int(thread_count)
    if count <= 0:
        raise ValueError("torch thread count must be positive")
    torch.set_num_threads(count)
    # Inter-op pools sit outside the bounded intra-op pool.  One coordinator
    # per process prevents N workers from silently multiplying thread counts.
    torch.set_num_interop_threads(1)


def _warm_policy_runtime(
    env: VisualRequiredEnv,
    policy: _InterventionPolicy,
    *,
    task_id: str,
    consumes_state: bool,
    consumes_vision: bool,
    seed: int,
    image_size: int,
) -> None:
    """Warm both one-frame and two-frame tensor shapes outside evidence."""

    observation, _ = env.reset(seed=seed, randomize=True)
    policy.reset()
    actions: list[np.ndarray] = []
    for step in range(2):
        view: dict[str, Any] = {
            "task": {
                "id": task_id,
                "text": VISUAL_REQUIRED_TASK_TEXTS[task_id],
            },
            "past_executed_actions": np.asarray(actions, dtype=np.float32).reshape(
                -1, 8
            ),
        }
        if consumes_state:
            view["proprioception"] = np.asarray(
                observation["proprioception"], dtype=np.float32
            ).copy()
        if consumes_vision:
            view["images"] = {
                "fixed": env.render(camera="fixed", width=image_size, height=image_size)
            }
            view["image_frame_indices"] = {"fixed": step}
            view["image_timestamps"] = {"fixed": 0.1 * step}
        action = policy.act(view)
        actions.append(action.copy())
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    policy.reset()


def _policy_config(config: Mapping[str, Any]) -> MultimodalJointWAMPolicyConfig:
    chunk = config["action_chunk"]
    return MultimodalJointWAMPolicyConfig(
        action_chunk=ActionChunkConfig(
            action_dim=int(config["data"]["action_dim"]),
            horizon=int(chunk["horizon"]),
            execution_steps=int(chunk["execution_steps"]),
            solver_steps=int(chunk["solver_steps"]),
            warm_start_mode=str(chunk["warm_start_mode"]),
        ),
        solver=str(chunk["solver"]),
        normalized_action_clip=float(chunk["normalized_action_clip"]),
        visual_residual_scale=float(chunk["anchor_residual_scale_visual"]),
        cooperative_residual_scale=float(chunk["anchor_residual_scale_cooperative"]),
        replan_warm_start_enabled=bool(chunk.get("replan_warm_start_enabled", True)),
        latency_budget_ms=float(
            config["acceptance"]["maximum_sensor_to_action_p95_ms"]
        ),
        maximum_visual_age_ms=float(
            config["acceptance"]["maximum_decimated_action_age_ms"]
        ),
        control_period_ms=1000.0 / float(config["evaluation"]["control_hz"]),
        visual_history_frames=int(config["data"]["visual_history_frames"]),
        fallback_enabled=False,
    )


def _smoke_gate(
    records: Sequence[M1EpisodeRecord], *, train_seeds: Sequence[int]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        grouped[(record.model_variant, record.intervention)].append(record.success)
    rate = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in grouped.items()
    }
    main = rate.get((PRIMARY_VARIANT, CLEAN), 0.0)
    state = rate.get(("state_only", CLEAN), 0.0)
    visual_drops = {
        condition: main - rate.get((PRIMARY_VARIANT, condition), main)
        for condition in (SHUFFLE_RGB, FREEZE_FIRST_RGB, OCCLUDE_CUE)
        if (PRIMARY_VARIANT, condition) in rate
    }
    checks = {
        "three_train_seeds": len(set(train_seeds)) == 3,
        "primary_clean_at_least_60pct": main >= 0.60,
        "primary_beats_state_by_5pp": main - state >= 0.05,
        "one_visual_intervention_drops_10pp": bool(visual_drops)
        and max(visual_drops.values()) >= 0.10,
    }
    return {
        "format_version": "wam.multimodal.m1.smoke/1",
        "passed": all(checks.values()),
        "checks": checks,
        "rates": {f"{key[0]}/{key[1]}": value for key, value in sorted(rate.items())},
        "primary_minus_state": main - state,
        "visual_drops": visual_drops,
        "records": len(records),
    }


def _evaluation_report(
    records: Sequence[M1EpisodeRecord],
    *,
    formal_protocol: bool,
    phase: str,
    expected_records: int | None,
    evaluation_seeds: Sequence[int],
    runtime_latencies: Sequence[float],
    runtime_ages: Sequence[float],
    deadline_misses: int,
    replan_events: int,
    cold_replan_events: int,
    warm_replan_events: int,
    elapsed: float,
    config_path: Path,
    training_summary_path: Path,
    episode_records_path: Path,
    control_hz: float,
    visual_hz: float,
    execution_steps: int,
    replan_warm_start_enabled: bool,
    training_warm_start_probability: float,
    workers: int,
    torch_threads_per_worker: int,
    direct_budget_ms: float,
    decimated_budget_ms: float,
    device: torch.device,
) -> dict[str, Any]:
    aggregation = aggregate_episode_records(records)
    return {
        "format_version": "wam.multimodal.m1.visual_evaluation/1",
        "formal_protocol": bool(formal_protocol),
        "phase": phase,
        "records": len(records),
        "expected_records": expected_records,
        "evaluation_seeds": list(evaluation_seeds),
        "aggregation": aggregation,
        "runtime": {
            "sensor_to_action_ms": list(runtime_latencies),
            "action_age_ms": list(runtime_ages),
            "deadline_misses": int(deadline_misses),
            "decimated": True,
            "control_hz": float(control_hz),
            "visual_hz": float(visual_hz),
            "worker_processes": int(workers),
            "torch_threads_per_worker": int(torch_threads_per_worker),
            "warmup_actions_excluded": True,
            "sensor_to_action_definition": "policy_act_wall_time_ms",
            "action_age_definition": (
                "frame-receipt-to-action wall time, lower-bounded by nominal "
                "decimation staleness plus current policy_act wall time"
            ),
            "resolved_device": str(device),
            "hardware": _hardware_description(device),
            "runtime_scope": {
                "model_variant": PRIMARY_VARIANT,
                "intervention": CLEAN,
                "all_configured_train_seeds": True,
                "all_visual_required_tasks": True,
            },
            "replan_contract": {
                "execution_steps": int(execution_steps),
                "warm_start_enabled": bool(replan_warm_start_enabled),
                "training_warm_start_probability": float(
                    training_warm_start_probability
                ),
                "observation_regrounding": "cold_start_every_execute_2_replan",
                "scope": "m1_latent_flow_visual_required_only",
                "observed_replan_events": int(replan_events),
                "observed_cold_replan_events": int(cold_replan_events),
                "observed_warm_replan_events": int(warm_replan_events),
            },
            "deadline_contract": {
                "direct_without_vision_ms": float(direct_budget_ms),
                "decimated_with_vision_ms": float(decimated_budget_ms),
            },
        },
        "elapsed_seconds": float(elapsed),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "training_summary": str(training_summary_path),
        "training_summary_sha256": _sha256(training_summary_path),
        "episode_records": str(episode_records_path.resolve()),
        "episode_records_sha256": _sha256(episode_records_path.resolve()),
    }


def _copy_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[str(key)] = _copy_observation(item)
        elif isinstance(item, np.ndarray):
            result[str(key)] = item.copy()
        else:
            result[str(key)] = item
    return result


def _fixed_rgb(observation: Mapping[str, Any]) -> np.ndarray:
    images = observation.get("images")
    if not isinstance(images, Mapping) or "fixed" not in images:
        raise KeyError("RGB intervention requires images.fixed")
    image = np.asarray(images["fixed"])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB intervention requires uint8 HWC fixed image")
    return image


def _occlude_high_chroma(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    values = result.astype(np.int16)
    maximum = values.max(axis=-1)
    minimum = values.min(axis=-1)
    mask = (maximum >= 90) & ((maximum - minimum) >= 55)
    # Dilate without OpenCV so the intervention has no optional dependency.
    expanded = mask.copy()
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            expanded |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    result[expanded] = np.asarray([32, 32, 34], dtype=np.uint8)
    return result


def _write_jsonl(path: Path, records: Sequence[M1EpisodeRecord]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_training_summary_binding(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    checkpoint_root: Path,
    train_seeds: Sequence[int],
) -> None:
    if summary.get("format_version") != "wam.multimodal.m1.training/1":
        raise ValueError("unsupported M1 training summary")
    if summary.get("config_sha256") != _sha256(config_path):
        raise ValueError("training summary does not bind the evaluation config")
    declared_config = Path(str(summary.get("config", "")))
    declared_config = (
        declared_config if declared_config.is_absolute() else ROOT / declared_config
    ).resolve()
    if declared_config != config_path:
        raise ValueError("training summary config path differs from evaluation config")
    declared_root = Path(str(summary.get("checkpoint_root", "")))
    declared_root = (
        declared_root if declared_root.is_absolute() else ROOT / declared_root
    ).resolve()
    if declared_root != checkpoint_root:
        raise ValueError(
            "training summary checkpoint root differs from evaluation root"
        )
    if summary.get("manifest_sha256") != str(
        config["data"]["expected_manifest_sha256"]
    ):
        raise ValueError("training summary does not bind the configured manifest")
    available_seeds = {int(value) for value in summary.get("train_seeds", ())}
    if not set(int(value) for value in train_seeds).issubset(available_seeds):
        raise ValueError("training summary does not cover the evaluation train seeds")
    validate_training_summary_vision(summary, config, project_root=ROOT)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M1 config must contain a mapping")
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def _hardware_description(device: torch.device) -> dict[str, Any]:
    description: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        description.update(
            {
                "accelerator_name": properties.name,
                "accelerator_total_memory_bytes": int(properties.total_memory),
                "cuda_runtime": torch.version.cuda,
            }
        )
    else:
        cpu_model = ""
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        description["cpu_model"] = cpu_model
    return description


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
