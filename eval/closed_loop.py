"""Closed-loop rollout records and final evaluation for Joint WAM."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from envs.runtime import RolloutSummary, SimulationTransition


@dataclass(frozen=True)
class ClosedLoopEpisode:
    policy: str
    seed: int
    steps: int
    success: bool
    failure: bool
    failure_reason: str
    total_reward: float
    response_delay_seconds: float
    mean_coordination_error: float
    gradual_brake_steps: int
    stop_hold_steps: int
    pre_brake_motion_valid: bool
    planner_latency_ms: tuple[float, ...]
    planner_modes: tuple[str, ...]
    planner_attempted_modes: tuple[str, ...]
    deadline_misses: int
    discarded_plans: int
    fallback_reasons: tuple[str, ...]
    predicted_returns: tuple[float, ...]
    applied_flow_residuals: tuple[float, ...]
    observation_residual_nrmse: tuple[float, ...]
    predicted_robot_distances: tuple[float, ...]
    actual_robot_distances: tuple[float, ...]
    actions_finite_and_bounded: bool
    privileged_state_seen: bool


class ClosedLoopEpisodeObserver:
    """Collect one episode without exposing environment truth to the policy."""

    def __init__(self, policy_name: str, policy: Any) -> None:
        self.policy_name = str(policy_name)
        self.policy = policy
        self.seed = -1
        self.coordination_errors: list[float] = []
        self.latencies: list[float] = []
        self.modes: list[str] = []
        self.attempted_modes: list[str] = []
        self.deadline_misses = 0
        self.discarded_plans = 0
        self.fallback_reasons: list[str] = []
        self.predicted_returns: list[float] = []
        self.applied_flow_residuals: list[float] = []
        self.observation_residual_nrmse: list[float] = []
        self.predicted_robot_distances: list[float] = []
        self.actual_robot_distances: list[float] = []
        self.actions_valid = True
        self.privileged_state_seen = False

    def on_episode_start(self, *, seed: int | None, **_: Any) -> None:
        self.seed = -1 if seed is None else int(seed)

    def on_transition(self, transition: SimulationTransition) -> None:
        self.coordination_errors.append(
            float(transition.info.get("coordination_error", np.nan))
        )
        action = np.asarray(transition.action)
        self.actions_valid = bool(
            self.actions_valid
            and np.isfinite(action).all()
            and np.all(action >= -1.0)
            and np.all(action <= 1.0)
        )
        diagnostics = dict(getattr(self.policy, "last_diagnostics", {}) or {})
        if "latency_ms" in diagnostics:
            self.latencies.append(float(diagnostics["latency_ms"]))
        if "executed_mode" in diagnostics:
            self.modes.append(str(diagnostics["executed_mode"]))
        planned_mode = str(diagnostics.get("planned_mode", "none"))
        if planned_mode != "none":
            self.attempted_modes.append(planned_mode)
        deadline_exceeded = bool(diagnostics.get("deadline_exceeded", False))
        self.deadline_misses += int(deadline_exceeded)
        self.discarded_plans += int(
            planned_mode != "none"
            and not bool(diagnostics.get("plan_executed", False))
        )
        reason = str(diagnostics.get("fallback_reason", "none"))
        if reason != "none":
            self.fallback_reasons.append(reason)
        if diagnostics.get("plan_executed", False) and "expected_return" in diagnostics:
            self.predicted_returns.append(float(diagnostics["expected_return"]))
        if "applied_flow_residual_max" in diagnostics:
            self.applied_flow_residuals.append(
                float(diagnostics["applied_flow_residual_max"])
            )
        if diagnostics.get("observation_residual_nrmse") is not None:
            self.observation_residual_nrmse.append(
                float(diagnostics["observation_residual_nrmse"])
            )
        if diagnostics.get("predicted_robot_distance") is not None:
            self.predicted_robot_distances.append(
                float(diagnostics["predicted_robot_distance"])
            )
        if transition.info.get("robot_distance") is not None:
            self.actual_robot_distances.append(
                float(transition.info["robot_distance"])
            )
        self.privileged_state_seen = bool(
            self.privileged_state_seen
            or diagnostics.get("privileged_state_seen", False)
            or "privileged_state" in diagnostics.get("observation_keys", ())
        )

    def on_episode_end(self, summary: RolloutSummary) -> None:
        del summary

    def finish(self, summary: RolloutSummary) -> ClosedLoopEpisode:
        info = summary.final_info
        finite_coordination = np.asarray(self.coordination_errors, dtype=np.float64)
        finite_coordination = finite_coordination[np.isfinite(finite_coordination)]
        return ClosedLoopEpisode(
            policy=self.policy_name,
            seed=self.seed,
            steps=summary.steps,
            success=bool(info.get("success", False)),
            failure=bool(info.get("failure", False)),
            failure_reason=str(info.get("failure_reason", "none")),
            total_reward=float(summary.total_reward),
            response_delay_seconds=float(info.get("response_delay_seconds", -1.0)),
            mean_coordination_error=(
                float(finite_coordination.mean()) if finite_coordination.size else 0.0
            ),
            gradual_brake_steps=int(info.get("follower_brake_steps", 0)),
            stop_hold_steps=int(info.get("stop_hold_steps", 0)),
            pre_brake_motion_valid=bool(info.get("pre_brake_motion_valid", False)),
            planner_latency_ms=tuple(self.latencies),
            planner_modes=tuple(self.modes),
            planner_attempted_modes=tuple(self.attempted_modes),
            deadline_misses=self.deadline_misses,
            discarded_plans=self.discarded_plans,
            fallback_reasons=tuple(self.fallback_reasons),
            predicted_returns=tuple(self.predicted_returns),
            applied_flow_residuals=tuple(self.applied_flow_residuals),
            observation_residual_nrmse=tuple(self.observation_residual_nrmse),
            predicted_robot_distances=tuple(self.predicted_robot_distances),
            actual_robot_distances=tuple(self.actual_robot_distances),
            actions_finite_and_bounded=self.actions_valid,
            privileged_state_seen=self.privileged_state_seen,
        )


def aggregate_closed_loop(
    episodes: list[ClosedLoopEpisode],
    *,
    exploitation_predicted_return_min: float = 10.0,
    exploitation_actual_return_max: float = 0.0,
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot aggregate zero closed-loop episodes")
    latencies = np.asarray(
        [value for episode in episodes for value in episode.planner_latency_ms],
        dtype=np.float64,
    )
    response_delays = np.asarray(
        [
            episode.response_delay_seconds
            for episode in episodes
            if episode.response_delay_seconds >= 0.0
        ],
        dtype=np.float64,
    )
    applied_flow_residuals = np.asarray(
        [
            value
            for episode in episodes
            for value in episode.applied_flow_residuals
        ],
        dtype=np.float64,
    )
    observation_residuals = np.asarray(
        [
            value
            for episode in episodes
            for value in episode.observation_residual_nrmse
        ],
        dtype=np.float64,
    )
    predicted_robot_distances = np.asarray(
        [
            value
            for episode in episodes
            for value in episode.predicted_robot_distances
        ],
        dtype=np.float64,
    )
    actual_robot_distances = np.asarray(
        [
            value
            for episode in episodes
            for value in episode.actual_robot_distances
        ],
        dtype=np.float64,
    )
    modes = Counter(mode for episode in episodes for mode in episode.planner_modes)
    attempted_modes = Counter(
        mode for episode in episodes for mode in episode.planner_attempted_modes
    )
    fallback_reasons = Counter(
        reason for episode in episodes for reason in episode.fallback_reasons
    )
    failures = Counter(
        episode.failure_reason for episode in episodes if not episode.success
    )
    exploitation = 0
    for episode in episodes:
        if not episode.predicted_returns:
            continue
        if (
            max(episode.predicted_returns) >= exploitation_predicted_return_min
            and episode.total_reward <= exploitation_actual_return_max
            and not episode.success
        ):
            exploitation += 1
    fallback_steps = sum(
        count
        for mode, count in modes.items()
        if not _is_primary_mode(mode)
    )
    total_diagnostic_steps = sum(modes.values())
    total_steps = sum(episode.steps for episode in episodes)
    deadline_misses = sum(episode.deadline_misses for episode in episodes)
    discarded_plans = sum(episode.discarded_plans for episode in episodes)
    return {
        "episodes": len(episodes),
        "seed_min": min(episode.seed for episode in episodes),
        "seed_max": max(episode.seed for episode in episodes),
        "success_rate": float(np.mean([episode.success for episode in episodes])),
        "failure_rate": float(np.mean([not episode.success for episode in episodes])),
        "mean_episode_return": float(
            np.mean([episode.total_reward for episode in episodes])
        ),
        "mean_response_delay_seconds": (
            float(response_delays.mean()) if response_delays.size else None
        ),
        "mean_coordination_error": float(
            np.mean([episode.mean_coordination_error for episode in episodes])
        ),
        "mean_gradual_brake_steps": float(
            np.mean([episode.gradual_brake_steps for episode in episodes])
        ),
        "mean_stop_hold_steps": float(
            np.mean([episode.stop_hold_steps for episode in episodes])
        ),
        "premature_stationary_successes": sum(
            episode.success and not episode.pre_brake_motion_valid
            for episode in episodes
        ),
        "failure_reasons": dict(sorted(failures.items())),
        "planner_latency_ms": _percentiles(latencies),
        "planner_modes": dict(sorted(modes.items())),
        "planner_attempted_modes": dict(sorted(attempted_modes.items())),
        "total_steps": total_steps,
        "action_source_diagnostic_steps": total_diagnostic_steps,
        "action_source_coverage": (
            total_diagnostic_steps / total_steps if total_steps else 0.0
        ),
        "deadline_misses": deadline_misses,
        "discarded_plans": discarded_plans,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "fallback_trigger_rate": (
            fallback_steps / total_diagnostic_steps if total_diagnostic_steps else 0.0
        ),
        "model_exploitation_events": exploitation,
        "online_world_state_nrmse": _nrmse_summary(observation_residuals),
        "predicted_robot_distance": _percentiles(predicted_robot_distances),
        "actual_robot_distance": _percentiles(actual_robot_distances),
        "robot_distance_violation_episodes": sum(
            episode.failure_reason == "robot_too_far" for episode in episodes
        ),
        "applied_flow_residual": {
            "samples": int(applied_flow_residuals.size),
            "mean": (
                float(applied_flow_residuals.mean())
                if applied_flow_residuals.size
                else None
            ),
            "max": (
                float(applied_flow_residuals.max())
                if applied_flow_residuals.size
                else None
            ),
        },
        "all_actions_finite_and_bounded": all(
            episode.actions_finite_and_bounded for episode in episodes
        ),
        "privileged_state_leakage": any(
            episode.privileged_state_seen for episode in episodes
        ),
    }


def paired_policy_statistics(
    first: list[ClosedLoopEpisode],
    second: list[ClosedLoopEpisode],
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired seed-level differences with a deterministic bootstrap interval."""

    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid paired bootstrap settings")
    first_by_seed = {episode.seed: episode for episode in first}
    second_by_seed = {episode.seed: episode for episode in second}
    seeds = sorted(set(first_by_seed) & set(second_by_seed))
    if not seeds:
        raise ValueError("paired policies have no common seeds")
    success = np.asarray(
        [
            float(first_by_seed[item].success)
            - float(second_by_seed[item].success)
            for item in seeds
        ],
        dtype=np.float64,
    )
    returns = np.asarray(
        [
            first_by_seed[item].total_reward
            - second_by_seed[item].total_reward
            for item in seeds
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    success_bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    return_bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 1024):
        stop = min(start + 1024, bootstrap_samples)
        indices = rng.integers(0, len(seeds), size=(stop - start, len(seeds)))
        success_bootstrap[start:stop] = success[indices].mean(axis=1)
        return_bootstrap[start:stop] = returns[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence)

    def summary(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "ci_lower": float(np.quantile(bootstrap, alpha)),
            "ci_upper": float(np.quantile(bootstrap, 1.0 - alpha)),
        }

    return {
        "paired_seeds": len(seeds),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "success_difference": summary(success, success_bootstrap),
        "return_difference": summary(returns, return_bootstrap),
        "success_wins": int((success > 0.0).sum()),
        "success_ties": int((success == 0.0).sum()),
        "success_losses": int((success < 0.0).sum()),
        "return_wins": int((returns > 0.0).sum()),
        "return_ties": int((returns == 0.0).sum()),
        "return_losses": int((returns < 0.0).sum()),
    }


def episode_to_dict(episode: ClosedLoopEpisode) -> dict[str, Any]:
    return asdict(episode)


def _percentiles(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    non_finite = int(values.size - finite.size)
    if not finite.size:
        return {
            "samples": int(values.size),
            "finite_samples": 0,
            "non_finite": non_finite,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "samples": int(values.size),
        "finite_samples": int(finite.size),
        "non_finite": non_finite,
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def _nrmse_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize per-step normalized RMSE into an aggregate state NRMSE."""

    finite = values[np.isfinite(values)]
    return {
        "samples": int(values.size),
        "finite_samples": int(finite.size),
        "non_finite": int(values.size - finite.size),
        "mean": float(finite.mean()) if finite.size else None,
        "rms": float(np.sqrt(np.mean(np.square(finite)))) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
    }


def _is_primary_mode(mode: str) -> bool:
    return mode in {
        "action_prior",
        "safe_stop",
        "scripted_oracle",
        "stationary",
        "joint_wam_flow",
    }


__all__ = [
    "ClosedLoopEpisode",
    "ClosedLoopEpisodeObserver",
    "aggregate_closed_loop",
    "episode_to_dict",
    "paired_policy_statistics",
]
