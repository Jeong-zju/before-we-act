"""Closed-loop rollout records and Gate D evaluation for Phase 3."""

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
            planned_mode.startswith("mppi")
            and not bool(diagnostics.get("plan_executed", False))
        )
        reason = str(diagnostics.get("fallback_reason", "none"))
        if reason != "none":
            self.fallback_reasons.append(reason)
        if diagnostics.get("plan_executed", False) and "expected_return" in diagnostics:
            self.predicted_returns.append(float(diagnostics["expected_return"]))
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
        if mode
        not in {
            "mppi_full",
            "mppi_reduced",
            "action_prior",
            "safe_stop",
            "scripted_oracle",
        }
    )
    total_diagnostic_steps = sum(modes.values())
    executed_mppi_steps = sum(
        count for mode, count in modes.items() if mode.startswith("mppi")
    )
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
        "mppi_execution_rate": (
            executed_mppi_steps / total_diagnostic_steps
            if total_diagnostic_steps
            else 0.0
        ),
        "deadline_misses": deadline_misses,
        "discarded_plans": discarded_plans,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "fallback_trigger_rate": (
            fallback_steps / total_diagnostic_steps if total_diagnostic_steps else 0.0
        ),
        "model_exploitation_events": exploitation,
        "all_actions_finite_and_bounded": all(
            episode.actions_finite_and_bounded for episode in episodes
        ),
        "privileged_state_leakage": any(
            episode.privileged_state_seen for episode in episodes
        ),
    }


def gate_d_report(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    full_evaluation: bool,
    protocol: str = "standard_noninferiority_v2",
    minimum_episodes: int = 500,
    minimum_success_improvement: float = 0.10,
    maximum_success_regression: float = 0.01,
    maximum_return_regression: float = 0.5,
    minimum_mppi_execution_rate: float = 0.5,
    maximum_model_exploitation_events: int = 0,
    latency_budget_ms: float = 50.0,
    held_out_seed_overlap: int = 0,
) -> dict[str, Any]:
    if protocol not in {"success_improvement_v1", "standard_noninferiority_v2"}:
        raise ValueError(f"unsupported Gate D protocol {protocol!r}")
    required = ("wam_mppi", "action_prior", "stationary", "scripted_oracle")
    missing = [name for name in required if name not in metrics]
    if missing:
        return {
            "passed": False,
            "checks": {
                "required_policy_coverage": {
                    "passed": False,
                    "missing": missing,
                }
            },
            "full_evaluation": False,
        }
    mppi = metrics["wam_mppi"]
    prior = metrics["action_prior"]
    stationary = metrics["stationary"]
    prior_latency = prior["planner_latency_ms"].get("p95")
    success_delta = float(mppi["success_rate"]) - float(prior["success_rate"])
    return_delta = float(mppi["mean_episode_return"]) - float(
        prior["mean_episode_return"]
    )
    performance_noninferior = bool(
        success_delta >= -maximum_success_regression
        and return_delta >= -maximum_return_regression
    )
    deadline_misses = int(mppi.get("deadline_misses", 0))
    discarded_plans = int(mppi.get("discarded_plans", 0))
    fallback_validated = bool(
        prior_latency is not None
        and float(prior_latency) <= latency_budget_ms
        and prior["all_actions_finite_and_bounded"]
        and mppi["all_actions_finite_and_bounded"]
        and deadline_misses > 0
        and discarded_plans >= deadline_misses
        and performance_noninferior
        and int(mppi["model_exploitation_events"])
        <= maximum_model_exploitation_events
    )
    mppi_p95 = mppi["planner_latency_ms"].get("p95")
    performance_checks: dict[str, dict[str, Any]]
    if protocol == "success_improvement_v1":
        performance_checks = {
            "mppi_vs_action_prior": {
                "passed": success_delta >= minimum_success_improvement,
                "mppi_success_rate": float(mppi["success_rate"]),
                "action_prior_success_rate": float(prior["success_rate"]),
                "absolute_improvement": success_delta,
                "minimum": minimum_success_improvement,
            }
        }
    else:
        performance_checks = {
            "standard_task_success_noninferiority": {
                "passed": success_delta >= -maximum_success_regression,
                "mppi_success_rate": float(mppi["success_rate"]),
                "action_prior_success_rate": float(prior["success_rate"]),
                "absolute_improvement": success_delta,
                "maximum_regression": maximum_success_regression,
            },
            "standard_task_return_noninferiority": {
                "passed": return_delta >= -maximum_return_regression,
                "mppi_mean_return": float(mppi["mean_episode_return"]),
                "action_prior_mean_return": float(prior["mean_episode_return"]),
                "absolute_improvement": return_delta,
                "maximum_regression": maximum_return_regression,
            },
            "mppi_execution_coverage": {
                "passed": float(mppi.get("mppi_execution_rate", 0.0))
                >= minimum_mppi_execution_rate,
                "rate": float(mppi.get("mppi_execution_rate", 0.0)),
                "minimum": minimum_mppi_execution_rate,
            },
            "no_model_exploitation": {
                "passed": int(mppi["model_exploitation_events"])
                <= maximum_model_exploitation_events,
                "events": int(mppi["model_exploitation_events"]),
                "maximum": maximum_model_exploitation_events,
            },
        }
    checks = {
        "full_held_out_evaluation": {
            "passed": bool(
                full_evaluation
                and int(mppi["episodes"]) >= minimum_episodes
                and held_out_seed_overlap == 0
            ),
            "episodes": int(mppi["episodes"]),
            "minimum": minimum_episodes,
            "training_seed_overlap": held_out_seed_overlap,
        },
        **performance_checks,
        "no_premature_stop_reward_hack": {
            "passed": bool(
                int(mppi["premature_stationary_successes"]) == 0
                and float(stationary["success_rate"]) == 0.0
                and float(mppi["success_rate"]) > float(stationary["success_rate"])
            ),
            "mppi_premature_stationary_successes": int(
                mppi["premature_stationary_successes"]
            ),
            "stationary_success_rate": float(stationary["success_rate"]),
        },
        "latency_or_safe_fallback": {
            "passed": bool(
                (
                    deadline_misses == 0
                    and mppi_p95 is not None
                    and float(mppi_p95) <= latency_budget_ms
                )
                or fallback_validated
            ),
            "mppi_p95_ms": mppi_p95,
            "budget_ms": latency_budget_ms,
            "fallback_validated": fallback_validated,
            "deadline_misses": deadline_misses,
            "discarded_plans": discarded_plans,
            "action_prior_p95_ms": prior_latency,
        },
        "no_privileged_state_leakage": {
            "passed": not bool(mppi["privileged_state_leakage"]),
        },
        "finite_bounded_actions": {
            "passed": bool(mppi["all_actions_finite_and_bounded"]),
        },
    }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "protocol": protocol,
        "checks": checks,
        "full_evaluation": bool(full_evaluation),
    }


def episode_to_dict(episode: ClosedLoopEpisode) -> dict[str, Any]:
    return asdict(episode)


def _percentiles(values: np.ndarray) -> dict[str, float | int | None]:
    if not values.size:
        return {"samples": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "samples": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


__all__ = [
    "ClosedLoopEpisode",
    "ClosedLoopEpisodeObserver",
    "aggregate_closed_loop",
    "episode_to_dict",
    "gate_d_report",
]
