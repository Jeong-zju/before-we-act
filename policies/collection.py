"""Deterministic behavior mixture for proprioceptive WAM data collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PHASE0_BEHAVIOR_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("scripted_oracle_v1", 30),
    ("oracle_ou_noise_v1", 25),
    ("delayed_response_v1", 15),
    ("response_rate_v1", 10),
    ("smooth_random_v1", 10),
    ("counterfactual_stop_v1", 5),
    ("induced_failure_v1", 5),
)


@dataclass(frozen=True)
class CollectionBehavior:
    behavior_id: str
    perturbation_config: dict[str, Any]


class CooperativeStopCollectionPolicy:
    """Generate the seven behavior groups specified by technical-plan §11.2.

    This policy is deliberately restricted to offline data generation. It may
    use the environment's oracle and event state to create controlled success,
    delayed-response, counterfactual, and failure trajectories; none of these
    values become WAM runtime inputs.
    """

    PROFILES = ("scripted_oracle_v1", "phase0_mixed_v1")

    def __init__(
        self,
        env: Any,
        *,
        profile: str = "scripted_oracle_v1",
        mixture_seed: int = 20260714,
    ) -> None:
        if profile not in self.PROFILES:
            raise ValueError(f"unknown collection behavior profile {profile!r}")
        self.env = env
        self.profile = profile
        self.mixture_seed = int(mixture_seed)
        self.behavior = CollectionBehavior("scripted_oracle_v1", {})
        self.rng = np.random.default_rng(self.mixture_seed)
        self._ou_noise = np.zeros(6, dtype=np.float64)
        self._smooth_action = np.zeros(6, dtype=np.float64)

    def configure_episode(
        self,
        *,
        episode_index: int,
        episode_seed: int,
    ) -> CollectionBehavior:
        behavior_id = (
            "scripted_oracle_v1"
            if self.profile == "scripted_oracle_v1"
            else self._scheduled_behavior(episode_index)
        )
        seed_sequence = np.random.SeedSequence(
            [self.mixture_seed, int(episode_index), int(episode_seed)]
        )
        self.rng = np.random.default_rng(seed_sequence)
        self._ou_noise[:] = 0.0
        self._smooth_action[:] = 0.0
        config = self._sample_config(behavior_id)
        self.behavior = CollectionBehavior(behavior_id, config)
        return self.behavior

    def act(self, observation: Any) -> np.ndarray:
        del observation
        behavior_id = self.behavior.behavior_id
        if behavior_id == "scripted_oracle_v1":
            return self._oracle()
        if behavior_id == "oracle_ou_noise_v1":
            return self._oracle_with_noise()
        if behavior_id == "delayed_response_v1":
            return self._delayed_response()
        if behavior_id == "response_rate_v1":
            return self._response_rate()
        if behavior_id == "smooth_random_v1":
            return self._smooth_random()
        if behavior_id == "counterfactual_stop_v1":
            return self._counterfactual_stop()
        if behavior_id == "induced_failure_v1":
            return self._induced_failure()
        raise RuntimeError(f"unsupported configured behavior {behavior_id!r}")

    def _scheduled_behavior(self, episode_index: int) -> str:
        cycle_size = sum(weight for _, weight in PHASE0_BEHAVIOR_WEIGHTS)
        block, offset = divmod(int(episode_index), cycle_size)
        schedule = [
            behavior_id
            for behavior_id, weight in PHASE0_BEHAVIOR_WEIGHTS
            for _ in range(weight)
        ]
        np.random.default_rng(
            np.random.SeedSequence([self.mixture_seed, block])
        ).shuffle(schedule)
        return schedule[offset]

    def _sample_config(self, behavior_id: str) -> dict[str, Any]:
        if behavior_id == "scripted_oracle_v1":
            return {}
        if behavior_id == "oracle_ou_noise_v1":
            return {
                "rho": 0.85,
                "sigma": float(self.rng.choice((0.05, 0.10, 0.20))),
                "motion_dimensions": [0, 1, 2, 4, 5, 6],
            }
        if behavior_id == "delayed_response_v1":
            return {
                "delay_steps": int(self.rng.choice((5, 10, 20, 30))),
                "cruise_command": 0.70,
            }
        if behavior_id == "response_rate_v1":
            return {"response_gain": float(self.rng.choice((0.45, 0.65, 1.35, 1.75)))}
        if behavior_id == "smooth_random_v1":
            return {
                "rho": 0.92,
                "sigma": float(self.rng.choice((0.08, 0.12, 0.18))),
                "forward_bias": float(self.rng.choice((0.25, 0.45, 0.65))),
            }
        if behavior_id == "counterfactual_stop_v1":
            return {
                "mode": str(self.rng.choice(("stationary", "early_stop"))),
                "stop_step": int(self.rng.integers(10, 31)),
            }
        if behavior_id == "induced_failure_v1":
            return {
                "released_agent": int(self.rng.integers(0, 2)),
                "release_step": int(self.rng.integers(8, 36)),
            }
        raise ValueError(f"unsupported behavior {behavior_id!r}")

    def _oracle(self) -> np.ndarray:
        return np.asarray(self.env.scripted_action(), dtype=np.float64).copy()

    def _oracle_with_noise(self) -> np.ndarray:
        action = self._oracle()
        config = self.behavior.perturbation_config
        rho = float(config["rho"])
        sigma = float(config["sigma"])
        self._ou_noise = rho * self._ou_noise + sigma * self.rng.normal(size=6)
        motion_indices = np.asarray((0, 1, 2, 4, 5, 6), dtype=np.int64)
        action[motion_indices] += self._ou_noise
        action[[3, 7]] = 1.0
        return np.clip(action, -1.0, 1.0)

    def _delayed_response(self) -> np.ndarray:
        action = self._oracle()
        if not bool(self.env.brake_event_active):
            return action
        delay_steps = int(self.behavior.perturbation_config["delay_steps"])
        elapsed = int(self.env.step_count - self.env.brake_event_step)
        if elapsed < delay_steps:
            offset = 4 * int(self.env.responding_agent)
            action[offset + 1] = float(
                self.behavior.perturbation_config["cruise_command"]
            )
        return action

    def _response_rate(self) -> np.ndarray:
        action = self._oracle()
        if bool(self.env.brake_event_active):
            offset = 4 * int(self.env.responding_agent)
            gain = float(self.behavior.perturbation_config["response_gain"])
            action[offset + 1] = np.clip(action[offset + 1] * gain, 0.0, 1.0)
        return action

    def _smooth_random(self) -> np.ndarray:
        config = self.behavior.perturbation_config
        rho = float(config["rho"])
        sigma = float(config["sigma"])
        target = np.asarray(
            [0.0, config["forward_bias"], 0.0, 0.0, config["forward_bias"], 0.0],
            dtype=np.float64,
        )
        self._smooth_action = (
            rho * self._smooth_action
            + (1.0 - rho) * target
            + sigma * self.rng.normal(size=6)
        )
        action = np.ones(8, dtype=np.float64)
        action[(0, 1, 2, 4, 5, 6),] = np.clip(self._smooth_action, -1.0, 1.0)
        return action

    def _counterfactual_stop(self) -> np.ndarray:
        config = self.behavior.perturbation_config
        if config["mode"] == "early_stop" and self.env.step_count < int(
            config["stop_step"]
        ):
            return self._oracle()
        action = np.zeros(8, dtype=np.float64)
        action[[3, 7]] = 1.0
        return action

    def _induced_failure(self) -> np.ndarray:
        action = self._oracle()
        config = self.behavior.perturbation_config
        if self.env.step_count >= int(config["release_step"]):
            action[4 * int(config["released_agent"]) + 3] = 0.0
        return action


__all__ = [
    "PHASE0_BEHAVIOR_WEIGHTS",
    "CollectionBehavior",
    "CooperativeStopCollectionPolicy",
]
