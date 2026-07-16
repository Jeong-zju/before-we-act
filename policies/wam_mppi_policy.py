"""Risk-aware MPPI and proprio-only runtime policy for Phase 3."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from models.wam import (
    RWMUEnsemble,
    WAMPlanningHeads,
    WorldModelRolloutInputs,
    WorldModelSequenceInputs,
)


@dataclass(frozen=True)
class MPPIConfig:
    planning_horizon: int = 20
    num_samples: int = 512
    num_elites: int = 64
    iterations: int = 4
    num_policy_trajectories: int = 32
    particles_per_candidate: int = 5
    candidate_batch_size: int = 128
    temperature: float = 0.5
    min_std: float = 0.05
    max_std: float = 1.0
    initial_std: float = 0.5
    discount: float = 0.99
    cvar_alpha: float = 0.2
    cvar_weight: float = 0.25
    warm_start: bool = True
    prior_action_max_delta: float = 0.15
    terminal_value_weight: float = 0.0

    def __post_init__(self) -> None:
        integer_names = (
            "planning_horizon",
            "num_samples",
            "num_elites",
            "iterations",
            "particles_per_candidate",
            "candidate_batch_size",
        )
        if any(int(getattr(self, name)) <= 0 for name in integer_names):
            raise ValueError("MPPI integer settings must be positive")
        if not 0 <= self.num_policy_trajectories <= self.num_samples:
            raise ValueError("num_policy_trajectories must be within num_samples")
        if self.num_elites > self.num_samples:
            raise ValueError("num_elites cannot exceed num_samples")
        if self.temperature <= 0.0 or self.min_std <= 0.0:
            raise ValueError("temperature and min_std must be positive")
        if not self.min_std <= self.initial_std <= self.max_std:
            raise ValueError("initial_std must lie within [min_std,max_std]")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0,1]")
        if not 0.0 < self.cvar_alpha <= 1.0 or not 0.0 <= self.cvar_weight <= 1.0:
            raise ValueError("invalid CVaR settings")
        if not 0.0 < self.prior_action_max_delta <= 2.0:
            raise ValueError("prior_action_max_delta must be in (0,2]")
        if not 0.0 <= self.terminal_value_weight <= 1.0:
            raise ValueError("terminal_value_weight must be in [0,1]")


@dataclass(frozen=True)
class MPPIRiskWeights:
    epistemic: float = 1.0
    aleatoric: float = 0.1
    failure: float = 2.0
    action_ood: float = 0.5
    action_magnitude: float = 0.01
    action_change: float = 0.05

    def __post_init__(self) -> None:
        if any(float(value) < 0.0 for value in vars(self).values()):
            raise ValueError("MPPI risk weights must be non-negative")


@dataclass(frozen=True)
class MPPISafetyConfig:
    latency_budget_ms: float = 50.0
    max_epistemic: float = 0.5
    max_action_ood: float = 1.0
    max_failure_probability: float = 0.9
    reduced_num_samples: int = 128
    reduced_num_elites: int = 16
    reduced_iterations: int = 2
    reduced_particles: int = 1
    recovery_steps: int = 20
    sticky_latency_fallback: bool = True
    discard_over_budget_plans: bool = True
    max_predicted_robot_distance: float = 1.15

    def __post_init__(self) -> None:
        if self.latency_budget_ms <= 0.0:
            raise ValueError("latency_budget_ms must be positive")
        for name in (
            "max_epistemic",
            "max_action_ood",
            "max_failure_probability",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if min(
            self.reduced_num_samples,
            self.reduced_num_elites,
            self.reduced_iterations,
            self.reduced_particles,
            self.recovery_steps,
        ) <= 0:
            raise ValueError("reduced planner settings must be positive")
        if self.reduced_num_elites > self.reduced_num_samples:
            raise ValueError("reduced elites cannot exceed reduced samples")
        if self.max_predicted_robot_distance <= 0.0:
            raise ValueError("max_predicted_robot_distance must be positive")


@dataclass(frozen=True)
class MPPIPlan:
    action: Tensor
    sequence: Tensor
    diagnostics: Mapping[str, float | int | str]


class RiskAwareMPPI:
    """CEM-style MPPI over fixed-member stochastic ensemble rollouts."""

    def __init__(
        self,
        ensemble: RWMUEnsemble,
        planning_heads: WAMPlanningHeads,
        config: MPPIConfig,
        *,
        risk_weights: MPPIRiskWeights | None = None,
        variance_scale: Tensor | np.ndarray | None = None,
        fixed_actions: Mapping[int, float] | None = None,
        outcome_positive_weights: Mapping[str, float] | None = None,
        max_predicted_robot_distance: float | None = None,
        seed: int = 0,
    ) -> None:
        self.ensemble = ensemble.eval()
        self.planning_heads = planning_heads.eval()
        self.config = config
        self.risk_weights = risk_weights or MPPIRiskWeights()
        self.device = next(ensemble.parameters()).device
        self.dtype = next(ensemble.parameters()).dtype
        expected_feature_dim = ensemble.members[0].planning_feature_dim
        if planning_heads.config.feature_dim != expected_feature_dim:
            raise ValueError("planning-head feature dimension does not match RWM")
        if planning_heads.config.action_dim != ensemble.member_config.action_dim:
            raise ValueError("planning-head action dimension does not match RWM")
        scale = torch.ones(ensemble.member_config.state_dim, dtype=torch.float32)
        if variance_scale is not None:
            scale = torch.as_tensor(variance_scale, dtype=torch.float32)
            if scale.shape != (ensemble.member_config.state_dim,):
                raise ValueError("variance_scale must have shape [state_dim]")
            if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
                raise ValueError("variance_scale must be finite and positive")
        self.variance_scale = scale.to(self.device)
        self.fixed_actions = {
            int(index): float(value) for index, value in (fixed_actions or {}).items()
        }
        for index, value in self.fixed_actions.items():
            if not 0 <= index < ensemble.member_config.action_dim:
                raise ValueError("fixed action index is out of range")
            if not -1.0 <= value <= 1.0:
                raise ValueError("fixed action values must be in [-1,1]")
        weights = {str(name): float(value) for name, value in (outcome_positive_weights or {}).items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
            raise ValueError("outcome positive weights must be finite and positive")
        self.outcome_logit_corrections = {
            name: float(np.log(weights.get(name, 1.0)))
            for name in ("done", "failure")
        }
        if max_predicted_robot_distance is not None and max_predicted_robot_distance <= 0.0:
            raise ValueError("max_predicted_robot_distance must be positive")
        self.max_predicted_robot_distance = max_predicted_robot_distance
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self._mean: Tensor | None = None

    def reset(self) -> None:
        self._mean = None

    @torch.inference_mode()
    def plan(
        self,
        history: WorldModelSequenceInputs,
        *,
        config: MPPIConfig | None = None,
    ) -> MPPIPlan:
        cfg = config or self.config
        history = _history_to(history, self.device, self.dtype)
        if history.states.shape[0] != 1:
            raise ValueError("online MPPI expects a single history")
        prior_mean = self._prior_sequences(
            history, 1, cfg.planning_horizon, deterministic=True
        )[0]
        if self._mean is not None and cfg.warm_start:
            mean = torch.cat((self._mean[1:], self._mean[-1:]), dim=0)
        else:
            mean = prior_mean
        mean = self._constrain_to_prior(mean, prior_mean, cfg.prior_action_max_delta)
        std = torch.full_like(mean, cfg.initial_std)
        latest: dict[str, Tensor] | None = None
        candidates: Tensor | None = None
        for _ in range(cfg.iterations):
            gaussian_count = cfg.num_samples - cfg.num_policy_trajectories
            noise = torch.randn(
                gaussian_count,
                cfg.planning_horizon,
                self.ensemble.member_config.action_dim,
                device=self.device,
                dtype=self.dtype,
                generator=self.generator,
            )
            gaussian = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidate_parts = [gaussian]
            if cfg.num_policy_trajectories:
                candidate_parts.append(
                    self._prior_sequences(
                        history,
                        cfg.num_policy_trajectories,
                        cfg.planning_horizon,
                    )
                )
            candidates = self._constrain_to_prior(
                self._constrain(torch.cat(candidate_parts, dim=0)),
                prior_mean,
                cfg.prior_action_max_delta,
            )
            latest = self._score_in_chunks(history, candidates, cfg)
            elite_scores, elite_indices = torch.topk(
                latest["score"], cfg.num_elites, sorted=False
            )
            elites = candidates[elite_indices]
            weights = torch.softmax(
                (elite_scores - elite_scores.max()) / cfg.temperature, dim=0
            )
            mean = (weights[:, None, None] * elites).sum(dim=0)
            variance = (
                weights[:, None, None] * (elites - mean).square()
            ).sum(dim=0)
            std = variance.clamp_min(cfg.min_std**2).sqrt().clamp_max(cfg.max_std)
            mean = self._constrain_to_prior(
                self._constrain(mean), prior_mean, cfg.prior_action_max_delta
            )
        if candidates is None or latest is None:
            raise RuntimeError("MPPI produced no candidates")
        best = int(latest["score"].argmax())
        sequence = candidates[best]
        self._mean = mean.detach().clone()
        diagnostics: dict[str, float | int | str] = {
            "score": float(latest["score"][best].cpu()),
            "expected_return": float(latest["expected_return"][best].cpu()),
            "cvar_return": float(latest["cvar_return"][best].cpu()),
            "terminal_value": float(latest["terminal_value"][best].cpu()),
            "epistemic": float(latest["epistemic"][best].cpu()),
            "aleatoric": float(latest["aleatoric"][best].cpu()),
            "failure_probability": float(
                latest["failure_probability"][best].cpu()
            ),
            "action_ood": float(latest["action_ood"][best].cpu()),
            "action_magnitude": float(latest["action_magnitude"][best].cpu()),
            "action_change": float(latest["action_change"][best].cpu()),
            "prior_action_max_delta": float(
                (sequence - prior_mean).abs().max().cpu()
            ),
            "predicted_robot_distance": float(
                latest["predicted_robot_distance"][best].cpu()
            ),
            "num_samples": cfg.num_samples,
            "particles_per_candidate": cfg.particles_per_candidate,
            "iterations": cfg.iterations,
        }
        return MPPIPlan(
            action=sequence[0].clone(),
            sequence=sequence.clone(),
            diagnostics=diagnostics,
        )

    def _prior_sequences(
        self,
        history: WorldModelSequenceInputs,
        count: int,
        horizon: int,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        repeated = _repeat_history(history, count)
        model = self.ensemble.members[0]
        hidden, state = model.encode_history(repeated)
        actions: list[Tensor] = []
        for _ in range(horizon):
            features = model.planning_features(hidden, state)
            action = (
                self.planning_heads.deterministic_action(features)
                if deterministic
                else self.planning_heads.sample_action(
                    features, generator=self.generator
                )
            )
            action = self._constrain(action)
            actions.append(action)
            hidden, state, _ = model.imagine_step(hidden, state, action)
        return torch.stack(actions, dim=1)

    def _terminal_values(
        self, history: WorldModelSequenceInputs, actions: Tensor
    ) -> Tensor:
        model = self.ensemble.members[0]
        repeated = _repeat_history(history, actions.shape[0])
        hidden, state = model.encode_history(repeated)
        for step in range(actions.shape[1]):
            hidden, state, _ = model.imagine_step(
                hidden, state, actions[:, step]
            )
        features = model.planning_features(hidden, state)
        return self.planning_heads(features).value.squeeze(-1)

    def _score_in_chunks(
        self,
        history: WorldModelSequenceInputs,
        candidates: Tensor,
        config: MPPIConfig,
    ) -> dict[str, Tensor]:
        collected: dict[str, list[Tensor]] = {}
        for start in range(0, candidates.shape[0], config.candidate_batch_size):
            actions = candidates[start : start + config.candidate_batch_size]
            scores = self._score(
                _repeat_history(history, actions.shape[0]), actions, config
            )
            for name, values in scores.items():
                collected.setdefault(name, []).append(values)
        return {name: torch.cat(parts, dim=0) for name, parts in collected.items()}

    def _score(
        self,
        history: WorldModelSequenceInputs,
        actions: Tensor,
        config: MPPIConfig,
    ) -> dict[str, Tensor]:
        output = self.ensemble(
            WorldModelRolloutInputs(
                history=history,
                candidate_actions=actions,
                num_particles=config.particles_per_candidate,
            )
        )
        rewards = output.rewards.squeeze(-1)
        done_probability = self._outcome_probability(
            output.termination["done_logit"], "done"
        ).squeeze(-1)
        leading, batch, horizon = rewards.shape
        discounts = torch.pow(
            torch.as_tensor(config.discount, device=self.device, dtype=self.dtype),
            torch.arange(horizon, device=self.device, dtype=self.dtype),
        )
        survival = torch.cat(
            (
                torch.ones(leading, batch, 1, device=self.device, dtype=self.dtype),
                torch.cumprod(1.0 - done_probability[..., :-1], dim=-1),
            ),
            dim=-1,
        )
        particle_return = (rewards * survival * discounts).sum(dim=-1)
        terminal_value = self._terminal_values(history, actions)
        terminal_survival = survival[..., -1] * (1.0 - done_probability[..., -1])
        particle_return = particle_return + config.terminal_value_weight * (
            config.discount**horizon
        ) * terminal_survival * terminal_value.unsqueeze(0)
        expected_return = particle_return.mean(dim=0)
        tail_count = max(1, int(np.ceil(config.cvar_alpha * leading)))
        cvar_return = particle_return.sort(dim=0).values[:tail_count].mean(dim=0)

        state_std = self.ensemble.state_std.to(self.device, self.dtype)
        continuous = self.ensemble.members[0].continuous_state_mask
        scale = self.variance_scale.to(self.dtype)
        epistemic = torch.sqrt(
            (
                output.uncertainty["epistemic_std"][..., continuous].square()
                * scale[continuous]
                / state_std[continuous].square()
            ).mean(dim=-1)
        )
        aleatoric = torch.sqrt(
            (
                output.uncertainty["aleatoric_std"][..., continuous].square()
                * scale[continuous]
                / state_std[continuous].square()
            ).mean(dim=-1)
        )
        failure_probability = self._outcome_probability(
            output.termination["failure_logit"], "failure"
        ).mean(dim=0).squeeze(-1)
        action_ood = output.diagnostics["risk_action_ood"]
        action_magnitude = actions.square().mean(dim=-1)
        previous_action = history.past_actions[:, -1]
        changes = torch.cat(
            (
                actions[:, :1] - previous_action.unsqueeze(1),
                actions[:, 1:] - actions[:, :-1],
            ),
            dim=1,
        )
        action_change = changes.square().mean(dim=-1)
        discounted = discounts.unsqueeze(0)

        def aggregate(values: Tensor) -> Tensor:
            return (values * discounted).sum(dim=-1)

        epistemic_cost = aggregate(epistemic)
        aleatoric_cost = aggregate(aleatoric)
        failure_cost = aggregate(failure_probability)
        action_ood_cost = aggregate(action_ood)
        magnitude_cost = aggregate(action_magnitude)
        change_cost = aggregate(action_change)
        state_mean = output.state_distribution["mean"]
        robot_distance = torch.linalg.vector_norm(
            state_mean[..., 0:2] - state_mean[..., 11:13], dim=-1
        )
        predicted_robot_distance = robot_distance.amax(dim=(0, 2))
        risk = self.risk_weights
        score = (
            (1.0 - config.cvar_weight) * expected_return
            + config.cvar_weight * cvar_return
            - risk.epistemic * epistemic_cost
            - risk.aleatoric * aleatoric_cost
            - risk.failure * failure_cost
            - risk.action_ood * action_ood_cost
            - risk.action_magnitude * magnitude_cost
            - risk.action_change * change_cost
        )
        if self.max_predicted_robot_distance is not None:
            excess = torch.relu(
                predicted_robot_distance - self.max_predicted_robot_distance
            )
            # A direct geometric guard is more reliable than asking a sparse
            # terminal-failure classifier to recognize every unsafe candidate.
            score = score - 10_000.0 * excess
        return {
            "score": score,
            "expected_return": expected_return,
            "cvar_return": cvar_return,
            "terminal_value": terminal_value,
            "epistemic": epistemic.max(dim=-1).values,
            "aleatoric": aleatoric.max(dim=-1).values,
            "failure_probability": failure_probability.max(dim=-1).values,
            "action_ood": action_ood.max(dim=-1).values,
            "action_magnitude": action_magnitude.mean(dim=-1),
            "action_change": action_change.mean(dim=-1),
            "predicted_robot_distance": predicted_robot_distance,
        }

    def _outcome_probability(self, logits: Tensor, name: str) -> Tensor:
        """Undo weighted-BCE prior shift before treating logits as probabilities."""

        correction = self.outcome_logit_corrections.get(name, 0.0)
        return torch.sigmoid(logits - correction)

    def _constrain(self, actions: Tensor) -> Tensor:
        result = actions.clamp(-1.0, 1.0)
        if not self.fixed_actions:
            return result
        result = result.clone()
        for index, value in self.fixed_actions.items():
            result[..., index] = value
        return result

    def _constrain_to_prior(
        self, actions: Tensor, prior: Tensor, maximum_delta: float
    ) -> Tensor:
        lower = prior - maximum_delta
        upper = prior + maximum_delta
        return self._constrain(torch.maximum(torch.minimum(actions, upper), lower))


class WAMMPPIActionPolicy:
    """Adapt risk-aware MPPI to ``Policy.act(observation)`` without leakage."""

    MODES = ("mppi", "action_prior", "safe_stop")

    def __init__(
        self,
        planner: RiskAwareMPPI,
        *,
        mode: str = "mppi",
        safety: MPPISafetyConfig | None = None,
        clock: Any = time.perf_counter,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported WAM policy mode {mode!r}")
        self.planner = planner
        self.mode = mode
        self.safety = safety or MPPISafetyConfig()
        self.clock = clock
        horizon = planner.ensemble.member_config.history_horizon
        self._states: deque[np.ndarray] = deque(maxlen=horizon)
        self._actions: deque[np.ndarray] = deque(maxlen=max(horizon - 1, 1))
        self._planner_profile = "full"
        self._fast_steps = 0
        self.last_diagnostics: dict[str, Any] = {}
        self.observation_keys: set[str] = set()

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._planner_profile = "full"
        self._fast_steps = 0
        self.last_diagnostics = {}
        self.observation_keys = set()
        self.planner.reset()

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        keys = {str(key) for key in observation}
        self.observation_keys.update(keys)
        if "privileged_state" in observation:
            raise RuntimeError("privileged_state leakage into WAM policy")
        if "proprioception" not in observation:
            raise KeyError("WAM policy requires observation['proprioception']")
        state = np.asarray(observation["proprioception"], dtype=np.float32)
        expected = self.planner.ensemble.member_config.state_dim
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError(f"proprioception must be finite with shape {(expected,)}")
        self._states.append(state.copy())
        history = self._history()
        start = self.clock()
        fallback_reason = "none"
        requested_profile = self._planner_profile
        planned_mode = "none"
        deadline_exceeded = False
        diagnostics: dict[str, Any] = {}
        if self.mode == "safe_stop":
            action = self._safe_stop()
            executed_mode = "safe_stop"
        elif self.mode == "action_prior":
            action, executed_mode, prior_error = self._prior_or_stop(
                history, "action_prior"
            )
            if prior_error:
                fallback_reason = prior_error
        else:
            if requested_profile == "prior":
                action, executed_mode, prior_error = self._prior_or_stop(
                    history, "action_prior_latency_fallback"
                )
                fallback_reason = prior_error or "planner_latency_budget"
            else:
                try:
                    plan = self.planner.plan(
                        history,
                        config=(
                            self.planner.config
                            if requested_profile == "full"
                            else self._reduced_config()
                        ),
                    )
                    action = plan.action.detach().cpu().numpy().astype(np.float32)
                    diagnostics.update(plan.diagnostics)
                    executed_mode = f"mppi_{requested_profile}"
                    planned_mode = executed_mode
                except (RuntimeError, FloatingPointError) as error:
                    fallback_reason = f"planner_error:{type(error).__name__}"
                    if requested_profile == "full":
                        try:
                            plan = self.planner.plan(
                                history, config=self._reduced_config()
                            )
                            action = plan.action.detach().cpu().numpy().astype(np.float32)
                            diagnostics.update(plan.diagnostics)
                            executed_mode = "mppi_reduced"
                            planned_mode = executed_mode
                            self._planner_profile = "reduced"
                        except (RuntimeError, FloatingPointError):
                            action, executed_mode, prior_error = self._prior_or_stop(
                                history, "action_prior_fallback"
                            )
                            if prior_error:
                                fallback_reason += f";{prior_error}"
                            self._planner_profile = "prior"
                    else:
                        action, executed_mode, prior_error = self._prior_or_stop(
                            history, "action_prior_fallback"
                        )
                        if prior_error:
                            fallback_reason += f";{prior_error}"
                        self._planner_profile = "prior"
                if executed_mode.startswith("mppi") and self._unsafe(diagnostics):
                    action, executed_mode, prior_error = self._prior_or_stop(
                        history, "action_prior_risk_fallback"
                    )
                    fallback_reason = prior_error or "planner_risk_threshold"
        planning_latency_ms = (self.clock() - start) * 1000.0
        if self.mode == "mppi":
            latency_mode = planned_mode if planned_mode != "none" else executed_mode
            self._update_latency_profile(planning_latency_ms, latency_mode)
            deadline_exceeded = bool(
                planned_mode.startswith("mppi")
                and planning_latency_ms > self.safety.latency_budget_ms
            )
            if deadline_exceeded and self.safety.discard_over_budget_plans:
                if executed_mode.startswith("mppi"):
                    action, executed_mode, prior_error = self._prior_or_stop(
                        history, "action_prior_deadline_fallback"
                    )
                    if prior_error:
                        fallback_reason = _join_reasons(fallback_reason, prior_error)
                fallback_reason = _join_reasons(
                    fallback_reason, "planner_deadline_exceeded"
                )
        latency_ms = planning_latency_ms
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (self.planner.ensemble.member_config.action_dim,):
            action = self._safe_stop()
            fallback_reason = "invalid_action_shape"
            executed_mode = "safe_stop_fallback"
        if not np.isfinite(action).all():
            action = self._safe_stop()
            fallback_reason = "non_finite_action"
            executed_mode = "safe_stop_fallback"
        self._actions.append(action.copy())
        self.last_diagnostics = {
            **diagnostics,
            "latency_ms": float(latency_ms),
            "planning_latency_ms": float(planning_latency_ms),
            "requested_profile": requested_profile,
            "planned_mode": planned_mode,
            "executed_mode": executed_mode,
            "plan_executed": executed_mode.startswith("mppi"),
            "deadline_exceeded": deadline_exceeded,
            "fallback_reason": fallback_reason,
            "observation_keys": sorted(keys),
            "privileged_state_seen": False,
        }
        return action

    def _history(self) -> WorldModelSequenceInputs:
        config = self.planner.ensemble.member_config
        count = len(self._states)
        offset = config.history_horizon - count
        states = torch.zeros(
            1,
            config.history_horizon,
            config.state_dim,
            device=self.planner.device,
            dtype=self.planner.dtype,
        )
        states[0, offset:] = torch.as_tensor(
            np.stack(self._states), device=self.planner.device, dtype=self.planner.dtype
        )
        past_actions = torch.zeros(
            1,
            config.history_horizon - 1,
            config.action_dim,
            device=self.planner.device,
            dtype=self.planner.dtype,
        )
        needed = max(count - 1, 0)
        if needed:
            recent = list(self._actions)[-needed:]
            past_actions[0, offset:] = torch.as_tensor(
                np.stack(recent),
                device=self.planner.device,
                dtype=self.planner.dtype,
            )
        valid_mask = torch.zeros(
            1, config.history_horizon, device=self.planner.device, dtype=torch.bool
        )
        valid_mask[0, offset:] = True
        return WorldModelSequenceInputs(states, past_actions, valid_mask)

    @torch.inference_mode()
    def _prior_action(self, history: WorldModelSequenceInputs) -> np.ndarray:
        model = self.planner.ensemble.members[0]
        _, _, features = model.encode_planning_history(history)
        action = self.planner.planning_heads.deterministic_action(features)[0]
        action = self.planner._constrain(action)
        return action.detach().cpu().numpy().astype(np.float32)

    def _reduced_config(self) -> MPPIConfig:
        return replace(
            self.planner.config,
            num_samples=self.safety.reduced_num_samples,
            num_elites=self.safety.reduced_num_elites,
            iterations=self.safety.reduced_iterations,
            particles_per_candidate=self.safety.reduced_particles,
            num_policy_trajectories=min(
                self.planner.config.num_policy_trajectories,
                self.safety.reduced_num_samples,
            ),
            candidate_batch_size=min(
                self.planner.config.candidate_batch_size,
                self.safety.reduced_num_samples,
            ),
        )

    def _unsafe(self, diagnostics: Mapping[str, Any]) -> bool:
        return bool(
            float(diagnostics.get("epistemic", float("inf")))
            > self.safety.max_epistemic
            or float(diagnostics.get("action_ood", float("inf")))
            > self.safety.max_action_ood
            or float(diagnostics.get("failure_probability", float("inf")))
            > self.safety.max_failure_probability
            or float(diagnostics.get("predicted_robot_distance", float("inf")))
            > self.safety.max_predicted_robot_distance
        )

    def _update_latency_profile(self, latency_ms: float, attempted_mode: str) -> None:
        if latency_ms > self.safety.latency_budget_ms:
            self._planner_profile = (
                "reduced" if attempted_mode == "mppi_full" else "prior"
            )
            self._fast_steps = 0
            return
        if self.safety.sticky_latency_fallback:
            return
        if attempted_mode in {"mppi_reduced", "action_prior_latency_fallback"}:
            self._fast_steps += 1
            if self._fast_steps >= self.safety.recovery_steps:
                self._planner_profile = (
                    "reduced"
                    if attempted_mode == "action_prior_latency_fallback"
                    else "full"
                )
                self._fast_steps = 0

    def _prior_or_stop(
        self, history: WorldModelSequenceInputs, mode: str
    ) -> tuple[np.ndarray, str, str]:
        try:
            return self._prior_action(history), mode, ""
        except (RuntimeError, FloatingPointError, ValueError) as error:
            return (
                self._safe_stop(),
                "safe_stop_fallback",
                f"action_prior_error:{type(error).__name__}",
            )

    def _safe_stop(self) -> np.ndarray:
        action = np.zeros(
            self.planner.ensemble.member_config.action_dim, dtype=np.float32
        )
        for index, value in self.planner.fixed_actions.items():
            action[index] = value
        return action


def _repeat_history(
    history: WorldModelSequenceInputs, count: int
) -> WorldModelSequenceInputs:
    if history.states.shape[0] == count:
        return history
    if history.states.shape[0] != 1:
        raise ValueError("history can only be expanded from batch size one")
    return WorldModelSequenceInputs(
        states=history.states.expand(count, -1, -1),
        past_actions=history.past_actions.expand(count, -1, -1),
        valid_mask=history.valid_mask.expand(count, -1),
    )


def _join_reasons(*reasons: str) -> str:
    return ";".join(reason for reason in reasons if reason and reason != "none") or "none"


def _history_to(
    history: WorldModelSequenceInputs,
    device: torch.device,
    dtype: torch.dtype,
) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=history.states.to(device=device, dtype=dtype),
        past_actions=history.past_actions.to(device=device, dtype=dtype),
        valid_mask=history.valid_mask.to(device=device),
    )


__all__ = [
    "MPPIConfig",
    "MPPIPlan",
    "MPPIRiskWeights",
    "MPPISafetyConfig",
    "RiskAwareMPPI",
    "WAMMPPIActionPolicy",
]
