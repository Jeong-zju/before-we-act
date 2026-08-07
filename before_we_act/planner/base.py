from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import time
from typing import Any, Mapping

import torch
import yaml

from before_we_act.contracts import PlannerDecision, TeamBeliefState
from before_we_act.world_model.base import CandidateConditionedWorldModel, load_r13_config


CANDIDATE_KINDS = {
    "p0": "world_in_world_revision",
    "p1": "dinowm_cem",
    "p2": "tdmpc2_mpc",
    "p3": "mbrl_lib_cem",
}
EXPECTED_TOP_LEVEL = {
    "schema_version", "round", "candidate_id", "parent_commit",
    "belief_checkpoint_sha256", "action_checkpoint_sha256",
    "world_checkpoint_sha256", "component", "planner", "deployment",
    "selection_rule",
}


@dataclass(frozen=True)
class R14DecisionConfig:
    raw: Mapping[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.raw["candidate_id"])

    @property
    def component(self) -> Mapping[str, Any]:
        return self.raw["component"]

    @property
    def planner(self) -> Mapping[str, Any]:
        return self.raw["planner"]

    @property
    def deployment(self) -> Mapping[str, Any]:
        return self.raw["deployment"]


def load_r14_config(path: str | Path) -> R14DecisionConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError("R14 config keys differ from the frozen schema")
    if payload["schema_version"] != 1 or payload["round"] != "R14":
        raise ValueError("R14 config identity differs")
    candidate = str(payload["candidate_id"])
    if candidate not in CANDIDATE_KINDS:
        raise ValueError("R14 candidate is not registered")
    if payload["component"].get("kind") != CANDIDATE_KINDS[candidate]:
        raise ValueError("R14 candidate/component kind differs")
    identities = {
        "parent_commit": 40,
        "belief_checkpoint_sha256": 64,
        "action_checkpoint_sha256": 64,
        "world_checkpoint_sha256": 64,
    }
    for key, length in identities.items():
        if len(str(payload[key])) != length:
            raise ValueError(f"R14 identity is invalid at {key}")
    expected_planner = {
        "action_prefix_steps", "population_size", "iterations", "elite_size",
        "initial_std", "max_delta", "min_utility_gain", "deadline_ms",
        "progress_weight", "failure_weight", "uncertainty_weight",
        "trust_region_weight", "seed",
    }
    if set(payload["planner"]) != expected_planner:
        raise ValueError("R14 planner keys differ")
    locked = {
        "action_prefix_steps": 16,
        "population_size": 8,
        "iterations": 2,
        "elite_size": 2,
        "deadline_ms": 250.0,
        "seed": 20260807,
    }
    for key, value in locked.items():
        if payload["planner"][key] != value:
            raise ValueError(f"R14 locked planner protocol differs at {key}")
    for key in (
        "initial_std", "max_delta", "min_utility_gain", "progress_weight",
        "failure_weight", "uncertainty_weight", "trust_region_weight",
    ):
        if float(payload["planner"][key]) < 0:
            raise ValueError(f"R14 planner value must be non-negative at {key}")
    if payload["deployment"] != {
        "specialist_tasks": ["three_robots_stack_cube"],
        "protected_tasks": [
            "lift_barrier", "camera_alignment", "long_pipeline_delivery", "take_photo"
        ],
        "routing": "world_guided_stack_exact_w12_protected_fallback",
        "fail_closed_to_w12_base": True,
        "core_runtime_forbidden": True,
    }:
        raise ValueError("R14 deployment contract differs")
    if payload["selection_rule"] != {
        "gate20_tasks": [
            "lift_barrier", "camera_alignment", "three_robots_stack_cube",
            "long_pipeline_delivery", "take_photo",
        ],
        "episodes_per_task": 20,
        "baseline_total_successes": 77,
        "winner_rule": "complete_100_episodes_and_total_successes_strictly_greater_than_77",
        "tie_break": [
            "paired_wins", "camera_plus_stack_successes", "worst_task_successes",
            "p95_latency_ms", "gpu_hours", "candidate_id",
        ],
    }:
        raise ValueError("R14 selection rule differs")
    return R14DecisionConfig(payload)


class WorldUtility:
    """Adapter from frozen W13 consequences to a scalar planner objective."""

    def __init__(self, model: CandidateConditionedWorldModel, config: R14DecisionConfig):
        self.model = model
        self.config = config

    @torch.no_grad()
    def __call__(
        self, belief: TeamBeliefState, candidates: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        if candidates.ndim != 4 or tuple(candidates.shape[1:]) != tuple(base.shape):
            raise ValueError("R14 utility candidates must be [P,A,H,D]")
        batch = len(candidates)
        actions = candidates.unsqueeze(0)
        valid = torch.ones((1, batch), dtype=torch.bool, device=actions.device)
        prediction = self.model(
            belief_tokens=belief.tokens,
            belief_agent_tokens=belief.agent_tokens,
            belief_consensus=belief.consensus_token,
            belief_uncertainty=belief.uncertainty,
            agent_mask=belief.agent_mask,
            candidate_actions=actions,
            candidate_valid_mask=valid,
        )
        weights = torch.tensor((0.2, 0.3, 0.5), device=actions.device)
        progress = (prediction.progress_by_horizon[0].float() * weights).sum(-1)
        failure = (
            prediction.failure_logits_by_horizon[0].float().sigmoid() * weights
        ).sum(-1)
        uncertainty = (
            prediction.uncertainty_by_horizon[0].float().tanh() * weights
        ).sum(-1)
        deviation = (candidates.float() - base.float()).square().mean(dim=(1, 2, 3))
        cfg = self.config.planner
        score = (
            float(cfg["progress_weight"]) * progress
            - float(cfg["failure_weight"]) * failure
            - float(cfg["uncertainty_weight"]) * uncertainty
            - float(cfg["trust_region_weight"]) * deviation
        )
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("R14 world utility is non-finite")
        return score


class WorldGuidedDecisionPlanner:
    """Candidate-specific upstream decision core with one common safety shell."""

    def __init__(
        self,
        config: R14DecisionConfig,
        world_config_path: str | Path,
        world_checkpoint_path: str | Path,
        device: torch.device,
    ) -> None:
        self.config = config
        world_config = load_r13_config(world_config_path)
        saved = torch.load(world_checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("round") != "R13" or saved.get("candidate_id") != "p0":
            raise ValueError("R14 requires the promoted W13-P0 checkpoint")
        self.world = CandidateConditionedWorldModel(world_config).to(device)
        self.world.load_state_dict(saved["model"], strict=True)
        self.world.eval()
        self.utility = WorldUtility(self.world, config)
        module = importlib.import_module("before_we_act.planner.candidate")
        if getattr(module, "CANDIDATE_ID", None) != config.candidate_id:
            raise ValueError("R14 candidate branch identity differs")
        self.core = module.build_decision_core(config)

    def reset(self) -> None:
        if hasattr(self.core, "reset"):
            self.core.reset()

    @torch.no_grad()
    def decide(
        self, belief: TeamBeliefState, base_actions: torch.Tensor, *, seed: int, step: int
    ) -> PlannerDecision:
        started = time.perf_counter_ns()
        reason = "planner_selected"
        candidate = base_actions
        source = f"r14_{self.config.candidate_id}_refined"
        fallback = False
        gain = 0.0
        diagnostics: dict[str, Any] = {}
        try:
            def score(values: torch.Tensor) -> torch.Tensor:
                values = values.float().clamp(-5.0, 5.0)
                return self.utility(belief, values, base_actions[0])

            refined, diagnostics = self.core.refine(
                base_actions[0], score, seed=int(seed), step=int(step)
            )
            if tuple(refined.shape) != tuple(base_actions[0].shape):
                raise ValueError("R14 decision core returned an invalid action shape")
            if not bool(torch.isfinite(refined).all()):
                raise FloatingPointError("R14 decision core returned NaN/Inf")
            refined = refined.float().clamp(-5.0, 5.0)
            maximum = float((refined - base_actions[0]).abs().max())
            if maximum > float(self.config.planner["max_delta"]) + 1e-6:
                raise ValueError("R14 refined action exceeded the frozen trust region")
            scores = score(torch.stack((base_actions[0], refined)))
            gain = float(scores[1] - scores[0])
            if gain >= float(self.config.planner["min_utility_gain"]):
                candidate = refined.unsqueeze(0).to(base_actions.dtype)
            else:
                fallback = True
                source = "w12_base_index_0"
                reason = "insufficient_world_utility_gain"
        except Exception as exc:  # fail-closed is part of the R14 contract
            fallback = True
            source = "w12_base_index_0"
            reason = f"planner_exception:{type(exc).__name__}"
            diagnostics = {"exception": str(exc)[:240]}
            candidate = base_actions
            gain = 0.0
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        if latency_ms > float(self.config.planner["deadline_ms"]):
            fallback = True
            source = "w12_base_index_0"
            reason = "planner_deadline_exceeded"
            candidate = base_actions
            gain = 0.0
        if fallback and not torch.equal(candidate, base_actions):
            raise AssertionError("R14 fallback is not bit-exact to W12 base")
        return PlannerDecision(
            actions=candidate,
            selected_source=source,
            fallback=fallback,
            utility_gain=gain,
            latency_ms=latency_ms,
            reason=reason,
            diagnostics=diagnostics,
        ).validate()
