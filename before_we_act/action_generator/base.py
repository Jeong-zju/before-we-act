from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml

from before_we_act.contracts import ActionProposalBatch, TeamBeliefState
from .registry import CANDIDATE_SPECS, build_action_core


EXPECTED_TOP_LEVEL = {
    "schema_version",
    "round",
    "candidate_id",
    "parent_commit",
    "belief_checkpoint_sha256",
    "component",
    "action",
    "training",
    "selection_rule",
}


@dataclass(frozen=True)
class R12Config:
    raw: Mapping[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.raw["candidate_id"])

    @property
    def component(self) -> Mapping[str, Any]:
        return self.raw["component"]

    @property
    def action(self) -> Mapping[str, Any]:
        return self.raw["action"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.raw["training"]


def load_r12_config(path: str | Path) -> R12Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError("R12 config keys differ from the frozen schema")
    if payload["schema_version"] != 1 or payload["round"] != "R12":
        raise ValueError("R12 config identity differs")
    candidate = payload["candidate_id"]
    if candidate not in CANDIDATE_SPECS:
        raise ValueError("R12 candidate is not registered")
    if payload["component"].get("kind") != CANDIDATE_SPECS[candidate]["kind"]:
        raise ValueError("R12 candidate and component kind differ")
    action = payload["action"]
    locked_action = {
        "horizon": 100,
        "max_agents": 4,
        "action_dim": 8,
        "belief_dim": 96,
        "normalization": "W10_mean_std_copied_into_R12_checkpoint",
        "normalized_clip": 5.0,
        "num_proposals": 1,
    }
    if action != locked_action:
        raise ValueError("locked R12 joint action contract differs")
    training = payload["training"]
    required_training = {
        "updates", "batch_size", "seed", "learning_rate", "weight_decay",
        "precision", "checkpoint_every", "progress_every", "grad_clip",
    }
    if set(training) != required_training:
        raise ValueError("R12 training keys differ from the frozen schema")
    if training["updates"] != 20_000 or training["seed"] != 20260805:
        raise ValueError("R12 update/seed freeze differs")
    if training["checkpoint_every"] != 2_000 or training["progress_every"] != 50:
        raise ValueError("R12-R1 checkpoint/progress cadence differs")
    if training["precision"] != "bfloat16":
        raise ValueError("R12 precision must be bfloat16")
    rule = payload["selection_rule"]
    expected_rule = {
        "gate20_tasks": [
            "lift_barrier",
            "camera_alignment",
            "three_robots_stack_cube",
            "long_pipeline_delivery",
            "take_photo",
        ],
        "episodes_per_task": 20,
        "baseline_total_successes": 74,
        "winner_rule": "complete_100_episodes_and_total_successes_strictly_greater_than_74",
        "tie_break": [
            "paired_wins",
            "camera_plus_stack_successes",
            "worst_task_successes",
            "p95_latency_ms",
            "gpu_hours",
            "candidate_id",
        ],
    }
    if rule != expected_rule:
        raise ValueError("R12 preregistered Gate20 selection rule differs")
    return R12Config(payload)


class JointActionGenerator(nn.Module):
    """Thin shared codec around exactly one transplanted action core."""

    def __init__(self, config: R12Config) -> None:
        super().__init__()
        self.config = config
        self.candidate_id = config.candidate_id
        self.horizon = int(config.action["horizon"])
        self.max_agents = int(config.action["max_agents"])
        self.action_dim = int(config.action["action_dim"])
        self.normalized_clip = float(config.action["normalized_clip"])
        component = dict(config.component)
        component.update(
            horizon=self.horizon,
            joint_action_dim=self.max_agents * self.action_dim,
            belief_dim=int(config.action["belief_dim"]),
        )
        self.core = build_action_core(config.candidate_id, component)

    @staticmethod
    def condition(belief: TeamBeliefState) -> tuple[torch.Tensor, torch.Tensor]:
        belief.validate()
        tokens = torch.cat(
            [belief.tokens, belief.agent_tokens, belief.consensus_token[:, None]], dim=1
        )
        token_mask = torch.cat(
            [
                torch.ones(
                    belief.tokens.shape[:2], device=tokens.device, dtype=torch.bool
                ),
                belief.agent_mask,
                torch.ones(
                    (len(tokens), 1), device=tokens.device, dtype=torch.bool
                ),
            ],
            dim=1,
        )
        return tokens, token_mask

    def flatten_actions(self, actions: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        expected = (len(actions), self.horizon, self.max_agents, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"joint action target {tuple(actions.shape)} != {expected}")
        masked = actions * agent_mask[:, None, :, None].to(actions.dtype)
        return masked.reshape(len(actions), self.horizon, -1)

    def training_loss(
        self,
        belief: TeamBeliefState,
        actions: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        tokens, token_mask = self.condition(belief)
        flat = self.flatten_actions(actions, belief.agent_mask)
        feature_mask = belief.agent_mask[:, None, :, None].expand(
            -1, self.horizon, -1, self.action_dim
        ).reshape(len(flat), self.horizon, -1)
        mask = feature_mask & step_mask[:, :, None]
        return self.core.training_loss(tokens, token_mask, flat, mask)

    @torch.no_grad()
    def sample(
        self,
        belief: TeamBeliefState,
        *,
        noise: torch.Tensor | None = None,
    ) -> ActionProposalBatch:
        tokens, token_mask = self.condition(belief)
        flat = self.core.sample(tokens, token_mask, noise=noise)
        expected = (len(tokens), self.horizon, self.max_agents * self.action_dim)
        if tuple(flat.shape) != expected:
            raise ValueError(f"action core output {tuple(flat.shape)} != {expected}")
        flat = flat.float().clamp(-self.normalized_clip, self.normalized_clip)
        actions = flat.reshape(
            len(flat), self.horizon, self.max_agents, self.action_dim
        ).permute(0, 2, 1, 3)
        actions = actions * belief.agent_mask[:, :, None, None].to(actions.dtype)
        return ActionProposalBatch(
            actions=actions[:, None],
            base_index=0,
            valid_mask=torch.ones((len(actions), 1), device=actions.device, dtype=torch.bool),
            agent_mask=belief.agent_mask,
            source=(f"r12_{self.candidate_id}_transplanted_action_core",),
            diagnostics={
                "core_free": True,
                "legacy_core_import": False,
                "normalized_clip": self.normalized_clip,
            },
        ).validate()
