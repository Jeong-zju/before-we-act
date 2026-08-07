from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TeamBeliefState:
    """Candidate-neutral R11 predictive representation contract.

    All tensors are derived from legal fixed-view observations, own/team qpos,
    executed-action history and masks.  Task IDs, simulator state and CoRE
    internals are deliberately absent.
    """

    tokens: torch.Tensor
    agent_tokens: torch.Tensor
    consensus_token: torch.Tensor
    uncertainty: torch.Tensor
    agent_mask: torch.Tensor

    def validate(self) -> "TeamBeliefState":
        if self.tokens.ndim != 3:
            raise ValueError("belief tokens must be [batch, tokens, dim]")
        if self.agent_tokens.ndim != 3:
            raise ValueError("agent tokens must be [batch, agents, dim]")
        if self.consensus_token.shape != (
            self.tokens.shape[0],
            self.tokens.shape[-1],
        ):
            raise ValueError("consensus token shape differs from belief tokens")
        if self.agent_mask.shape != self.agent_tokens.shape[:2]:
            raise ValueError("agent mask shape differs from agent tokens")
        values = (self.tokens, self.agent_tokens, self.consensus_token, self.uncertainty)
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("belief state contains non-finite values")
        return self


@dataclass(frozen=True)
class ActionProposalBatch:
    """Core-free normalized joint action proposals produced from team belief.

    Actions use ``[batch, proposal, agent, horizon, action_dim]``.  Candidate
    zero is always the transplanted generator's deterministic base proposal;
    it is never a legacy CoRE/forced-role action.
    """

    actions: torch.Tensor
    base_index: int
    valid_mask: torch.Tensor
    agent_mask: torch.Tensor
    source: Sequence[str]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "ActionProposalBatch":
        if self.actions.ndim != 5:
            raise ValueError(
                "actions must be [batch,proposal,agent,horizon,dim], got "
                f"{tuple(self.actions.shape)}"
            )
        batch, proposals, agents = self.actions.shape[:3]
        if self.base_index != 0 or not 0 <= self.base_index < proposals:
            raise ValueError("R12 base proposal index must be zero")
        if self.valid_mask.shape != (batch, proposals):
            raise ValueError("proposal valid_mask must be [batch,proposal]")
        if self.agent_mask.shape != (batch, agents):
            raise ValueError("agent_mask must be [batch,agent]")
        if len(self.source) != proposals:
            raise ValueError("proposal source count differs")
        if not bool(torch.isfinite(self.actions).all()):
            raise ValueError("non-finite action proposal")
        absent = ~self.agent_mask[:, None, :, None, None]
        if bool((self.actions.masked_select(absent) != 0).any()):
            raise ValueError("absent-agent action must be exactly zero")
        return self


@dataclass(frozen=True)
class ConsequencePrediction:
    """Off-path R13 consequence prediction for a candidate action batch.

    The leading axes are always ``[batch, proposal, horizon]``.  This object
    intentionally has no action-selection or actuator field: R13 may describe
    candidate consequences, but it cannot change the frozen W12 action path.
    """

    latent_by_horizon: torch.Tensor
    qpos_delta_by_horizon: torch.Tensor
    progress_by_horizon: torch.Tensor
    failure_logits_by_horizon: torch.Tensor
    uncertainty_by_horizon: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "ConsequencePrediction":
        if self.latent_by_horizon.ndim != 5:
            raise ValueError(
                "latent consequence must be [batch,proposal,horizon,token,dim]"
            )
        batch, proposals, horizons = self.latent_by_horizon.shape[:3]
        if self.qpos_delta_by_horizon.ndim != 5:
            raise ValueError(
                "qpos consequence must be [batch,proposal,horizon,agent,qpos]"
            )
        if self.qpos_delta_by_horizon.shape[:3] != (batch, proposals, horizons):
            raise ValueError("qpos consequence leading axes differ")
        scalar_shape = (batch, proposals, horizons)
        for name, value in (
            ("progress", self.progress_by_horizon),
            ("failure", self.failure_logits_by_horizon),
            ("uncertainty", self.uncertainty_by_horizon),
        ):
            if value.shape != scalar_shape:
                raise ValueError(f"{name} consequence axes differ")
        if self.valid_mask.shape != (batch, proposals):
            raise ValueError("consequence valid mask must be [batch,proposal]")
        values = (
            self.latent_by_horizon,
            self.qpos_delta_by_horizon,
            self.progress_by_horizon,
            self.failure_logits_by_horizon,
            self.uncertainty_by_horizon,
        )
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("consequence prediction contains non-finite values")
        return self


@dataclass(frozen=True)
class PlannerDecision:
    """R14 fail-closed decision over a frozen W12 action proposal.

    ``actions`` is the selected normalized joint chunk ``[B,A,H,D]``.  A
    fallback decision must be bit-exact to the caller-provided W12 base.
    """

    actions: torch.Tensor
    selected_source: str
    fallback: bool
    utility_gain: float
    latency_ms: float
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "PlannerDecision":
        if self.actions.ndim != 4:
            raise ValueError("planner actions must be [batch,agent,horizon,dim]")
        if not bool(torch.isfinite(self.actions).all()):
            raise ValueError("planner decision contains non-finite actions")
        if not self.selected_source or not self.reason:
            raise ValueError("planner decision requires source and reason")
        if not float("-inf") < float(self.utility_gain) < float("inf"):
            raise ValueError("planner utility gain must be finite")
        if not 0 <= float(self.latency_ms) < float("inf"):
            raise ValueError("planner latency must be finite and non-negative")
        return self
