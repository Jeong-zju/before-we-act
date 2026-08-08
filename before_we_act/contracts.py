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
