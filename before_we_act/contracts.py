from __future__ import annotations

from dataclasses import dataclass

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
