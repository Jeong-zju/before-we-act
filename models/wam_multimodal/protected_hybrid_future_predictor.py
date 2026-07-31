"""Evaluate-only S2-R4 hybrid with an exact protected own path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from models.wam_multimodal.team_shared_future_predictor import (
    TeamSharedFuturePrediction,
    TeamSharedFuturePredictor,
    TeamSharedFuturePredictorConfig,
)


@dataclass(frozen=True)
class ProtectedHybridFuturePrediction:
    own_state: Tensor
    own_visual: Tensor
    peer_state: Tensor
    peer_visual: Tensor
    shared_visual: Tensor


class ProtectedHybridFuturePredictor(nn.Module):
    """Compose P0 own outputs with P1 peer/shared modules without training.

    P1's inherited local weights are replaced by P0 before evaluation. The
    existing team encoder therefore reads protected P0 projections through its
    detached state/visual/action path. P1 own heads and residual gate never
    enter the returned own function.
    """

    def __init__(
        self,
        local_config: LocalFuturePredictorConfig,
        team_config: TeamSharedFuturePredictorConfig,
    ) -> None:
        super().__init__()
        self.protected_own = LocalActionConditionedFuturePredictor(local_config)
        self.team_source = TeamSharedFuturePredictor(local_config, team_config)

    def load_sources(
        self,
        *,
        own_state_dict: Mapping[str, Tensor],
        team_state_dict: Mapping[str, Tensor],
    ) -> None:
        self.protected_own.load_state_dict(own_state_dict, strict=True)
        self.team_source.load_state_dict(team_state_dict, strict=True)
        # Discard the P1 local trajectory and make all team input projections
        # an exact P0 clone. P1 peer/shared blocks remain untouched.
        self.team_source.local_predictor.load_state_dict(
            own_state_dict,
            strict=True,
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ProtectedHybridFuturePredictor":
        if mode:
            raise RuntimeError(
                "S2-R4 protected hybrid is evaluate-only and cannot train"
            )
        super().train(False)
        return self

    def forward(
        self,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        candidate_actions: Tensor,
        valid_agent_mask: Tensor,
        *,
        actions_by_focal: Tensor | None = None,
    ) -> ProtectedHybridFuturePrediction:
        own_state, own_visual = self.protected_own(
            current_state,
            current_visual_latent,
            candidate_actions,
            valid_agent_mask,
            valid_agent_mask,
        )
        team: TeamSharedFuturePrediction = self.team_source(
            current_state,
            current_visual_latent,
            shared_visual_latent,
            candidate_actions,
            valid_agent_mask,
            actions_by_focal=actions_by_focal,
        )
        return ProtectedHybridFuturePrediction(
            own_state=own_state,
            own_visual=own_visual,
            peer_state=team.peer_state,
            peer_visual=team.peer_visual,
            shared_visual=team.shared_visual,
        )

    @property
    def discarded_team_source_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.team_source.state_dict()
            if key.startswith(
                (
                    "local_predictor.",
                    "own_state_residual.",
                    "own_visual_residual.",
                    "own_residual_gate",
                )
            )
        )


def exact_own_difference(
    reference: tuple[Tensor, Tensor],
    hybrid: ProtectedHybridFuturePrediction,
) -> dict[str, float | bool]:
    state_reference, visual_reference = reference
    state_difference = (state_reference - hybrid.own_state).abs()
    visual_difference = (visual_reference - hybrid.own_visual).abs()
    state_max = float(state_difference.max())
    visual_max = float(visual_difference.max())
    return {
        "state_elementwise_exact": bool(
            torch.equal(state_reference, hybrid.own_state)
        ),
        "visual_elementwise_exact": bool(
            torch.equal(visual_reference, hybrid.own_visual)
        ),
        "state_max_abs_diff": state_max,
        "visual_max_abs_diff": visual_max,
        "max_abs_diff": max(state_max, visual_max),
    }


__all__ = [
    "ProtectedHybridFuturePrediction",
    "ProtectedHybridFuturePredictor",
    "exact_own_difference",
]
