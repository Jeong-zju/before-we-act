"""Frozen Flow plus zero-init gated residual injection for S3-R6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from models.wam_multimodal.agent_factorized_flow_wam import AgentFactorizedFlowWAM
from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
)
from models.wam_multimodal.protected_team_future_predictor import (
    ProtectedTeamFuturePredictor,
)


@dataclass(frozen=True)
class WorldToFlowAdapterConfig:
    flow_dim: int = 384
    state_dim: int = 18
    visual_dim: int = 256
    hidden_dim: int = 384
    action_dim: int = 8
    max_gate: float = 0.25

    def __post_init__(self) -> None:
        if min(
            self.flow_dim,
            self.state_dim,
            self.visual_dim,
            self.hidden_dim,
            self.action_dim,
        ) <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not 0.0 < self.max_gate <= 1.0:
            raise ValueError("max_gate must lie in (0,1]")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorldToFlowAdapterConfig":
        return cls(**dict(value))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictedFutureLatents:
    own_state: Tensor
    own_visual: Tensor
    peer_state: Tensor
    peer_visual: Tensor
    shared_visual: Tensor


class WorldToFlowResidualAdapter(nn.Module):
    """Map protected predicted futures to a velocity residual.

    Local and team candidates use the same module and tensor shapes. Local
    candidates mask peer/shared slots; team candidates expose all three slots.
    """

    def __init__(self, config: WorldToFlowAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.own_state = nn.Linear(config.state_dim, config.hidden_dim)
        self.own_visual = nn.Linear(config.visual_dim, config.hidden_dim)
        self.peer_state = nn.Linear(config.state_dim, config.hidden_dim)
        self.peer_visual = nn.Linear(config.visual_dim, config.hidden_dim)
        self.shared_visual = nn.Linear(config.visual_dim, config.hidden_dim)
        self.future_norm = nn.LayerNorm(3 * config.hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(config.flow_dim + 3 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        # A non-zero adapter output is required for alpha to receive a gradient
        # at alpha=0. Only the bounded multiplicative gate is initialized to 0.
        self.gate_alpha = nn.Parameter(torch.zeros(()))

    def bounded_gate(self) -> Tensor:
        return self.config.max_gate * torch.tanh(self.gate_alpha)

    def forward(
        self,
        flow_features: Tensor,
        futures: PredictedFutureLatents,
        valid_agent_mask: Tensor,
        *,
        future_scope: str,
        force_gate_zero: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if flow_features.ndim != 4:
            raise ValueError("flow_features must be [B,A,H,D]")
        batch_size, agents, horizon, width = flow_features.shape
        if width != self.config.flow_dim:
            raise ValueError("flow feature width differs from adapter config")
        if valid_agent_mask.shape != (batch_size, agents):
            raise ValueError("valid_agent_mask must be [B,A]")
        if future_scope not in {"local", "team_shared"}:
            raise ValueError("future_scope must be local or team_shared")

        own = self.own_state(futures.own_state.mean(dim=2)) + self.own_visual(
            futures.own_visual.mean(dim=(2, 3))
        )
        if future_scope == "team_shared":
            pair_valid = valid_agent_mask[:, :, None] & valid_agent_mask[:, None, :]
            peer_count = pair_valid.sum(dim=2, keepdim=True).clamp_min(1)
            peer_state = (
                futures.peer_state.mean(dim=3)
                * pair_valid[..., None]
            ).sum(dim=2) / peer_count
            peer_visual = (
                futures.peer_visual.mean(dim=(3, 4))
                * pair_valid[..., None]
            ).sum(dim=2) / peer_count
            peer = self.peer_state(peer_state) + self.peer_visual(peer_visual)
            shared = self.shared_visual(futures.shared_visual.mean(dim=(2, 3)))
        else:
            peer = torch.zeros_like(own)
            shared = torch.zeros_like(own)
        future_features = self.future_norm(torch.cat((own, peer, shared), dim=-1))
        expanded = future_features[:, :, None].expand(-1, -1, horizon, -1)
        delta = self.residual(torch.cat((flow_features, expanded), dim=-1))
        gate = (
            self.gate_alpha.new_zeros(())
            if force_gate_zero
            else self.bounded_gate()
        )
        enabled = valid_agent_mask[:, :, None, None].to(delta)
        return gate * delta * enabled, gate


class CrossAgentWorldConditionedFlow(nn.Module):
    """Execute the candidate-action contract at every Flow solver evaluation."""

    def __init__(
        self,
        base_flow: AgentFactorizedFlowWAM,
        future_predictor: (
            LocalActionConditionedFuturePredictor | ProtectedTeamFuturePredictor
        ),
        adapter_config: WorldToFlowAdapterConfig,
        *,
        future_scope: str,
        injection: bool,
    ) -> None:
        super().__init__()
        if future_scope not in {"local", "team_shared"}:
            raise ValueError("future_scope must be local or team_shared")
        if future_scope == "team_shared" and not isinstance(
            future_predictor, ProtectedTeamFuturePredictor
        ):
            raise TypeError("team_shared injection requires the protected team model")
        self.base_flow = base_flow
        self.future_predictor = future_predictor
        self.adapter = WorldToFlowResidualAdapter(adapter_config)
        self.future_scope = future_scope
        self.injection = bool(injection)
        self._freeze_parents()

    def _freeze_parents(self) -> None:
        for module in (self.base_flow, self.future_predictor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "CrossAgentWorldConditionedFlow":
        super().train(mode)
        self.base_flow.eval()
        self.future_predictor.eval()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        if not self.injection:
            return ()
        return tuple(self.adapter.parameters())

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return dict(self.adapter.state_dict())

    def load_adapter_state_dict(self, value: Mapping[str, Tensor]) -> None:
        self.adapter.load_state_dict(dict(value), strict=True)

    def velocity(
        self,
        base_vision_tokens: Tensor,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
        valid_agent_mask: Tensor,
        *,
        force_gate_zero: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if base_vision_tokens.ndim != 4 or current_state.ndim != 3:
            raise ValueError("base vision/state must retain [B,A,...]")
        batch_size, agents = current_state.shape[:2]
        expected_actions = (
            batch_size,
            agents,
            self.base_flow.config.horizon,
            self.base_flow.config.action_dim,
        )
        if action_inputs.shape != expected_actions:
            raise ValueError(f"action_inputs must have shape {expected_actions}")
        if flow_time.shape != (batch_size,):
            raise ValueError("flow_time must be [B]")
        flat_vision = base_vision_tokens.flatten(0, 1)
        flat_state = current_state.flatten(0, 1)
        flat_actions = action_inputs.flatten(0, 1)
        flat_time = flow_time[:, None].expand(-1, agents).reshape(-1)
        with torch.no_grad():
            base_velocity, router_aux, flow_features = self.base_flow.forward_features(
                flat_vision,
                flat_state,
                flat_actions,
                flat_time,
            )
        base_velocity = base_velocity.reshape(expected_actions)
        flow_features = flow_features.reshape(
            batch_size,
            agents,
            self.base_flow.config.horizon,
            -1,
        )
        valid = valid_agent_mask[:, :, None, None].to(base_velocity)
        base_velocity = base_velocity * valid
        if not self.injection or force_gate_zero:
            return base_velocity, {
                "gate": base_velocity.new_zeros(()),
                "router_aux": router_aux.detach(),
                "residual_rms": base_velocity.new_zeros(()),
            }

        # Rectified-Flow clean endpoint estimate x_1 = x_tau + (1-tau)v.
        clean_actions = action_inputs + (
            1.0 - flow_time[:, None, None, None]
        ) * base_velocity
        with torch.no_grad():
            futures = self._predict_futures(
                current_state,
                current_visual_latent,
                shared_visual_latent,
                clean_actions.detach(),
                valid_agent_mask,
            )
        residual, gate = self.adapter(
            flow_features.detach(),
            futures,
            valid_agent_mask,
            future_scope=self.future_scope,
        )
        return base_velocity + residual, {
            "gate": gate.detach(),
            "router_aux": router_aux.detach(),
            "residual_rms": residual.float().square().mean().sqrt().detach(),
        }

    def _predict_futures(
        self,
        state: Tensor,
        local_visual: Tensor,
        shared_visual: Tensor,
        clean_actions: Tensor,
        valid: Tensor,
    ) -> PredictedFutureLatents:
        if isinstance(self.future_predictor, ProtectedTeamFuturePredictor):
            output = self.future_predictor(
                state,
                local_visual,
                shared_visual,
                clean_actions,
                valid,
            )
            return PredictedFutureLatents(
                own_state=output.own_state,
                own_visual=output.own_visual,
                peer_state=output.peer_state,
                peer_visual=output.peer_visual,
                shared_visual=output.shared_visual,
            )
        own_state, own_visual = self.future_predictor(
            state,
            local_visual,
            clean_actions,
            valid,
            valid,
        )
        batch, agents, futures = own_state.shape[:3]
        return PredictedFutureLatents(
            own_state=own_state,
            own_visual=own_visual,
            peer_state=own_state.new_zeros(
                batch, agents, agents, futures, own_state.shape[-1]
            ),
            peer_visual=own_visual.new_zeros(
                batch,
                agents,
                agents,
                futures,
                own_visual.shape[-2],
                own_visual.shape[-1],
            ),
            shared_visual=own_visual.new_zeros(
                batch,
                agents,
                futures,
                own_visual.shape[-2],
                own_visual.shape[-1],
            ),
        )

    def integrate_actions(
        self,
        base_vision_tokens: Tensor,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        valid_agent_mask: Tensor,
        *,
        initial_actions: Tensor,
        solver_steps: int,
        solver: str,
        normalized_clip: float,
    ) -> Tensor:
        if solver_steps <= 0 or normalized_clip <= 0:
            raise ValueError("solver_steps and normalized_clip must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        current = initial_actions.clone()
        batch_size = current.shape[0]
        dt = 1.0 / solver_steps
        for step in range(solver_steps):
            tau = torch.full(
                (batch_size,), step * dt, device=current.device, dtype=current.dtype
            )
            velocity = self.velocity(
                base_vision_tokens,
                current_state,
                current_visual_latent,
                shared_visual_latent,
                current,
                tau,
                valid_agent_mask,
            )[0]
            if solver == "euler":
                current = current + dt * velocity
            else:
                proposal = current + dt * velocity
                correction = self.velocity(
                    base_vision_tokens,
                    current_state,
                    current_visual_latent,
                    shared_visual_latent,
                    proposal,
                    torch.full_like(tau, (step + 1) * dt),
                    valid_agent_mask,
                )[0]
                current = current + 0.5 * dt * (velocity + correction)
            current = current.clamp(-normalized_clip, normalized_clip)
        return current


__all__ = [
    "CrossAgentWorldConditionedFlow",
    "PredictedFutureLatents",
    "WorldToFlowAdapterConfig",
    "WorldToFlowResidualAdapter",
]
