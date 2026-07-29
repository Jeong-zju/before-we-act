"""Per-agent shared-parameter Rectified Flow over the frozen S0 context."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from models.static_rgb_act import StaticRGBMoEACTConfig, _MoEDecoder


class AgentFactorizedFlowWAM(nn.Module):
    """Cold-start action flow with the exact S0 local context and decoder family."""

    def __init__(self, config: StaticRGBMoEACTConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(config.vision_dim),
            nn.Linear(config.vision_dim, config.d_model),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.action_projection = nn.Linear(config.action_dim, config.d_model)
        frequency_count = max(math.ceil(config.d_model / 2), 1)
        self.register_buffer(
            "flow_time_frequencies",
            torch.logspace(0.0, 3.0, frequency_count),
            persistent=True,
        )
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(2 * frequency_count, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.action_position = nn.Parameter(
            torch.randn(1, config.horizon, config.d_model) * 0.02
        )
        self.decoder = _MoEDecoder(config)
        self.velocity_head = nn.Linear(config.d_model, config.action_dim)

    def forward(
        self,
        vision_tokens: Tensor,
        state: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = self._validate_inputs(
            vision_tokens=vision_tokens,
            state=state,
            action_inputs=action_inputs,
            flow_time=flow_time,
        )
        frequencies = self.flow_time_frequencies.to(flow_time)
        phase = flow_time[:, None] * frequencies[None]
        time_features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        query = (
            self.action_projection(action_inputs)
            + self.flow_time_mlp(time_features)[:, None, :]
            + self.action_position
        )
        memory = torch.cat(
            (
                self.state_projection(state).unsqueeze(1),
                self.vision_projection(vision_tokens),
            ),
            dim=1,
        )
        decoded, router_aux = self.decoder(query, memory)
        velocity = self.velocity_head(decoded)
        if velocity.shape != (
            batch_size,
            self.config.horizon,
            self.config.action_dim,
        ):
            raise RuntimeError("velocity head violated the per-agent chunk contract")
        return velocity, router_aux

    def integrate_actions(
        self,
        vision_tokens: Tensor,
        state: Tensor,
        *,
        initial_actions: Tensor | None = None,
        solver_steps: int = 4,
        solver: str = "euler",
        normalized_clip: float = 10.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Integrate from a standard-normal cold source to normalized actions."""

        if solver_steps <= 0 or normalized_clip <= 0.0:
            raise ValueError("solver_steps and normalized_clip must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        batch_size = vision_tokens.shape[0]
        shape = (batch_size, self.config.horizon, self.config.action_dim)
        initial = (
            torch.randn(
                shape,
                device=vision_tokens.device,
                dtype=vision_tokens.dtype,
                generator=generator,
            )
            if initial_actions is None
            else initial_actions.clone()
        )
        if initial.shape != shape:
            raise ValueError(f"initial_actions must have shape {shape}")
        current = initial.clone()
        dt = 1.0 / solver_steps
        for step in range(solver_steps):
            tau = torch.full(
                (batch_size,),
                step * dt,
                device=current.device,
                dtype=current.dtype,
            )
            velocity = self(vision_tokens, state, current, tau)[0]
            if solver == "euler":
                current = current + dt * velocity
            else:
                proposal = current + dt * velocity
                correction = self(
                    vision_tokens,
                    state,
                    proposal,
                    torch.full_like(tau, (step + 1) * dt),
                )[0]
                current = current + 0.5 * dt * (velocity + correction)
            current = current.clamp(-normalized_clip, normalized_clip)
        return current

    @torch.no_grad()
    def generate_actions(
        self,
        vision_tokens: Tensor,
        state: Tensor,
        *,
        initial_actions: Tensor | None = None,
        solver_steps: int = 4,
        solver: str = "euler",
        normalized_clip: float = 10.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.integrate_actions(
            vision_tokens,
            state,
            initial_actions=initial_actions,
            solver_steps=solver_steps,
            solver=solver,
            normalized_clip=normalized_clip,
            generator=generator,
        )

    def _validate_inputs(
        self,
        *,
        vision_tokens: Tensor,
        state: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
    ) -> int:
        if (
            vision_tokens.ndim != 3
            or vision_tokens.shape[-1] != self.config.vision_dim
        ):
            raise ValueError("vision_tokens must be [B,N,vision_dim]")
        batch_size = vision_tokens.shape[0]
        if state.shape != (batch_size, self.config.state_dim):
            raise ValueError("state must be [B,state_dim]")
        if action_inputs.shape != (
            batch_size,
            self.config.horizon,
            self.config.action_dim,
        ):
            raise ValueError("action_inputs violate the per-agent chunk contract")
        if flow_time.shape != (batch_size,):
            raise ValueError("flow_time must be [B]")
        if not bool(torch.isfinite(flow_time).all()):
            raise ValueError("flow_time must be finite")
        return batch_size


__all__ = ["AgentFactorizedFlowWAM"]
