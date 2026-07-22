"""Stateful single-expert Flow Matching action chunks for Joint WAM."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from models.wam.action_prior import ActionPrior
from models.wam.config import ActionPriorConfig, StatefulActionFlowConfig
from models.wam.normalizer import NormalizationStats


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(value)


class StatefulActionFlow(nn.Module):
    """Rectified-flow velocity conditioned on belief and the chunk start.

    ``initial_chunk`` is part of the conditioning signal.  A cold generation
    starts at normalized zero (the dataset action mean); a stateful generation
    starts from the actually shifted previous chunk.  Keeping the initial
    condition explicit lets the same expert learn both cold generation and
    observation-conditioned correction without averaging multiple action modes.
    """

    def __init__(
        self,
        config: StatefulActionFlowConfig,
        stats: NormalizationStats,
    ) -> None:
        super().__init__()
        if stats.action_mean.shape != (config.action_dim,):
            raise ValueError("action normalization mean has the wrong dimension")
        if stats.action_std.shape != (config.action_dim,):
            raise ValueError("action normalization std has the wrong dimension")
        self.config = config
        self.anchor_prior: ActionPrior | None = None
        if config.anchor_mode == "frozen_prior":
            self.anchor_prior = ActionPrior(
                ActionPriorConfig(
                    feature_dim=config.feature_dim,
                    action_dim=config.action_dim,
                    hidden_dim=config.anchor_hidden_dim,
                    hidden_layers=config.anchor_hidden_layers,
                    min_log_std=config.anchor_min_log_std,
                    max_log_std=config.anchor_max_log_std,
                )
            )
        chunk_width = config.horizon * config.action_dim
        input_width = (
            config.feature_dim
            + 2 * chunk_width
            + config.time_embedding_dim
            + 1
        )
        self.input_projection = nn.Sequential(
            nn.Linear(input_width, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _ResidualBlock(config.hidden_dim)
            for _ in range(config.hidden_layers)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, chunk_width),
        )
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                config.time_embedding_dim // 2,
            )
        )
        self.register_buffer("time_frequencies", frequencies)
        self.register_buffer(
            "action_mean", torch.as_tensor(stats.action_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.as_tensor(stats.action_std, dtype=torch.float32)
        )
        self.freeze_anchor()

    def set_anchor_from_prior(self, prior: ActionPrior) -> None:
        """Copy the accepted prior into the flow artifact, then freeze it."""

        if self.anchor_prior is None:
            raise RuntimeError("anchor_mode='none' cannot accept an action prior")
        if prior.config != self.anchor_prior.config:
            raise ValueError("action prior architecture does not match flow anchor")
        self.anchor_prior.load_state_dict(prior.state_dict(), strict=True)
        self.freeze_anchor()

    def freeze_anchor(self) -> None:
        if self.anchor_prior is None:
            return
        self.anchor_prior.eval()
        for parameter in self.anchor_prior.parameters():
            parameter.requires_grad_(False)

    def anchor_action(self, features: Tensor) -> Tensor:
        if self.anchor_prior is None:
            raise RuntimeError("action flow has no anchor prior")
        return self.anchor_prior.deterministic_action(features)

    @property
    def has_anchor(self) -> bool:
        return self.anchor_prior is not None

    def forward(
        self,
        action_chunk: Tensor,
        flow_time: Tensor,
        features: Tensor,
        initial_chunk: Tensor,
        warm_start: Tensor,
    ) -> Tensor:
        expected = (
            features.shape[0],
            self.config.horizon,
            self.config.action_dim,
        )
        if tuple(action_chunk.shape) != expected:
            raise ValueError(f"action_chunk must have shape {expected}")
        if tuple(initial_chunk.shape) != expected:
            raise ValueError(f"initial_chunk must have shape {expected}")
        if features.ndim != 2 or features.shape[-1] != self.config.feature_dim:
            raise ValueError("features must have shape [B,feature_dim]")
        if flow_time.ndim == 2 and flow_time.shape[-1] == 1:
            flow_time = flow_time[:, 0]
        if tuple(flow_time.shape) != (features.shape[0],):
            raise ValueError("flow_time must have shape [B] or [B,1]")
        if warm_start.ndim == 1:
            warm_start = warm_start[:, None]
        if tuple(warm_start.shape) != (features.shape[0], 1):
            raise ValueError("warm_start must have shape [B] or [B,1]")
        tensors = (action_chunk, flow_time, features, initial_chunk, warm_start)
        if len({tensor.device for tensor in tensors}) != 1:
            raise TypeError("all flow inputs must share a device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise TypeError("all flow inputs must share a floating dtype")
        phase = flow_time[:, None] * self.time_frequencies.to(flow_time)[None]
        time_embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        inputs = torch.cat(
            (
                features,
                action_chunk.flatten(start_dim=1),
                initial_chunk.flatten(start_dim=1),
                time_embedding,
                warm_start,
            ),
            dim=-1,
        )
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden).reshape(expected)

    def normalize_actions(self, actions: Tensor) -> Tensor:
        return (actions - self.action_mean.to(actions)) / self.action_std.to(actions)

    def denormalize_actions(self, actions: Tensor) -> Tensor:
        return actions * self.action_std.to(actions) + self.action_mean.to(actions)

    @torch.no_grad()
    def generate(
        self,
        features: Tensor,
        *,
        initial_actions: Tensor | None = None,
        solver_steps: int = 4,
        solver: str = "euler",
        normalized_clip: float = 10.0,
    ) -> Tensor:
        """Deterministically integrate one cold or warm-started action chunk."""

        if solver_steps <= 0:
            raise ValueError("solver_steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        if normalized_clip <= 0.0:
            raise ValueError("normalized_clip must be positive")
        batch_size = features.shape[0]
        shape = (batch_size, self.config.horizon, self.config.action_dim)
        if initial_actions is None:
            initial = torch.zeros(shape, device=features.device, dtype=features.dtype)
            warm = torch.zeros(
                (batch_size, 1), device=features.device, dtype=features.dtype
            )
        else:
            if tuple(initial_actions.shape) != shape:
                raise ValueError(f"initial_actions must have shape {shape}")
            initial = self.normalize_actions(initial_actions).clamp(
                -normalized_clip, normalized_clip
            )
            warm = torch.ones(
                (batch_size, 1), device=features.device, dtype=features.dtype
            )
        current = initial.clone()
        dt = 1.0 / solver_steps
        for step in range(solver_steps):
            tau = torch.full(
                (batch_size,),
                step * dt,
                device=features.device,
                dtype=features.dtype,
            )
            velocity = self(current, tau, features, initial, warm)
            if solver == "euler":
                current = current + dt * velocity
            else:
                proposal = current + dt * velocity
                next_tau = torch.full_like(tau, (step + 1) * dt)
                correction = self(proposal, next_tau, features, initial, warm)
                current = current + 0.5 * dt * (velocity + correction)
            current.clamp_(-normalized_clip, normalized_clip)
        return self.denormalize_actions(current).clamp(-1.0, 1.0)


__all__ = ["StatefulActionFlow"]
