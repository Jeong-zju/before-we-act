"""S2-R5 team predictors with an immutable S2-R4 P0 own path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from models.wam_multimodal.team_shared_future_predictor import (
    TeamSharedFuturePrediction,
)


@dataclass(frozen=True)
class ProtectedTeamFuturePredictorConfig:
    """The only paired-branch degree of freedom is ``team_mixer``."""

    layers: int = 2
    heads: int = 8
    ffn_dim: int = 1536
    dropout: float = 0.1
    team_mixer: str = "shared"

    def __post_init__(self) -> None:
        if min(self.layers, self.heads, self.ffn_dim) <= 0:
            raise ValueError("team dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.team_mixer not in {"shared", "role_mot"}:
            raise ValueError("team_mixer must be shared or role_mot")

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "ProtectedTeamFuturePredictorConfig":
        return cls(**dict(value))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _RoleMixer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        layers: int,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )

    def forward(self, tokens: Tensor, padding: Tensor) -> Tensor:
        return self.encoder(tokens, src_key_padding_mask=padding)


class ProtectedTeamFuturePredictor(nn.Module):
    """Train only peer/shared modules while returning exact P0 own outputs.

    ``shared`` uses one transformer for peer and shared role reads. ``role_mot``
    hard-routes the same input tokens through two private transformer blocks.
    There is no learned router and no team-to-own residual path.
    """

    def __init__(
        self,
        local_config: LocalFuturePredictorConfig,
        team_config: ProtectedTeamFuturePredictorConfig,
    ) -> None:
        super().__init__()
        if local_config.d_model % team_config.heads:
            raise ValueError("local d_model must be divisible by team heads")
        self.local_config = local_config
        self.team_config = team_config
        self.protected_own = LocalActionConditionedFuturePredictor(local_config)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(local_config.visual_latent_dim),
            nn.Linear(local_config.visual_latent_dim, local_config.d_model),
            nn.GELU(),
            nn.Linear(local_config.d_model, local_config.d_model),
        )
        self.team_agent_norm = nn.LayerNorm(local_config.d_model)
        self.slot_embedding = nn.Parameter(
            torch.randn(
                1, 1, local_config.max_agents + 1, local_config.d_model
            ) * 0.02
        )
        mixer_arguments = {
            "d_model": local_config.d_model,
            "heads": team_config.heads,
            "ffn_dim": team_config.ffn_dim,
            "dropout": team_config.dropout,
            "layers": team_config.layers,
        }
        self.shared_mixer = _RoleMixer(**mixer_arguments)
        # Private Role-MoT construction must not perturb initialization of the
        # common heads on the paired shared branch.
        with torch.random.fork_rng(enabled=team_config.team_mixer == "role_mot"):
            self.peer_mixer = (
                _RoleMixer(**mixer_arguments)
                if team_config.team_mixer == "role_mot"
                else None
            )
        self.peer_state_head = _head(local_config.d_model, local_config.state_dim)
        self.peer_visual_head = _head(
            local_config.d_model, local_config.visual_latent_dim
        )
        self.shared_visual_head = _head(
            local_config.d_model, local_config.visual_latent_dim
        )
        self._protected_loaded = False

    def load_protected_own(self, state_dict: Mapping[str, Tensor]) -> None:
        self.protected_own.load_state_dict(state_dict, strict=True)
        self._protected_loaded = True
        self.protected_own.eval()
        for parameter in self.protected_own.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ProtectedTeamFuturePredictor":
        super().train(mode)
        # P0 dropout must never activate after protection.
        self.protected_own.eval()
        return self

    def team_state_dict(self) -> dict[str, Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("protected_own.")
        }

    def load_team_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        expected = set(self.team_state_dict())
        if set(state_dict) != expected:
            missing = sorted(expected - set(state_dict))
            extra = sorted(set(state_dict) - expected)
            raise ValueError(
                f"S2-R5 team state mismatch; missing={missing}, extra={extra}"
            )
        self.load_state_dict(dict(state_dict), strict=False)

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("protected_own.") and parameter.requires_grad
        )

    def parameter_audit(self) -> dict[str, object]:
        total = sum(parameter.numel() for parameter in self.parameters())
        protected = sum(
            parameter.numel() for parameter in self.protected_own.parameters()
        )
        trainable = sum(
            parameter.numel() for parameter in self.trainable_parameters()
        )
        common = sum(
            parameter.numel()
            for module in (
                self.shared_projection,
                self.team_agent_norm,
            )
            for parameter in module.parameters()
        ) + self.slot_embedding.numel()
        shared_mixer = sum(
            parameter.numel() for parameter in self.shared_mixer.parameters()
        )
        peer_mixer = (
            sum(parameter.numel() for parameter in self.peer_mixer.parameters())
            if self.peer_mixer is not None
            else shared_mixer
        )
        peer_head = sum(
            parameter.numel()
            for module in (self.peer_state_head, self.peer_visual_head)
            for parameter in module.parameters()
        )
        shared_head = sum(
            parameter.numel()
            for parameter in self.shared_visual_head.parameters()
        )
        return {
            "total_parameters": total,
            "protected_parameters": protected,
            "trainable_parameters": trainable,
            "team_mixer": self.team_config.team_mixer,
            "activated_parameters_by_role": {
                "peer": common + peer_mixer + peer_head,
                "shared": common + shared_mixer + shared_head,
            },
            "mixer_invocations_per_sample": 2,
            "active_depth": self.team_config.layers,
            "active_width": self.local_config.d_model,
            "strict_parameter_matched": False,
            "trainable_names": [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            ],
        }

    def forward(
        self,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        candidate_actions: Tensor,
        valid_agent_mask: Tensor,
        *,
        actions_by_focal: Tensor | None = None,
    ) -> TeamSharedFuturePrediction:
        if not self._protected_loaded:
            raise RuntimeError("load_protected_own must run before forward")
        config = self.local_config
        batch_size = current_state.shape[0]
        if shared_visual_latent.shape != (
            batch_size,
            config.visual_grid_tokens,
            config.visual_latent_dim,
        ):
            raise ValueError("shared_visual_latent has an invalid shape")
        if actions_by_focal is None:
            actions_by_focal = candidate_actions[:, None].expand(
                -1, config.max_agents, -1, -1, -1
            )
        if actions_by_focal.shape != (
            batch_size,
            config.max_agents,
            config.max_agents,
            config.action_horizon,
            config.action_dim,
        ):
            raise ValueError(
                "actions_by_focal must be [B,focal,target,H,action_dim]"
            )

        with torch.no_grad():
            own_context = self.protected_own.encode_context(
                current_state,
                current_visual_latent,
                candidate_actions,
                valid_agent_mask,
                valid_agent_mask,
            )
            own_state, own_visual = self.protected_own.decode_future(
                own_context, valid_agent_mask
            )
            state_token = self.protected_own.state_projection(current_state)
            visual_token = self.protected_own.visual_projection(
                current_visual_latent
            ).mean(dim=2)
            action_token = self.protected_own.action_projection(actions_by_focal)
            action_token = (
                action_token + self.protected_own.action_position[:, None]
            ).mean(dim=3)

        agent_tokens = self.team_agent_norm(
            state_token[:, None] + visual_token[:, None] + action_token
        )
        shared_token = self.shared_projection(shared_visual_latent).mean(dim=1)
        shared_tokens = shared_token[:, None, None].expand(
            -1, config.max_agents, -1, -1
        )
        tokens = torch.cat((shared_tokens, agent_tokens), dim=2)
        tokens = tokens + self.slot_embedding
        valid_sequence = torch.cat(
            (
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=valid_agent_mask.device,
                ),
                valid_agent_mask,
            ),
            dim=1,
        )
        padding = (~valid_sequence[:, None].expand(
            -1, config.max_agents, -1
        )).reshape(
            batch_size * config.max_agents, config.max_agents + 1
        )
        flat_tokens = tokens.reshape(
            batch_size * config.max_agents,
            config.max_agents + 1,
            config.d_model,
        )
        shared_encoded = self.shared_mixer(flat_tokens, padding).reshape(
            batch_size,
            config.max_agents,
            config.max_agents + 1,
            config.d_model,
        )
        # Both candidates execute two equal-shape role passes. P0 reuses the
        # same weights; P1 hard-routes each role to a private block.
        peer_encoded = (
            (self.peer_mixer or self.shared_mixer)(flat_tokens, padding).reshape(
                batch_size,
                config.max_agents,
                config.max_agents + 1,
                config.d_model,
            )
            if self.peer_mixer is not None
            else shared_encoded
        )
        encoded_shared = shared_encoded[:, :, 0]
        encoded_agents = peer_encoded[:, :, 1:]
        focal_index = torch.arange(config.max_agents, device=current_state.device)
        focal_context = encoded_agents[:, focal_index, focal_index]
        pair_context = encoded_agents + focal_context[:, :, None]
        future_position = self.protected_own.future_position
        grid_position = self.protected_own.grid_position
        pair_future = pair_context[:, :, :, None] + future_position[:, None]
        peer_state = self.peer_state_head(pair_future)
        peer_visual = self.peer_visual_head(
            pair_future[:, :, :, :, None] + grid_position[:, None]
        )
        shared_future = encoded_shared[:, :, None] + future_position
        shared_visual = self.shared_visual_head(
            shared_future[:, :, :, None] + grid_position
        )
        pair_valid = valid_agent_mask[:, :, None] & valid_agent_mask[:, None, :]
        focal_valid = valid_agent_mask[:, :, None, None, None]
        return TeamSharedFuturePrediction(
            own_state=own_state,
            own_visual=own_visual,
            peer_state=peer_state * pair_valid[:, :, :, None, None],
            peer_visual=peer_visual * pair_valid[:, :, :, None, None, None],
            shared_visual=shared_visual * focal_valid,
        )


def _head(d_model: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, output_dim),
    )


__all__ = [
    "ProtectedTeamFuturePredictor",
    "ProtectedTeamFuturePredictorConfig",
]
