"""Horizon-causal action aggregation for the S4-R8 method comparison.

R8 changes only how projected action tokens are summarized for each future
horizon.  The P0 path uses a strict prefix mean.  P1 adds a rank-limited
attention residual whose output projection is initialized to exact zero, so a
fresh P1 is elementwise identical to P0 while retaining a trainable causal
path.  Neither path can inspect action tokens after its declared horizon.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from models.wam_multimodal.cross_agent_world_conditioned_flow import (
    PredictedFutureLatents,
)
from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
)
from models.wam_multimodal.protected_team_future_predictor import (
    ProtectedTeamFuturePredictor,
)
from models.wam_multimodal.utility_calibrated_world_flow import (
    S4ActiveTeamFutureProvider,
)


R8_ACTION_PREFIX_AGGREGATORS = frozenset({"prefix_mean", "causal_prefix_attention"})


class HorizonCausalActionAggregator(nn.Module):
    """Return one action summary per configured future horizon.

    Input has shape ``[..., action_horizon, d_model]`` and output has shape
    ``[..., number_of_future_horizons, d_model]``.  P1 attention is computed
    independently for every horizon and masked before softmax.  Its residual
    output projection and bias are exact zeros at initialization.
    """

    def __init__(
        self,
        *,
        d_model: int,
        action_horizon: int,
        future_horizons: tuple[int, ...],
        kind: str,
        rank: int = 32,
    ) -> None:
        super().__init__()
        if kind not in R8_ACTION_PREFIX_AGGREGATORS:
            raise ValueError(f"unregistered R8 action prefix aggregator: {kind!r}")
        if d_model <= 0 or action_horizon <= 0 or rank <= 0:
            raise ValueError("R8 aggregator dimensions must be positive")
        if not future_horizons or any(
            value <= 0 or value > action_horizon for value in future_horizons
        ):
            raise ValueError("R8 future horizons must lie inside the action chunk")
        if tuple(sorted(set(future_horizons))) != future_horizons:
            raise ValueError("R8 future horizons must be strictly increasing")
        self.d_model = int(d_model)
        self.action_horizon = int(action_horizon)
        self.future_horizons = tuple(int(value) for value in future_horizons)
        self.kind = kind
        self.rank = int(rank)
        self.register_buffer(
            "prefix_mask",
            torch.arange(action_horizon)[None, :]
            < torch.tensor(self.future_horizons)[:, None],
            persistent=True,
        )
        if kind == "causal_prefix_attention":
            horizons = len(self.future_horizons)
            self.query_weight = nn.Parameter(torch.empty(horizons, rank, d_model))
            self.key_weight = nn.Parameter(torch.empty(horizons, rank, d_model))
            self.value_weight = nn.Parameter(torch.empty(horizons, rank, d_model))
            self.output_weight = nn.Parameter(torch.zeros(horizons, d_model, rank))
            self.output_bias = nn.Parameter(torch.zeros(horizons, d_model))
            nn.init.xavier_uniform_(self.query_weight)
            nn.init.xavier_uniform_(self.key_weight)
            nn.init.xavier_uniform_(self.value_weight)
        else:
            self.register_parameter("query_weight", None)
            self.register_parameter("key_weight", None)
            self.register_parameter("value_weight", None)
            self.register_parameter("output_weight", None)
            self.register_parameter("output_bias", None)

    def forward(self, action_tokens: Tensor) -> Tensor:
        if action_tokens.ndim < 2 or action_tokens.shape[-2:] != (
            self.action_horizon,
            self.d_model,
        ):
            raise ValueError("projected actions must end with [action_horizon,d_model]")
        cumulative = action_tokens.cumsum(dim=-2)
        indices = torch.tensor(
            [value - 1 for value in self.future_horizons],
            dtype=torch.long,
            device=action_tokens.device,
        )
        prefix = cumulative.index_select(-2, indices)
        divisors = action_tokens.new_tensor(self.future_horizons)
        shape = (1,) * (prefix.ndim - 2) + (len(self.future_horizons), 1)
        prefix = prefix / divisors.view(shape)
        if self.kind == "prefix_mean":
            return prefix
        if any(
            value is None
            for value in (
                self.query_weight,
                self.key_weight,
                self.value_weight,
                self.output_weight,
                self.output_bias,
            )
        ):
            raise RuntimeError("causal attention parameters are unavailable")
        query = torch.einsum("...hd,hrd->...hr", prefix, self.query_weight)
        key = torch.einsum("...td,hrd->...htr", action_tokens, self.key_weight)
        value = torch.einsum("...td,hrd->...htr", action_tokens, self.value_weight)
        scores = torch.einsum("...hr,...htr->...ht", query, key)
        scores = scores / math.sqrt(self.rank)
        mask_shape = (1,) * (scores.ndim - 2) + self.prefix_mask.shape
        scores = scores.masked_fill(
            ~self.prefix_mask.view(mask_shape),
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("...ht,...htr->...hr", weights, value)
        residual = (
            torch.einsum("...hr,hdr->...hd", attended, self.output_weight)
            + self.output_bias
        )
        return prefix + residual

    def audit(self) -> dict[str, object]:
        output_zero = True
        if self.kind == "causal_prefix_attention":
            output_zero = bool(
                torch.count_nonzero(self.output_weight).item() == 0
                and torch.count_nonzero(self.output_bias).item() == 0
            )
        return {
            "kind": self.kind,
            "rank": self.rank,
            "future_horizons": list(self.future_horizons),
            "strict_prefix_mask": True,
            "output_projection_zero_initialized": output_zero,
        }


class HorizonCausalActiveTeamFutureProvider(S4ActiveTeamFutureProvider):
    """R8 own/peer/shared provider with a strict per-horizon action prefix."""

    def __init__(
        self,
        local_predictor: LocalActionConditionedFuturePredictor,
        r5_p0_source: ProtectedTeamFuturePredictor,
        *,
        action_prefix_aggregator: str,
        action_prefix_rank: int = 32,
    ) -> None:
        super().__init__(local_predictor, r5_p0_source)
        config = self.local_config
        self.action_prefix_aggregator = HorizonCausalActionAggregator(
            d_model=config.d_model,
            action_horizon=config.action_horizon,
            future_horizons=config.future_horizons,
            kind=action_prefix_aggregator,
            rank=action_prefix_rank,
        )

    @classmethod
    def from_active_provider(
        cls,
        source: S4ActiveTeamFutureProvider,
        *,
        action_prefix_aggregator: str,
        action_prefix_rank: int = 32,
    ) -> "HorizonCausalActiveTeamFutureProvider":
        """Clone an already assembled R7 provider without reloading ancestors."""

        if isinstance(source, cls):
            raise ValueError("R8 provider source must be the common R7 clone")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.local_predictor = copy.deepcopy(source.local_predictor)
        result.local_config = result.local_predictor.config
        result.shared_projection = copy.deepcopy(source.shared_projection)
        result.team_agent_norm = copy.deepcopy(source.team_agent_norm)
        result.slot_embedding = nn.Parameter(source.slot_embedding.detach().clone())
        result.shared_mixer = copy.deepcopy(source.shared_mixer)
        result.peer_state_head = copy.deepcopy(source.peer_state_head)
        result.peer_visual_head = copy.deepcopy(source.peer_visual_head)
        result.shared_visual_head = copy.deepcopy(source.shared_visual_head)
        config = result.local_config
        result.action_prefix_aggregator = HorizonCausalActionAggregator(
            d_model=config.d_model,
            action_horizon=config.action_horizon,
            future_horizons=config.future_horizons,
            kind=action_prefix_aggregator,
            rank=action_prefix_rank,
        )
        for parameter in result.parameters():
            parameter.requires_grad_(True)
        result.train(source.training)
        return result

    def forward(
        self,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        candidate_actions: Tensor,
        valid_agent_mask: Tensor,
        *,
        actions_by_focal: Tensor | None = None,
    ) -> PredictedFutureLatents:
        config = self.local_config
        batch_size = current_state.shape[0]
        expected_state = (batch_size, config.max_agents, config.state_dim)
        expected_visual = (
            batch_size,
            config.max_agents,
            config.visual_grid_tokens,
            config.visual_latent_dim,
        )
        expected_actions = (
            batch_size,
            config.max_agents,
            config.action_horizon,
            config.action_dim,
        )
        if current_state.shape != expected_state:
            raise ValueError("current_state has an invalid shape")
        if current_visual_latent.shape != expected_visual:
            raise ValueError("current_visual_latent has an invalid shape")
        if candidate_actions.shape != expected_actions:
            raise ValueError("candidate_actions has an invalid shape")
        if shared_visual_latent.shape != (
            batch_size,
            config.visual_grid_tokens,
            config.visual_latent_dim,
        ):
            raise ValueError("shared_visual_latent has an invalid shape")
        if valid_agent_mask.shape != (batch_size, config.max_agents):
            raise ValueError("valid_agent_mask must be [B,A]")
        if valid_agent_mask.dtype != torch.bool:
            raise TypeError("valid_agent_mask must have dtype bool")
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
            raise ValueError("actions_by_focal must be [B,focal,target,H,action_dim]")

        local = self.local_predictor
        horizon_count = len(config.future_horizons)
        state_token = local.state_projection(current_state)
        visual_tokens = local.visual_projection(current_visual_latent)
        own_action_tokens = (
            local.action_projection(candidate_actions) + local.action_position
        )
        own_action_summary = self.action_prefix_aggregator(own_action_tokens)
        own_action_summary = own_action_summary * valid_agent_mask[..., None, None]
        own_tokens = torch.cat(
            (
                state_token[:, :, None, None].expand(-1, -1, horizon_count, 1, -1),
                visual_tokens[:, :, None].expand(-1, -1, horizon_count, -1, -1),
                own_action_summary[:, :, :, None],
            ),
            dim=3,
        )
        own_context = (
            local.context_encoder(
                own_tokens.reshape(-1, own_tokens.shape[3], config.d_model)
            )
            .mean(dim=1)
            .reshape(batch_size, config.max_agents, horizon_count, config.d_model)
        )
        own_context = own_context * valid_agent_mask[..., None, None]
        own_future = own_context + local.future_position
        own_state = local.state_head(own_future)
        own_visual = local.visual_head(own_future[:, :, :, None] + local.grid_position)

        team_action_tokens = local.action_projection(actions_by_focal)
        team_action_tokens = team_action_tokens + local.action_position[:, None]
        team_action_summary = self.action_prefix_aggregator(team_action_tokens)
        team_agent_tokens = self.team_agent_norm(
            state_token[:, None, :, None]
            + visual_tokens.mean(dim=2)[:, None, :, None]
            + team_action_summary
        ).permute(0, 1, 3, 2, 4)
        shared_token = self.shared_projection(shared_visual_latent).mean(dim=1)
        shared_tokens = shared_token[:, None, None, None].expand(
            -1, config.max_agents, horizon_count, 1, -1
        )
        team_tokens = torch.cat((shared_tokens, team_agent_tokens), dim=3)
        team_tokens = team_tokens + self.slot_embedding[:, :, None]
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
        padding = (
            ~valid_sequence[:, None, None].expand(
                -1, config.max_agents, horizon_count, -1
            )
        ).reshape(
            batch_size * config.max_agents * horizon_count,
            config.max_agents + 1,
        )
        encoded = self.shared_mixer(
            team_tokens.reshape(
                batch_size * config.max_agents * horizon_count,
                config.max_agents + 1,
                config.d_model,
            ),
            padding,
        ).reshape(
            batch_size,
            config.max_agents,
            horizon_count,
            config.max_agents + 1,
            config.d_model,
        )
        encoded_shared = encoded[:, :, :, 0]
        encoded_agents = encoded[:, :, :, 1:]
        focal_index = torch.arange(config.max_agents, device=current_state.device).view(
            1, config.max_agents, 1, 1, 1
        )
        focal_context = encoded_agents.gather(
            3,
            focal_index.expand(
                batch_size,
                config.max_agents,
                horizon_count,
                1,
                config.d_model,
            ),
        ).squeeze(3)
        pair_context = encoded_agents + focal_context[:, :, :, None]
        pair_context = pair_context.permute(0, 1, 3, 2, 4)
        pair_future = pair_context + local.future_position[:, None]
        peer_state = self.peer_state_head(pair_future)
        peer_visual = self.peer_visual_head(
            pair_future[:, :, :, :, None] + local.grid_position[:, None]
        )
        shared_future = encoded_shared + local.future_position
        shared_visual = self.shared_visual_head(
            shared_future[:, :, :, None] + local.grid_position
        )
        pair_valid = valid_agent_mask[:, :, None] & valid_agent_mask[:, None, :]
        focal_valid = valid_agent_mask[:, :, None, None, None]
        own_valid = valid_agent_mask[:, :, None, None]
        return PredictedFutureLatents(
            own_state=own_state * own_valid,
            own_visual=own_visual * own_valid[..., None],
            peer_state=peer_state * pair_valid[:, :, :, None, None],
            peer_visual=peer_visual * pair_valid[:, :, :, None, None, None],
            shared_visual=shared_visual * focal_valid,
        )


__all__ = [
    "HorizonCausalActionAggregator",
    "HorizonCausalActiveTeamFutureProvider",
    "R8_ACTION_PREFIX_AGGREGATORS",
]
