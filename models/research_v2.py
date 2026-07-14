"""Research-v2 model family with explicit deployable information boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.plan_tokenizer import ActionOnlyPlanTokenizer, ActionOnlyPlanTokenizerConfig


ROLE_NAMES = ("self", "object-belief", "teammate-belief", "task-context")


def _transformer(d: int, heads: int, ffn: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d,
        nhead=heads,
        dim_feedforward=ffn,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


def _action_causal_mask(
    *,
    context_tokens: int,
    action_tokens: int,
    query_tokens: int,
    next_state_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a leak-free mask for action-conditioned transition tokens.

    Context is already encoded before entering the transition stack.  Action
    token ``t`` and output query ``t`` may therefore see only actions up to
    ``t``.  End-of-block state queries may see the complete block.  This keeps
    the parallel query implementation while preventing an early prediction
    from using a later action as a shortcut.
    """

    total = context_tokens + action_tokens + query_tokens + next_state_tokens
    mask = torch.ones(total, total, dtype=torch.bool, device=device)
    context_stop = context_tokens
    action_start = context_stop
    query_start = action_start + action_tokens
    next_start = query_start + query_tokens

    mask[:context_stop, :context_stop] = False
    for index in range(action_tokens):
        row = action_start + index
        mask[row, :context_stop] = False
        mask[row, action_start : action_start + index + 1] = False
    for index in range(query_tokens):
        row = query_start + index
        causal_action_stop = min(index + 1, action_tokens)
        mask[row, :context_stop] = False
        mask[row, action_start : action_start + causal_action_stop] = False
        mask[row, query_start : query_start + index + 1] = False
    if next_state_tokens:
        mask[next_start:, :] = False
    return mask


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()))
            width = hidden_dim
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@dataclass(frozen=True)
class PlanTokenizerV2Config:
    horizon: int = 16
    action_dim: int = 4
    codebook_size: int = 64
    residual_dim: int = 16
    hidden_dim: int = 256
    residual_dropout: float = 0.2

    def to_legacy_config(self) -> ActionOnlyPlanTokenizerConfig:
        return ActionOnlyPlanTokenizerConfig(
            horizon=self.horizon,
            action_dim=self.action_dim,
            latent_dim=self.residual_dim,
            hidden_dim=self.hidden_dim,
            codebook_size=self.codebook_size,
            residual_dropout=self.residual_dropout,
        )


class PlanTokenizerV2(ActionOnlyPlanTokenizer):
    """Action-only VQ tokenizer with a 16-dimensional communication residual."""

    def __init__(self, cfg: PlanTokenizerV2Config):
        self.v2_cfg = cfg
        super().__init__(cfg.to_legacy_config())


@dataclass(frozen=True)
class BeliefEncoderV2Config:
    history: int = 8
    local_dim: int = 21
    object_dim: int = 3
    model_dim: int = 256
    num_heads: int = 8
    temporal_layers: int = 3
    role_layers: int = 3
    ffn_dim: int = 1024
    num_agents: int = 2
    dropout: float = 0.1


class BeliefEncoderV2(nn.Module):
    """Causal-by-input local history encoder producing four fixed role tokens."""

    INPUT_NAMES = (
        "local_history",
        "history_mask",
        "ego_id",
        "object_observation",
        "object_valid",
        "object_age",
        "object_confidence",
    )
    NUM_ROLES = 4

    def __init__(self, cfg: BeliefEncoderV2Config):
        super().__init__()
        if cfg.model_dim % cfg.num_heads:
            raise ValueError("belief model_dim must be divisible by num_heads")
        self.cfg = cfg
        self.local_proj = nn.Linear(cfg.local_dim, cfg.model_dim)
        self.object_proj = nn.Linear(cfg.object_dim, cfg.model_dim)
        self.object_status_proj = nn.Linear(3, cfg.model_dim)
        self.time_embed = nn.Parameter(torch.randn(1, cfg.history, cfg.model_dim) * 0.02)
        self.agent_embed = nn.Embedding(cfg.num_agents, cfg.model_dim)
        self.temporal = _transformer(
            cfg.model_dim, cfg.num_heads, cfg.ffn_dim, cfg.temporal_layers, cfg.dropout
        )
        self.role_queries = nn.Parameter(torch.randn(1, 4, cfg.model_dim) * 0.02)
        self.role_embed = nn.Embedding(4, cfg.model_dim)
        self.role_cross_attention = nn.MultiheadAttention(
            cfg.model_dim,
            cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.role_cross_norm = nn.LayerNorm(cfg.model_dim)
        self.role = _transformer(
            cfg.model_dim, cfg.num_heads, cfg.ffn_dim, cfg.role_layers, cfg.dropout
        )
        self.norm = nn.LayerNorm(cfg.model_dim)

    def forward(
        self,
        local_history: torch.Tensor,
        history_mask: torch.Tensor,
        ego_id: torch.Tensor,
        *,
        object_observation: torch.Tensor,
        object_valid: torch.Tensor,
        object_age: torch.Tensor,
        object_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        if local_history.ndim != 3 or local_history.shape[1:] != (cfg.history, cfg.local_dim):
            raise ValueError("local_history does not match the V2 belief contract")
        B = local_history.shape[0]
        if history_mask.shape != (B, cfg.history) or history_mask.dtype != torch.bool:
            raise TypeError("history_mask must be boolean [B,L]")
        if object_observation.shape != (B, cfg.history, cfg.object_dim):
            raise ValueError("object_observation shape mismatch")
        for name, value in (
            ("object_valid", object_valid),
            ("object_age", object_age),
            ("object_confidence", object_confidence),
        ):
            if value.shape != (B, cfg.history):
                raise ValueError(f"{name} must have shape [B,L]")
        valid = object_valid.to(local_history.dtype)
        status = torch.stack((valid, object_confidence, object_age.clamp_min(0).log1p()), dim=-1)
        object_token = self.object_proj(object_observation * object_confidence.unsqueeze(-1))
        history = (
            self.local_proj(local_history)
            + object_token
            + self.object_status_proj(status)
            + self.time_embed
            + self.agent_embed(ego_id.long()).unsqueeze(1)
        )
        encoded = self.temporal(history, src_key_padding_mask=~history_mask)
        denominator = history_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        pooled = (encoded * history_mask.unsqueeze(-1)).sum(dim=1) / denominator
        role_ids = torch.arange(4, device=local_history.device)
        roles = self.role_queries.expand(B, -1, -1) + self.role_embed(role_ids).unsqueeze(0)
        roles = roles + pooled.unsqueeze(1)
        role_evidence, _ = self.role_cross_attention(
            roles,
            encoded,
            encoded,
            key_padding_mask=~history_mask,
            need_weights=False,
        )
        roles = self.role_cross_norm(roles + role_evidence)
        # Each role gathers different evidence from the local history before
        # roles exchange information.  No future observation or privileged
        # target is accepted by this path.
        slots = self.norm(self.role(roles))
        return {"belief": slots, "slots": slots, "history_encoding": encoded}


class BeliefRoleTargetHeads(nn.Module):
    """Training-only probes; omitted from the exported runtime bundle."""

    def __init__(self, model_dim: int, target_dims: Mapping[str, tuple[int, int]]):
        super().__init__()
        self.target_roles = {name: int(role_and_dim[0]) for name, role_and_dim in target_dims.items()}
        self.heads = nn.ModuleDict(
            {name: MLP(model_dim, model_dim * 2, int(role_and_dim[1]), depth=2) for name, role_and_dim in target_dims.items()}
        )

    def forward(self, belief: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(belief[:, self.target_roles[name]]) for name, head in self.heads.items()}


@dataclass(frozen=True)
class PlanDistributionV2Config:
    belief_dim: int = 256
    codebook_size: int = 64
    residual_dim: int = 16
    model_dim: int = 256
    layers: int = 4
    heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1


class _PlanDistributionBackbone(nn.Module):
    def __init__(self, cfg: PlanDistributionV2Config, *, extra_tokens: int):
        super().__init__()
        self.cfg = cfg
        self.belief_proj = nn.Linear(cfg.belief_dim, cfg.model_dim)
        self.role_embed = nn.Embedding(4, cfg.model_dim)
        self.agent_embed = nn.Embedding(2, cfg.model_dim)
        self.query = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.02)
        self.extra_embed = nn.Embedding(max(extra_tokens, 1), cfg.model_dim)
        self.encoder = _transformer(cfg.model_dim, cfg.heads, cfg.ffn_dim, cfg.layers, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.model_dim)
        self.code_head = MLP(cfg.model_dim, cfg.ffn_dim, cfg.codebook_size, depth=2)
        conditional = cfg.codebook_size * cfg.residual_dim
        self.mu_head = MLP(cfg.model_dim, cfg.ffn_dim, conditional, depth=2)
        self.logvar_head = MLP(cfg.model_dim, cfg.ffn_dim, conditional, depth=2)

    def _distribution(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        state = self.norm(self.encoder(tokens)[:, 0])
        logits = self.code_head(state)
        mu = self.mu_head(state).reshape(-1, cfg.codebook_size, cfg.residual_dim)
        logvar = self.logvar_head(state).reshape(-1, cfg.codebook_size, cfg.residual_dim).clamp(-6, 3)
        return {
            "code_logits": logits,
            "code_probabilities": logits.softmax(dim=-1),
            "residual_mu_by_code": mu,
            "residual_logvar_by_code": logvar,
        }


class PlanProposalV2(_PlanDistributionBackbone):
    """State-conditioned own-plan distribution and deterministic top-K API."""

    def __init__(self, cfg: PlanDistributionV2Config):
        super().__init__(cfg, extra_tokens=1)

    def forward(self, belief: torch.Tensor, ego_id: torch.Tensor) -> dict[str, torch.Tensor]:
        B = belief.shape[0]
        role_ids = torch.arange(4, device=belief.device)
        belief_tokens = self.belief_proj(belief) + self.role_embed(role_ids).unsqueeze(0)
        query = self.query.expand(B, -1, -1) + self.agent_embed(ego_id.long()).unsqueeze(1)
        return self._distribution(torch.cat((query, belief_tokens), dim=1))

    def topk(
        self,
        belief: torch.Tensor,
        ego_id: torch.Tensor,
        active_code_mask: torch.Tensor,
        *,
        k: int = 8,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(belief, ego_id)
        mask = active_code_mask.to(device=belief.device, dtype=torch.bool).reshape(1, -1)
        logits = out["code_logits"].masked_fill(~mask, -torch.inf)
        active_count = int(mask.sum().item())
        if active_count < k:
            raise ValueError(f"proposal requires at least {k} active codes, got {active_count}")
        _, codes = logits.topk(k, dim=-1)
        residuals = out["residual_mu_by_code"].gather(
            1, codes.unsqueeze(-1).expand(-1, -1, self.cfg.residual_dim)
        )
        return {**out, "topk_codes": codes, "topk_residuals": residuals}


class IntentionPosteriorV2(_PlanDistributionBackbone):
    """Teammate-plan posterior from local belief and received envelope facts."""

    def __init__(self, cfg: PlanDistributionV2Config, message_metadata_dim: int = 4):
        super().__init__(cfg, extra_tokens=4)
        self.message_metadata_dim = int(message_metadata_dim)
        self.own_code_embed = nn.Embedding(cfg.codebook_size, cfg.model_dim)
        self.own_residual_proj = nn.Linear(cfg.residual_dim, cfg.model_dim)
        self.metadata_proj = nn.Linear(self.message_metadata_dim, cfg.model_dim)

    def forward(
        self,
        belief: torch.Tensor,
        own_plan_code: torch.Tensor,
        own_plan_residual: torch.Tensor,
        ego_id: torch.Tensor,
        received_message_metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = belief.shape[0]
        role_ids = torch.arange(4, device=belief.device)
        belief_tokens = self.belief_proj(belief) + self.role_embed(role_ids).unsqueeze(0)
        query = self.query.expand(B, -1, -1) + self.agent_embed(ego_id.long()).unsqueeze(1)
        own = self.own_code_embed(own_plan_code.long()).unsqueeze(1)
        own = own + self.own_residual_proj(own_plan_residual).unsqueeze(1) + self.extra_embed.weight[1]
        metadata = self.metadata_proj(received_message_metadata).unsqueeze(1) + self.extra_embed.weight[2]
        out = self._distribution(torch.cat((query, belief_tokens, own, metadata), dim=1))
        probabilities = out["code_probabilities"]
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        within = (probabilities.unsqueeze(-1) * out["residual_logvar_by_code"].exp()).sum(dim=(1, 2))
        out["uncertainty"] = entropy + within / float(self.cfg.residual_dim)
        return out


@dataclass(frozen=True)
class WorldModelV2Config:
    horizon: int = 16
    block_length: int = 4
    belief_tokens: int = 4
    belief_dim: int = 256
    action_dim: int = 4
    model_dim: int = 512
    context_layers: int = 4
    transition_layers: int = 6
    heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    return_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
        if self.horizon % self.block_length:
            raise ValueError("horizon must be divisible by block_length")
        if self.model_dim % self.heads:
            raise ValueError("world model_dim must be divisible by heads")


class _WorldHeads(nn.Module):
    def __init__(self, cfg: WorldModelV2Config):
        super().__init__()
        self.cfg = cfg
        self.belief_head = MLP(
            cfg.model_dim, cfg.ffn_dim, cfg.belief_tokens * cfg.belief_dim, depth=2
        )
        self.step_head = MLP(cfg.model_dim, cfg.ffn_dim, 4, depth=2)
        self.return_head = MLP(cfg.model_dim, cfg.ffn_dim, len(cfg.return_quantiles), depth=2)
        self.terminal_head = MLP(cfg.model_dim, cfg.ffn_dim, 2, depth=2)

    def forward(self, step_features: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        belief = self.belief_head(step_features).reshape(
            *step_features.shape[:2], cfg.belief_tokens, cfg.belief_dim
        )
        step = self.step_head(step_features)
        terminal = self.terminal_head(step_features[:, -1])
        raw_quantiles = self.return_head(step_features[:, -1])
        if raw_quantiles.shape[-1] > 1:
            increments = F.softplus(raw_quantiles[:, 1:])
            return_quantiles = torch.cat(
                (raw_quantiles[:, :1], raw_quantiles[:, :1] + increments.cumsum(dim=-1)),
                dim=-1,
            )
        else:
            return_quantiles = raw_quantiles
        return {
            "future_belief": belief,
            "contact_logits": step[..., 0],
            "force": F.softplus(step[..., 1]),
            "progress": step[..., 2],
            "step_reward": step[..., 3],
            "return_quantiles": return_quantiles,
            "success_logits": terminal[:, 0],
            "constraint_logits": terminal[:, 1],
            "features": step_features,
        }


class _ActionConditionedWorldBase(nn.Module):
    INPUT_NAMES = ("belief", "ego_actions", "teammate_actions")

    def __init__(self, cfg: WorldModelV2Config):
        super().__init__()
        self.cfg = cfg
        self.belief_proj = nn.Linear(cfg.belief_dim, cfg.model_dim)
        self.role_embed = nn.Embedding(cfg.belief_tokens, cfg.model_dim)
        self.context = _transformer(
            cfg.model_dim, cfg.heads, cfg.ffn_dim, cfg.context_layers, cfg.dropout
        )
        self.action_proj = nn.Linear(2 * cfg.action_dim, cfg.model_dim)
        self.time_embed = nn.Embedding(cfg.horizon, cfg.model_dim)
        self.query_embed = nn.Parameter(torch.randn(1, cfg.horizon, cfg.model_dim) * 0.02)
        self.transition = _transformer(
            cfg.model_dim, cfg.heads, cfg.ffn_dim, cfg.transition_layers, cfg.dropout
        )
        self.norm = nn.LayerNorm(cfg.model_dim)
        self.heads = _WorldHeads(cfg)

    def _validate(
        self, belief: torch.Tensor, ego_actions: torch.Tensor, teammate_actions: torch.Tensor
    ) -> None:
        cfg = self.cfg
        if belief.ndim != 3 or belief.shape[1:] != (cfg.belief_tokens, cfg.belief_dim):
            raise ValueError("belief shape does not match WorldModelV2")
        expected = (belief.shape[0], cfg.horizon, cfg.action_dim)
        if ego_actions.shape != expected or teammate_actions.shape != expected:
            raise ValueError(f"world actions must both have shape {expected}")

    def _context(self, belief: torch.Tensor) -> torch.Tensor:
        role_ids = torch.arange(self.cfg.belief_tokens, device=belief.device)
        return self.context(self.belief_proj(belief) + self.role_embed(role_ids).unsqueeze(0))

    def _action_tokens(
        self, ego_actions: torch.Tensor, teammate_actions: torch.Tensor, start: int = 0
    ) -> torch.Tensor:
        length = ego_actions.shape[1]
        times = torch.arange(start, start + length, device=ego_actions.device)
        return self.action_proj(torch.cat((ego_actions, teammate_actions), dim=-1)) + self.time_embed(times)


class DirectParallelWorldModelV2(_ActionConditionedWorldBase):
    """Matched action-conditioned direct baseline with the V2 output contract."""

    def __init__(self, cfg: WorldModelV2Config):
        super().__init__(cfg)
        self.register_buffer(
            "transition_mask",
            _action_causal_mask(
                context_tokens=cfg.belief_tokens,
                action_tokens=cfg.horizon,
                query_tokens=cfg.horizon,
                next_state_tokens=0,
                device=torch.device("cpu"),
            ),
            persistent=False,
        )

    def forward(
        self, belief: torch.Tensor, ego_actions: torch.Tensor, teammate_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self._validate(belief, ego_actions, teammate_actions)
        B = belief.shape[0]
        context = self._context(belief)
        actions = self._action_tokens(ego_actions, teammate_actions)
        queries = self.query_embed.expand(B, -1, -1)
        tokens = torch.cat((context, actions, queries), dim=1)
        encoded = self.norm(self.transition(tokens, mask=self.transition_mask))
        return self.heads(encoded[:, -self.cfg.horizon :])


class BlockTransitionWorldModelV2(_ActionConditionedWorldBase):
    """Four-step parallel transitions with a shared block-autoregressive cell."""

    def __init__(self, cfg: WorldModelV2Config):
        super().__init__(cfg)
        self.block_query = nn.Parameter(torch.randn(1, cfg.block_length, cfg.model_dim) * 0.02)
        self.next_belief_query = nn.Parameter(torch.randn(1, cfg.belief_tokens, cfg.model_dim) * 0.02)
        self.next_belief_head = MLP(cfg.model_dim, cfg.ffn_dim, cfg.belief_dim, depth=2)
        self.block_embed = nn.Embedding(cfg.horizon // cfg.block_length, cfg.model_dim)
        self.register_buffer(
            "block_transition_mask",
            _action_causal_mask(
                context_tokens=cfg.belief_tokens,
                action_tokens=cfg.block_length,
                query_tokens=cfg.block_length,
                next_state_tokens=cfg.belief_tokens,
                device=torch.device("cpu"),
            ),
            persistent=False,
        )

    def _run(
        self,
        belief: torch.Tensor,
        ego_actions: torch.Tensor,
        teammate_actions: torch.Tensor,
        teacher_block_beliefs: torch.Tensor | None = None,
        teacher_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate(belief, ego_actions, teammate_actions)
        cfg = self.cfg
        B = belief.shape[0]
        current = belief
        step_features: list[torch.Tensor] = []
        block_boundaries: list[torch.Tensor] = []
        blocks = cfg.horizon // cfg.block_length
        for block in range(blocks):
            start = block * cfg.block_length
            stop = start + cfg.block_length
            context = self._context(current)
            actions = self._action_tokens(
                ego_actions[:, start:stop], teammate_actions[:, start:stop], start
            )
            marker = self.block_embed.weight[block]
            queries = self.block_query.expand(B, -1, -1) + marker
            next_queries = self.next_belief_query.expand(B, -1, -1) + marker
            tokens = torch.cat((context, actions, queries, next_queries), dim=1)
            encoded = self.norm(self.transition(tokens, mask=self.block_transition_mask))
            query_start = cfg.belief_tokens + cfg.block_length
            features = encoded[:, query_start : query_start + cfg.block_length]
            next_tokens = encoded[:, -cfg.belief_tokens :]
            predicted_next = current + self.next_belief_head(next_tokens)
            step_features.append(features)
            block_boundaries.append(predicted_next)
            if teacher_block_beliefs is not None and block < blocks - 1:
                if teacher_mask is None:
                    raise ValueError("teacher_mask is required with teacher_block_beliefs")
                use_teacher = teacher_mask[:, block].reshape(B, 1, 1)
                current = torch.where(use_teacher, teacher_block_beliefs[:, block], predicted_next)
            else:
                current = predicted_next
        out = self.heads(torch.cat(step_features, dim=1))
        out["block_boundary_belief"] = torch.stack(block_boundaries, dim=1)
        return out

    def forward(
        self, belief: torch.Tensor, ego_actions: torch.Tensor, teammate_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self._run(belief, ego_actions, teammate_actions)

    def forward_train(
        self,
        belief: torch.Tensor,
        ego_actions: torch.Tensor,
        teammate_actions: torch.Tensor,
        teacher_block_beliefs: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Training-only scheduled sampling; the deployable forward has no target input."""

        return self._run(
            belief,
            ego_actions,
            teammate_actions,
            teacher_block_beliefs=teacher_block_beliefs,
            teacher_mask=teacher_mask,
        )


def decode_plan_batch(
    tokenizer: ActionOnlyPlanTokenizer,
    codes: torch.Tensor,
    residuals: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    """Decode arbitrary leading plan dimensions into physical action chunks."""

    leading = codes.shape
    flat_codes = codes.reshape(-1)
    flat_residuals = residuals.reshape(-1, residuals.shape[-1])
    decoded = tokenizer.decode_plan_latent(flat_codes, flat_residuals)["recon_actions"]
    decoded = decoded * action_std.reshape(1, 1, -1) + action_mean.reshape(1, 1, -1)
    return decoded.reshape(*leading, tokenizer.cfg.horizon, tokenizer.cfg.action_dim)


def count_deployable_parameters(modules: Mapping[str, nn.Module]) -> dict[str, int]:
    counts = {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in modules.items()}
    counts["total"] = sum(counts.values())
    return counts
