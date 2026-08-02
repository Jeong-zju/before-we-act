"""Token-preserving world-evidence routing for S4-R7.

The public contract in this module deliberately keeps source-agent, future
horizon, and spatial-token axes intact until the low-rank attention read.  A
router may summarize each of the twelve ``source x horizon`` groups, but the
evidence adapter itself never averages the tokens it attends to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn


EVIDENCE_SOURCES = ("own", "peer", "shared")
EVIDENCE_HORIZONS = (1, 25, 50, 100)


@dataclass(frozen=True)
class WorldEvidenceRouterConfig:
    """Shape and initialization contract shared by both R7 candidates."""

    max_agents: int = 4
    future_horizons: tuple[int, ...] = EVIDENCE_HORIZONS
    visual_grid_tokens: int = 4
    state_dim: int = 18
    visual_dim: int = 256
    d_model: int = 384
    evidence_rank: int = 32
    action_dim: int = 8
    new_gate_max: float = 0.25

    def __post_init__(self) -> None:
        if self.max_agents != 4:
            raise ValueError("S4 evidence contract requires exactly four agent slots")
        if tuple(self.future_horizons) != EVIDENCE_HORIZONS:
            raise ValueError(
                "S4 evidence horizons must be exactly (1, 25, 50, 100)"
            )
        if self.visual_grid_tokens != 4:
            raise ValueError("S4 evidence contract requires a 2x2 visual grid")
        if self.d_model != 384:
            raise ValueError("S4 evidence token width must be 384")
        if self.evidence_rank != 32:
            raise ValueError("S4 low-rank evidence adapters must have rank 32")
        if min(self.state_dim, self.visual_dim, self.action_dim) <= 0:
            raise ValueError("S4 evidence dimensions must be positive")
        if not 0.0 < self.new_gate_max <= 1.0:
            raise ValueError("new_gate_max must lie in (0, 1]")

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "WorldEvidenceRouterConfig":
        payload = dict(value)
        if "future_horizons" in payload:
            payload["future_horizons"] = tuple(
                int(item) for item in payload["future_horizons"]  # type: ignore[arg-type]
            )
        return cls(**payload)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["future_horizons"] = list(self.future_horizons)
        return value

    @property
    def group_count(self) -> int:
        return len(EVIDENCE_SOURCES) * len(self.future_horizons)

    @property
    def token_count(self) -> int:
        return 1 + self.visual_grid_tokens


@dataclass(frozen=True)
class EvidenceTokens:
    """Padded S4 evidence and its fail-closed validity mask.

    ``tokens`` is ``[B, focal, source=3, source_agent=4, future=4,
    token=5, dim=384]``. ``mask`` has the same axes without ``dim``.
    """

    tokens: Tensor
    mask: Tensor


@dataclass(frozen=True)
class EvidenceAdapterOutput:
    z: Tensor
    group_mask: Tensor


@dataclass(frozen=True)
class EvidenceRouterOutput:
    logits: Tensor
    pi: Tensor
    group_mask: Tensor
    group_summary: Tensor


class S4WorldEvidenceProvider(nn.Module):
    """Project predicted own/peer/shared futures into the canonical token grid.

    The optional ``future_predictor`` is an active, trainable clone supplied by
    the training assembly.  Synthetic tests and causal evaluation can instead
    pass ``predicted_futures`` directly, so this class never requires a real
    checkpoint merely to validate the tensor and mask contract.
    """

    def __init__(
        self,
        config: WorldEvidenceRouterConfig,
        future_predictor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.future_predictor = future_predictor
        self.own_state_projection = nn.Linear(config.state_dim, config.d_model)
        self.own_visual_projection = nn.Linear(config.visual_dim, config.d_model)
        self.peer_state_projection = nn.Linear(config.state_dim, config.d_model)
        self.peer_visual_projection = nn.Linear(config.visual_dim, config.d_model)
        self.shared_visual_projection = nn.Linear(config.visual_dim, config.d_model)
        self.source_embedding = nn.Embedding(len(EVIDENCE_SOURCES), config.d_model)
        self.source_agent_embedding = nn.Embedding(
            config.max_agents, config.d_model
        )
        self.horizon_embedding = nn.Embedding(
            len(config.future_horizons), config.d_model
        )
        self.token_type_embedding = nn.Embedding(2, config.d_model)
        self.grid_position_embedding = nn.Embedding(
            config.visual_grid_tokens, config.d_model
        )

    def forward(
        self,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        active_clean_actions: Tensor,
        valid_agent_mask: Tensor,
        *,
        predicted_futures: object | None = None,
    ) -> EvidenceTokens:
        if predicted_futures is None:
            if self.future_predictor is None:
                raise RuntimeError(
                    "predicted_futures is required when no active future predictor "
                    "was supplied"
                )
            # Future prediction must never optimize the Flow through its clean
            # endpoint estimate.  The active predictor itself remains trainable.
            predicted_futures = self.future_predictor(
                current_state,
                current_visual_latent,
                shared_visual_latent,
                active_clean_actions.detach(),
                valid_agent_mask,
            )
        return self.pack(predicted_futures, valid_agent_mask)

    def pack(
        self,
        predicted_futures: object,
        valid_agent_mask: Tensor,
    ) -> EvidenceTokens:
        config = self.config
        own_state = _future_value(predicted_futures, "own_state")
        own_visual = _future_value(predicted_futures, "own_visual")
        peer_state = _future_value(predicted_futures, "peer_state")
        peer_visual = _future_value(predicted_futures, "peer_visual")
        shared_visual = _future_value(predicted_futures, "shared_visual")
        batch_size = own_state.shape[0]
        agents = config.max_agents
        futures = len(config.future_horizons)
        grid = config.visual_grid_tokens
        if own_state.shape != (batch_size, agents, futures, config.state_dim):
            raise ValueError("own_state must be [B,4,4,state_dim]")
        if own_visual.shape != (
            batch_size,
            agents,
            futures,
            grid,
            config.visual_dim,
        ):
            raise ValueError("own_visual must be [B,4,4,4,visual_dim]")
        if peer_state.shape != (
            batch_size,
            agents,
            agents,
            futures,
            config.state_dim,
        ):
            raise ValueError("peer_state must be [B,focal=4,source_agent=4,4,state_dim]")
        if peer_visual.shape != (
            batch_size,
            agents,
            agents,
            futures,
            grid,
            config.visual_dim,
        ):
            raise ValueError(
                "peer_visual must be [B,focal=4,source_agent=4,4,4,visual_dim]"
            )
        if shared_visual.shape != (
            batch_size,
            agents,
            futures,
            grid,
            config.visual_dim,
        ):
            raise ValueError("shared_visual must be [B,focal=4,4,4,visual_dim]")
        if valid_agent_mask.shape != (batch_size, agents):
            raise ValueError("valid_agent_mask must be bool [B,4]")
        if valid_agent_mask.dtype != torch.bool:
            raise TypeError("valid_agent_mask must have dtype bool")

        own_base = torch.cat(
            (
                self.own_state_projection(own_state).unsqueeze(-2),
                self.own_visual_projection(own_visual),
            ),
            dim=-2,
        )
        peer_base = torch.cat(
            (
                self.peer_state_projection(peer_state).unsqueeze(-2),
                self.peer_visual_projection(peer_visual),
            ),
            dim=-2,
        )
        shared_base = torch.cat(
            (
                shared_visual.new_zeros(
                    batch_size,
                    agents,
                    futures,
                    1,
                    config.d_model,
                ),
                self.shared_visual_projection(shared_visual),
            ),
            dim=-2,
        )

        identity = torch.eye(agents, dtype=own_base.dtype, device=own_base.device)
        own = own_base[:, :, None] * identity[None, :, :, None, None, None]
        shared_slot = own_base.new_zeros(agents)
        shared_slot[0] = 1
        shared = shared_base[:, :, None] * shared_slot[
            None, None, :, None, None, None
        ]
        tokens = torch.stack((own, peer_base, shared), dim=2)

        identity_mask = identity.to(dtype=torch.bool)
        focal_valid = valid_agent_mask[:, :, None]
        source_valid = valid_agent_mask[:, None, :]
        pair_valid = focal_valid & source_valid
        own_slot_mask = pair_valid & identity_mask[None]
        peer_slot_mask = pair_valid & ~identity_mask[None]
        shared_slot_mask = focal_valid & torch.nn.functional.one_hot(
            torch.tensor(0, device=valid_agent_mask.device),
            num_classes=agents,
        ).to(dtype=torch.bool)[None, None]
        all_token_mask = torch.ones(
            futures,
            config.token_count,
            dtype=torch.bool,
            device=valid_agent_mask.device,
        )
        shared_token_mask = all_token_mask.clone()
        shared_token_mask[:, 0] = False
        own_mask = own_slot_mask[:, :, :, None, None] & all_token_mask[
            None, None, None
        ]
        peer_mask = peer_slot_mask[:, :, :, None, None] & all_token_mask[
            None, None, None
        ]
        shared_mask = shared_slot_mask[:, :, :, None, None] & shared_token_mask[
            None, None, None
        ]
        mask = torch.stack((own_mask, peer_mask, shared_mask), dim=2)

        token_types = torch.tensor(
            [0, 1, 1, 1, 1], dtype=torch.long, device=tokens.device
        )
        token_position = torch.cat(
            (
                tokens.new_zeros(1, config.d_model),
                self.grid_position_embedding.weight,
            ),
            dim=0,
        )
        additive = (
            self.source_embedding.weight[None, None, :, None, None, None]
            + self.source_agent_embedding.weight[None, None, None, :, None, None]
            + self.horizon_embedding.weight[None, None, None, None, :, None]
            + self.token_type_embedding(token_types)[None, None, None, None, None]
            + token_position[None, None, None, None, None]
        )
        tokens = torch.where(mask[..., None], tokens + additive, 0)
        evidence = EvidenceTokens(tokens=tokens, mask=mask)
        _validate_evidence(evidence, config)
        return evidence


class LowRankEvidenceAdapterBank(nn.Module):
    """Twelve independent rank-32 token-preserving cross-attention reads."""

    def __init__(self, config: WorldEvidenceRouterConfig) -> None:
        super().__init__()
        self.config = config
        count = config.group_count
        self.query_projections = nn.ModuleList(
            nn.Linear(config.d_model, config.evidence_rank, bias=False)
            for _ in range(count)
        )
        self.key_projections = nn.ModuleList(
            nn.Linear(config.d_model, config.evidence_rank, bias=False)
            for _ in range(count)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(config.d_model, config.evidence_rank, bias=False)
            for _ in range(count)
        )
        self.output_projections = nn.ModuleList(
            nn.Linear(config.evidence_rank, config.d_model, bias=False)
            for _ in range(count)
        )

    def forward(self, q: Tensor, evidence: EvidenceTokens) -> EvidenceAdapterOutput:
        _validate_query(q, self.config)
        _validate_evidence(evidence, self.config, batch_agents=q.shape[:2])
        grouped_tokens, grouped_mask = _flatten_evidence_groups(
            evidence, self.config
        )
        outputs: list[Tensor] = []
        scale = self.config.evidence_rank**-0.5
        for group in range(self.config.group_count):
            token_mask = grouped_mask[:, :, group]
            # Invalid/padded values are selected away before any projection, so
            # even adversarial NaNs in padding cannot poison a valid attention.
            token_values = torch.where(
                token_mask[..., None], grouped_tokens[:, :, group], 0
            )
            query = self.query_projections[group](q)
            key = self.key_projections[group](token_values)
            value = self.value_projections[group](token_values)
            scores = torch.einsum("baqr,banr->baqn", query, key) * scale
            attention = _masked_softmax(
                scores, token_mask[:, :, None, :], dim=-1
            )
            context = torch.einsum("baqn,banr->baqr", attention, value)
            output = self.output_projections[group](context)
            available = token_mask.any(dim=-1)
            outputs.append(output * available[:, :, None, None].to(output))
        z = torch.stack(outputs, dim=3)
        return EvidenceAdapterOutput(z=z, group_mask=grouped_mask.any(dim=-1))


class FutureEvidenceRouter(nn.Module):
    """Dense masked router with detached query/evidence inputs."""

    def __init__(self, config: WorldEvidenceRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.query_norm = nn.LayerNorm(config.d_model)
        self.summary_norm = nn.LayerNorm(config.d_model)
        self.query_projection = nn.Linear(config.d_model, config.d_model)
        self.summary_projection = nn.Linear(config.d_model, config.d_model)
        self.group_prototypes = nn.Parameter(
            torch.randn(config.group_count, config.d_model) * 0.02
        )
        self.group_bias = nn.Parameter(torch.zeros(config.group_count))

    def forward(
        self,
        q: Tensor,
        evidence: EvidenceTokens,
        *,
        group_mask: Tensor | None = None,
    ) -> EvidenceRouterOutput:
        _validate_query(q, self.config)
        _validate_evidence(evidence, self.config, batch_agents=q.shape[:2])
        grouped_tokens, grouped_token_mask = _flatten_evidence_groups(
            evidence, self.config
        )
        calculated_group_mask = grouped_token_mask.any(dim=-1)
        if group_mask is not None:
            if group_mask.shape != calculated_group_mask.shape:
                raise ValueError("group_mask must be [B,A,12]")
            if group_mask.dtype != torch.bool:
                raise TypeError("group_mask must have dtype bool")
            if bool((group_mask & ~calculated_group_mask).any()):
                raise ValueError("group_mask cannot enable a group without valid tokens")
            calculated_group_mask = group_mask
        safe_tokens = torch.where(
            grouped_token_mask[..., None], grouped_tokens, 0
        )
        count = grouped_token_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        summary = safe_tokens.sum(dim=-2) / count.to(safe_tokens)
        summary = summary * calculated_group_mask[..., None].to(summary)

        # These detach calls are the gradient-scope boundary used by both the
        # normal Flow loss and the WUC-only backward audit.
        query_hidden = self.query_projection(self.query_norm(q.detach()))
        summary_hidden = self.summary_projection(
            self.summary_norm(summary.detach())
        )
        group_hidden = summary_hidden + self.group_prototypes[None, None]
        logits = (
            torch.einsum("baqd,bamd->baqm", query_hidden, group_hidden)
            / math.sqrt(self.config.d_model)
            + self.group_bias
        )
        query_group_mask = calculated_group_mask[:, :, None].expand(
            -1, -1, q.shape[2], -1
        )
        masked_logits = logits.masked_fill(~query_group_mask, float("-inf"))
        pi = _masked_softmax(masked_logits, query_group_mask, dim=-1)
        return EvidenceRouterOutput(
            logits=masked_logits,
            pi=pi,
            group_mask=calculated_group_mask,
            group_summary=summary,
        )


class UtilityCalibratedResidual(nn.Module):
    """Dense evidence mixture followed by an exact-zero query-wise gate."""

    def __init__(self, config: WorldEvidenceRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.output = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.action_dim),
        )
        self.query_gate = nn.Linear(2 * config.d_model, 1)
        nn.init.zeros_(self.query_gate.weight)
        nn.init.zeros_(self.query_gate.bias)

    def bounded_gate(self, q: Tensor, mixture: Tensor) -> Tensor:
        return self.config.new_gate_max * torch.tanh(
            self.query_gate(torch.cat((q, mixture), dim=-1))
        )

    def forward(
        self,
        q: Tensor,
        z: Tensor,
        pi: Tensor,
        valid_agent_mask: Tensor,
        *,
        force_gate_zero: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _validate_query(q, self.config)
        expected = (*q.shape[:3], self.config.group_count)
        if pi.shape != expected:
            raise ValueError("pi must be [B,A,Q,12]")
        if z.shape != (*expected, self.config.d_model):
            raise ValueError("z must be [B,A,Q,12,384]")
        if valid_agent_mask.shape != q.shape[:2] or valid_agent_mask.dtype != torch.bool:
            raise ValueError("valid_agent_mask must be bool [B,A]")
        mixture = torch.sum(pi[..., None] * z, dim=3)
        joined = torch.cat((q, mixture), dim=-1)
        raw = self.output(joined)
        gate = (
            raw.new_zeros(*raw.shape[:-1], 1)
            if force_gate_zero
            else self.config.new_gate_max * torch.tanh(self.query_gate(joined))
        )
        enabled = valid_agent_mask[:, :, None, None].to(raw)
        return gate * raw * enabled, gate * enabled, mixture


def group_index(source: str, horizon: int) -> int:
    """Return the stable source-major checkpoint index for an evidence group."""

    if source not in EVIDENCE_SOURCES:
        raise ValueError(f"unknown evidence source: {source}")
    if horizon not in EVIDENCE_HORIZONS:
        raise ValueError(f"unknown evidence horizon: {horizon}")
    return EVIDENCE_SOURCES.index(source) * len(EVIDENCE_HORIZONS) + EVIDENCE_HORIZONS.index(horizon)


def _future_value(value: object, name: str) -> Tensor:
    item = value[name] if isinstance(value, Mapping) else getattr(value, name, None)
    if not isinstance(item, Tensor):
        raise TypeError(f"predicted_futures.{name} must be a Tensor")
    return item


def _validate_query(q: Tensor, config: WorldEvidenceRouterConfig) -> None:
    if q.ndim != 4 or q.shape[-1] != config.d_model:
        raise ValueError("Flow query must be [B,A,Q,384]")


def _validate_evidence(
    evidence: EvidenceTokens,
    config: WorldEvidenceRouterConfig,
    *,
    batch_agents: tuple[int, int] | None = None,
) -> None:
    if evidence.tokens.ndim != 7:
        raise ValueError("evidence tokens must have seven non-feature axes")
    expected_tail = (
        len(EVIDENCE_SOURCES),
        config.max_agents,
        len(config.future_horizons),
        config.token_count,
        config.d_model,
    )
    if evidence.tokens.shape[2:] != expected_tail:
        raise ValueError("evidence tokens violate [B,A,3,4,4,5,384]")
    if evidence.mask.shape != evidence.tokens.shape[:-1]:
        raise ValueError("evidence mask must match every non-feature token axis")
    if evidence.mask.dtype != torch.bool:
        raise TypeError("evidence mask must have dtype bool")
    if batch_agents is not None and evidence.tokens.shape[:2] != batch_agents:
        raise ValueError("Flow queries and evidence have different B/A axes")


def _flatten_evidence_groups(
    evidence: EvidenceTokens,
    config: WorldEvidenceRouterConfig,
) -> tuple[Tensor, Tensor]:
    batch_size, agents = evidence.tokens.shape[:2]
    tokens = evidence.tokens.permute(0, 1, 2, 4, 3, 5, 6).reshape(
        batch_size,
        agents,
        config.group_count,
        config.max_agents * config.token_count,
        config.d_model,
    )
    mask = evidence.mask.permute(0, 1, 2, 4, 3, 5).reshape(
        batch_size,
        agents,
        config.group_count,
        config.max_agents * config.token_count,
    )
    return tokens, mask


def _masked_softmax(logits: Tensor, mask: Tensor, *, dim: int) -> Tensor:
    mask = mask.expand_as(logits)
    masked = logits.masked_fill(~mask, float("-inf"))
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    probabilities = torch.softmax(safe, dim=dim)
    return torch.where(mask, probabilities, 0)


__all__ = [
    "EVIDENCE_HORIZONS",
    "EVIDENCE_SOURCES",
    "EvidenceAdapterOutput",
    "EvidenceRouterOutput",
    "EvidenceTokens",
    "FutureEvidenceRouter",
    "LowRankEvidenceAdapterBank",
    "S4WorldEvidenceProvider",
    "UtilityCalibratedResidual",
    "WorldEvidenceRouterConfig",
    "group_index",
]
