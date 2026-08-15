"""Step 3-N2 symmetric predictive team-belief core.

The runtime path consumes only current/past frozen-DINO evidence, own qpos,
executed own actions and legal task text.  A structurally separate training
teacher may additionally consume synchronized current team state and the four
future visual anchors.  The teacher is never called by runtime ``forward`` and
can be physically removed before a deployment checkpoint is written.

The four capacity choices intentionally have no defaults.  They must come from
the preceding 3-N1 receipt rather than being tuned on 3-N2 Validation5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn
import torch.nn.functional as F


FUTURE_OFFSETS_STEPS: Final[tuple[int, ...]] = (4, 8, 16, 32)
FUTURE_OFFSETS_SECONDS: Final[tuple[float, ...]] = (0.2, 0.4, 0.8, 1.6)


@dataclass(frozen=True)
class B3N2Config:
    """Frozen architecture receipt values for one 3-N2 candidate."""

    n_belief_tokens: int
    n_evidence_queries: int
    event_capacity: int
    temporal_layers: int
    d_model: int = 384
    vision_dim: int = 768
    state_dim: int = 9
    action_dim: int = 8
    n_agent_anchors: int = 2
    max_views: int = 3
    heads: int = 8
    dropout: float = 0.1
    belief_factors: int = 12
    belief_classes: int = 32
    belief_unimix: float = 0.01
    belief_free_nats: float = 1.0
    belief_representation_scale: float = 0.1
    source_frequency_hz: int = 20
    future_offsets_steps: tuple[int, ...] = FUTURE_OFFSETS_STEPS
    future_offsets_seconds: tuple[float, ...] = FUTURE_OFFSETS_SECONDS

    def __post_init__(self) -> None:
        positive = {
            "n_belief_tokens": self.n_belief_tokens,
            "n_evidence_queries": self.n_evidence_queries,
            "event_capacity": self.event_capacity,
            "temporal_layers": self.temporal_layers,
            "d_model": self.d_model,
            "vision_dim": self.vision_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_views": self.max_views,
            "heads": self.heads,
            "belief_factors": self.belief_factors,
            "belief_classes": self.belief_classes,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_agent_anchors != 2:
            raise ValueError("the frozen V7.3 first implementation has two agents")
        if self.n_belief_tokens < self.n_agent_anchors:
            raise ValueError("belief tokens must retain both relative agent anchors")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by attention heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.belief_unimix < 1.0:
            raise ValueError("belief_unimix must be in (0, 1)")
        if self.belief_free_nats < 0.0:
            raise ValueError("belief_free_nats must be non-negative")
        if not 0.0 <= self.belief_representation_scale <= 1.0:
            raise ValueError("belief_representation_scale must be in [0, 1]")
        if self.source_frequency_hz != 20:
            raise ValueError("3-N2 is frozen to the audited 20 Hz source corpus")
        if tuple(self.future_offsets_steps) != FUTURE_OFFSETS_STEPS:
            raise ValueError("future step anchors must remain [4, 8, 16, 32]")
        if tuple(self.future_offsets_seconds) != FUTURE_OFFSETS_SECONDS:
            raise ValueError("future second anchors must remain [0.2, 0.4, 0.8, 1.6]")


@dataclass
class CompressedEvidence:
    tokens: torch.Tensor
    step_valid: torch.Tensor
    conflict: torch.Tensor
    missing_fraction: torch.Tensor
    raw_dino_pool: torch.Tensor
    raw_dino_view_pool: torch.Tensor
    view_valid: torch.Tensor


@dataclass
class EventMemoryState:
    tokens: torch.Tensor
    scores: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class BeliefRuntimeState:
    """Carry state for sequential deployment; reset masks remain mandatory."""

    hidden: torch.Tensor
    events: EventMemoryState
    previous_valid: torch.Tensor


@dataclass
class BeliefCoreOutput:
    mu: torch.Tensor
    sigma: torch.Tensor
    categorical_log_probs: torch.Tensor
    categorical_probs: torch.Tensor
    categorical_entropy: torch.Tensor
    free_nats: float
    representation_scale: float
    mu_sequence: torch.Tensor
    sigma_sequence: torch.Tensor
    reliability: torch.Tensor
    surprise: torch.Tensor
    evidence_conflict: torch.Tensor
    event_memory: torch.Tensor
    event_scores: torch.Tensor
    event_mask: torch.Tensor
    current_visual_reference: torch.Tensor
    current_visual_view_mask: torch.Tensor
    future_latent_prediction: torch.Tensor
    teammate_state_delta_prediction: torch.Tensor
    teammate_action_mean: torch.Tensor
    teammate_action_logvar: torch.Tensor
    runtime_state: BeliefRuntimeState


@dataclass
class TeacherBeliefInputs:
    """Privileged tensors which are legal only in the training teacher."""

    current_visual_tokens: torch.Tensor
    current_visual_mask: torch.Tensor
    future_visual_tokens: torch.Tensor
    future_visual_mask: torch.Tensor
    future_anchor_mask: torch.Tensor
    agent_state: torch.Tensor
    agent_mask: torch.Tensor
    relative_agent_role: torch.Tensor


@dataclass
class TeacherBeliefOutput:
    mu: torch.Tensor
    sigma: torch.Tensor
    categorical_log_probs: torch.Tensor
    categorical_probs: torch.Tensor
    categorical_entropy: torch.Tensor
    future_latent_reconstruction: torch.Tensor
    future_latent_target: torch.Tensor
    future_anchor_mask: torch.Tensor
    future_view_mask: torch.Tensor


class ActionConditionedFuturePredictor(nn.Module):
    """Predict action-conditioned DINO deltas above a persistence reference.

    This is the small recurrent form of the action-conditioned latent predictor
    used by V-JEPA 2-AC and DINO-WM. Predicting a delta, with a zero-initialized
    output layer, makes the untrained model exactly equal to legal-view
    persistence instead of asking an unconditional MLP to relearn persistence.
    """

    def __init__(self, config: B3N2Config) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.belief_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.visual_projection = nn.Linear(config.vision_dim, d)
        self.view_role = nn.Embedding(config.max_views, d)
        self.action_projection = nn.Linear(config.action_dim, d)
        self.recurrent = nn.GRU(
            d,
            d,
            num_layers=config.temporal_layers,
            dropout=config.dropout if config.temporal_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d)
        self.delta_head = nn.Linear(d, config.vision_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        belief: torch.Tensor,
        current_visual: torch.Tensor,
        current_view_mask: torch.Tensor,
        future_action: torch.Tensor | None,
        future_action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if belief.ndim != 3:
            raise ValueError("future predictor belief must be [batch,token,width]")
        batch = belief.shape[0]
        expected_visual = (
            batch,
            self.config.max_views,
            self.config.vision_dim,
        )
        if tuple(current_visual.shape) != expected_visual:
            raise ValueError("future predictor current-visual shape differs")
        if (
            current_view_mask.shape != expected_visual[:2]
            or current_view_mask.dtype != torch.bool
        ):
            raise ValueError("future predictor view mask must be boolean [batch,view]")

        horizon = max(FUTURE_OFFSETS_STEPS)
        if future_action is None:
            future_action = current_visual.new_zeros(
                batch, horizon, self.config.action_dim
            )
            future_action_mask = torch.zeros(
                batch, horizon, dtype=torch.bool, device=current_visual.device
            )
        elif (
            future_action.ndim != 3
            or future_action.shape[0] != batch
            or future_action.shape[2] != self.config.action_dim
        ):
            raise ValueError("future action must be [batch,horizon,action_dim]")
        if future_action_mask is None:
            future_action_mask = torch.ones(
                future_action.shape[:2], dtype=torch.bool, device=future_action.device
            )
        if (
            future_action_mask.shape != future_action.shape[:2]
            or future_action_mask.dtype != torch.bool
        ):
            raise ValueError("future action mask must be boolean [batch,horizon]")

        # Short synthetic/test horizons are padded rather than allowed to alter
        # the frozen 0.2/0.4/0.8/1.6-second anchor contract.
        if future_action.shape[1] < horizon:
            missing = horizon - future_action.shape[1]
            future_action = F.pad(future_action, (0, 0, 0, missing))
            future_action_mask = F.pad(future_action_mask, (0, missing))
        else:
            future_action = future_action[:, :horizon]
            future_action_mask = future_action_mask[:, :horizon]

        view_ids = torch.arange(self.config.max_views, device=belief.device)
        initial = (
            self.belief_projection(belief.mean(1)).unsqueeze(1)
            + self.visual_projection(current_visual)
            + self.view_role(view_ids).unsqueeze(0).to(current_visual.dtype)
        )
        initial = initial * current_view_mask.unsqueeze(-1).to(initial.dtype)
        initial = initial.reshape(batch * self.config.max_views, self.config.d_model)
        hidden = initial.unsqueeze(0).expand(
            self.config.temporal_layers, -1, -1
        ).contiguous()

        projected_action = self.action_projection(future_action)
        projected_action = projected_action * future_action_mask.unsqueeze(-1).to(
            projected_action.dtype
        )
        boundaries = (0, *FUTURE_OFFSETS_STEPS)
        action_segments = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            segment_mask = future_action_mask[:, start:end]
            count = segment_mask.sum(1, keepdim=True).clamp_min(1)
            summary = projected_action[:, start:end].sum(1) / count
            summary = summary * segment_mask.any(1, keepdim=True).to(summary.dtype)
            action_segments.append(summary)
        # Four fixed 20-Hz intervals preserve the registered prediction times
        # while avoiding a slow 32-step recurrent rollout in every train batch.
        action = torch.stack(action_segments, dim=1)
        action = action[:, None].expand(
            -1, self.config.max_views, -1, -1
        ).reshape(
            batch * self.config.max_views,
            len(FUTURE_OFFSETS_STEPS),
            self.config.d_model,
        )
        rollout, _ = self.recurrent(action, hidden)
        anchored = rollout.view(
            batch,
            self.config.max_views,
            len(FUTURE_OFFSETS_STEPS),
            self.config.d_model,
        ).transpose(1, 2)
        delta = self.delta_head(self.output_norm(anchored))
        delta = delta * current_view_mask[:, None, :, None].to(delta.dtype)
        return current_visual[:, None] + delta


class MultiViewEvidenceCompressor(nn.Module):
    """Compress frozen DINO patch tokens with shared learnable queries."""

    def __init__(self, config: B3N2Config) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.input_projection = nn.Linear(config.vision_dim, d)
        self.view_role = nn.Embedding(config.max_views, d)
        self.queries = nn.Parameter(torch.randn(config.n_evidence_queries, d) * 0.02)
        self.attention = nn.MultiheadAttention(
            d, config.heads, dropout=config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self, visual_tokens: torch.Tensor, visual_mask: torch.Tensor
    ) -> CompressedEvidence:
        if visual_tokens.ndim != 5:
            raise ValueError("visual tokens must be [batch,time,view,patch,vision_dim]")
        batch, steps, views, patches, width = visual_tokens.shape
        if width != self.config.vision_dim:
            raise ValueError(f"expected DINO width {self.config.vision_dim}, got {width}")
        if not 1 <= views <= self.config.max_views or patches < 1:
            raise ValueError("visual view/patch dimensions are outside the frozen config")
        if visual_mask.shape != visual_tokens.shape[:-1]:
            raise ValueError("visual token mask shape differs")
        if visual_mask.dtype != torch.bool:
            raise TypeError("visual token mask must be boolean")

        content = self.input_projection(visual_tokens)
        role_ids = torch.arange(views, device=visual_tokens.device)
        role = self.view_role(role_ids).view(1, 1, views, 1, -1)
        projected = content + role.to(dtype=content.dtype)
        flat = projected.reshape(batch * steps, views * patches, -1)
        flat_mask = visual_mask.reshape(batch * steps, views * patches)
        flat_step_valid = flat_mask.any(-1)
        safe_mask = flat_mask.clone()
        # MultiheadAttention returns NaN for an all-masked row.  Invalid padded
        # timesteps get one zero-valued dummy key and are zeroed again below.
        invalid = ~flat_step_valid
        if invalid.any():
            safe_mask[invalid, 0] = True
            flat = flat.clone()
            flat[invalid, 0] = 0
        query = self.queries.unsqueeze(0).expand(batch * steps, -1, -1)
        attended = self.attention(
            query,
            flat,
            flat,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )[0]
        compressed = self.norm(query + attended)
        compressed = compressed * flat_step_valid[:, None, None].to(compressed.dtype)
        compressed = compressed.reshape(
            batch, steps, self.config.n_evidence_queries, self.config.d_model
        )
        step_valid = flat_step_valid.reshape(batch, steps)

        mask_float = visual_mask.unsqueeze(-1).to(visual_tokens.dtype)
        raw_count = mask_float.sum((2, 3)).clamp_min(1)
        raw_pool = (visual_tokens * mask_float).sum((2, 3)) / raw_count
        raw_pool = raw_pool * step_valid.unsqueeze(-1).to(raw_pool.dtype)

        view_mask = visual_mask.any(-1)
        raw_view_count = mask_float.sum(3).clamp_min(1)
        raw_view_pool = (visual_tokens * mask_float).sum(3) / raw_view_count
        raw_view_pool = raw_view_pool * view_mask.unsqueeze(-1).to(
            raw_view_pool.dtype
        )
        view_count = visual_mask.sum(-1, keepdim=True).clamp_min(1)
        view_pool = (content * visual_mask.unsqueeze(-1)).sum(-2) / view_count
        view_weights = view_mask.unsqueeze(-1).to(view_pool.dtype)
        mean_view = (view_pool * view_weights).sum(2) / view_weights.sum(2).clamp_min(1)
        conflict = (
            (view_pool - mean_view.unsqueeze(2)).square().mean(-1)
            * view_mask.to(view_pool.dtype)
        ).sum(2) / view_mask.sum(2).clamp_min(1)
        conflict = conflict * step_valid.to(conflict.dtype)
        missing_fraction = 1.0 - view_mask.float().mean(2)
        missing_fraction = missing_fraction * step_valid.to(missing_fraction.dtype)
        return CompressedEvidence(
            compressed,
            step_valid,
            conflict,
            missing_fraction,
            raw_pool,
            raw_view_pool,
            view_mask,
        )


class BeliefUpdateLayer(nn.Module):
    """One recurrent belief update; the Python time loop enforces causality."""

    def __init__(self, config: B3N2Config) -> None:
        super().__init__()
        d = config.d_model
        self.query_norm = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(
            d, config.heads, dropout=config.dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(
            d, config.heads, dropout=config.dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * d, d),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, belief: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor
    ) -> torch.Tensor:
        query = self.query_norm(belief)
        normalized_memory = self.memory_norm(memory)
        belief = belief + self.dropout(
            self.cross_attention(
                query,
                normalized_memory,
                normalized_memory,
                key_padding_mask=~memory_mask,
                need_weights=False,
            )[0]
        )
        normalized = self.self_norm(belief)
        belief = belief + self.dropout(
            self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        )
        return belief + self.ff(self.ff_norm(belief))


class TopKEventMemory(nn.Module):
    """Episode-local, hard top-k storage selected by detached surprise scores."""

    def __init__(self, config: B3N2Config) -> None:
        super().__init__()
        self.capacity = config.event_capacity
        d = config.d_model
        self.event_projection = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d)
        )

    def empty(self, reference: torch.Tensor) -> EventMemoryState:
        batch, width = reference.shape[0], reference.shape[-1]
        return EventMemoryState(
            tokens=reference.new_zeros(batch, self.capacity, width),
            scores=reference.new_zeros(batch, self.capacity),
            valid_mask=torch.zeros(
                batch, self.capacity, dtype=torch.bool, device=reference.device
            ),
        )

    def reset_where(
        self, state: EventMemoryState, reset: torch.Tensor
    ) -> EventMemoryState:
        keep = (~reset).unsqueeze(-1)
        return EventMemoryState(
            tokens=state.tokens * keep.unsqueeze(-1).to(state.tokens.dtype),
            scores=state.scores * keep.to(state.scores.dtype),
            valid_mask=state.valid_mask & keep,
        )

    def update(
        self,
        state: EventMemoryState,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        surprise: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> EventMemoryState:
        candidate = self.event_projection(
            torch.cat((predicted, actual, actual - predicted), dim=-1)
        )
        candidate_score = surprise.detach().unsqueeze(-1)
        all_tokens = torch.cat((state.tokens, candidate.unsqueeze(1)), dim=1)
        all_scores = torch.cat((state.scores, candidate_score), dim=1)
        all_valid = torch.cat((state.valid_mask, candidate_valid.unsqueeze(-1)), dim=1)
        ranked = all_scores.masked_fill(~all_valid, -torch.inf)
        selected_scores, selected = ranked.topk(self.capacity, dim=1)
        gather_index = selected.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1])
        selected_tokens = all_tokens.gather(1, gather_index)
        selected_valid = torch.isfinite(selected_scores)
        return EventMemoryState(
            tokens=selected_tokens
            * selected_valid.unsqueeze(-1).to(selected_tokens.dtype),
            scores=torch.where(
                selected_valid, selected_scores, torch.zeros_like(selected_scores)
            ),
            valid_mask=selected_valid,
        )


class TrainingTeacherBranch(nn.Module):
    """Future-informed posterior that is removable after training."""

    def __init__(self, config: B3N2Config) -> None:
        super().__init__()
        self.config = config
        self.compressor = MultiViewEvidenceCompressor(config)
        self.agent_state = nn.Linear(config.state_dim, config.d_model, bias=False)
        self.relative_role = nn.Embedding(2, config.d_model)
        self.layers = nn.ModuleList(
            BeliefUpdateLayer(config) for _ in range(config.temporal_layers)
        )

    def forward(
        self,
        inputs: TeacherBeliefInputs,
        slot_queries: torch.Tensor,
        task_token: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        current = inputs.current_visual_tokens
        future = inputs.future_visual_tokens
        if current.ndim != 4 or future.ndim != 5:
            raise ValueError("teacher visual tensors have the wrong rank")
        if current.shape[0] != future.shape[0]:
            raise ValueError("teacher current/future batch differs")
        if current.shape[1:] != future.shape[2:]:
            raise ValueError("teacher current/future view and patch layouts differ")
        if future.shape[1] != len(FUTURE_OFFSETS_STEPS):
            raise ValueError("teacher must preserve all four future anchor slots")
        if inputs.current_visual_mask.shape != current.shape[:-1]:
            raise ValueError("teacher current visual mask differs")
        if inputs.future_visual_mask.shape != future.shape[:-1]:
            raise ValueError("teacher future visual mask differs")
        if inputs.future_anchor_mask.shape != future.shape[:2]:
            raise ValueError("teacher future anchor mask differs")
        if inputs.future_anchor_mask.dtype != torch.bool:
            raise TypeError("teacher future anchor mask must be boolean")
        if (
            inputs.current_visual_mask.dtype != torch.bool
            or inputs.future_visual_mask.dtype != torch.bool
        ):
            raise TypeError("teacher visual masks must be boolean")
        masked_future = inputs.future_visual_mask & inputs.future_anchor_mask[
            :, :, None, None
        ]
        combined_tokens = torch.cat((current.unsqueeze(1), future), dim=1)
        combined_mask = torch.cat(
            (inputs.current_visual_mask.unsqueeze(1), masked_future), dim=1
        )
        compressed = self.compressor(combined_tokens, combined_mask)

        if inputs.agent_state.ndim != 3:
            raise ValueError("teacher agent state must be [batch,agent,state_dim]")
        if inputs.agent_state.shape[-1] != self.config.state_dim:
            raise ValueError("teacher agent-state width differs")
        if inputs.agent_mask.shape != inputs.agent_state.shape[:2]:
            raise ValueError("teacher agent mask differs")
        if inputs.relative_agent_role.shape != inputs.agent_mask.shape:
            raise ValueError("teacher relative agent role shape differs")
        if inputs.agent_mask.dtype != torch.bool:
            raise TypeError("teacher agent mask must be boolean")
        if inputs.relative_agent_role.dtype not in (torch.int32, torch.int64):
            raise TypeError("teacher relative agent roles must be integer tensors")
        active_roles = inputs.relative_agent_role[inputs.agent_mask]
        if active_roles.numel() and not torch.all((active_roles == 0) | (active_roles == 1)):
            raise ValueError("teacher roles must be relative ego/teammate values")
        agent = self.agent_state(inputs.agent_state) + self.relative_role(
            inputs.relative_agent_role.clamp(0, 1)
        )
        batch = current.shape[0]
        visual_memory = compressed.tokens.flatten(1, 2)
        visual_mask = compressed.step_valid.unsqueeze(-1).expand(
            -1, -1, self.config.n_evidence_queries
        ).flatten(1)
        memory = torch.cat((visual_memory, agent, task_token.unsqueeze(1)), dim=1)
        memory_mask = torch.cat(
            (
                visual_mask,
                inputs.agent_mask,
                torch.ones(batch, 1, dtype=torch.bool, device=current.device),
            ),
            dim=1,
        )
        belief = slot_queries.expand(batch, -1, -1)
        for layer in self.layers:
            belief = layer(belief, memory, memory_mask)
        future_target = compressed.raw_dino_view_pool[:, 1:].detach()
        future_view_mask = compressed.view_valid[:, 1:]
        if future_target.shape[2] < self.config.max_views:
            missing = self.config.max_views - future_target.shape[2]
            future_target = F.pad(future_target, (0, 0, 0, missing))
            future_view_mask = F.pad(future_view_mask, (0, missing))
        return (
            belief,
            compressed.conflict,
            future_target,
            inputs.future_anchor_mask,
            future_view_mask,
        )


class PredictiveTeamBeliefCore(nn.Module):
    """Causal discrete team belief, event memory and training-only teacher."""

    def __init__(self, config: B3N2Config, *, include_teacher: bool = True) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.evidence_compressor = MultiViewEvidenceCompressor(config)
        self.shared_agent_anchor = nn.Parameter(torch.randn(1, d) * 0.02)
        self.relative_agent_role = nn.Embedding(2, d)
        free = config.n_belief_tokens - config.n_agent_anchors
        self.free_interaction_slots = nn.Parameter(torch.randn(free, d) * 0.02)
        self.qpos_projection = nn.Linear(config.state_dim, d, bias=False)
        self.action_projection = nn.Linear(config.action_dim, d, bias=False)
        self.update_layers = nn.ModuleList(
            BeliefUpdateLayer(config) for _ in range(config.temporal_layers)
        )
        self.next_evidence = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.event_memory = TopKEventMemory(config)
        self.belief_norm = nn.LayerNorm(d)
        self.categorical_head = nn.Linear(
            d, config.belief_factors * config.belief_classes
        )
        self.categorical_projection = nn.Linear(
            config.belief_factors * config.belief_classes, d, bias=False
        )
        self.belief_feature_norm = nn.LayerNorm(d)
        self.uncertainty_gain_raw = nn.Parameter(torch.tensor(-2.25))
        self.future_predictor = ActionConditionedFuturePredictor(config)
        self.teammate_delta_head = nn.Linear(
            d, len(FUTURE_OFFSETS_STEPS) * config.state_dim
        )
        self.teammate_action_head = nn.Linear(d, 16 * config.action_dim * 2)
        self.teacher_branch: TrainingTeacherBranch | None = (
            TrainingTeacherBranch(config) if include_teacher else None
        )

    def slot_queries(self) -> torch.Tensor:
        roles = torch.tensor((0, 1), device=self.shared_agent_anchor.device)
        agents = self.shared_agent_anchor + self.relative_agent_role(roles)
        return torch.cat((agents, self.free_interaction_slots), dim=0).unsqueeze(0)

    def _distribution(
        self,
        hidden: torch.Tensor,
        uncertainty_signal: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return a bounded factorized categorical belief and its feature view.

        This is the DreamerV3 discrete-state idea in the local tensor contract:
        every factor mixes in at least ``belief_unimix`` uniform probability.
        Missing/conflicting evidence may add more uniform mass, but can never
        remove that floor.  ``sigma`` remains only a backwards-compatible
        entropy diagnostic; it is not used as a Gaussian scale.
        """

        normalized = self.belief_norm(hidden)
        batch, tokens = hidden.shape[:2]
        raw_logits = self.categorical_head(normalized).view(
            batch,
            tokens,
            self.config.belief_factors,
            self.config.belief_classes,
        )
        base_probs = raw_logits.float().softmax(-1)
        gain = F.softplus(self.uncertainty_gain_raw)
        adaptive_mix = 1.0 - torch.exp(
            -gain * torch.log1p(uncertainty_signal.float().clamp_min(0))
        )
        mix = self.config.belief_unimix + (
            1.0 - self.config.belief_unimix
        ) * adaptive_mix
        mix = mix[:, None, None, None]
        uniform = 1.0 / self.config.belief_classes
        probs = (1.0 - mix) * base_probs + mix * uniform
        log_probs = probs.log()
        entropy = -(probs * log_probs).sum(-1) / torch.log(
            probs.new_tensor(float(self.config.belief_classes))
        )
        centered = (probs - uniform) * self.config.belief_classes**0.5
        mu = self.belief_feature_norm(
            self.categorical_projection(centered.flatten(-2)).to(hidden.dtype)
        )
        # Keep the public B=(mu,sigma) interface during the revision.  Each
        # feature receives the token's normalized categorical entropy, so no
        # reciprocal scale can explode.
        sigma = entropy.mean(-1, keepdim=True).expand(-1, -1, self.config.d_model)
        weight = valid[:, None, None].to(mu.dtype)
        categorical_weight = valid[:, None, None, None].to(probs.dtype)
        return (
            mu * weight,
            sigma.to(mu.dtype) * weight,
            log_probs * categorical_weight,
            probs * categorical_weight,
            entropy * valid[:, None, None].to(entropy.dtype),
        )

    @staticmethod
    def reliability_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim != 3:
            raise ValueError("belief sigma must be [batch,token,width]")
        return (1.0 / (1.0 + sigma.mean((1, 2)))).view(-1, 1, 1)

    def forward(
        self,
        runtime_visual_tokens: torch.Tensor,
        runtime_visual_mask: torch.Tensor,
        history_qpos: torch.Tensor,
        history_action: torch.Tensor,
        history_mask: torch.Tensor,
        action_history_mask: torch.Tensor,
        task_token: torch.Tensor,
        episode_reset_mask: torch.Tensor,
        initial_state: BeliefRuntimeState | None = None,
        *,
        future_action: torch.Tensor | None = None,
        future_action_mask: torch.Tensor | None = None,
    ) -> BeliefCoreOutput:
        if history_qpos.ndim != 3 or history_qpos.shape[-1] != self.config.state_dim:
            raise ValueError("runtime qpos contract differs")
        batch, steps = history_qpos.shape[:2]
        expected_action = (batch, steps, self.config.action_dim)
        if tuple(history_action.shape) != expected_action:
            raise ValueError("runtime executed-action contract differs")
        for name, value in {
            "history_mask": history_mask,
            "action_history_mask": action_history_mask,
            "episode_reset_mask": episode_reset_mask,
        }.items():
            if value.shape != (batch, steps) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean [batch,time] tensor")
        if task_token.shape != (batch, self.config.d_model):
            raise ValueError("runtime task token contract differs")
        if runtime_visual_tokens.shape[:2] != (batch, steps):
            raise ValueError("runtime visual/history time axes differ")
        if not torch.all(history_mask[:, -1]):
            raise ValueError("current runtime timestep must be valid")
        if torch.any(episode_reset_mask & ~history_mask):
            raise ValueError("episode reset cannot occur in padding")
        if torch.any(action_history_mask & ~history_mask):
            raise ValueError("executed actions cannot be exposed in history padding")

        evidence = self.evidence_compressor(runtime_visual_tokens, runtime_visual_mask)
        if torch.any(history_mask & ~evidence.step_valid):
            raise ValueError("every valid runtime timestep requires visual evidence")
        if torch.any(evidence.step_valid & ~history_mask):
            raise ValueError("padded runtime timesteps cannot expose visual evidence")

        initial = self.slot_queries().expand(batch, -1, -1)
        if initial_state is None:
            hidden = initial
            events = self.event_memory.empty(initial)
            previous_valid = torch.zeros(
                batch, dtype=torch.bool, device=history_qpos.device
            )
        else:
            if initial_state.hidden.shape != initial.shape:
                raise ValueError("carried belief hidden state shape differs")
            if initial_state.events.tokens.shape != (
                batch,
                self.config.event_capacity,
                self.config.d_model,
            ):
                raise ValueError("carried event-memory shape differs")
            if initial_state.events.scores.shape != (
                batch,
                self.config.event_capacity,
            ) or initial_state.events.valid_mask.shape != (
                batch,
                self.config.event_capacity,
            ):
                raise ValueError("carried event-memory metadata shape differs")
            if initial_state.previous_valid.shape != (batch,):
                raise ValueError("carried previous-valid shape differs")
            if initial_state.previous_valid.dtype != torch.bool:
                raise TypeError("carried previous-valid state must be boolean")
            if initial_state.events.valid_mask.dtype != torch.bool:
                raise TypeError("carried event validity must be boolean")
            hidden = initial_state.hidden
            events = initial_state.events
            previous_valid = initial_state.previous_valid
        mu_rows, sigma_rows, surprise_rows = [], [], []
        final_log_probs = final_probs = final_entropy = None
        for index in range(steps):
            step_valid = history_mask[:, index]
            reset = episode_reset_mask[:, index] & step_valid
            hidden = torch.where(reset[:, None, None], initial, hidden)
            events = self.event_memory.reset_where(events, reset)
            previous_valid = previous_valid & ~reset

            actual = evidence.tokens[:, index].mean(1)
            predicted = self.next_evidence(hidden.mean(1))
            surprise = (actual - predicted).float().square().mean(-1).to(actual.dtype)
            candidate_valid = step_valid & previous_valid
            events = self.event_memory.update(
                events, predicted, actual, surprise, candidate_valid
            )

            qpos = self.qpos_projection(history_qpos[:, index]).unsqueeze(1)
            action = self.action_projection(history_action[:, index]).unsqueeze(1)
            memory = torch.cat(
                (
                    evidence.tokens[:, index],
                    qpos,
                    action,
                    task_token.unsqueeze(1),
                    events.tokens,
                ),
                dim=1,
            )
            memory_mask = torch.cat(
                (
                    step_valid[:, None].expand(-1, self.config.n_evidence_queries),
                    step_valid[:, None],
                    action_history_mask[:, index : index + 1],
                    step_valid[:, None],
                    events.valid_mask,
                ),
                dim=1,
            )
            no_memory = ~memory_mask.any(1)
            if no_memory.any():
                memory = memory.clone()
                memory_mask = memory_mask.clone()
                memory[no_memory, 0] = 0
                memory_mask[no_memory, 0] = True
            candidate_hidden = hidden
            for layer in self.update_layers:
                candidate_hidden = layer(candidate_hidden, memory, memory_mask)
            hidden = torch.where(
                step_valid[:, None, None], candidate_hidden, hidden
            )
            uncertainty = surprise * candidate_valid.to(surprise.dtype)
            uncertainty = (
                uncertainty
                + evidence.conflict[:, index]
                + evidence.missing_fraction[:, index]
            )
            mu, sigma, log_probs, probs, entropy = self._distribution(
                hidden, uncertainty, step_valid
            )
            mu_rows.append(mu)
            sigma_rows.append(sigma)
            final_log_probs = log_probs
            final_probs = probs
            final_entropy = entropy
            surprise_rows.append(surprise * candidate_valid.to(surprise.dtype))
            previous_valid = step_valid

        mu_sequence = torch.stack(mu_rows, dim=1)
        sigma_sequence = torch.stack(sigma_rows, dim=1)
        surprise_sequence = torch.stack(surprise_rows, dim=1)
        final_mu = mu_sequence[:, -1]
        final_sigma = sigma_sequence[:, -1]
        assert final_log_probs is not None
        assert final_probs is not None
        assert final_entropy is not None
        reliability = self.reliability_from_sigma(final_sigma)
        current_visual = evidence.raw_dino_view_pool[:, -1]
        current_view_mask = evidence.view_valid[:, -1]
        if current_visual.shape[1] < self.config.max_views:
            missing = self.config.max_views - current_visual.shape[1]
            current_visual = F.pad(current_visual, (0, 0, 0, missing))
            current_view_mask = F.pad(current_view_mask, (0, missing))
        future = self.predict_future(
            final_mu,
            current_visual,
            current_view_mask,
            future_action,
            future_action_mask,
        )
        teammate_delta = self.teammate_delta_head(final_mu[:, 1]).view(
            batch, len(FUTURE_OFFSETS_STEPS), self.config.state_dim
        )
        teammate_action_parameters = self.teammate_action_head(final_mu[:, 1]).view(
            batch, 16, self.config.action_dim, 2
        )
        return BeliefCoreOutput(
            mu=final_mu,
            sigma=final_sigma,
            categorical_log_probs=final_log_probs,
            categorical_probs=final_probs,
            categorical_entropy=final_entropy,
            free_nats=self.config.belief_free_nats,
            representation_scale=self.config.belief_representation_scale,
            mu_sequence=mu_sequence,
            sigma_sequence=sigma_sequence,
            reliability=reliability,
            surprise=surprise_sequence,
            evidence_conflict=evidence.conflict,
            event_memory=events.tokens,
            event_scores=events.scores,
            event_mask=events.valid_mask,
            current_visual_reference=current_visual,
            current_visual_view_mask=current_view_mask,
            future_latent_prediction=future,
            teammate_state_delta_prediction=teammate_delta,
            teammate_action_mean=teammate_action_parameters[..., 0],
            teammate_action_logvar=teammate_action_parameters[..., 1].clamp(-8.0, 5.0),
            runtime_state=BeliefRuntimeState(hidden, events, previous_valid),
        )

    def predict_future(
        self,
        belief: torch.Tensor,
        current_visual: torch.Tensor,
        current_view_mask: torch.Tensor,
        future_action: torch.Tensor | None,
        future_action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.future_predictor(
            belief,
            current_visual,
            current_view_mask,
            future_action,
            future_action_mask,
        )

    def forward_teacher(
        self,
        inputs: TeacherBeliefInputs,
        task_token: torch.Tensor,
        *,
        future_action: torch.Tensor | None = None,
        future_action_mask: torch.Tensor | None = None,
    ) -> TeacherBeliefOutput:
        if self.teacher_branch is None:
            raise RuntimeError("the privileged teacher has been stripped")
        (
            hidden,
            conflict,
            future_target,
            anchor_mask,
            future_view_mask,
        ) = self.teacher_branch(
            inputs, self.slot_queries(), task_token
        )
        valid = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        teacher_step_mask = torch.cat(
            (
                torch.ones(
                    anchor_mask.shape[0],
                    1,
                    dtype=torch.bool,
                    device=anchor_mask.device,
                ),
                anchor_mask,
            ),
            dim=1,
        )
        aggregate_conflict = (
            conflict * teacher_step_mask.to(conflict.dtype)
        ).sum(1) / teacher_step_mask.sum(1).clamp_min(1)
        mu, sigma, log_probs, probs, entropy = self._distribution(
            hidden, aggregate_conflict, valid
        )
        # The privileged posterior must reconstruct through the same belief
        # bottleneck used at runtime; reading raw hidden bypassed estimability.
        current_mask = inputs.current_visual_mask.any(-1)
        current_count = inputs.current_visual_mask.unsqueeze(-1).sum(2).clamp_min(1)
        current_visual = (
            inputs.current_visual_tokens
            * inputs.current_visual_mask.unsqueeze(-1).to(
                inputs.current_visual_tokens.dtype
            )
        ).sum(2) / current_count
        current_visual = current_visual * current_mask.unsqueeze(-1).to(
            current_visual.dtype
        )
        future_reconstruction = self.predict_future(
            mu,
            current_visual,
            current_mask,
            future_action,
            future_action_mask,
        )
        return TeacherBeliefOutput(
            mu=mu,
            sigma=sigma,
            categorical_log_probs=log_probs,
            categorical_probs=probs,
            categorical_entropy=entropy,
            future_latent_reconstruction=future_reconstruction,
            future_latent_target=future_target,
            future_anchor_mask=anchor_mask,
            future_view_mask=future_view_mask,
        )

    def strip_teacher_(self) -> "PredictiveTeamBeliefCore":
        """Physically remove privileged parameters before deployment export."""

        self.teacher_branch = None
        return self


__all__ = [
    "ActionConditionedFuturePredictor",
    "B3N2Config",
    "BeliefCoreOutput",
    "BeliefRuntimeState",
    "EventMemoryState",
    "FUTURE_OFFSETS_SECONDS",
    "FUTURE_OFFSETS_STEPS",
    "MultiViewEvidenceCompressor",
    "PredictiveTeamBeliefCore",
    "TeacherBeliefInputs",
    "TeacherBeliefOutput",
    "TopKEventMemory",
]
