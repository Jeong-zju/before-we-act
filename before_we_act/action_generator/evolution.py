"""R12-E1 image-primary, task-conditioned action specialist.

Native 480x640 images remain the primary input and are encoded before spatial
compression.  Frozen W11 TeamBeliefState, an explicit task identifier and a
bounded agent-slot identity are supplemental signals.  Deployment uses the
specialist only on preregistered tasks and routes every protected task through
the exact W10 policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml

from before_we_act.contracts import ActionProposalBatch, TeamBeliefState
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.spatial_observation import locked_r12_full_episode_observation
from .r4_base import R4JointActionGenerator
from .registry import CANDIDATE_SPECS


EXPECTED_TOP_LEVEL = {
    "schema_version",
    "round",
    "candidate_id",
    "parent_commit",
    "belief_checkpoint_sha256",
    "component",
    "observation",
    "action",
    "training",
    "deployment",
    "selection_rule",
}


@dataclass(frozen=True)
class R12EvolutionConfig:
    raw: Mapping[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.raw["candidate_id"])

    @property
    def component(self) -> Mapping[str, Any]:
        return self.raw["component"]

    @property
    def observation(self) -> Mapping[str, Any]:
        return self.raw["observation"]

    @property
    def action(self) -> Mapping[str, Any]:
        return self.raw["action"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.raw["training"]

    @property
    def deployment(self) -> Mapping[str, Any]:
        return self.raw["deployment"]


def load_r12_evolution_config(path: str | Path) -> R12EvolutionConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError("R12-E1 config keys differ from the evolution schema")
    if payload["schema_version"] != 4 or payload["round"] != "R12-E1":
        raise ValueError("R12-E1 config identity differs")
    candidate = str(payload["candidate_id"])
    if candidate not in CANDIDATE_SPECS:
        raise ValueError("R12-E1 candidate is not registered")
    if payload["component"].get("kind") != CANDIDATE_SPECS[candidate]["kind"]:
        raise ValueError("R12-E1 candidate/component kind differs")
    if payload["observation"] != locked_r12_full_episode_observation():
        raise ValueError("R12-E1 native-resolution observation contract differs")
    if payload["action"] != {
        "horizon": 100,
        "max_agents": 4,
        "action_dim": 8,
        "belief_dim": 96,
        "normalization": "W10_mean_std_copied_into_R12_checkpoint",
        "normalized_clip": 5.0,
        "num_proposals": 1,
        "condition_tokens": 37,
    }:
        raise ValueError("R12-E1 joint action contract differs")

    training = payload["training"]
    required_training = {
        "updates",
        "bridge_updates",
        "batch_size",
        "rows_per_task",
        "seed",
        "learning_rate",
        "weight_decay",
        "precision",
        "checkpoint_every",
        "progress_every",
        "grad_clip",
        "warm_start_checkpoint",
        "warm_start_sha256",
        "warm_start_update",
        "joint_warmup_steps",
        "joint_decay_steps",
        "decay_lr_ratio",
        "history_augmentation_probability",
        "history_augmentation_ramp_updates",
        "history_noise_scale",
        "recovery_cache",
        "task_film_hidden_dim",
        "task_film_scale",
        "agent_slot_scale",
    }
    if set(training) != required_training:
        raise ValueError("R12-E1 training keys differ")
    rows = training["rows_per_task"]
    if not isinstance(rows, dict) or set(rows) != set(TASKS):
        raise ValueError("R12-E1 rows_per_task must cover all five tasks")
    rows = {task: int(rows[task]) for task in TASKS}
    if any(value < 1 for value in rows.values()):
        raise ValueError("R12-E1 cannot discard a training task")
    if int(training["batch_size"]) != sum(rows.values()):
        raise ValueError("R12-E1 batch size does not match task rows")
    locked = {
        "updates": 130_000,
        "bridge_updates": 10_000,
        "seed": 20260806,
        "precision": "bfloat16",
        "checkpoint_every": 10_000,
        "progress_every": 50,
        "warm_start_update": 60_000,
        "joint_warmup_steps": 1_000,
        "joint_decay_steps": 120_000,
        "history_augmentation_probability": 0.25,
        "history_augmentation_ramp_updates": 10_000,
        "recovery_cache": "",
    }
    for key, value in locked.items():
        if training[key] != value:
            raise ValueError(f"R12-E1 frozen training protocol differs at {key}")
    for key in ("learning_rate", "weight_decay", "grad_clip"):
        if float(training[key]) <= 0:
            raise ValueError(f"R12-E1 {key} must be positive")
    if float(training["history_noise_scale"]) < 0:
        raise ValueError("R12-E1 history noise scale must be non-negative")
    if int(training["task_film_hidden_dim"]) < 1:
        raise ValueError("R12-E1 task FiLM hidden size must be positive")
    if not 0 < float(training["task_film_scale"]) <= 1:
        raise ValueError("R12-E1 task FiLM scale must be in (0,1]")
    if not 0 < float(training["agent_slot_scale"]) <= 1:
        raise ValueError("R12-E1 agent-slot scale must be in (0,1]")
    warm = str(training["warm_start_checkpoint"])
    digest = str(training["warm_start_sha256"])
    if not warm or len(digest) != 64:
        raise ValueError("R12-E1 requires a hash-pinned R12-R3 core warm start")

    deployment = payload["deployment"]
    required_deployment = {
        "specialist_tasks",
        "protected_tasks",
        "w10_checkpoint",
        "w10_checkpoint_sha256",
        "routing",
    }
    if not isinstance(deployment, dict) or set(deployment) != required_deployment:
        raise ValueError("R12-E1 deployment keys differ")
    specialist = tuple(deployment["specialist_tasks"])
    protected = tuple(deployment["protected_tasks"])
    if (
        not specialist
        or set(specialist) & set(protected)
        or set(specialist) | set(protected) != set(TASKS)
    ):
        raise ValueError("R12-E1 specialist/protected task partition differs")
    if deployment["routing"] != "explicit_task_id_exact_w10_fallback":
        raise ValueError("R12-E1 deployment routing differs")
    if len(str(deployment["w10_checkpoint_sha256"])) != 64:
        raise ValueError("R12-E1 W10 fallback digest is invalid")

    expected_rule = {
        "gate20_tasks": list(TASKS),
        "episodes_per_task": 20,
        "baseline_total_successes": 74,
        "winner_rule": "complete_100_episodes_and_total_successes_strictly_greater_than_74",
        "tie_break": [
            "paired_wins",
            "camera_plus_stack_successes",
            "worst_task_successes",
            "p95_latency_ms",
            "gpu_hours",
            "candidate_id",
        ],
    }
    if payload["selection_rule"] != expected_rule:
        raise ValueError("R12-E1 Gate20 selection rule differs")
    return R12EvolutionConfig(payload)


def bind_role_conditioned_spatial_queries(
    slot_delta: torch.Tensor,
    agent_mask: torch.Tensor,
    query_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bind equal groups of existing spatial queries to joint-action slots.

    Stack exposes four named cameras that are pixel-identical, so view identity
    cannot associate the shared spatial evidence with a particular robot.  The
    E1 checkpoint already contains learned agent-slot embeddings.  Reusing
    those embeddings on four equal query groups introduces no new checkpoint
    tensors and preserves the 37-token/core contract while giving each action
    slot an explicit role-conditioned read of the legal current RGB.
    """

    if slot_delta.ndim != 3 or tuple(agent_mask.shape) != tuple(slot_delta.shape[:2]):
        raise ValueError("R15 role-query slot/mask shape differs")
    batch, agents, width = slot_delta.shape
    if (
        query_count < agents
        or query_count % agents
    ):
        raise ValueError("R15 role-query grouping contract differs")
    queries_per_agent = query_count // agents
    role_delta = slot_delta[:, :, None, :].expand(
        -1, -1, queries_per_agent, -1
    ).reshape(batch, query_count, width)
    role_mask = agent_mask[:, :, None].expand(
        -1, -1, queries_per_agent
    ).reshape(batch, query_count).bool()
    return role_delta, role_mask


def deduplicate_exact_spatial_views(
    spatial_tokens: torch.Tensor,
    spatial_view_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep the first active copy of every bit-exact spatial view.

    The Stack dataset exposes four differently named cameras from one physical
    pose.  Their DINO tokens are therefore exact duplicates before the bridge
    adds learned view embeddings.  Retaining only the first physical copy
    prevents four artificial view identities from reweighting the same visual
    evidence.  Near-equal views are deliberately preserved so the operation is
    fail-closed and cannot silently merge distinct cameras.
    """

    if spatial_tokens.ndim != 4:
        raise ValueError("R15 view-dedup spatial tokens must be [batch,view,patch,dim]")
    if tuple(spatial_view_mask.shape) != tuple(spatial_tokens.shape[:2]):
        raise ValueError("R15 view-dedup token/mask shape differs")
    active = spatial_view_mask.bool()
    if not bool(active.any(dim=1).all()):
        raise ValueError("every R15 sample requires an active spatial view")
    deduplicated = active.clone()
    batch, views = active.shape
    for view in range(1, views):
        duplicate = torch.zeros(batch, dtype=torch.bool, device=active.device)
        for previous in range(view):
            exact = (spatial_tokens[:, view] == spatial_tokens[:, previous]).reshape(
                batch, -1
            ).all(dim=1)
            duplicate |= active[:, previous] & exact
        deduplicated[:, view] &= ~duplicate
    if not bool(deduplicated.any(dim=1).all()):
        raise ValueError("R15 view deduplication removed every spatial view")
    return deduplicated


class TaskConditionedActionGenerator(R4JointActionGenerator):
    """Full-resolution R12 specialist with bounded task and slot conditioning."""

    def __init__(
        self, config: R12EvolutionConfig, *, core: nn.Module | None = None
    ) -> None:
        super().__init__(config, core=core)
        belief_dim = int(config.action["belief_dim"])
        hidden = int(config.training["task_film_hidden_dim"])
        self.task_film_scale = float(config.training["task_film_scale"])
        self.agent_slot_scale = float(config.training["agent_slot_scale"])
        self.task_embedding = nn.Embedding(len(TASKS), belief_dim)
        # Stack's four configured camera names are pixel-identical and the
        # three arms start from identical local qpos.  Preserve the existing
        # 37-token contract while adding identity only to the four existing
        # TeamBeliefState agent-token positions.  Native image tokens remain
        # the high-bandwidth primary input.
        self.agent_slot_embedding = nn.Parameter(
            torch.empty(self.max_agents, belief_dim)
        )
        self.task_film = nn.Sequential(
            nn.LayerNorm(belief_dim),
            nn.Linear(belief_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, belief_dim * 2),
        )
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.normal_(self.agent_slot_embedding, std=0.02)
        nn.init.zeros_(self.task_film[-1].weight)
        nn.init.zeros_(self.task_film[-1].bias)

    def condition(
        self,
        belief: TeamBeliefState,
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
        task_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if belief.agent_tokens.shape[1] != self.max_agents:
            raise ValueError("R12-E1 agent-slot count differs")
        agent_start = belief.tokens.shape[1]
        agent_stop = agent_start + self.max_agents
        slot_delta = self.agent_slot_scale * self.agent_slot_embedding[None]
        slot_delta = slot_delta * belief.agent_mask[:, :, None].to(slot_delta.dtype)
        query_bias, query_mask = bind_role_conditioned_spatial_queries(
            slot_delta,
            belief.agent_mask,
            self.bridge.query_count,
        )
        spatial_view_mask = deduplicate_exact_spatial_views(
            spatial_tokens, spatial_view_mask
        )
        tokens, mask = self.bridge(
            belief,
            spatial_tokens,
            spatial_view_mask,
            query_bias=query_bias,
            query_mask=query_mask,
        )
        # Avoid an in-place update so autograd retains a clean path into the
        # explicit identity embeddings during bridge alignment.
        tokens = torch.cat(
            [
                tokens[:, :agent_start],
                tokens[:, agent_start:agent_stop] + slot_delta,
                tokens[:, agent_stop:],
            ],
            dim=1,
        )
        if tuple(task_index.shape) != (len(tokens),):
            raise ValueError("R12-E1 task index must be [batch]")
        if task_index.dtype != torch.long:
            task_index = task_index.long()
        if bool(((task_index < 0) | (task_index >= len(TASKS))).any()):
            raise ValueError("R12-E1 task index is out of range")
        gamma, beta = self.task_film(self.task_embedding(task_index)).chunk(2, -1)
        scale = self.task_film_scale
        tokens = tokens * (1 + scale * torch.tanh(gamma[:, None]))
        tokens = tokens + scale * torch.tanh(beta[:, None])
        expected = int(self.config.action["condition_tokens"])
        if tokens.shape[1:] != (expected, int(self.config.action["belief_dim"])):
            raise ValueError("R12-E1 condition token contract differs")
        return tokens, mask

    def set_training_stage(self, stage: str) -> list[str]:
        super().set_training_stage(stage)
        if stage == "bridge":
            self.agent_slot_embedding.requires_grad_(True)
            for parameter in self.task_embedding.parameters():
                parameter.requires_grad_(True)
            for parameter in self.task_film.parameters():
                parameter.requires_grad_(True)
        return [name for name, value in self.named_parameters() if value.requires_grad]

    def training_loss(
        self,
        belief: TeamBeliefState,
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
        task_index: torch.Tensor,
        actions: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        tokens, token_mask = self.condition(
            belief, spatial_tokens, spatial_view_mask, task_index
        )
        flat = self.flatten_actions(actions, belief.agent_mask)
        feature_mask = belief.agent_mask[:, None, :, None].expand(
            -1, self.horizon, -1, self.action_dim
        ).reshape(len(flat), self.horizon, -1)
        return self.core.training_loss(
            tokens, token_mask, flat, feature_mask & step_mask[:, :, None]
        )

    @torch.no_grad()
    def sample(
        self,
        belief: TeamBeliefState,
        *,
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
        task_index: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> ActionProposalBatch:
        tokens, token_mask = self.condition(
            belief, spatial_tokens, spatial_view_mask, task_index
        )
        flat = self.core.sample(tokens, token_mask, noise=noise)
        expected = (len(tokens), self.horizon, self.max_agents * self.action_dim)
        if tuple(flat.shape) != expected:
            raise ValueError(f"R12-E1 core output {tuple(flat.shape)} != {expected}")
        actions = flat.float().clamp(
            -self.normalized_clip, self.normalized_clip
        ).reshape(len(flat), self.horizon, self.max_agents, self.action_dim)
        actions = actions.permute(0, 2, 1, 3)
        actions = actions * belief.agent_mask[:, :, None, None].to(actions.dtype)
        return ActionProposalBatch(
            actions=actions[:, None],
            base_index=0,
            valid_mask=torch.ones(
                (len(actions), 1), dtype=torch.bool, device=actions.device
            ),
            agent_mask=belief.agent_mask,
            source=(
                f"r15_{self.candidate_id}_role_query_view_dedup_stack_specialist",
            ),
            diagnostics={
                "core_free": True,
                "legacy_core_import": False,
                "observation_mode": self.config.observation["mode"],
                "primary_input": "native_480x640_fixed_view_rgb",
                "supplemental_inputs": "W11_TeamBeliefState+task_id+agent_slot_id",
                "task_film_scale": self.task_film_scale,
                "agent_slot_scale": self.agent_slot_scale,
                "role_conditioned_spatial_queries": True,
                "exact_spatial_view_deduplication": True,
                "spatial_queries_per_action_slot": (
                    self.bridge.query_count // self.max_agents
                ),
            },
        ).validate()


__all__ = [
    "R12EvolutionConfig",
    "TaskConditionedActionGenerator",
    "bind_role_conditioned_spatial_queries",
    "deduplicate_exact_spatial_views",
    "load_r12_evolution_config",
]
