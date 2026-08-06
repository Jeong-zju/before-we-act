"""R12-R4 full-data action codec with a direct spatial query bridge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml

from before_we_act.contracts import ActionProposalBatch, TeamBeliefState
from before_we_act.spatial_observation import locked_r12_full_episode_observation
from .registry import CANDIDATE_SPECS, build_action_core
from .spatial_bridge import SpatialQueryBridge


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
    "selection_rule",
}


@dataclass(frozen=True)
class R12R4Config:
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


def load_r12_r4_config(path: str | Path) -> R12R4Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError("R12-R4 config keys differ from the frozen schema")
    if payload["schema_version"] != 3 or payload["round"] != "R12-R4":
        raise ValueError("R12-R4 config identity differs")
    candidate = str(payload["candidate_id"])
    if candidate not in CANDIDATE_SPECS:
        raise ValueError("R12-R4 candidate is not registered")
    if payload["component"].get("kind") != CANDIDATE_SPECS[candidate]["kind"]:
        raise ValueError("R12-R4 candidate/component kind differs")
    if payload["observation"] != locked_r12_full_episode_observation():
        raise ValueError("R12-R4 rectangular observation contract differs")
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
        raise ValueError("R12-R4 joint action contract differs")
    training = payload["training"]
    required = {
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
    }
    if set(training) != required:
        raise ValueError("R12-R4 training keys differ from the frozen schema")
    locked = {
        "updates": 130_000,
        "bridge_updates": 10_000,
        "batch_size": 10,
        "rows_per_task": 2,
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
            raise ValueError(f"R12-R4 frozen training protocol differs at {key}")
    for key in ("learning_rate", "weight_decay", "grad_clip"):
        if float(training[key]) <= 0:
            raise ValueError(f"R12-R4 {key} must be positive")
    if not 0 < float(training["decay_lr_ratio"]) <= 1:
        raise ValueError("R12-R4 decay LR ratio must be in (0,1]")
    if float(training["history_noise_scale"]) < 0:
        raise ValueError("R12-R4 history noise scale must be non-negative")
    warm = str(training["warm_start_checkpoint"])
    digest = str(training["warm_start_sha256"])
    if not warm or len(digest) != 64:
        raise ValueError("R12-R4 requires a hash-pinned R12-R3 core warm start")
    expected_rule = {
        "gate20_tasks": [
            "lift_barrier",
            "camera_alignment",
            "three_robots_stack_cube",
            "long_pipeline_delivery",
            "take_photo",
        ],
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
        raise ValueError("R12-R4 Gate20 selection rule differs")
    return R12R4Config(payload)


class R4JointActionGenerator(nn.Module):
    """Shared full-observation codec around exactly one candidate action core."""

    def __init__(self, config: R12R4Config, *, core: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.candidate_id = config.candidate_id
        self.horizon = int(config.action["horizon"])
        self.max_agents = int(config.action["max_agents"])
        self.action_dim = int(config.action["action_dim"])
        self.normalized_clip = float(config.action["normalized_clip"])
        component = dict(config.component)
        component.update(
            horizon=self.horizon,
            joint_action_dim=self.max_agents * self.action_dim,
            belief_dim=int(config.action["belief_dim"]),
            condition_tokens=int(config.action["condition_tokens"]),
        )
        self.core = core if core is not None else build_action_core(
            config.candidate_id, component
        )
        self.bridge = SpatialQueryBridge()

    def condition(
        self,
        belief: TeamBeliefState,
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.bridge(belief, spatial_tokens, spatial_view_mask)
        expected = int(self.config.action["condition_tokens"])
        if tokens.shape[1:] != (expected, int(self.config.action["belief_dim"])):
            raise ValueError("R12-R4 condition token contract differs")
        return tokens, mask

    def set_training_stage(self, stage: str) -> list[str]:
        if stage not in ("bridge", "joint"):
            raise ValueError("R12-R4 training stage must be bridge or joint")
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "joint")
        if stage == "bridge":
            for parameter in self.bridge.parameters():
                parameter.requires_grad_(True)
            # New condition positions and P2's learned current-condition plan
            # proposal belong to the adapter alignment phase, not the frozen
            # historical core skill.
            allowed = (
                "condition_position",
                "model.cond_pos_emb",
                "plan_proposal",
            )
            for name, parameter in self.core.named_parameters():
                if name.startswith(allowed):
                    parameter.requires_grad_(True)
        return [name for name, value in self.named_parameters() if value.requires_grad]

    def flatten_actions(
        self, actions: torch.Tensor, agent_mask: torch.Tensor
    ) -> torch.Tensor:
        expected = (len(actions), self.horizon, self.max_agents, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"R12-R4 action target {tuple(actions.shape)} != {expected}")
        return (
            actions * agent_mask[:, None, :, None].to(actions.dtype)
        ).reshape(len(actions), self.horizon, -1)

    def training_loss(
        self,
        belief: TeamBeliefState,
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
        actions: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        tokens, token_mask = self.condition(
            belief, spatial_tokens, spatial_view_mask
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
        noise: torch.Tensor | None = None,
    ) -> ActionProposalBatch:
        tokens, token_mask = self.condition(
            belief, spatial_tokens, spatial_view_mask
        )
        flat = self.core.sample(tokens, token_mask, noise=noise)
        expected = (len(tokens), self.horizon, self.max_agents * self.action_dim)
        if tuple(flat.shape) != expected:
            raise ValueError(f"R12-R4 core output {tuple(flat.shape)} != {expected}")
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
            source=(f"r12r4_{self.candidate_id}_full_query",),
            diagnostics={
                "core_free": True,
                "legacy_core_import": False,
                "observation_mode": self.config.observation["mode"],
                "spatial_scalar_gate": False,
                "condition_tokens": tokens.shape[1],
            },
        ).validate()


def load_r3_core_warm_start(
    model: R4JointActionGenerator, checkpoint: Mapping[str, object]
) -> dict[str, object]:
    """Load only shape-compatible R3 action-core tensors, never its adapter."""

    source_model = checkpoint.get("model")
    if not isinstance(source_model, Mapping):
        raise ValueError("R12-R3 warm start has no model state")
    source = {
        name.removeprefix("core."): value
        for name, value in source_model.items()
        if str(name).startswith("core.") and isinstance(value, torch.Tensor)
    }
    target = model.core.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in target and tuple(value.shape) == tuple(target[name].shape)
    }
    skipped = sorted(set(source) - set(compatible))
    incompatible = model.core.load_state_dict(compatible, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("unexpected R12-R3 action-core warm-start key")
    loaded = sorted(compatible)
    if not loaded:
        raise ValueError("R12-R3 action-core warm start loaded no tensors")
    return {
        "mode": "r12r3_core_only_shape_compatible",
        "loaded_keys": loaded,
        "skipped_source_keys": skipped,
        "new_target_keys": sorted(incompatible.missing_keys),
        "adapter_loaded": False,
    }


__all__ = [
    "R12R4Config",
    "R4JointActionGenerator",
    "load_r12_r4_config",
    "load_r3_core_warm_start",
]
