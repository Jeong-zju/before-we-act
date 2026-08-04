from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml

from before_we_act.contracts import TeamBeliefState
from .registry import CANDIDATE_SPECS, build_candidate_encoder


EXPECTED_TOP_LEVEL = {
    "schema_version",
    "round",
    "candidate_id",
    "parent_commit",
    "parent_checkpoint_sha256",
    "component",
    "observation",
    "training",
    "loss_weights",
    "selection_rule",
}
EXPECTED_TRAINING = {
    "batch_size",
    "updates",
    "seed",
    "learning_rate",
    "weight_decay",
    "precision",
    "checkpoint_every",
    "progress_every",
}


@dataclass(frozen=True)
class R11Config:
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
    def training(self) -> Mapping[str, Any]:
        return self.raw["training"]


def load_r11_config(path: str | Path) -> R11Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError("R11 config keys differ from the frozen schema")
    if payload["schema_version"] != 1 or payload["round"] != "R11":
        raise ValueError("R11 config identity differs")
    candidate = payload["candidate_id"]
    if candidate not in CANDIDATE_SPECS:
        raise ValueError("R11 candidate is not registered")
    if payload["component"].get("kind") != CANDIDATE_SPECS[candidate]["kind"]:
        raise ValueError("R11 candidate and component kind differ")
    if set(payload["training"]) != EXPECTED_TRAINING:
        raise ValueError("R11 training keys differ from the frozen schema")
    training = payload["training"]
    locked = {
        "batch_size": 64,
        "updates": 10_000,
        "seed": 20260805,
        "precision": "bfloat16",
        "checkpoint_every": 1000,
        "progress_every": 20,
    }
    for key, value in locked.items():
        if training[key] != value:
            raise ValueError(f"locked R11 training protocol differs at {key}")
    observation = payload["observation"]
    expected_observation = {
        "history_steps": 3,
        "grid_size": 4,
        "max_agents": 4,
        "max_views": 5,
        "qpos_dim": 9,
        "action_dim": 8,
        "visual_channels": 15,
        "visual_mask_channels": 5,
    }
    if observation != expected_observation:
        raise ValueError("locked R11 observation contract differs")
    weights = payload["loss_weights"]
    if weights != {"future_feature": 0.5, "partner_action": 0.3, "shared_progress": 0.2}:
        raise ValueError("locked R11 loss weights differ")
    rule = payload["selection_rule"]
    if rule != {
        "future_feature_gain": 0.50,
        "partner_action_gain": 0.25,
        "shared_progress_r2": 0.20,
        "throughput": 0.05,
        "throughput_saturation_windows_per_second": 512,
        "minimum_score": None,
    }:
        raise ValueError("R11 representation selection rule differs")
    return R11Config(payload)


class PredictiveBeliefModel(nn.Module):
    """Shared adapter/readouts around one transplanted predictive core."""

    def __init__(self, config: R11Config) -> None:
        super().__init__()
        component = dict(config.component)
        self.candidate_id = config.candidate_id
        self.embed_dim = int(component["embed_dim"])
        self.max_agents = int(config.observation["max_agents"])
        visual_input_dim = int(config.observation["visual_channels"]) + int(
            config.observation["visual_mask_channels"]
        )
        self.visual_adapter = nn.Linear(visual_input_dim, self.embed_dim)
        self.state_adapter = nn.Linear(9, self.embed_dim)
        self.action_adapter = nn.Linear(8, self.embed_dim)
        self.core = build_candidate_encoder(config.candidate_id, component)
        self.future_head = nn.Linear(self.embed_dim, 15)
        self.partner_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, 8)
        )
        self.progress_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, 1)
        )

    def forward(self, batch: Mapping[str, torch.Tensor]):
        visual = torch.cat(
            [
                batch["visual"],
                batch["view_mask"][:, :, None, :].expand(
                    -1, -1, batch["visual"].shape[2], -1
                ),
            ],
            dim=-1,
        )
        tokens = self.visual_adapter(visual)
        mask = batch["agent_mask"].to(tokens.dtype)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        state = self.state_adapter(batch["qpos"])
        action = self.action_adapter(batch["actions"])
        team_state = (state * mask[:, None, :, None]).sum(dim=2) / denominator[:, None]
        team_action = (action * mask[:, None, :, None]).sum(dim=2) / denominator[:, None]
        tokens = tokens + team_state[:, :, None, :] + team_action[:, :, None, :]
        predicted_tokens = self.core(
            tokens=tokens,
            actions=batch["actions"],
            qpos=batch["qpos"],
            agent_mask=batch["agent_mask"],
        )
        if predicted_tokens.ndim != 3 or predicted_tokens.shape[-1] != self.embed_dim:
            raise ValueError("candidate encoder returned an invalid token contract")
        consensus = predicted_tokens.mean(dim=1)
        agent_tokens = self.state_adapter(batch["qpos"][:, -1]) + consensus[:, None, :]
        uncertainty = predicted_tokens.var(dim=1, unbiased=False).mean(dim=-1, keepdim=True)
        belief = TeamBeliefState(
            tokens=predicted_tokens,
            agent_tokens=agent_tokens,
            consensus_token=consensus,
            uncertainty=uncertainty,
            agent_mask=batch["agent_mask"],
        ).validate()
        return {
            "belief": belief,
            "future_visual": self.future_head(predicted_tokens),
            "partner_action": self.partner_head(agent_tokens),
            "shared_progress": self.progress_head(consensus).squeeze(-1).sigmoid(),
        }


def masked_action_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    weights = mask.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)
