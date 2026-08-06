from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml

from before_we_act.contracts import ConsequencePrediction
from .registry import CANDIDATE_SPECS, build_world_core


EXPECTED_TOP_LEVEL = {
    "schema_version",
    "round",
    "candidate_id",
    "parent_commit",
    "belief_checkpoint_sha256",
    "action_checkpoint_sha256",
    "component",
    "world",
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
    "grad_clip",
}


@dataclass(frozen=True)
class R13Config:
    raw: Mapping[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.raw["candidate_id"])

    @property
    def component(self) -> Mapping[str, Any]:
        return self.raw["component"]

    @property
    def world(self) -> Mapping[str, Any]:
        return self.raw["world"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.raw["training"]


def load_r13_config(path: str | Path) -> R13Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError("R13 config keys differ from the frozen schema")
    if payload["schema_version"] != 1 or payload["round"] != "R13":
        raise ValueError("R13 config identity differs")
    candidate = str(payload["candidate_id"])
    if candidate not in CANDIDATE_SPECS:
        raise ValueError("R13 candidate is not registered")
    if payload["component"].get("kind") != CANDIDATE_SPECS[candidate]["kind"]:
        raise ValueError("R13 candidate/component kind differs")
    if set(payload["training"]) != EXPECTED_TRAINING:
        raise ValueError("R13 training keys differ")
    locked_training = {
        "batch_size": 64,
        "updates": 10_000,
        "seed": 20260806,
        "precision": "bfloat16",
        "checkpoint_every": 1_000,
        "progress_every": 20,
    }
    for key, value in locked_training.items():
        if payload["training"][key] != value:
            raise ValueError(f"locked R13 training protocol differs at {key}")
    for key in ("learning_rate", "weight_decay", "grad_clip"):
        if float(payload["training"][key]) <= 0:
            raise ValueError(f"R13 {key} must be positive")
    expected_world = {
        "belief_dim": 96,
        "belief_tokens": 16,
        "max_agents": 4,
        "action_horizon": 100,
        "action_dim": 8,
        "action_prefix_steps": 16,
        "prediction_horizons": [1, 5, 15],
        "target_tokens": 1,
        "future_inputs_forbidden": True,
        "planner_enabled": False,
        "rerank_enabled": False,
    }
    if payload["world"] != expected_world:
        raise ValueError("R13 world contract differs")
    if payload["loss_weights"] != {
        "latent": 0.60,
        "qpos": 0.20,
        "progress": 0.15,
        "failure": 0.05,
    }:
        raise ValueError("R13 loss weights differ")
    if payload["selection_rule"] != {
        "latent_gain": 0.50,
        "qpos_gain": 0.20,
        "progress_r2": 0.20,
        "throughput": 0.10,
        "throughput_saturation_windows_per_second": 1024,
        "minimum_score": None,
        "winner_rule": "highest_world_screen_score_among_valid_candidates",
    }:
        raise ValueError("R13 world selection rule differs")
    identity_lengths = {
        "parent_commit": 40,
        "belief_checkpoint_sha256": 64,
        "action_checkpoint_sha256": 64,
    }
    for key, expected_length in identity_lengths.items():
        if len(str(payload[key])) != expected_length:
            raise ValueError(f"R13 {key} identity is invalid")
    return R13Config(payload)


class CandidateConditionedWorldModel(nn.Module):
    """Shared legal-input adapter and outcome heads around one upstream core."""

    def __init__(self, config: R13Config) -> None:
        super().__init__()
        self.config = config
        self.candidate_id = config.candidate_id
        dim = int(config.world["belief_dim"])
        self.dim = dim
        self.max_agents = int(config.world["max_agents"])
        self.horizons = tuple(int(value) for value in config.world["prediction_horizons"])
        self.action_adapter = nn.Sequential(
            nn.Linear(int(config.world["action_dim"]), dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.core = build_world_core(config.candidate_id, dict(config.component))
        self.horizon_embedding = nn.Parameter(torch.empty(len(self.horizons), dim))
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.latent_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.qpos_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, self.max_agents * 9)
        )
        self.progress_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.failure_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(
        self,
        *,
        belief_tokens: torch.Tensor,
        belief_agent_tokens: torch.Tensor,
        belief_consensus: torch.Tensor,
        belief_uncertainty: torch.Tensor,
        agent_mask: torch.Tensor,
        candidate_actions: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> ConsequencePrediction:
        del belief_agent_tokens, belief_uncertainty
        if belief_tokens.ndim != 3 or belief_tokens.shape[-1] != self.dim:
            raise ValueError("R13 belief tokens must be [B,T,96]")
        if candidate_actions.ndim != 5:
            raise ValueError("R13 candidates must be [B,P,A,H,D]")
        batch, proposals, agents, horizon, action_dim = candidate_actions.shape
        if (
            agents != self.max_agents
            or horizon != int(self.config.world["action_horizon"])
            or action_dim != int(self.config.world["action_dim"])
            or agent_mask.shape != (batch, agents)
            or candidate_valid_mask.shape != (batch, proposals)
        ):
            raise ValueError("R13 candidate/agent contract differs")
        prefix = candidate_actions[:, :, :, : int(self.config.world["action_prefix_steps"])]
        action = self.action_adapter(prefix).mean(dim=3)
        weights = agent_mask[:, None, :, None].to(action.dtype)
        action = (action * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        tokens = belief_tokens[:, : int(self.config.world["belief_tokens"])]
        expanded_tokens = tokens[:, None].expand(-1, proposals, -1, -1)
        flat_tokens = expanded_tokens.reshape(batch * proposals, tokens.shape[1], self.dim)
        flat_action = action.reshape(batch * proposals, self.dim)
        core_tokens, core_uncertainty = self.core(flat_tokens, flat_action)
        if core_tokens.shape != flat_tokens.shape:
            raise ValueError("R13 upstream core returned an invalid token shape")
        summary = core_tokens.mean(dim=1).reshape(batch, proposals, self.dim)
        # The consensus residual keeps the candidate-neutral W11 state explicit;
        # all candidate variation must still enter through the action-conditioned core.
        summary = summary + belief_consensus[:, None]
        query = summary[:, :, None] + self.horizon_embedding[None, None]
        latent = self.latent_head(query).unsqueeze(3)
        qpos = self.qpos_head(query).reshape(
            batch, proposals, len(self.horizons), self.max_agents, 9
        )
        qpos = qpos * agent_mask[:, None, None, :, None].to(qpos.dtype)
        progress = self.progress_head(query).squeeze(-1).sigmoid()
        failure = self.failure_head(query).squeeze(-1)
        uncertainty = core_uncertainty.reshape(batch, proposals, 1).expand(
            -1, -1, len(self.horizons)
        ).abs()
        valid = candidate_valid_mask.bool()
        return ConsequencePrediction(
            latent_by_horizon=latent,
            qpos_delta_by_horizon=qpos,
            progress_by_horizon=progress,
            failure_logits_by_horizon=failure,
            uncertainty_by_horizon=uncertainty,
            valid_mask=valid,
            diagnostics={
                "candidate_id": self.candidate_id,
                "planner_enabled": False,
                "rerank_enabled": False,
            },
        ).validate()


def world_losses(
    prediction: ConsequencePrediction,
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    # The shared cache has one observed W12 candidate.  Additional candidates
    # remain supported by the deployed contract but require branch outcomes.
    latent = prediction.latent_by_horizon[:, 0]
    qpos = prediction.qpos_delta_by_horizon[:, 0]
    progress = prediction.progress_by_horizon[:, 0]
    failure = prediction.failure_logits_by_horizon[:, 0]
    horizon_mask = batch["horizon_mask"].to(latent.dtype)
    latent_target = batch["future_latent"]
    latent_error = (latent - latent_target).square().mean(dim=(-1, -2))
    latent_loss = (latent_error * horizon_mask).sum() / horizon_mask.sum().clamp_min(1)
    qpos_error = (qpos - batch["future_qpos_delta"]).square().mean(dim=(-1, -2))
    qpos_loss = (qpos_error * horizon_mask).sum() / horizon_mask.sum().clamp_min(1)
    progress_error = (progress - batch["future_progress"]).square()
    progress_loss = (progress_error * horizon_mask).sum() / horizon_mask.sum().clamp_min(1)
    failure_error = torch.nn.functional.binary_cross_entropy_with_logits(
        failure, batch["future_failure"].to(failure.dtype), reduction="none"
    )
    failure_loss = (failure_error * horizon_mask).sum() / horizon_mask.sum().clamp_min(1)
    total = (
        float(weights["latent"]) * latent_loss
        + float(weights["qpos"]) * qpos_loss
        + float(weights["progress"]) * progress_loss
        + float(weights["failure"]) * failure_loss
    )
    return {
        "loss": total,
        "latent_loss": latent_loss,
        "qpos_loss": qpos_loss,
        "progress_loss": progress_loss,
        "failure_loss": failure_loss,
    }
