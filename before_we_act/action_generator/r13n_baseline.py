"""From-scratch six-task ACT baseline used by R13N."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml

from before_we_act.contracts import ActionProposalBatch, TeamBeliefState
from before_we_act.r13n import TASKS, observation_contract
from .act_chunk import build_core
from .spatial_bridge import SpatialQueryBridge


EXPECTED_TOP_LEVEL = {
    "schema_version", "round", "model_id", "component", "observation",
    "action", "training", "evaluation",
}


@dataclass(frozen=True)
class R13NConfig:
    raw: Mapping[str, Any]

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


def load_r13n_config(path: str | Path) -> R13NConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError("R13N config keys differ")
    if payload["schema_version"] != 1 or payload["round"] != "R13N" or payload["model_id"] != "b6_act_six_task":
        raise ValueError("R13N config identity differs")
    if payload["observation"] != observation_contract():
        raise ValueError("R13N observation contract differs")
    action = payload["action"]
    expected_action = {
        "horizon": 100, "max_agents": 4, "action_dim": 8,
        "belief_dim": 96, "normalized_clip": 5.0,
        "num_proposals": 1, "condition_tokens": 37,
        "normalization": "six_task_active_agent_training_moments_v1",
    }
    if action != expected_action:
        raise ValueError("R13N action contract differs")
    component = payload["component"]
    expected_component = {
        "kind":"act_action_chunk_transformer","hidden_dim":256,"num_heads":8,
        "encoder_layers":4,"decoder_layers":6,"dim_feedforward":2048,
        "dropout":0.1,"latent_dim":32,"condition_tokens":37,"chunk_size":100,
        "temporal_ensemble":True,"plan_prior":"learned_current_condition",
        "plan_prior_hidden_dim":256,"plan_kl_weight":0.01,"kl_balance_alpha":0.8,
    }
    if component != expected_component:
        raise ValueError("R13N ACT component contract differs")
    training = payload["training"]
    required_training = {
        "updates","batch_size","rows_per_task","seed","learning_rate",
        "weight_decay","precision","checkpoint_every","progress_every",
        "grad_clip","warmup_steps","decay_steps","decay_lr_ratio",
        "history_augmentation_probability","history_augmentation_ramp_updates",
        "history_noise_scale","task_film_scale","agent_slot_scale",
    }
    if set(training) != required_training:
        raise ValueError("R13N training keys differ")
    locked = {
        "updates":130_000,"batch_size":12,"rows_per_task":2,
        "seed":20260808,"precision":"bfloat16","checkpoint_every":10_000,
        "progress_every":50,"warmup_steps":1_000,"decay_steps":130_000,
        "history_augmentation_probability":0.25,
        "history_augmentation_ramp_updates":10_000,
    }
    for key, value in locked.items():
        if training[key] != value:
            raise ValueError(f"R13N frozen training protocol differs at {key}")
    if int(training["batch_size"]) != int(training["rows_per_task"]) * len(TASKS):
        raise ValueError("R13N batch is not exactly task balanced")
    for key in ("learning_rate","weight_decay","grad_clip"):
        if float(training[key]) <= 0:
            raise ValueError(f"R13N {key} must be positive")
    for key in ("decay_lr_ratio","task_film_scale","agent_slot_scale"):
        if not 0 < float(training[key]) <= 1:
            raise ValueError(f"R13N {key} must be in (0,1]")
    evaluation = payload["evaluation"]
    if evaluation != {
        "tasks":list(TASKS),"stages":["discovery","validation","formal"],
        "episodes_per_task":20,"candidate_native_coverage":1.0,
        "task_result_reuse":False,"temporal_ensemble_decay":0.01,
    }:
        raise ValueError("R13N evaluation contract differs")
    return R13NConfig(payload)


class CausalTeamEncoder(nn.Module):
    """Small trainable adapter for legal RGB summaries and robot histories."""

    def __init__(
        self, width: int = 96, max_agents: int = 4, agent_slot_scale: float = 0.25
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.max_agents = int(max_agents)
        self.agent_slot_scale = float(agent_slot_scale)
        self.visual = nn.Linear(20, width)
        self.qpos = nn.Linear(9, width)
        self.action = nn.Linear(8, width)
        self.agent_slot = nn.Parameter(torch.empty(max_agents, width))
        self.temporal = nn.Embedding(3, width)
        self.visual_norm = nn.LayerNorm(width)
        self.agent_norm = nn.LayerNorm(width)
        nn.init.normal_(self.agent_slot, std=0.02)
        nn.init.normal_(self.temporal.weight, std=0.02)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> TeamBeliefState:
        required = {"visual","view_mask","qpos","actions","agent_mask"}
        if not required.issubset(batch):
            raise ValueError("R13N causal team batch is incomplete")
        visual = batch["visual"]
        view_mask = batch["view_mask"]
        qpos = batch["qpos"]
        actions = batch["actions"]
        agent_mask = batch["agent_mask"].bool()
        if visual.shape[1:] != (3,16,15) or view_mask.shape[1:] != (3,5):
            raise ValueError("R13N causal visual/history shape differs")
        if qpos.shape[1:] != (3,4,9) or actions.shape[1:] != (3,4,8) or agent_mask.shape[1:] != (4,):
            raise ValueError("R13N causal robot history shape differs")
        latest = torch.cat([visual[:,-1], view_mask[:,-1,None,:].expand(-1,16,-1)], dim=-1)
        visual_tokens = self.visual(latest)
        history = self.qpos(qpos) + self.action(actions) + self.temporal.weight[None,:,None,:]
        mask = agent_mask[:,None,:,None].to(history.dtype)
        history_context = (history * mask).sum(dim=(1,2)) / (mask.sum(dim=(1,2)).clamp_min(1))
        tokens = self.visual_norm(visual_tokens + history_context[:,None])
        agent_tokens = self.agent_norm(
            self.qpos(qpos[:,-1])
            + self.action(actions[:,-1])
            + self.agent_slot_scale * self.agent_slot[None]
        )
        agent_tokens = agent_tokens * agent_mask[:,:,None].to(agent_tokens.dtype)
        denominator = agent_mask.sum(dim=1,keepdim=True).clamp_min(1).to(agent_tokens.dtype)
        consensus = tokens.mean(dim=1) + agent_tokens.sum(dim=1) / denominator
        uncertainty = tokens.var(dim=1,unbiased=False).mean(dim=-1,keepdim=True)
        return TeamBeliefState(
            tokens=tokens,agent_tokens=agent_tokens,consensus_token=consensus,
            uncertainty=uncertainty,agent_mask=agent_mask,
        ).validate()


class R13NActionGenerator(nn.Module):
    def __init__(self, config: R13NConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon = int(config.action["horizon"])
        self.max_agents = int(config.action["max_agents"])
        self.action_dim = int(config.action["action_dim"])
        self.normalized_clip = float(config.action["normalized_clip"])
        width = int(config.action["belief_dim"])
        self.agent_slot_scale = float(config.training["agent_slot_scale"])
        self.team_encoder = CausalTeamEncoder(
            width, self.max_agents, self.agent_slot_scale
        )
        self.bridge = SpatialQueryBridge(belief_dim=width)
        self.task_embedding = nn.Embedding(len(TASKS), width)
        self.task_film = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2))
        self.task_film_scale = float(config.training["task_film_scale"])
        nn.init.normal_(self.task_embedding.weight,std=0.02)
        nn.init.zeros_(self.task_film[-1].weight)
        nn.init.zeros_(self.task_film[-1].bias)
        component = dict(config.component)
        component.update(
            horizon=self.horizon,
            joint_action_dim=self.max_agents*self.action_dim,
            belief_dim=width,
        )
        self.core = build_core(component)

    def condition(
        self,
        batch: Mapping[str, torch.Tensor],
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
        task_index: torch.Tensor,
    ) -> tuple[TeamBeliefState, torch.Tensor, torch.Tensor]:
        belief = self.team_encoder(batch)
        tokens, mask = self.bridge(belief, spatial_tokens, spatial_view_mask)
        if tuple(tokens.shape[1:]) != (37,96) or tuple(task_index.shape) != (len(tokens),):
            raise ValueError("R13N condition/task shape differs")
        if bool(((task_index < 0) | (task_index >= len(TASKS))).any()):
            raise ValueError("R13N task index out of range")
        scale, shift = self.task_film(self.task_embedding(task_index)).chunk(2,dim=-1)
        amplitude = self.task_film_scale
        tokens = tokens * (1 + amplitude*torch.tanh(scale)[:,None]) + amplitude*torch.tanh(shift)[:,None]
        return belief, tokens, mask

    def training_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
        task_index: torch.Tensor,
        actions: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        belief, tokens, mask = self.condition(batch,spatial_tokens,spatial_view_mask,task_index)
        expected = (len(tokens),self.horizon,self.max_agents,self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError("R13N target action shape differs")
        flat = (actions*belief.agent_mask[:,None,:,None].to(actions.dtype)).reshape(len(tokens),self.horizon,-1)
        features = belief.agent_mask[:,None,:,None].expand(-1,self.horizon,-1,self.action_dim).reshape(len(tokens),self.horizon,-1)
        return self.core.training_loss(tokens,mask,flat,features & step_mask[:,:,None])

    @torch.no_grad()
    def sample(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
        task_index: torch.Tensor,
    ) -> ActionProposalBatch:
        belief, tokens, mask = self.condition(batch,spatial_tokens,spatial_view_mask,task_index)
        flat = self.core.sample(tokens,mask)
        expected = (len(tokens),self.horizon,self.max_agents*self.action_dim)
        if tuple(flat.shape) != expected:
            raise ValueError("R13N ACT output shape differs")
        actions = flat.float().clamp(-self.normalized_clip,self.normalized_clip).reshape(len(tokens),self.horizon,self.max_agents,self.action_dim).permute(0,2,1,3)
        actions = actions*belief.agent_mask[:,:,None,None].to(actions.dtype)
        return ActionProposalBatch(
            actions=actions[:,None],base_index=0,
            valid_mask=torch.ones((len(tokens),1),dtype=torch.bool,device=tokens.device),
            agent_mask=belief.agent_mask,source=("r13n_b6_act",),
            diagnostics={"round":"R13N","model_id":"b6_act_six_task","candidate_native":True,"condition_tokens":37},
        ).validate()


__all__ = ["CausalTeamEncoder","R13NActionGenerator","R13NConfig","load_r13n_config"]
