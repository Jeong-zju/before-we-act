from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from before_we_act.upstream_components.r13_tdmpc2.tdmpc2.common.world_model import (
    WorldModel,
)


class TDMPC2WorldCore(nn.Module):
    """Legal-input adapter around the exact TD-MPC2 world-model heads."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = int(config["embed_dim"])
        cfg = SimpleNamespace(
            multitask=False,
            tasks=(),
            task_dim=0,
            action_dims=(),
            action_dim=dim,
            obs_shape={"state": (dim,)},
            obs="state",
            latent_dim=dim,
            mlp_dim=int(config["mlp_dim"]),
            num_bins=1,
            episodic=True,
            dropout=0.0,
            num_q=int(config["num_q"]),
            log_std_min=-10.0,
            log_std_max=2.0,
            tau=0.01,
            simnorm_dim=int(config["simnorm_dim"]),
            num_enc_layers=int(config["depth"]),
            enc_dim=int(config["mlp_dim"]),
            num_channels=32,
        )
        self.world = WorldModel(cfg)

    def forward(self, tokens: torch.Tensor, action: torch.Tensor):
        state = tokens.mean(dim=1)
        latent = self.world.encode(state, None)
        next_latent = self.world.next(latent, action, None)
        reward = self.world.reward(latent, action, None)
        values = self.world.Q(latent, action, None, return_type="all")
        termination = self.world.termination(next_latent, None, unnormalized=True)
        value_mean = values.mean(dim=0)
        delta = next_latent + reward + value_mean + termination
        output = tokens + delta[:, None]
        uncertainty = values.float().var(dim=0, unbiased=False).mean(dim=-1)
        return output, uncertainty.to(output.dtype)


def build_world_core(config: dict) -> nn.Module:
    return TDMPC2WorldCore(config)
