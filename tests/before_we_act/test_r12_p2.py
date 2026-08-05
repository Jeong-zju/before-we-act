from __future__ import annotations

import torch

from before_we_act.action_generator.base import load_r12_config
from before_we_act.action_generator.act_chunk import build_core


def test_act_core_train_and_inference_contract():
    config = load_r12_config("configs/before_we_act/r12_action/p2.yaml")
    core = build_core({**config.component, "horizon": 100, "joint_action_dim": 32, "belief_dim": 96})
    tokens = torch.randn(2, 21, 96)
    token_mask = torch.ones(2, 21, dtype=torch.bool)
    actions = torch.randn(2, 100, 32)
    mask = torch.ones_like(actions, dtype=torch.bool)
    losses = core.training_loss(tokens, token_mask, actions, mask)
    assert losses.keys() == {"loss", "l1", "kl"}
    assert all(torch.isfinite(value) for value in losses.values())
    core.eval()
    with torch.no_grad():
        first = core.sample(tokens, token_mask)
        second = core.sample(tokens, token_mask, noise=torch.randn_like(actions))
    assert first.shape == (2, 100, 32)
    torch.testing.assert_close(first, second)
