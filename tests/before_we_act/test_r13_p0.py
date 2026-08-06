from __future__ import annotations

import importlib.util

import pytest
import torch


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tensordict") is None,
    reason="TD-MPC2 pins tensordict in requirements/r13-p0.txt",
)


def test_tdmpc2_world_core_shape_and_action_effect():
    from before_we_act.world_model.tdmpc2_world import TDMPC2WorldCore

    config = {
        "embed_dim": 96,
        "mlp_dim": 384,
        "depth": 2,
        "num_q": 2,
        "simnorm_dim": 8,
    }
    torch.manual_seed(1302)
    model = TDMPC2WorldCore(config).eval()
    tokens = torch.randn(2, 16, 96)
    action = torch.randn(2, 96)
    first, uncertainty = model(tokens, action)
    second, _ = model(tokens, action + 0.5)
    assert first.shape == tokens.shape
    assert uncertainty.shape == (2,)
    assert torch.isfinite(first).all()
    assert not torch.equal(first, second)
