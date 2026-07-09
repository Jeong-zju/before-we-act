import numpy as np
import torch

from policies.closed_loop import PolicyConfig, get_action_from_scripted, unwrap_reset, unwrap_step


def test_policy_config_defaults():
    cfg = PolicyConfig()
    assert cfg.horizon > 0
    assert cfg.k_exec > 0
    assert cfg.num_candidates > 0


def test_scripted_fallback_action_shape():
    a = get_action_from_scripted(None, None)
    assert isinstance(a, np.ndarray)
    assert a.shape == (8,)
    assert np.isfinite(a).all()


def test_unwrap_reset_and_step():
    obs, info = unwrap_reset(({"x": 1}, {"i": 2}))
    assert obs["x"] == 1
    assert info["i"] == 2

    obs, reward, done, info = unwrap_step(({"x": 1}, 1.0, False, True, {"a": 3}))
    assert done is True
    assert reward == 1.0

    obs, reward, done, info = unwrap_step(({"x": 1}, 1.0, False, {"a": 3}))
    assert done is False
