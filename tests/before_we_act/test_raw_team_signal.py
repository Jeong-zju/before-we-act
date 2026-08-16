from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from before_we_act.raw_team_signal_data import CAPACITY_CANDIDATES  # noqa: E402
from before_we_act.raw_team_signal_model import (  # noqa: E402
    TeamActionProbeSet,
    RawTeamSignalEncoder,
    representation_losses,
)


def batch(batch_size: int = 3):
    generator = torch.Generator().manual_seed(23)
    history_mask = torch.ones(batch_size, 16, dtype=torch.bool)
    history_mask[1, :5] = False
    action_mask = history_mask.clone()
    action_mask[:, -1] = False
    return {
        "history_visual": torch.randn(batch_size, 16, 2, 768, generator=generator),
        "history_qpos": torch.randn(batch_size, 16, 9, generator=generator),
        "history_action": torch.randn(batch_size, 16, 8, generator=generator),
        "history_mask": history_mask,
        "action_history_mask": action_mask,
        "task_index": torch.tensor([0, 1, 2]),
        "future_visual": torch.randn(batch_size, 4, 2, 768, generator=generator),
        "future_mask": torch.tensor([[True] * 4, [True, True, False, False], [True] * 4]),
        "teammate_qpos": torch.randn(batch_size, 9, generator=generator),
        "teammate_delta": torch.randn(batch_size, 4, 9, generator=generator),
    }


def model():
    return RawTeamSignalEncoder(d_model=32, temporal_layers=1, heads=4, dropout=0.0)


def test_all_capacity_candidates_are_trained_and_non_future_inputs_are_explicit():
    values = batch()
    runtime = {key: values[key] for key in ("history_visual", "history_qpos", "history_action", "history_mask", "action_history_mask", "task_index")}
    output = model()(**runtime)
    assert tuple(output.capacities) == CAPACITY_CANDIDATES
    for capacity, item in output.capacities.items():
        assert item.tokens.shape == (3, capacity, 32)
        assert item.future_visual.shape == (3, 4, 2, 768)
    losses = representation_losses(output, values)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_history_encoder_does_not_accept_teacher_or_audit_fields():
    values = batch()
    runtime = {key: values[key] for key in ("history_visual", "history_qpos", "history_action", "history_mask", "action_history_mask", "task_index")}
    with pytest.raises(TypeError):
        model()(**runtime, future_visual=values["future_visual"])


def test_matched_probe_heads_have_identical_parameter_counts():
    probes = TeamActionProbeSet(d_model=32)
    counts = {
        key: sum(parameter.numel() for parameter in head.parameters())
        for key, head in probes.probes.items()
    }
    assert len(set(counts.values())) == 1
    assert len(counts) == len(CAPACITY_CANDIDATES) * len(probes.CONDITIONS)
