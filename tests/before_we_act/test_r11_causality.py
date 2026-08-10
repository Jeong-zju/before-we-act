from __future__ import annotations

from pathlib import Path

import torch

from before_we_act.evaluate_r11_causality import (
    PROBE_SEED,
    PROBE_UPDATES,
    SAMPLES_PER_TASK,
    _masked_representation_sse,
)


def test_masked_representation_error_excludes_invalid_future_offsets():
    target = torch.zeros(2, 4, 3)
    prediction = torch.ones_like(target)
    prediction[:, 2:] = 100
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    sse, count = _masked_representation_sse(prediction, target, mask)
    assert sse == 12
    assert count == 12


def test_causal_protocol_is_frozen_and_deployment_call_excludes_targets():
    assert (PROBE_SEED, PROBE_UPDATES, SAMPLES_PER_TASK) == (20260811, 4, 32)
    source = (
        Path(__file__).resolve().parents[2]
        / "before_we_act/evaluate_r11_causality.py"
    ).read_text()
    deployment = source[source.index("deployment_keys = {") : source.index("action_outputs = {}")]
    assert '"action"' not in deployment
    assert '"future_rgb"' not in deployment
    assert '"task_text"' in deployment
