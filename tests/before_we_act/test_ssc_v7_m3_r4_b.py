from __future__ import annotations

import numpy as np
import pytest

from scripts.before_we_act import run_ssc_v7_m3 as m3
from scripts.before_we_act import run_ssc_v7_m3_r4 as r4
from scripts.before_we_act import run_ssc_v7_m3_r4_b as r4b
from scripts.before_we_act import run_ssc_v7_m3_r4_successor as successor


def probe() -> m3.ProbeData:
    rows = 6
    return m3.ProbeData(
        legal=np.zeros((rows, 3), dtype=np.float32),
        e0=np.zeros((rows, 3), dtype=np.float32),
        social=np.zeros((rows, 192), dtype=np.float32),
        time=np.zeros((rows, 192), dtype=np.float32),
        target=np.zeros((rows, 100, 8), dtype=np.float32),
        target_mask=np.ones((rows, 100), dtype=np.float32),
        tasks=np.asarray(["lift_barrier"] * rows, dtype="U32"),
        episode_ids=np.asarray(["a", "a", "a", "b", "b", "b"], dtype="U64"),
        frame_indices=np.asarray([16, 24, 32, 16, 24, 32], dtype=np.int32),
        agent_slots=np.zeros(rows, dtype=np.int16),
    )


def test_schema_has_36_variable_heads_and_12_frozen_constants() -> None:
    assert len(r4b.ACTIVE_HEADS) == 36
    assert len(r4b.CONSTANT_HEAD_VALUES) == 12
    assert set(r4b.ACTIVE_HEADS).isdisjoint(r4b.CONSTANT_HEAD_VALUES)
    assert set(r4b.ACTIVE_HEADS) | set(r4b.CONSTANT_HEAD_VALUES) == set(range(48))


def test_probability_calibration_preserves_frozen_schema_fields() -> None:
    logits = np.zeros((4, 48), dtype=np.float32)
    prior = np.full(48, 0.25, dtype=np.float32)
    alphas = np.ones(48, dtype=np.float32)
    output = r4b.calibrated_probabilities(logits, prior, alphas)
    for index, value in r4b.CONSTANT_HEAD_VALUES.items():
        assert np.all(output[:, index] == value)


def test_shrinkage_grid_cannot_be_worse_than_constant_on_fit_rows() -> None:
    rng = np.random.default_rng(17)
    logits = rng.normal(size=(100, 48)).astype(np.float32)
    targets = (rng.uniform(size=(100, 48)) > 0.7).astype(np.float32)
    prior = targets.mean(axis=0)
    alphas = r4b.fit_shrinkage(logits, targets, prior)
    calibrated = r4b.calibrated_probabilities(logits, prior, alphas)
    for head in r4b.ACTIVE_HEADS:
        candidate = np.mean((calibrated[:, head] - targets[:, head]) ** 2)
        constant = np.mean((prior[head] - targets[:, head]) ** 2)
        assert candidate <= constant + 1e-7


def test_stale_mapping_never_crosses_episode_reset() -> None:
    data = probe()
    probabilities = np.zeros((6, 48), dtype=np.float32)
    probabilities[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    for index, value in r4b.CONSTANT_HEAD_VALUES.items():
        probabilities[:, index] = value
    reliability = np.ones((6, 1), dtype=np.float32)
    stale, stale_reliability = r4b.stale_values(
        probabilities, reliability, data, delay=8
    )
    assert stale[1, 0] == pytest.approx(0.1)
    assert stale[2, 0] == pytest.approx(0.2)
    assert stale[4, 0] == pytest.approx(0.7)
    assert stale[5, 0] == pytest.approx(0.8)
    assert stale_reliability[0, 0] == 0.0
    assert stale_reliability[3, 0] == 0.0


def test_missing_input_forces_zero_reliability() -> None:
    probabilities = np.full((5, 48), 0.9, dtype=np.float32)
    assert np.all(r4b.predictor_reliability(probabilities, available=False) == 0.0)


def test_direct_residual_is_exact_hc_when_reliability_is_zero() -> None:
    torch = pytest.importorskip("torch")
    hc = m3.build_action_model(successor.HC_INPUT_WIDTH, 256, 101)
    payload = {
        "state_dict": hc.state_dict(),
        "input_width": successor.HC_INPUT_WIDTH,
        "hidden_width": 256,
    }
    rng = np.random.default_rng(5)
    hc_input = rng.normal(size=(7, successor.HC_INPUT_WIDTH)).astype(np.float32)
    arb = rng.normal(size=(7, successor.FEATURE_WIDTH)).astype(np.float32)
    reliability = np.zeros((7, 1), dtype=np.float32)
    values = torch.from_numpy(np.concatenate((hc_input, arb, reliability), axis=1))
    model = successor.DirectResidualFactory.create(payload, 202).eval()
    baseline = r4.HCWrapper.create(payload).eval()
    with torch.no_grad():
        assert torch.equal(model(values), baseline(torch.from_numpy(hc_input)))
