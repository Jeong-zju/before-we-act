from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def stub_runtime_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "robofactory", types.ModuleType("robofactory"))
    torchvision = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")
    models.resnet18 = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("resnet18 is outside this evaluator contract test")
    )
    torchvision.models = models
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)


def _legacy_step(histories, arms, step, chunks):
    action = {}
    for local_index, arm in enumerate(arms):
        histories[local_index].append((step, chunks[local_index]))
        histories[local_index] = [
            item for item in histories[local_index] if step - item[0] < len(item[1])
        ]
        candidates = np.asarray(
            [chunk[step - start] for start, chunk in histories[local_index]]
        )
        weights = np.exp(-0.01 * np.arange(len(candidates) - 1, -1, -1))
        weights /= weights.sum()
        action[f"panda-{arm}"] = np.sum(candidates * weights[:, None], axis=0)
    return action


def test_temporal_chunk_ensembler_is_bit_exact_to_parent_loop():
    from stereo_core.evaluate_no_wrist_pair import TemporalChunkEnsembler

    arms = (0, 1, 2)
    legacy_histories = [[] for _ in arms]
    ensembler = TemporalChunkEnsembler(arms)
    rng = np.random.default_rng(45)
    for step in range(8):
        chunks = rng.normal(size=(3, 5, 8)).astype(np.float32)
        expected = _legacy_step(legacy_histories, arms, step, chunks)
        actual = ensembler.append_and_select(step, chunks)
        assert expected.keys() == actual.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_r15_recent_ensemble_emphasizes_latest_and_latest_chunk_is_exact():
    from before_we_act.evaluate_action_generator_r4 import TemporalChunkEnsembler

    chunks0 = np.zeros((1, 4, 8), dtype=np.float32)
    chunks1 = np.ones((1, 4, 8), dtype=np.float32)
    frozen = TemporalChunkEnsembler((0,), decay=0.01)
    mild = TemporalChunkEnsembler((0,), decay=0.02)
    balanced = TemporalChunkEnsembler((0,), decay=0.05)
    recent = TemporalChunkEnsembler((0,), decay=0.10)
    responsive = TemporalChunkEnsembler((0,), decay=0.20)
    frozen.append_and_select(0, chunks0)
    mild.append_and_select(0, chunks0)
    balanced.append_and_select(0, chunks0)
    recent.append_and_select(0, chunks0)
    responsive.append_and_select(0, chunks0)
    frozen_value = frozen.append_and_select(1, chunks1)["panda-0"]
    mild_value = mild.append_and_select(1, chunks1)["panda-0"]
    balanced_value = balanced.append_and_select(1, chunks1)["panda-0"]
    recent_value = recent.append_and_select(1, chunks1)["panda-0"]
    responsive_value = responsive.append_and_select(1, chunks1)["panda-0"]
    assert np.all(mild_value > frozen_value)
    assert np.all(balanced_value > mild_value)
    assert np.all(recent_value > balanced_value)
    assert np.all(responsive_value > recent_value)
    np.testing.assert_array_equal(chunks1[0, 0], np.ones(8, dtype=np.float32))


def test_r15_temporal_grid_has_distinct_resume_routes():
    source = (
        Path(__file__).resolve().parents[2]
        / "before_we_act/evaluate_action_generator_evolution.py"
    ).read_text()
    for mode, route, decay in (
        ("mild_temporal_ensemble", "r15_w12_mild_decay_0p02", "0.02"),
        ("balanced_temporal_ensemble", "r15_w12_balanced_decay_0p05", "0.05"),
        ("recent_temporal_ensemble", "r15_w12_recent_decay_0p10", "0.10"),
        ("responsive_temporal_ensemble", "r15_w12_responsive_decay_0p20", "0.20"),
    ):
        assert mode in source
        assert route in source
        assert f"decay={decay}" in source


def test_cogact_adaptive_ensemble_matches_pinned_equation_and_horizon():
    from before_we_act.upstream_components.cogact import AdaptiveEnsembler

    ensembler = AdaptiveEnsembler(pred_action_horizon=2, adaptive_ensemble_alpha=0.1)
    chunks = [
        np.arange(32, dtype=np.float32).reshape(4, 8),
        np.arange(32, 64, dtype=np.float32).reshape(4, 8),
        np.arange(64, 96, dtype=np.float32).reshape(4, 8),
    ]
    np.testing.assert_array_equal(ensembler.ensemble_action(chunks[0]), chunks[0][0])
    candidates = np.stack([chunks[0][1], chunks[1][0]])
    reference = candidates[-1]
    cosine = np.sum(candidates * reference, axis=1) / (
        np.linalg.norm(candidates, axis=1) * np.linalg.norm(reference) + 1e-7
    )
    weights = np.exp(0.1 * cosine)
    expected = np.sum((weights / weights.sum())[:, None] * candidates, axis=0)
    np.testing.assert_allclose(
        ensembler.ensemble_action(chunks[1]), expected, rtol=0, atol=0
    )
    ensembler.ensemble_action(chunks[2])
    assert len(ensembler.action_history) == 2


def test_cogact_transplant_source_and_license_are_pinned():
    root = Path(__file__).resolve().parents[2]
    component = root / "before_we_act/upstream_components/cogact"
    source = component / "adaptive_ensemble.py"
    license_path = component / "LICENSE"
    source_map = json.loads((component / "SOURCE_MAP.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "41fb978ff46cca961690f67df54ea89873412040e0a8d117fa8f0ccff90fc927"
    )
    assert hashlib.sha256(license_path.read_bytes()).hexdigest() == (
        "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"
    )
    assert source_map["repository_commit"] == (
        "b174a1b86deedfab4d198d935207e7bb0527994e"
    )
    assert source_map["files"][0]["algorithm_changes"] == 0


def test_cogact_team_adapter_preserves_arm_keys_and_route_identity():
    from before_we_act.evaluate_action_generator_evolution import (
        CogACTAdaptiveTeamEnsembler,
    )

    team = CogACTAdaptiveTeamEnsembler((0, 2), horizon=2, alpha=0.1)
    chunks = np.arange(2 * 4 * 8, dtype=np.float32).reshape(2, 4, 8)
    selected = team.append_and_select(0, chunks)
    assert set(selected) == {"panda-0", "panda-2"}
    np.testing.assert_array_equal(selected["panda-0"], chunks[0, 0])
    np.testing.assert_array_equal(selected["panda-2"], chunks[1, 0])
    root = Path(__file__).resolve().parents[2]
    evaluator = (root / "before_we_act/evaluate_action_generator_evolution.py").read_text()
    launcher = (root / "scripts/before_we_act/launch_r15_temporal_screens_tmux.sh").read_text()
    assert "r15_cogact_adaptive_alpha0p1_h2_stack_specialist" in evaluator
    assert "cogact_adaptive_alpha0p1_h2" in launcher


def test_prepare_and_denormalize_preserve_exact_contract():
    from stereo_core.evaluate_no_wrist_pair import (
        denormalize_action_chunks,
        prepare_no_wrist_batch,
    )

    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    observation = {
        "sensor_data": {
            "head_camera_global": {"rgb": image[None]},
            "head_camera_agent0": {"rgb": (image + 1)[None]},
        },
        "agent": {"panda-0": {"qpos": np.arange(9, dtype=np.float32)[None]}},
    }
    stats = {
        "q_mean": torch.arange(9, dtype=torch.float32),
        "q_std": torch.ones(9),
        "a_mean": torch.arange(8, dtype=torch.float32),
        "a_std": torch.full((8,), 2.0),
    }
    global_rgb, local_rgb, qpos = prepare_no_wrist_batch(
        observation, (0,), stats, torch.device("cpu")
    )
    assert global_rgb.shape == (1, 3, 4, 5)
    assert local_rgb.shape == (1, 3, 4, 5)
    torch.testing.assert_close(qpos, torch.zeros_like(qpos), rtol=0, atol=0)
    chunks = torch.ones(1, 3, 8)
    expected = chunks * stats["a_std"] + stats["a_mean"]
    torch.testing.assert_close(denormalize_action_chunks(chunks, stats), expected, rtol=0, atol=0)


def test_deployment_context_rejects_privileged_future_keys():
    from stereo_core.bwa_contracts import CoreDeploymentContext

    with pytest.raises(ValueError, match="privileged"):
        CoreDeploymentContext(fixed_camera_metadata={"future_rgb": "forbidden"})
