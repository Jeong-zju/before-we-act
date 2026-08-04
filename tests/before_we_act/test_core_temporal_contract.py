from __future__ import annotations

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
