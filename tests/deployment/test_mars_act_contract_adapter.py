from __future__ import annotations

import json
from pathlib import Path
import types

import h5py
import numpy as np
import pytest

from before_we_act.mars_action_contract import (
    ACTION_HIGH,
    ACTION_LOW,
    canonicalize_action,
)
from deployment.mars_care.act_contract_adapter import (
    ACTActionMomentAccumulator,
    act_checkpoint_contract,
    canonicalize_act_target,
    canonicalize_runtime_action,
    validate_act_checkpoint_contract,
    write_corpus_receipt,
)


def test_act_target_is_canonicalized_before_repeat_padding() -> None:
    raw = np.stack((ACTION_LOW - 1, ACTION_HIGH + 1)).astype(np.float32)
    target, mask = canonicalize_act_target(raw, horizon=5)
    expected = canonicalize_action(raw)
    assert target.shape == (5, 8)
    assert target.dtype == np.float32
    np.testing.assert_array_equal(target[:2], expected)
    np.testing.assert_array_equal(target[2:], np.repeat(expected[-1:], 3, axis=0))
    np.testing.assert_array_equal(mask, np.asarray((1, 1, 0, 0, 0), np.float32))


def test_streaming_stats_match_direct_canonical_population_moments() -> None:
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(31, 8)).astype(np.float32) * 5
    accumulator = ACTActionMomentAccumulator(std_floor=1e-4)
    accumulator.update(raw[:7], source="first")
    accumulator.update(raw[7:], source="second")
    stats, audit = accumulator.finalize()
    expected = canonicalize_action(raw)
    np.testing.assert_allclose(stats["a_mean"], expected.astype(np.float64).mean(0), atol=3e-7)
    np.testing.assert_allclose(
        stats["a_std"], np.maximum(expected.astype(np.float64).std(0), 1e-4), atol=3e-7
    )
    assert audit["raw_rows"] == len(raw)
    assert audit["raw_values"] == raw.size
    assert audit["out_of_bounds_values"] > 0
    assert audit["sources"] == 2


def test_runtime_action_requires_exact_live_bounds_then_clips() -> None:
    space = types.SimpleNamespace(low=ACTION_LOW.copy(), high=ACTION_HIGH.copy())
    result = canonicalize_runtime_action(ACTION_HIGH + 10, space)
    np.testing.assert_array_equal(result, ACTION_HIGH)
    bad = types.SimpleNamespace(low=ACTION_LOW - 0.1, high=ACTION_HIGH.copy())
    with pytest.raises(ValueError, match="bounds"):
        canonicalize_runtime_action(ACTION_HIGH, bad)


def test_checkpoint_contract_binds_action_stats() -> None:
    stats = {
        "a_mean": np.zeros(8, dtype=np.float32),
        "a_std": np.ones(8, dtype=np.float32),
    }
    metadata = act_checkpoint_contract(stats, corpus_receipt_sha256="a" * 64)
    checkpoint = {"action_contract": metadata, "stats": stats}
    assert validate_act_checkpoint_contract(checkpoint) == metadata
    checkpoint["stats"]["a_mean"] = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="statistics hash"):
        validate_act_checkpoint_contract(checkpoint)


def _write_minimal_corpus(root: Path) -> None:
    tasks = {
        "place_cube_in_cup": 2,
        "strike_cube_hard": 2,
        "three_robots_place_shoes": 3,
        "four_robots_stack_cube": 4,
    }
    for task_index, (task, arms) in enumerate(tasks.items()):
        directory = root / task / "motionplanning"
        directory.mkdir(parents=True)
        path = directory / f"{task}.shard00.h5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("traj_0")
            obs = group.create_group("obs").create_group("agent")
            actions = group.create_group("actions")
            group.create_dataset("success", data=np.asarray((True,)))
            for arm in range(arms):
                action = np.broadcast_to(
                    (ACTION_HIGH + task_index + arm + 1)[None], (2, 8)
                ).copy()
                actions.create_dataset(f"panda-{arm}", data=action)
                obs.create_group(f"panda-{arm}").create_dataset(
                    "qpos", data=np.zeros((2, 9), dtype=np.float32)
                )


def test_corpus_receipt_is_json_safe_and_records_canonical_stats(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_minimal_corpus(root)
    output = tmp_path / "receipt.json"
    receipt = write_corpus_receipt(
        root,
        output,
        expected_episodes_per_task=1,
        expected_shards_per_task=1,
        expected_local_steps=None,
    )
    assert receipt["status"] == "PASSED"
    assert receipt["episodes"] == 4
    assert receipt["local_steps"] == 22
    assert receipt["action_audit"]["out_of_bounds_values"] > 0
    assert json.loads(output.read_text()) == receipt
