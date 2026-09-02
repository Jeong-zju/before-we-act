from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest

from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    action_contract_hash,
    audit_action_array,
    canonicalize_action,
    canonicalize_normalized_action,
    checkpoint_action_contract,
    contract_metadata,
    validate_action_space_bounds,
    validate_checkpoint_action_contract,
)


def test_bounds_are_immutable_and_have_the_frozen_eight_dimensional_layout() -> None:
    assert ACTION_DIM == 8
    assert ACTION_LOW.dtype == np.float32
    assert ACTION_HIGH.dtype == np.float32
    assert ACTION_LOW.shape == ACTION_HIGH.shape == (8,)
    np.testing.assert_array_equal(
        ACTION_LOW,
        np.asarray(
            (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        ACTION_HIGH,
        np.asarray(
            (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0),
            dtype=np.float32,
        ),
    )
    assert not ACTION_LOW.flags.writeable
    assert not ACTION_HIGH.flags.writeable


def test_canonicalization_is_finite_shape_checked_idempotent_and_auditable() -> None:
    midpoint = (ACTION_LOW.astype(np.float64) + ACTION_HIGH.astype(np.float64)) / 2
    raw = np.broadcast_to(midpoint, (2, 3, ACTION_DIM)).copy()
    raw[0, 0] = ACTION_LOW.astype(np.float64) - 1.0
    raw[1, 2] = ACTION_HIGH.astype(np.float64) + 1.0
    canonical, report = audit_action_array(raw)
    assert canonical.dtype == np.float32
    assert canonical.shape == raw.shape
    assert report["raw_values"] == 6 * ACTION_DIM
    assert report["changed_values"] == 2 * ACTION_DIM
    assert report["out_of_bounds_values"] == 2 * ACTION_DIM
    assert report["max_abs_change"] > 0
    np.testing.assert_array_equal(canonical, canonicalize_action(canonical))
    assert report["contract_version"] == ACTION_CONTRACT_VERSION

    with pytest.raises(ValueError, match="finite"):
        canonicalize_action(np.full((ACTION_DIM,), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="width"):
        canonicalize_action(np.zeros((7,), dtype=np.float32))


def test_normalized_decode_clip_reencode_uses_the_same_contract() -> None:
    mean = np.arange(ACTION_DIM, dtype=np.float32)
    std = np.full(ACTION_DIM, 2.0, dtype=np.float32)
    normalized = np.full((4, ACTION_DIM), 100.0, dtype=np.float32)
    result = canonicalize_normalized_action(normalized, mean, std)
    expected = (ACTION_HIGH - mean) / std
    np.testing.assert_array_equal(result, np.broadcast_to(expected, result.shape))
    assert np.array_equal(result, canonicalize_normalized_action(result, mean, std))
    with pytest.raises(ValueError, match="standard deviation"):
        canonicalize_normalized_action(normalized, mean, np.zeros(ACTION_DIM))


def test_contract_hash_and_metadata_are_stable_and_json_serializable() -> None:
    metadata = contract_metadata()
    assert metadata["version"] == ACTION_CONTRACT_VERSION
    assert metadata["sha256"] == action_contract_hash()
    assert metadata["action_dim"] == ACTION_DIM
    assert metadata["encoding"] == "absolute_pd_joint_pos"
    assert json.loads(json.dumps(metadata)) == metadata
    assert action_contract_hash() == action_contract_hash()


def test_frozen_config_points_to_the_code_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/before_we_act/mars_action_contract_v1.json").read_text()
    )
    assert config["status"] == "frozen"
    assert config["contract_version"] == ACTION_CONTRACT_VERSION
    assert config["contract_sha256"] == action_contract_hash()
    assert config["per_arm_action_dimension"] == ACTION_DIM
    assert config["canonicalization"]["raw_hdf5_policy"].startswith("immutable")


def test_live_action_space_bounds_must_match_exactly() -> None:
    space = types.SimpleNamespace(low=ACTION_LOW.copy(), high=ACTION_HIGH.copy())
    assert validate_action_space_bounds(space) is True
    bad = types.SimpleNamespace(low=ACTION_LOW.copy(), high=ACTION_HIGH.copy())
    bad.high[0] = np.nextafter(bad.high[0], np.float32(np.inf))
    with pytest.raises(ValueError, match="bounds"):
        validate_action_space_bounds(bad)


def test_checkpoint_contract_is_explicit_and_fail_closed() -> None:
    metadata = checkpoint_action_contract()
    assert metadata["version"] == ACTION_CONTRACT_VERSION
    assert metadata["sha256"] == action_contract_hash()
    payload = {"action_contract": metadata}
    assert validate_checkpoint_action_contract(payload) == metadata
    with pytest.raises(ValueError, match="missing"):
        validate_checkpoint_action_contract({})
    wrong = {"action_contract": dict(metadata, sha256="0" * 64)}
    with pytest.raises(ValueError, match="hash"):
        validate_checkpoint_action_contract(wrong)
