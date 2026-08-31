from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deployment.bicoord_care.branch_collection import (
    BRANCHES_PER_FAMILY,
    CANDIDATES,
    HORIZONS,
    _targets,
    candidate_plan,
)
from deployment.bicoord_care.config import ACTION_DIM, ACTION_HORIZON, TASKS
from deployment.bicoord_care.paired_evaluate import (
    TRAINING_FORMAT,
    _load_care,
    _read_completed_pair,
    _write_pair,
)
from deployment.bicoord_care.prepare_branches import _records
from deployment.bicoord_care.select_calibrate import finite_sample_quantile
from deployment.bicoord_care.train_belief import (
    ETA_MIN,
    LEARNING_RATE,
    fold_assignment_receipt,
    fold_assignments,
)


def test_candidate_transforms_preserve_native_shape_without_clipping() -> None:
    reference = np.linspace(-3.0, 4.0, ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(ACTION_HORIZON, ACTION_DIM)
    base = reference + np.float32(5.0)
    current = np.linspace(-2.0, 2.0, ACTION_DIM, dtype=np.float32)
    rows = [candidate_plan(candidate, reference, base, current) for candidate in range(CANDIDATES)]
    assert all(row.shape == (ACTION_HORIZON, ACTION_DIM) for row in rows)
    assert np.array_equal(rows[0], reference)
    assert np.array_equal(rows[1], base)
    # Values outside a nominal [-1, 1] population range survive unchanged;
    # the candidate generator is not an action projector.
    assert float(rows[0].max()) == pytest.approx(float(reference.max()))
    assert float(rows[1].max()) == pytest.approx(float(base.max()))
    assert np.all(rows[5][:, -1] == current[-1])


def _branch(candidate: int, regime: str, repeat: int) -> dict[str, object]:
    baseline = float(repeat)
    direct = 0.1 * candidate
    response = 0.01 * candidate
    utility = baseline + (direct if regime == "replay" else direct + response)
    return {
        "candidate_id": candidate,
        "regime": regime,
        "repeat_id": repeat,
        "physical_simulator_outcome": True,
        "outcomes": {
            str(horizon): {
                "utility_main": utility,
                "hard_safety_violation": bool(candidate == 5 and regime == "reactive"),
            }
            for horizon in HORIZONS
        },
    }


def test_physical_family_has_exactly_24_keys_and_target_decomposition() -> None:
    branches = [
        _branch(candidate, regime, repeat)
        for candidate in range(CANDIDATES)
        for regime in ("reactive", "replay")
        for repeat in (0, 1)
    ]
    assert len(branches) == BRANCHES_PER_FAMILY == 24
    targets, safety, usable = _targets(branches)
    assert targets.shape == (4, 6, 2, 3)
    assert np.allclose(targets[..., 2], targets[..., 0] + targets[..., 1])
    assert np.array_equal(targets[:, 0], np.zeros_like(targets[:, 0]))
    assert safety[:, 5].all() and usable.all()
    with pytest.raises(RuntimeError, match="24 unique"):
        _targets(branches[:-1])


def test_oof_folds_are_balanced_within_each_task_and_hash_stable() -> None:
    task_ids = torch.tensor([task for task in range(len(TASKS)) for _ in range(5)])
    prepared = {"task_id": task_ids}
    folds = fold_assignments(prepared)
    assert folds.tolist() == [0, 1, 2, 0, 1] * len(TASKS)
    for task in range(len(TASKS)):
        counts = torch.bincount(folds[task_ids == task], minlength=3)
        assert int(counts.max() - counts.min()) <= 1
    assert fold_assignment_receipt(prepared) == fold_assignment_receipt(prepared)
    assert LEARNING_RATE == pytest.approx(3e-4)
    assert ETA_MIN == pytest.approx(3e-6)


def test_finite_sample_conformal_quantile_uses_n_plus_one_correction() -> None:
    # n=19, coverage=.9 -> ceil(20*.9)=18th order statistic.
    assert finite_sample_quantile(np.arange(19), 0.9) == 17.0
    # n=9 has no spare point at 90%, so the finite-sample rule returns max.
    assert finite_sample_quantile(np.arange(9), 0.9) == 8.0
    with pytest.raises(ValueError):
        finite_sample_quantile([], 0.9)


def test_paired_seed_resume_is_hash_bound_and_fails_closed(tmp_path: Path) -> None:
    identity = {
        "task": TASKS[0],
        "seed": 100000,
        "max_steps": 300,
        "reference_checkpoint_sha256": "a" * 64,
        "care_checkpoint_sha256": "b" * 64,
        "seed_manifest_sha256": "c" * 64,
        "normalization_sha256": "d" * 64,
        "operation": "validation20-paired",
    }
    row = {
        "task": TASKS[0],
        "seed": 100000,
        "paired": True,
        "selector_off": {"success": False},
        "care": {"success": True},
    }
    for mode in ("selector_off", "care"):
        progress = tmp_path / f"{mode}.jsonl"
        progress.write_text(json.dumps({"mode": mode}) + "\n")
        import hashlib
        row[f"{mode}_progress"] = str(progress.resolve())
        row[f"{mode}_progress_sha256"] = hashlib.sha256(progress.read_bytes()).hexdigest()
    path = tmp_path / "seed_100000.json"
    _write_pair(path, row, identity=identity)
    loaded = _read_completed_pair(path, identity=identity)
    assert loaded["care"]["success"] is True
    with pytest.raises(RuntimeError, match="provenance differs"):
        _read_completed_pair(path, identity={**identity, "care_checkpoint_sha256": "e" * 64})
    value = json.loads(path.read_text())
    value["selector_off"] = None
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="payload hash differs"):
        _read_completed_pair(path, identity=identity)
    _write_pair(path, {**row, "selector_off": None}, identity=identity)
    with pytest.raises(RuntimeError, match="incomplete"):
        _read_completed_pair(path, identity=identity)


def test_smoke_care_checkpoint_must_bind_exact_bcore_reference(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "belief_smoke.pt"
    torch.save(
        {
            "format_version": TRAINING_FORMAT,
            "benchmark_adapter": "BiCoord",
            "method_family": "CARE",
            "policy_family": "CAREBeliefHead",
            "reference_policy": "B-core/TUNE",
            "reference_checkpoint_sha256": "a" * 64,
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="not bound"):
        _load_care(
            checkpoint,
            device=torch.device("cpu"),
            reference_checkpoint_sha256="b" * 64,
            action_std=np.ones(ACTION_DIM, dtype=np.float32),
            formal=False,
        )


def test_branch_preparation_rejects_smoke_manifest_in_formal_namespace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts" / "branches"
    shard = root / "rank_0" / "manifest.json"
    shard.parent.mkdir(parents=True)
    shard.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "provider_policy": "B-core/TUNE",
                "physical_simulator_outcomes": True,
                "offline_demonstration_error_used": False,
                "smoke": True,
                "records": [],
            }
        )
    )
    with pytest.raises(ValueError, match="smoke/formal provenance differs"):
        _records([shard], smoke=False, root=root)


def test_branch_preparation_rejects_artifacts_outside_selected_namespace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts" / "branches"
    shard = root / "rank_0" / "manifest.json"
    shard.parent.mkdir(parents=True)
    outside_npz = tmp_path / "artifacts" / "branch_smoke" / "family.npz"
    outside_json = tmp_path / "artifacts" / "branch_smoke" / "family.json"
    outside_npz.parent.mkdir(parents=True)
    outside_npz.write_bytes(b"smoke")
    outside_json.write_text("{}")
    shard.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "provider_policy": "B-core/TUNE",
                "physical_simulator_outcomes": True,
                "offline_demonstration_error_used": False,
                "smoke": False,
                "records": [
                    {
                        "npz": str(outside_npz),
                        "npz_sha256": "0" * 64,
                        "manifest": str(outside_json),
                        "manifest_sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="escapes namespace"):
        _records([shard], smoke=False, root=root)
