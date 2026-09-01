from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.bicoord_care import asset_runtime
from deployment.bicoord_care import seed_discovery
from deployment.bicoord_care.asset_contract import (
    CONTACT_KEY,
    LEGACY_CONTACT_KEY,
    LEGACY_TRANSFORM_KEY,
    canonical_json_sha256,
    overlay_legacy_contact_metadata,
)
from deployment.bicoord_care.asset_runtime import (
    REQUIRED_ENV,
    RuntimeAssetError,
    SHOVEL_OVERLAY_ENV,
    apply_configured_task_overlay,
    apply_task_overlay,
)
from deployment.bicoord_care.config import DATASET_REPO_ID, DATASET_REVISION, TASKS
from deployment.bicoord_care.preflight import EXPECTED_BENCHMARK_COMMIT
from deployment.bicoord_care.seed_discovery import (
    RepeatedStructuralSeedError,
    STRUCTURAL_ERROR_STREAK_LIMIT,
    discover,
)
from deployment.bicoord_care.stage_common import read_json


SHOVEL_TASK = "sweep_block"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _pose(offset: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.15 + offset],
        [0.0, 0.0, 1.0, -0.6],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _legacy_shovel() -> dict[str, object]:
    return {
        "center": [0.0, 0.17515499716553293, -0.029588341057975368],
        LEGACY_CONTACT_KEY: [_pose()],
        "extents": [1.225306775101383, 0.7410235277713523, 1.716424184732401],
        "scale": [0.167, 0.167, 0.167],
        "stable": False,
        "target_pose": [_pose(0.1)],
        LEGACY_TRANSFORM_KEY: [
            [0.0007963267107332633, 0.0, -0.9999996829318346, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9999996829318346, 0.0, 0.0007963267107332633, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _dual_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    benchmark_objects = tmp_path / "benchmark" / "assets" / "objects"
    source = benchmark_objects / "082_smallshovel" / "model_data3.json"
    pristine = _legacy_shovel()
    _write_json(source, pristine)
    converted, proof = overlay_legacy_contact_metadata(pristine)

    contract_root = tmp_path / "run" / "artifacts" / "asset_contract"
    plate_overlay = contract_root / "overlay" / "003_plate" / "model_data0.json"
    shovel_overlay = (
        contract_root
        / "overlay"
        / "082_smallshovel"
        / "model_data3.json"
    )
    _write_json(
        plate_overlay,
        {"scale": [0.025, 0.025, 0.025], CONTACT_KEY: [_pose(), _pose(1.0), _pose(2.0)]},
    )
    _write_json(shovel_overlay, converted)

    mesh_rows = []
    mesh_values = {
        "collision/base3.glb": b"fixture collision mesh",
        "visual/base3.glb": b"fixture visual mesh",
    }
    for relative, payload in mesh_values.items():
        path = source.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        mesh_rows.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "bytes": len(payload),
                "sha256": _sha(path),
            }
        )

    source_sha = _sha(source)
    monkeypatch.setattr(asset_runtime, "PRISTINE_SHOVEL_METADATA_SHA256", source_sha)
    monkeypatch.setattr(
        asset_runtime, "SHOVEL_COLLISION_BYTES", len(mesh_values["collision/base3.glb"])
    )
    monkeypatch.setattr(
        asset_runtime,
        "SHOVEL_COLLISION_SHA256",
        _sha(source.parent / "collision/base3.glb"),
    )
    monkeypatch.setattr(
        asset_runtime, "SHOVEL_VISUAL_BYTES", len(mesh_values["visual/base3.glb"])
    )
    monkeypatch.setattr(
        asset_runtime,
        "SHOVEL_VISUAL_SHA256",
        _sha(source.parent / "visual/base3.glb"),
    )

    shovel_row = {
        **proof,
        "source_metadata": str(source.resolve()),
        "source_metadata_sha256": source_sha,
        "target_metadata_sha256": _sha(shovel_overlay),
        "overlay_metadata": str(shovel_overlay.resolve()),
        "pristine_source_metadata_sha256": source_sha,
        "benchmark_asset_source_modified": False,
        "mutation_scope": "run_artifact_and_actor_config_in_memory_only",
        "asset_identity": {
            "object": "082_smallshovel",
            "model_id": 3,
            "source_metadata": str(source.resolve()),
            "source_metadata_sha256": source_sha,
            "meshes": mesh_rows,
            "mesh_and_metadata_identity": "PASSED",
        },
    }
    receipt = contract_root / "asset_contract.json"
    _write_json(
        receipt,
        {
            "schema": "before-we-act.bicoord.asset-contract/1",
            "status": "PASSED",
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_revision": DATASET_REVISION,
            "benchmark_revision": EXPECTED_BENCHMARK_COMMIT,
            "tasks": list(TASKS),
            "supplemental_assets_installed": True,
            "benchmark_tracked_source_modified": False,
            "task_source_modified": False,
            "upstream_model_modified": False,
            "normalization_modified": False,
            "plate_overlay": {
                "overlay_metadata": str(plate_overlay.resolve()),
                "target_metadata_sha256": _sha(plate_overlay),
            },
            "shovel_overlay": shovel_row,
        },
    )
    return plate_overlay, shovel_overlay, receipt, pristine, converted


def test_shovel_runtime_adds_only_converted_contact_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, shovel_overlay, receipt, pristine, converted = _dual_fixture(
        tmp_path, monkeypatch
    )
    actor = SimpleNamespace(config=copy.deepcopy(pristine))
    env = SimpleNamespace(shovel=actor)

    result = apply_task_overlay(env, SHOVEL_TASK, shovel_overlay)

    expected = copy.deepcopy(pristine)
    expected[CONTACT_KEY] = converted[CONTACT_KEY]
    assert actor.config == expected
    assert pristine == _legacy_shovel()
    assert result["applied"] is True
    assert result["overlay"] == str(shovel_overlay.resolve())
    assert result["receipt"] == str(receipt.resolve())
    assert result["contact_points_pose_sha256"] == canonical_json_sha256(
        converted[CONTACT_KEY]
    )
    assert result["derived_fields"] == [CONTACT_KEY]
    assert result["source_fields"] == [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY]
    assert result["actors"] == {
        "shovel": {
            "before_sha256": canonical_json_sha256(None),
            "after_sha256": result["contact_points_pose_sha256"],
            "contact_points_pose_count": 1,
            "scale_preserved": True,
            "changed_fields": [CONTACT_KEY],
        }
    }


def test_shovel_runtime_fails_before_actor_mutation_on_source_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, shovel_overlay, receipt, pristine, _ = _dual_fixture(tmp_path, monkeypatch)
    value = read_json(receipt)
    value["shovel_overlay"]["source_metadata_sha256"] = "0" * 64
    _write_json(receipt, value)
    actor = SimpleNamespace(config=copy.deepcopy(pristine))

    with pytest.raises(RuntimeAssetError, match="source hash differs"):
        apply_task_overlay(SimpleNamespace(shovel=actor), SHOVEL_TASK, shovel_overlay)

    assert actor.config == pristine


def test_required_configured_shovel_rejects_mixed_receipts_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plate_overlay, shovel_overlay, _, pristine, _ = _dual_fixture(
        tmp_path, monkeypatch
    )
    foreign = tmp_path / "foreign" / "003_plate" / "model_data0.json"
    _write_json(foreign, read_json(plate_overlay))
    actor = SimpleNamespace(config=copy.deepcopy(pristine))
    monkeypatch.setenv(REQUIRED_ENV, "1")
    monkeypatch.setenv(SHOVEL_OVERLAY_ENV, str(shovel_overlay))
    monkeypatch.setenv(asset_runtime.PLATE_OVERLAY_ENV, str(foreign))

    with pytest.raises(RuntimeAssetError, match="do not share one asset receipt"):
        apply_configured_task_overlay(SimpleNamespace(shovel=actor), SHOVEL_TASK)

    assert actor.config == pristine


def test_sweep_setup_failure_receipt_records_explicit_overlay_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured" / "model_data3.json"
    monkeypatch.setenv(SHOVEL_OVERLAY_ENV, str(configured))

    def fail_setup(_root: Path, _task: str, _seed: int) -> object:
        raise RuntimeError("setup_demo failed before shovel overlay")

    monkeypatch.setattr(seed_discovery, "_make_env", fail_setup)
    with pytest.raises(RepeatedStructuralSeedError):
        discover(
            tmp_path,
            episodes=1,
            max_attempts=10,
            task=SHOVEL_TASK,
            progress_dir=tmp_path,
        )

    progress = read_json(tmp_path / f"progress_{SHOVEL_TASK}.json")
    assert len(progress["recent_attempts"]) == STRUCTURAL_ERROR_STREAK_LIMIT
    for row in progress["recent_attempts"]:
        assert row["asset_overlay"] == {
            "task": SHOVEL_TASK,
            "applied": False,
            "reason": "environment_construction_failed_before_overlay_receipt",
            "overlay": str(configured),
            "contact_points_pose_sha256": None,
            "receipt": None,
            "receipt_sha256": None,
            "actors": {},
            "derived_fields": [CONTACT_KEY],
            "source_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
            "legacy_conversion": True,
        }


def test_sweep_success_attempt_preserves_complete_runtime_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = {
        "task": SHOVEL_TASK,
        "applied": True,
        "overlay": "/run/overlay/082_smallshovel/model_data3.json",
        "receipt": "/run/asset_contract.json",
        "receipt_sha256": "a" * 64,
        "contact_points_pose_sha256": "b" * 64,
        "actors": {
            "shovel": {
                "before_sha256": "c" * 64,
                "after_sha256": "b" * 64,
                "contact_points_pose_count": 1,
                "scale_preserved": True,
                "changed_fields": [CONTACT_KEY],
            }
        },
        "derived_fields": [CONTACT_KEY],
        "source_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
        "legacy_conversion": True,
        "task_source_modified": False,
    }

    class SuccessEnv:
        plan_success = True
        stage_eval_score = 1.0
        _bicoord_asset_overlay = overlay

        def play_once(self) -> dict[str, bool]:
            return {"official_expert": True}

        def check_success(self) -> bool:
            return True

        def close_env(self) -> None:
            pass

    monkeypatch.setattr(seed_discovery, "_make_env", lambda *_args: SuccessEnv())
    manifest = discover(
        tmp_path,
        episodes=1,
        max_attempts=1,
        task=SHOVEL_TASK,
        progress_dir=tmp_path,
    )

    row = manifest["attempts"][SHOVEL_TASK][0]
    assert row["asset_overlay"] == overlay
    seed_receipt = read_json(manifest["seed_receipts"][SHOVEL_TASK][0])
    assert seed_receipt["row"]["asset_overlay"] == overlay
