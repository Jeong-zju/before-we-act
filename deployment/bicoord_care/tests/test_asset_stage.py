from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from deployment.bicoord_care import asset_contract, asset_stage
from deployment.bicoord_care.asset_runtime import (
    RuntimeAssetError,
    apply_task_overlay,
)
from deployment.bicoord_care.config import DATASET_REPO_ID, DATASET_REVISION, TASKS
from deployment.bicoord_care.preflight import EXPECTED_BENCHMARK_COMMIT


def _pose(offset: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, offset],
        [0.0, 0.0, 1.0, -offset],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _plate(*, contacts: list[object]) -> dict[str, object]:
    return {
        "center": [0.0, 0.0, 0.0],
        "scale": [0.025, 0.025, 0.025],
        "contact_points_pose": contacts,
        "functional_matrix": [_pose(0.1)],
        "target_pose": [_pose(0.2)],
    }


def _legacy_shovel() -> dict[str, object]:
    return {
        "center": [0.0, 0.17515499716553293, -0.029588341057975368],
        "contact_pose": [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.15],
                [0.0, 0.0, 1.0, -0.6],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        "extents": [1.225306775101383, 0.7410235277713523, 1.716424184732401],
        "scale": [0.167, 0.167, 0.167],
        "stable": False,
        "target_pose": [_pose(0.8)],
        "trans_matrix": [
            [0.0007963267107332633, 0.0, -0.9999996829318346, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9999996829318346, 0.0, 0.0007963267107332633, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _overlay_fixture(tmp_path: Path, value: dict[str, object]) -> Path:
    overlay = (
        tmp_path
        / "asset_contract"
        / "overlay"
        / "003_plate"
        / "model_data0.json"
    )
    overlay.parent.mkdir(parents=True)
    overlay.write_text(json.dumps(value), encoding="utf-8")
    pristine = (
        tmp_path
        / "benchmark"
        / "assets"
        / "objects"
        / "003_plate"
        / "model_data0.json"
    )
    pristine.parent.mkdir(parents=True)
    pristine.write_text(json.dumps(_plate(contacts=[])), encoding="utf-8")
    shovel_overlay = (
        tmp_path
        / "asset_contract"
        / "overlay"
        / "082_smallshovel"
        / "model_data3.json"
    )
    shovel_overlay.parent.mkdir(parents=True)
    shovel_overlay.write_text("{}\n", encoding="utf-8")
    contact_hash = hashlib.sha256(
        json.dumps(
            value["contact_points_pose"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    receipt = {
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
            "overlay_metadata": str(overlay.resolve()),
            "target_metadata_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
            "target_contact_points_pose_sha256": contact_hash,
            "source_small_metadata": str(pristine.resolve()),
            "source_small_metadata_sha256": hashlib.sha256(
                pristine.read_bytes()
            ).hexdigest(),
            "pristine_small_metadata_sha256": hashlib.sha256(
                pristine.read_bytes()
            ).hexdigest(),
            "copied_fields": ["contact_points_pose"],
            "benchmark_asset_source_modified": False,
            "mutation_scope": "run_artifact_and_actor_config_in_memory_only",
        },
        "shovel_overlay": {
            "overlay_metadata": str(shovel_overlay.resolve()),
            "target_metadata_sha256": hashlib.sha256(
                shovel_overlay.read_bytes()
            ).hexdigest(),
        },
    }
    (tmp_path / "asset_contract" / "asset_contract.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return overlay


def test_runtime_overlay_changes_only_contact_metadata(tmp_path: Path) -> None:
    contacts = [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)]
    overlay = _overlay_fixture(tmp_path, _plate(contacts=contacts))
    first = SimpleNamespace(config=_plate(contacts=[]))
    second = SimpleNamespace(config=_plate(contacts=[]))
    env = SimpleNamespace(plate=first, plate_2=second)

    result = apply_task_overlay(env, "place_plate_and_cup", overlay)

    assert result["applied"] is True
    assert result["copied_fields"] == ["contact_points_pose"]
    assert first.config["contact_points_pose"] == contacts
    assert second.config["contact_points_pose"] == contacts
    assert first.config["scale"] == [0.025, 0.025, 0.025]


def test_runtime_overlay_rejects_non_contact_drift(tmp_path: Path) -> None:
    value = _plate(contacts=[_pose(0.1), _pose(0.2), _pose(0.3)])
    value["center"] = [1.0, 0.0, 0.0]
    overlay = _overlay_fixture(tmp_path, value)
    actor = SimpleNamespace(config=_plate(contacts=[]))
    env = SimpleNamespace(plate=actor, plate_2=SimpleNamespace(config=_plate(contacts=[])))

    with pytest.raises(
        RuntimeAssetError,
        match="(?:differs from overlay|overlay differs from pristine source) at center",
    ):
        apply_task_overlay(env, "place_plate_and_cup", overlay)


def test_runtime_overlay_rejects_non_object_stage_receipt(tmp_path: Path) -> None:
    contacts = [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)]
    overlay = _overlay_fixture(tmp_path, _plate(contacts=contacts))
    receipt = tmp_path / "asset_contract" / "asset_contract.json"
    receipt.write_text("[]\n", encoding="utf-8")
    env = SimpleNamespace(
        plate=SimpleNamespace(config=_plate(contacts=[])),
        plate_2=SimpleNamespace(config=_plate(contacts=[])),
    )

    with pytest.raises(RuntimeAssetError, match="receipt is not a JSON object"):
        apply_task_overlay(env, "place_plate_and_cup", overlay)


def test_runtime_overlay_rejects_symbolic_stage_receipt(tmp_path: Path) -> None:
    contacts = [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)]
    overlay = _overlay_fixture(tmp_path, _plate(contacts=contacts))
    receipt = tmp_path / "asset_contract" / "asset_contract.json"
    outside = tmp_path / "outside-receipt.json"
    receipt.rename(outside)
    receipt.symlink_to(outside)
    env = SimpleNamespace(
        plate=SimpleNamespace(config=_plate(contacts=[])),
        plate_2=SimpleNamespace(config=_plate(contacts=[])),
    )

    with pytest.raises(RuntimeAssetError, match="receipt must not be a symlink"):
        apply_task_overlay(env, "place_plate_and_cup", overlay)


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("dataset_repo_id", "other/dataset"),
        ("dataset_revision", "f" * 40),
        ("benchmark_revision", "e" * 40),
        ("tasks", ["place_plate_and_cup"]),
        ("supplemental_assets_installed", False),
        ("benchmark_tracked_source_modified", True),
        ("task_source_modified", True),
        ("upstream_model_modified", True),
        ("normalization_modified", True),
    ],
)
def test_runtime_overlay_rejects_stage_identity_drift(
    tmp_path: Path, field: str, drifted: object
) -> None:
    contacts = [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)]
    overlay = _overlay_fixture(tmp_path, _plate(contacts=contacts))
    receipt_path = tmp_path / "asset_contract" / "asset_contract.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = drifted
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    env = SimpleNamespace(
        plate=SimpleNamespace(config=_plate(contacts=[])),
        plate_2=SimpleNamespace(config=_plate(contacts=[])),
    )

    with pytest.raises(RuntimeAssetError, match="stage receipt/hash differs"):
        apply_task_overlay(env, "place_plate_and_cup", overlay)


def test_plate_stage_rejects_an_already_modified_source_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contacts = [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)]
    objects = tmp_path / "benchmark" / "assets" / "objects"
    small = objects / "003_plate" / "model_data0.json"
    donor = objects / "003_plate_large" / "model_data0.json"
    small.parent.mkdir(parents=True)
    donor.parent.mkdir(parents=True)
    # Simulate a prior in-place workaround.  Even though the contact values
    # are usable, the formal stage must start from the exact released record
    # and create its own run-local overlay.
    small.write_text(json.dumps(_plate(contacts=contacts)), encoding="utf-8")
    donor.write_text(json.dumps(_plate(contacts=contacts)), encoding="utf-8")
    monkeypatch.setattr(
        asset_stage,
        "DONOR_METADATA_SHA256",
        hashlib.sha256(donor.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        asset_stage,
        "PRISTINE_SMALL_METADATA_SHA256",
        hashlib.sha256(json.dumps(_plate(contacts=[])).encode()).hexdigest(),
    )

    with pytest.raises(
        asset_stage.AssetStageError,
        match="not the pristine released record",
    ):
        asset_stage._plate_overlay(
            tmp_path / "benchmark", tmp_path / "run" / "overlay"
        )


def test_shovel_stage_builds_a_run_local_derived_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = tmp_path / "benchmark"
    object_root = benchmark / "assets" / "objects" / "082_smallshovel"
    metadata = object_root / "model_data3.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps(_legacy_shovel()), encoding="utf-8")
    monkeypatch.setattr(
        asset_contract,
        "PRISTINE_SHOVEL_METADATA_SHA256",
        hashlib.sha256(metadata.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        asset_stage,
        "PRISTINE_SHOVEL_METADATA_SHA256",
        hashlib.sha256(metadata.read_bytes()).hexdigest(),
    )
    for relative, payload, size_name, hash_name in (
        (
            "collision/base3.glb",
            b"official-collision-fixture",
            "SHOVEL_COLLISION_BYTES",
            "SHOVEL_COLLISION_SHA256",
        ),
        (
            "visual/base3.glb",
            b"official-visual-fixture",
            "SHOVEL_VISUAL_BYTES",
            "SHOVEL_VISUAL_SHA256",
        ),
    ):
        path = object_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        monkeypatch.setattr(asset_contract, size_name, len(payload))
        monkeypatch.setattr(
            asset_contract, hash_name, hashlib.sha256(payload).hexdigest()
        )
    expected_overlay, proof = asset_contract.overlay_legacy_contact_metadata(
        _legacy_shovel()
    )
    monkeypatch.setattr(
        asset_stage,
        "SHOVEL_CONTACT_POINTS_POSE_SHA256",
        proof["contact_points_pose_sha256"],
    )
    monkeypatch.setattr(
        asset_stage,
        "SHOVEL_OVERLAY_METADATA_CANONICAL_SHA256",
        asset_contract.canonical_json_sha256(expected_overlay),
    )
    source_before = metadata.read_bytes()
    result = asset_stage._shovel_overlay(
        benchmark, tmp_path / "run" / "asset_contract" / "overlay"
    )

    overlay = Path(result["overlay_metadata"])
    value = json.loads(overlay.read_text(encoding="utf-8"))
    assert metadata.read_bytes() == source_before
    assert result["modelname"] == "082_smallshovel"
    assert result["model_id"] == 3
    assert result["added_fields"] == ["contact_points_pose"]
    assert result["derived_fields"] == ["contact_points_pose"]
    assert result["source_fields"] == ["contact_pose", "trans_matrix"]
    assert result["contact_points_pose_count"] == 1
    assert result["max_scale_equivalence_error"] <= 1e-12
    assert result["mutation_scope"] == (
        "run_artifact_and_actor_config_in_memory_only"
    )
    assert set(value) == set(_legacy_shovel()) | {"contact_points_pose"}
    assert value["contact_pose"] == _legacy_shovel()["contact_pose"]
    assert value["trans_matrix"] == _legacy_shovel()["trans_matrix"]
    assert value["scale"] == [0.167, 0.167, 0.167]


def test_supplemental_installer_is_safe_exact_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "objects.zip"
    roots = ("plate", "cup")
    files = {
        "plate/model.json": b'{"plate": true}\n',
        "cup/model.json": b'{"cup": true}\n',
    }
    with zipfile.ZipFile(archive, "w") as stream:
        for root in roots:
            stream.writestr(f"{root}/", b"")
        for name, payload in files.items():
            stream.writestr(name, payload)
    monkeypatch.setattr(asset_stage, "BICOORD_OBJECT_ROOTS", roots)
    monkeypatch.setattr(asset_stage, "BICOORD_OBJECT_MEMBERS", 4)
    monkeypatch.setattr(asset_stage, "BICOORD_OBJECT_FILES", 2)
    destination = tmp_path / "assets" / "objects"

    first = asset_stage.install_supplemental_archive(archive, destination)
    second = asset_stage.install_supplemental_archive(archive, destination)

    assert first["files_changed"] == 2
    assert second["files_changed"] == 0
    assert first["roots"] == sorted(roots)
    assert {
        row["member"]: row["sha256"] for row in second["files"]
    } == {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in files.items()
    }


def test_supplemental_installer_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "objects.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape", b"no")
    monkeypatch.setattr(asset_stage, "BICOORD_OBJECT_ROOTS", ("plate",))
    monkeypatch.setattr(asset_stage, "BICOORD_OBJECT_MEMBERS", 1)
    monkeypatch.setattr(asset_stage, "BICOORD_OBJECT_FILES", 1)

    with pytest.raises(asset_stage.AssetStageError, match="unsafe BiCoord ZIP member"):
        asset_stage.install_supplemental_archive(
            archive, tmp_path / "assets" / "objects"
        )
    assert not (tmp_path / "escape").exists()
