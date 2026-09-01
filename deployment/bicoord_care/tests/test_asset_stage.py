from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from deployment.bicoord_care import asset_stage
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
            "copied_fields": ["contact_points_pose"],
            "benchmark_asset_source_modified": False,
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

    with pytest.raises(RuntimeAssetError, match="differs from overlay at center"):
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
