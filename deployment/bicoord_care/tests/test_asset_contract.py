from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from deployment.bicoord_care.asset_contract import (
    AssetContractError,
    CONTACT_KEY,
    MODEL_METADATA_NAME,
    SCHEMA,
    apply_contact_points_overlay,
    main,
    overlay_metadata,
    sha256_file,
    validate_asset_pair,
)


def _pose(offset: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, offset],
        [0.0, 0.0, 1.0, -offset],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _metadata(*, scale: float, contacts: list[object]) -> dict[str, object]:
    return {
        "center": [-0.00035, 0.5594, -0.00018],
        "contact_points_discription": [],
        "contact_points_group": [],
        "contact_points_mask": [],
        CONTACT_KEY: contacts,
        "extents": [9.18, 1.11, 9.20],
        "functional_matrix": [_pose(0.123)],
        "functional_point_discription": [""],
        "orientation_point": [],
        "orientation_point_discription": [""],
        "scale": [scale, scale, scale],
        "stable": True,
        "target_point_discription": ["The center point of the plate"],
        "target_pose": [_pose(0.0743)],
        "transform_matrix": _pose(0.0),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _objects(root: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    small = root / "003_plate"
    large = root / "003_plate_large"
    small_metadata = _metadata(scale=0.025, contacts=[])
    large_metadata = _metadata(
        scale=0.035,
        contacts=[_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)],
    )
    _write_json(small / MODEL_METADATA_NAME, small_metadata)
    _write_json(large / MODEL_METADATA_NAME, large_metadata)
    for relative, payload in (
        (Path("collision/base0.glb"), b"same collision mesh"),
        (Path("visual/base0.glb"), b"same visual mesh"),
    ):
        for directory in (small, large):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return small, large, small_metadata, large_metadata


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_validate_pair_records_metadata_scales_and_equal_mesh_hashes(tmp_path: Path) -> None:
    small, large, _, _ = _objects(tmp_path)

    result = validate_asset_pair(small, large)

    assert result["schema"] == SCHEMA
    assert result["status"] == "PASSED"
    assert result["overlay"] == "contact_points_pose_only"
    assert result["small_scale"] == [0.025, 0.025, 0.025]
    assert result["large_scale"] == [0.035, 0.035, 0.035]
    assert result["small_contact_points_pose_count"] == 0
    assert result["large_contact_points_pose_count"] == 4
    assert result["mesh_hashes_equal"] is True
    assert [row["relative_path"] for row in result["small_meshes"]] == [
        "collision/base0.glb",
        "visual/base0.glb",
    ]
    assert [row["sha256"] for row in result["small_meshes"]] == [
        row["sha256"] for row in result["large_meshes"]
    ]
    assert result["small_metadata_sha256"] == sha256_file(
        small / MODEL_METADATA_NAME
    )


def test_in_memory_overlay_is_contact_only_and_does_not_mutate_inputs() -> None:
    small = _metadata(scale=0.025, contacts=[])
    large = _metadata(
        scale=0.035,
        contacts=[_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)],
    )
    small_before = copy.deepcopy(small)
    large_before = copy.deepcopy(large)

    result = overlay_metadata(small, large)

    expected = copy.deepcopy(small)
    expected[CONTACT_KEY] = large[CONTACT_KEY]
    assert result == expected
    assert result["scale"] == [0.025, 0.025, 0.025]
    assert small == small_before
    assert large == large_before
    assert result is not small
    assert result[CONTACT_KEY] is not large[CONTACT_KEY]


def test_in_place_overlay_changes_only_contact_field_and_is_idempotent(tmp_path: Path) -> None:
    small, large, small_metadata, large_metadata = _objects(tmp_path)
    small_meshes_before = {
        relative: sha256_file(small / relative)
        for relative in ("collision/base0.glb", "visual/base0.glb")
    }

    first = apply_contact_points_overlay(
        small / MODEL_METADATA_NAME,
        large / MODEL_METADATA_NAME,
    )
    after_first_bytes = (small / MODEL_METADATA_NAME).read_bytes()
    second = apply_contact_points_overlay(
        small / MODEL_METADATA_NAME,
        large / MODEL_METADATA_NAME,
    )

    overlaid = json.loads((small / MODEL_METADATA_NAME).read_text(encoding="utf-8"))
    expected = copy.deepcopy(small_metadata)
    expected[CONTACT_KEY] = large_metadata[CONTACT_KEY]
    assert overlaid == expected
    assert overlaid["scale"] == [0.025, 0.025, 0.025]
    assert first["changed"] is True
    assert first["small_scale_preserved"] is True
    assert first["contact_points_pose_count"] == 4
    assert second["changed"] is False
    assert second["idempotent"] is True
    assert (small / MODEL_METADATA_NAME).read_bytes() == after_first_bytes
    assert {
        relative: sha256_file(small / relative)
        for relative in ("collision/base0.glb", "visual/base0.glb")
    } == small_meshes_before


def test_explicit_output_keeps_benchmark_metadata_untouched(tmp_path: Path) -> None:
    small, large, small_metadata, large_metadata = _objects(tmp_path)
    before = _tree_hash(tmp_path)
    output = tmp_path / "overlay" / MODEL_METADATA_NAME

    result = apply_contact_points_overlay(
        small / MODEL_METADATA_NAME,
        large / MODEL_METADATA_NAME,
        output_path=output,
    )

    assert json.loads((small / MODEL_METADATA_NAME).read_text()) == small_metadata
    expected = copy.deepcopy(small_metadata)
    expected[CONTACT_KEY] = large_metadata[CONTACT_KEY]
    assert json.loads(output.read_text()) == expected
    assert result["changed"] is True
    assert result["target_metadata_before_sha256"] is None
    assert result["target_metadata_sha256"] == sha256_file(output)
    after_without_overlay = {
        key: value
        for key, value in _tree_hash(tmp_path).items()
        if not key.startswith("overlay/")
    }
    assert after_without_overlay == before


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("mesh", "GLB hash mismatch"),
        ("small_scale", "small scale drift"),
        ("large_scale", "large scale drift"),
        ("metadata", "metadata invariant drift"),
        ("conflicting_contacts", "differs from the reference"),
        ("missing_contacts", "small metadata lacks contact_points_pose"),
    ],
)
def test_contract_drift_fails_before_any_write(
    tmp_path: Path, mutation: str, match: str
) -> None:
    small, large, _, _ = _objects(tmp_path)
    if mutation == "mesh":
        (large / "collision/base0.glb").write_bytes(b"different")
    else:
        path = small / MODEL_METADATA_NAME
        metadata = json.loads(path.read_text())
        if mutation == "small_scale":
            metadata["scale"] = [0.035, 0.035, 0.035]
        elif mutation == "large_scale":
            path = large / MODEL_METADATA_NAME
            metadata = json.loads(path.read_text())
            metadata["scale"] = [0.025, 0.025, 0.025]
        elif mutation == "metadata":
            metadata["center"] = [1, 2, 3]
        elif mutation == "conflicting_contacts":
            metadata[CONTACT_KEY] = [_pose(9.0), _pose(8.0), _pose(7.0)]
        elif mutation == "missing_contacts":
            del metadata[CONTACT_KEY]
        _write_json(path, metadata)
    before = _tree_hash(tmp_path)

    with pytest.raises(AssetContractError, match=match):
        apply_contact_points_overlay(
            small / MODEL_METADATA_NAME,
            large / MODEL_METADATA_NAME,
        )

    assert _tree_hash(tmp_path) == before


def test_invalid_reference_pose_fails_closed(tmp_path: Path) -> None:
    small, large, _, _ = _objects(tmp_path)
    large_path = large / MODEL_METADATA_NAME
    metadata = json.loads(large_path.read_text())
    metadata[CONTACT_KEY][2][3] = [0.0, 0.0, 0.0, 0.0]
    _write_json(large_path, metadata)
    before = _tree_hash(tmp_path)

    with pytest.raises(AssetContractError, match="invalid homogeneous row"):
        apply_contact_points_overlay(
            small / MODEL_METADATA_NAME,
            large / MODEL_METADATA_NAME,
        )

    assert _tree_hash(tmp_path) == before


def test_cli_writes_hashed_receipt_without_binary_assets(tmp_path: Path) -> None:
    objects = tmp_path / "objects"
    small, _, _, _ = _objects(objects)
    receipt = tmp_path / "receipt.json"

    assert main(["--assets-root", str(objects), "--receipt", str(receipt)]) == 0

    value = json.loads(receipt.read_text())
    assert value["schema"] == SCHEMA
    assert value["status"] == "PASSED"
    assert value["target_metadata_sha256"] == sha256_file(
        small / MODEL_METADATA_NAME
    )
    assert value["contact_points_pose_count"] == 4
    assert not list(tmp_path.rglob("*.tmp"))
