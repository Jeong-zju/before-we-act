from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from deployment.bicoord_care.asset_contract import (
    AssetContractError,
    CONTACT_KEY,
    LEGACY_CONTACT_KEY,
    LEGACY_TRANSFORM_KEY,
    MODEL_METADATA_NAME,
    SCHEMA,
    apply_contact_points_overlay,
    apply_legacy_contact_overlay,
    legacy_contact_points_pose,
    main,
    overlay_legacy_contact_metadata,
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


def _legacy_shovel_metadata(
    *,
    scale: float = 0.167,
    contact_pose: list[object] | None = None,
    trans_matrix: list[list[float]] | None = None,
) -> dict[str, object]:
    return {
        "center": [0.0, 0.17515499716553293, -0.029588341057975368],
        LEGACY_CONTACT_KEY: contact_pose
        if contact_pose is not None
        else [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.15],
                [0.0, 0.0, 1.0, -0.6],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        "extents": [1.225306775101383, 0.7410235277713523, 1.716424184732401],
        "scale": [scale, scale, scale],
        "stable": False,
        "target_pose": [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.15],
                [0.0, 0.0, 1.0, 0.8],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        LEGACY_TRANSFORM_KEY: trans_matrix
        if trans_matrix is not None
        else [
            [0.0007963267107332633, -0.0, -0.9999996829318346, 0.0],
            [0.0, 1.0, -0.0, 0.0],
            [0.9999996829318346, 0.0, 0.0007963267107332633, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
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


def test_low_level_pair_allows_arbitrary_directories_but_optional_name_gate_does_not(
    tmp_path: Path,
) -> None:
    small, large, _, _ = _objects(tmp_path)
    arbitrary_small = tmp_path / "small-fixture"
    arbitrary_large = tmp_path / "large-fixture"
    small.rename(arbitrary_small)
    large.rename(arbitrary_large)

    assert validate_asset_pair(arbitrary_small, arbitrary_large)["status"] == "PASSED"
    with pytest.raises(AssetContractError, match="must be named 003_plate"):
        validate_asset_pair(
            arbitrary_small,
            arbitrary_large,
            require_canonical_names=True,
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


def test_legacy_shovel_conversion_preserves_old_scaled_matrix() -> None:
    metadata = _legacy_shovel_metadata()

    converted, proof = legacy_contact_points_pose(metadata)

    assert len(converted) == 1
    assert proof["schema"] == "before-we-act.bicoord.legacy-contact-overlay/1"
    assert proof["conversion"] == (
        "scale(contact_pose) @ trans_matrix -> scale(contact_points_pose)"
    )
    # The current Actor loader scales only the translation of each local point.
    # Reconstruct the old expression explicitly and compare all matrix entries.
    old = metadata[LEGACY_CONTACT_KEY][0]
    transform = metadata[LEGACY_TRANSFORM_KEY]
    scale = metadata["scale"]
    old_scaled = [[float(item) for item in row] for row in old]
    for axis in range(3):
        old_scaled[axis][3] *= scale[axis]
    expected = [
        [sum(old_scaled[row][k] * transform[k][column] for k in range(4)) for column in range(4)]
        for row in range(4)
    ]
    reconstructed = [[float(item) for item in row] for row in converted[0]]
    for axis in range(3):
        reconstructed[axis][3] *= scale[axis]
    assert all(
        reconstructed[row][column] == pytest.approx(expected[row][column], abs=1e-12)
        for row in range(4)
        for column in range(4)
    )
    assert proof["max_scale_equivalence_error"] <= 1e-12


def test_legacy_shovel_overlay_adds_only_current_contact_field() -> None:
    metadata = _legacy_shovel_metadata()
    before = copy.deepcopy(metadata)

    output, proof = overlay_legacy_contact_metadata(metadata)

    assert set(output) == set(metadata) | {CONTACT_KEY}
    assert output[LEGACY_CONTACT_KEY] == before[LEGACY_CONTACT_KEY]
    assert output[LEGACY_TRANSFORM_KEY] == before[LEGACY_TRANSFORM_KEY]
    assert output["scale"] == [0.167, 0.167, 0.167]
    assert proof["changed_fields"] == [CONTACT_KEY]
    assert proof["source_fields"] == [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY]
    assert proof["derived_fields"] == [CONTACT_KEY]
    assert metadata == before


def test_legacy_shovel_overlay_rejects_conflicting_current_field() -> None:
    metadata = _legacy_shovel_metadata()
    metadata[CONTACT_KEY] = [_pose(99.0)]

    with pytest.raises(AssetContractError, match="conflicting legacy contact_points_pose"):
        overlay_legacy_contact_metadata(metadata)


def test_legacy_shovel_overlay_rejects_scale_or_transform_drift() -> None:
    with pytest.raises(AssetContractError, match="legacy shovel scale drift"):
        legacy_contact_points_pose(_legacy_shovel_metadata(scale=0.2))
    bad_transform = _legacy_shovel_metadata()
    bad_transform[LEGACY_TRANSFORM_KEY] = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(AssetContractError, match="not orthonormal"):
        legacy_contact_points_pose(bad_transform)


def test_legacy_shovel_file_overlay_is_atomic_and_run_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "assets" / "objects" / "082_smallshovel"
    source_dir.mkdir(parents=True)
    source = source_dir / "model_data3.json"
    source.write_text(json.dumps(_legacy_shovel_metadata()), encoding="utf-8")
    # Use fixture identities while retaining the production function's
    # fail-closed path/mesh checks.
    monkeypatch.setattr(
        "deployment.bicoord_care.asset_contract.PRISTINE_SHOVEL_METADATA_SHA256",
        sha256_file(source),
    )
    for relative, payload in (
        ("collision/base3.glb", b"collision"),
        ("visual/base3.glb", b"visual"),
    ):
        path = source_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        constant = (
            "SHOVEL_COLLISION_BYTES"
            if relative.startswith("collision")
            else "SHOVEL_VISUAL_BYTES"
        )
        digest_constant = (
            "SHOVEL_COLLISION_SHA256"
            if relative.startswith("collision")
            else "SHOVEL_VISUAL_SHA256"
        )
        monkeypatch.setattr(
            f"deployment.bicoord_care.asset_contract.{constant}",
            len(payload),
        )
        monkeypatch.setattr(
            f"deployment.bicoord_care.asset_contract.{digest_constant}",
            sha256_file(path),
        )
    target = tmp_path / "run" / "overlay" / "082_smallshovel" / "model_data3.json"
    result = apply_legacy_contact_overlay(source, output_path=target)
    assert result["status"] == "PASSED"
    assert result["benchmark_asset_source_modified"] is False
    assert result["mutation_scope"] == "run_artifact_only"
    assert json.loads(source.read_text()) == _legacy_shovel_metadata()
    assert json.loads(target.read_text())[CONTACT_KEY]
    second = apply_legacy_contact_overlay(source, output_path=target)
    assert second["changed"] is False
    assert second["idempotent"] is True


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

    second = apply_contact_points_overlay(
        small / MODEL_METADATA_NAME,
        large / MODEL_METADATA_NAME,
        output_path=output,
    )
    assert second["changed"] is False
    assert second["idempotent"] is True


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
