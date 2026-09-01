"""Fail-closed compatibility overlays for released BiCoord object metadata.

The public BiCoord task ``place_plate_and_cup`` refers to ``003_plate`` and
requests contact point ``2``.  The accompanying RoboTwin base asset contains
the same meshes under ``003_plate_large`` but only that metadata record has
contact poses.  This module makes the smallest possible compatibility fix:
copy *only* ``contact_points_pose`` from the metadata record with the matching
mesh, while retaining the small plate's scale and every other field.

No simulator, task, model, or normalisation code is imported here.  The
functions are consequently usable by a preflight process before SAPIEN is
initialised.  Every write is atomic and all checks fail closed; in particular,
an existing non-empty but different contact list is never silently replaced.

The released ``sweep_block`` task exposes a second, independent schema-boundary
defect.  It selects ``082_smallshovel/model_data3.json``, a RoboTwin-1.0 record
whose grasp field is named ``contact_pose`` and whose pose convention includes
a right-multiplied ``trans_matrix``.  Current RoboTwin consumes
``contact_points_pose`` directly.  Merely renaming/copying ``contact_pose``
would therefore rotate the grasp frame incorrectly.  The legacy adapter below
preserves the old runtime matrix ``scale_translation(contact_pose) @
trans_matrix`` under the current loader's scaling convention and adds only the
current field to a run-local copy of the pristine record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


SCHEMA: Final[str] = "before-we-act.bicoord.asset-contact-overlay/1"
MODEL_METADATA_NAME: Final[str] = "model_data0.json"
SMALL_OBJECT_NAME: Final[str] = "003_plate"
LARGE_OBJECT_NAME: Final[str] = "003_plate_large"
CONTACT_KEY: Final[str] = "contact_points_pose"
SCALE_KEY: Final[str] = "scale"
DEFAULT_SMALL_SCALE: Final[tuple[float, float, float]] = (0.025, 0.025, 0.025)
DEFAULT_LARGE_SCALE: Final[tuple[float, float, float]] = (0.035, 0.035, 0.035)
MIN_CONTACT_POINTS: Final[int] = 3

LEGACY_CONTACT_SCHEMA: Final[str] = (
    "before-we-act.bicoord.legacy-contact-overlay/1"
)
SHOVEL_OBJECT_NAME: Final[str] = "082_smallshovel"
SHOVEL_MODEL_ID: Final[int] = 3
SHOVEL_METADATA_NAME: Final[str] = f"model_data{SHOVEL_MODEL_ID}.json"
LEGACY_CONTACT_KEY: Final[str] = "contact_pose"
LEGACY_TRANSFORM_KEY: Final[str] = "trans_matrix"
DEFAULT_SHOVEL_SCALE: Final[tuple[float, float, float]] = (0.167, 0.167, 0.167)
PRISTINE_SHOVEL_METADATA_SHA256: Final[str] = (
    "61be803fb503312ce14856826d8a06b027971b7d7ed33b65b7bf87aa7dfdbf0e"
)
SHOVEL_CONTACT_POINTS_POSE_SHA256: Final[str] = (
    "0c60f46ff0ee41427e9357256163cf0d26e0153cca6e182c8fbed2e7c74b5be3"
)
SHOVEL_OVERLAY_METADATA_CANONICAL_SHA256: Final[str] = (
    "631f80ef76c02ddbeed2a163a6ca4621ae907b219fd9262a29e95638424f4a5a"
)
SHOVEL_COLLISION_BYTES: Final[int] = 455_048
SHOVEL_COLLISION_SHA256: Final[str] = (
    "574a9dec00d7712ee125665d04264f4d151a3cea5a56187ff009f8da645aba7e"
)
SHOVEL_VISUAL_BYTES: Final[int] = 1_872_352
SHOVEL_VISUAL_SHA256: Final[str] = (
    "c53c4dd1cfa855c1f18ad8f2e05c135751edcf4a98639ecbed4ffe834335cc9b"
)


class AssetContractError(RuntimeError):
    """Raised when a benchmark asset violates the frozen compatibility gate."""


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb", buffering=0) as stream:
            while block := stream.read(16 * 1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise AssetContractError(f"cannot hash asset file {source}: {error}") from error
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _regular_file(path: str | Path, *, label: str) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise AssetContractError(f"{label} must not be a symlink: {value}")
    try:
        resolved = value.resolve(strict=True)
    except OSError as error:
        raise AssetContractError(f"{label} is unavailable: {value}: {error}") from error
    if not resolved.is_file():
        raise AssetContractError(f"{label} is not a regular file: {value}")
    return resolved


def _metadata(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label="metadata")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetContractError(f"invalid metadata JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise AssetContractError(f"metadata must be a JSON object: {source}")
    return source, value


def _validate_metadata_pair(
    small: Mapping[str, Any],
    large: Mapping[str, Any],
    *,
    expected_small_scale: Sequence[float] = DEFAULT_SMALL_SCALE,
    expected_large_scale: Sequence[float] = DEFAULT_LARGE_SCALE,
) -> tuple[list[float], list[float], list[list[list[float]]], list[list[list[float]]]]:
    """Validate JSON records and return normalized scales/contact poses.

    This helper intentionally has no filesystem dependency.  It is used by
    both the receipt-producing file path and the in-memory actor adapter.
    """

    small_scale = _vector(small.get(SCALE_KEY), name="small scale")
    large_scale = _vector(large.get(SCALE_KEY), name="large scale")
    _close_vector(small_scale, list(expected_small_scale), name="small scale")
    _close_vector(large_scale, list(expected_large_scale), name="large scale")
    if CONTACT_KEY not in small:
        raise AssetContractError(f"small metadata lacks {CONTACT_KEY}")
    if CONTACT_KEY not in large:
        raise AssetContractError(f"large metadata lacks {CONTACT_KEY}")
    small_raw = small[CONTACT_KEY]
    if not isinstance(small_raw, list):
        raise AssetContractError("small contact_points_pose must be a list")
    large_contacts = _validate_pose_list(
        large[CONTACT_KEY], name="large contact_points_pose", minimum=MIN_CONTACT_POINTS
    )
    if small_raw:
        small_contacts = _validate_pose_list(
            small_raw,
            name="small contact_points_pose",
            minimum=MIN_CONTACT_POINTS,
        )
        if small_raw != large[CONTACT_KEY]:
            raise AssetContractError(
                "small contact_points_pose is non-empty and differs from the reference"
            )
    else:
        small_contacts = []
    small_invariants = _invariant_metadata(small)
    large_invariants = _invariant_metadata(large)
    if small_invariants != large_invariants:
        differing = sorted(
            key
            for key in set(small_invariants) | set(large_invariants)
            if small_invariants.get(key) != large_invariants.get(key)
        )
        raise AssetContractError(f"small/large metadata invariant drift: {differing}")
    return small_scale, large_scale, small_contacts, large_contacts


def overlay_metadata(
    small_metadata: Mapping[str, Any],
    large_metadata: Mapping[str, Any],
    *,
    expected_small_scale: Sequence[float] = DEFAULT_SMALL_SCALE,
    expected_large_scale: Sequence[float] = DEFAULT_LARGE_SCALE,
) -> dict[str, Any]:
    """Return a copy of *small_metadata* with only contact poses overlaid.

    This is the preferred call for simulator adapters: assign the returned
    mapping to the newly-created actor's ``config`` in memory.  It never writes
    a file and never mutates either input mapping.
    """

    if not isinstance(small_metadata, Mapping) or not isinstance(large_metadata, Mapping):
        raise AssetContractError("small and large metadata must be mappings")
    _validate_metadata_pair(
        small_metadata,
        large_metadata,
        expected_small_scale=expected_small_scale,
        expected_large_scale=expected_large_scale,
    )
    value = copy.deepcopy(dict(small_metadata))
    value[CONTACT_KEY] = copy.deepcopy(large_metadata[CONTACT_KEY])
    return value


def _vector(value: object, *, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AssetContractError(f"{name} must be a length-3 numeric vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AssetContractError(f"{name} contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise AssetContractError(f"{name} contains a non-finite value")
        result.append(number)
    return result


def _close_vector(actual: Sequence[float], expected: Sequence[float], *, name: str) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(actual, expected)
    ):
        raise AssetContractError(f"{name} drift: expected {list(expected)!r}, got {list(actual)!r}")


def _validate_pose_list(value: object, *, name: str, minimum: int) -> list[list[list[float]]]:
    if not isinstance(value, list) or len(value) < minimum:
        raise AssetContractError(f"{name} must contain at least {minimum} 4x4 poses")
    poses: list[list[list[float]]] = []
    for pose_index, pose in enumerate(value):
        if not isinstance(pose, list) or len(pose) != 4:
            raise AssetContractError(f"{name}[{pose_index}] is not a 4x4 matrix")
        rows: list[list[float]] = []
        for row_index, row in enumerate(pose):
            if not isinstance(row, list) or len(row) != 4:
                raise AssetContractError(
                    f"{name}[{pose_index}][{row_index}] is not a length-4 row"
                )
            numbers: list[float] = []
            for item in row:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise AssetContractError(f"{name} contains a non-numeric value")
                number = float(item)
                if not math.isfinite(number):
                    raise AssetContractError(f"{name} contains a non-finite value")
                numbers.append(number)
            rows.append(numbers)
        # Homogeneous transforms are part of the metadata contract.  A loose
        # tolerance accepts harmless JSON float formatting differences but not
        # a pose accidentally represented in a different convention.
        if any(
            not math.isclose(rows[3][column], (0.0, 0.0, 0.0, 1.0)[column], abs_tol=1e-9)
            for column in range(4)
        ):
            raise AssetContractError(f"{name}[{pose_index}] has an invalid homogeneous row")
        poses.append(rows)
    return poses


def _validate_matrix4(value: object, *, name: str) -> list[list[float]]:
    """Validate and normalize one homogeneous 4x4 transform.

    This deliberately mirrors :func:`_validate_pose_list` but is kept
    separate because the legacy ``trans_matrix`` is a single matrix rather
    than a list of contact poses.  Returning fresh Python floats also makes
    the conversion deterministic across JSON number spellings and avoids a
    dependency on NumPy in the preflight process.
    """

    if not isinstance(value, list) or len(value) != 4:
        raise AssetContractError(f"{name} must be a 4x4 matrix")
    rows: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 4:
            raise AssetContractError(f"{name}[{row_index}] is not a length-4 row")
        numbers: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise AssetContractError(f"{name} contains a non-numeric value")
            number = float(item)
            if not math.isfinite(number):
                raise AssetContractError(f"{name} contains a non-finite value")
            numbers.append(number)
        rows.append(numbers)
    if any(
        not math.isclose(rows[3][column], (0.0, 0.0, 0.0, 1.0)[column], abs_tol=1e-9)
        for column in range(4)
    ):
        raise AssetContractError(f"{name} has an invalid homogeneous row")
    return rows


def _validate_rigid_matrix4(value: object, *, name: str) -> list[list[float]]:
    """Validate one finite, right-handed homogeneous rigid transform."""

    rows = _validate_matrix4(value, name=name)
    rotation = [row[:3] for row in rows[:3]]
    for first in range(3):
        for second in range(3):
            dot = sum(
                rotation[row][first] * rotation[row][second]
                for row in range(3)
            )
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, rel_tol=0.0, abs_tol=2e-5):
                raise AssetContractError(
                    f"{name} rotation is not orthonormal at ({first}, {second})"
                )
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise AssetContractError(
            f"{name} rotation determinant is not +1: {determinant}"
        )
    return rows


def _matmul4(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    """Multiply two validated homogeneous 4x4 matrices without NumPy."""

    return [
        [
            sum(float(left[row][inner]) * float(right[inner][column]) for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _scale_pose_translation(
    pose: Sequence[Sequence[float]], scale: Sequence[float]
) -> list[list[float]]:
    """Apply the benchmark metadata scale to a pose's translation only."""

    value = [list(map(float, row)) for row in pose]
    for axis in range(3):
        value[axis][3] *= float(scale[axis])
    return value


def legacy_contact_points_pose(
    metadata: Mapping[str, Any],
    *,
    expected_scale: Sequence[float] = DEFAULT_SHOVEL_SCALE,
) -> tuple[list[list[list[float]]], dict[str, Any]]:
    """Convert RoboTwin-1.0 ``contact_pose`` metadata exactly.

    The old RoboTwin grasp helper evaluates each point as::

        actor_pose @ scale_translation(contact_pose) @ trans_matrix

    followed by its fixed gripper-axis conversion.  The current BiCoord
    helper evaluates::

        actor_pose @ scale_translation(contact_points_pose)

    followed by the same conversion.  We therefore derive a current-format
    point whose *scaled* matrix equals the old product.  This is more precise
    than a field rename and remains correct even if a future legacy record has
    a non-zero ``trans_matrix`` translation.  The returned proof dictionary is
    suitable for a run receipt and records the two matrices' equivalence.
    """

    if not isinstance(metadata, Mapping):
        raise AssetContractError("legacy shovel metadata must be a mapping")
    scale = _vector(metadata.get(SCALE_KEY), name="legacy shovel scale")
    _close_vector(scale, list(expected_scale), name="legacy shovel scale")
    if LEGACY_CONTACT_KEY not in metadata:
        raise AssetContractError(
            f"legacy shovel metadata lacks {LEGACY_CONTACT_KEY}"
        )
    if LEGACY_TRANSFORM_KEY not in metadata:
        raise AssetContractError(
            f"legacy shovel metadata lacks {LEGACY_TRANSFORM_KEY}"
        )
    legacy_contacts = _validate_pose_list(
        metadata[LEGACY_CONTACT_KEY],
        name=f"legacy shovel {LEGACY_CONTACT_KEY}",
        minimum=1,
    )
    legacy_contacts = [
        _validate_rigid_matrix4(
            pose,
            name=f"legacy shovel {LEGACY_CONTACT_KEY}[{index}]",
        )
        for index, pose in enumerate(legacy_contacts)
    ]
    trans_matrix = _validate_rigid_matrix4(
        metadata[LEGACY_TRANSFORM_KEY],
        name=f"legacy shovel {LEGACY_TRANSFORM_KEY}",
    )
    # A zero scale axis cannot be represented by the current loader because
    # it scales the derived translation at runtime.  Reject rather than
    # silently inventing a coordinate value.
    if any(abs(float(axis)) <= 1e-15 for axis in scale):
        raise AssetContractError("legacy shovel scale axes must be non-zero")

    converted: list[list[list[float]]] = []
    equivalence_errors: list[float] = []
    for index, legacy_pose in enumerate(legacy_contacts):
        old_scaled = _scale_pose_translation(legacy_pose, scale)
        old_local = _matmul4(old_scaled, trans_matrix)
        # Undo the current loader's translation scaling so that applying it
        # later reconstructs old_local exactly (within floating arithmetic).
        current = [list(row) for row in old_local]
        for axis in range(3):
            current[axis][3] /= float(scale[axis])
        # Keep homogeneous structure explicit; this catches accidental matrix
        # arithmetic or malformed legacy data before a simulator is started.
        current = _validate_rigid_matrix4(
            current,
            name=f"derived {CONTACT_KEY}[{index}]",
        )
        reconstructed = _scale_pose_translation(current, scale)
        error = max(
            abs(reconstructed[row][column] - old_local[row][column])
            for row in range(4)
            for column in range(4)
        )
        equivalence_errors.append(error)
        if error > 1e-12:
            raise AssetContractError(
                f"legacy contact conversion is not scale-equivalent at index {index}: {error}"
            )
        converted.append(current)

    return converted, {
        "schema": LEGACY_CONTACT_SCHEMA,
        "conversion": "scale(contact_pose) @ trans_matrix -> scale(contact_points_pose)",
        "legacy_contact_pose_count": len(legacy_contacts),
        "contact_points_pose_count": len(converted),
        "legacy_contact_pose_sha256": canonical_json_sha256(legacy_contacts),
        "trans_matrix_sha256": canonical_json_sha256(trans_matrix),
        "contact_points_pose_sha256": canonical_json_sha256(converted),
        "max_scale_equivalence_error": max(equivalence_errors, default=0.0),
        "scale": list(scale),
        "scale_preserved": True,
    }


def overlay_legacy_contact_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_scale: Sequence[float] = DEFAULT_SHOVEL_SCALE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a legacy shovel record with only ``contact_points_pose`` added.

    Existing current-format contacts are accepted only when they exactly equal
    the deterministic conversion.  A conflicting list is never overwritten.
    All legacy fields (including ``contact_pose`` and ``trans_matrix``) remain
    in the returned actor config for provenance and backward compatibility.
    """

    converted, proof = legacy_contact_points_pose(
        metadata,
        expected_scale=expected_scale,
    )
    existing = metadata.get(CONTACT_KEY, [])
    if not isinstance(existing, list):
        raise AssetContractError(f"legacy metadata {CONTACT_KEY} must be a list if present")
    if existing and existing != converted:
        raise AssetContractError(
            f"refusing to replace conflicting legacy {CONTACT_KEY}"
        )
    output = copy.deepcopy(dict(metadata))
    output[CONTACT_KEY] = copy.deepcopy(converted)
    changed_fields = sorted(
        key for key in set(metadata) | set(output) if metadata.get(key) != output.get(key)
    )
    if changed_fields not in ([], [CONTACT_KEY]):
        raise AssetContractError(
            "legacy shovel overlay must add only contact_points_pose; "
            f"changed fields: {changed_fields}"
        )
    proof = dict(proof)
    proof.update(
        {
            # Keep the same receipt vocabulary as the plate overlay while
            # separately stating that this field is *derived*, not copied
            # byte-for-byte from the legacy record.
            "copied_fields": [CONTACT_KEY],
            "added_fields": [CONTACT_KEY] if changed_fields else [],
            "derived_fields": [CONTACT_KEY],
            "source_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
            "preserved_fields": "all_except_contact_points_pose",
            "changed_fields": changed_fields,
            "idempotent": not changed_fields,
        }
    )
    return output, proof


def validate_legacy_shovel_asset(
    source_metadata_path: str | Path,
) -> dict[str, Any]:
    """Bind the conversion to the exact released model-3 record and meshes."""

    source_path, source = _metadata(source_metadata_path)
    if source_path.name != SHOVEL_METADATA_NAME or source_path.parent.name != SHOVEL_OBJECT_NAME:
        raise AssetContractError(
            f"legacy contact overlay must target {SHOVEL_OBJECT_NAME}/{SHOVEL_METADATA_NAME}"
        )
    source_sha256 = sha256_file(source_path)
    if source_sha256 != PRISTINE_SHOVEL_METADATA_SHA256:
        raise AssetContractError(
            "legacy shovel metadata is not the pristine released record: "
            f"{source_sha256} != {PRISTINE_SHOVEL_METADATA_SHA256}"
        )
    if CONTACT_KEY in source:
        raise AssetContractError(
            f"pristine legacy shovel metadata unexpectedly contains {CONTACT_KEY}"
        )
    meshes: list[dict[str, Any]] = []
    for relative, expected_bytes, expected_sha256 in (
        (
            "collision/base3.glb",
            SHOVEL_COLLISION_BYTES,
            SHOVEL_COLLISION_SHA256,
        ),
        ("visual/base3.glb", SHOVEL_VISUAL_BYTES, SHOVEL_VISUAL_SHA256),
    ):
        candidate = _regular_file(
            source_path.parent / relative,
            label=f"legacy shovel {relative}",
        )
        observed_bytes = candidate.stat().st_size
        observed_sha256 = sha256_file(candidate)
        if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
            raise AssetContractError(
                f"legacy shovel {relative} identity drift: "
                f"bytes={observed_bytes}, sha256={observed_sha256}"
            )
        meshes.append(
            {
                "relative_path": relative,
                "path": str(candidate),
                "bytes": observed_bytes,
                "sha256": observed_sha256,
            }
        )
    return {
        "object": SHOVEL_OBJECT_NAME,
        "model_id": SHOVEL_MODEL_ID,
        "source_metadata": str(source_path),
        "source_metadata_sha256": source_sha256,
        "meshes": meshes,
        "mesh_and_metadata_identity": "PASSED",
    }


def apply_legacy_contact_overlay(
    source_metadata_path: str | Path,
    *,
    output_path: str | Path,
    expected_scale: Sequence[float] = DEFAULT_SHOVEL_SCALE,
) -> dict[str, Any]:
    """Atomically write a run-local current-format shovel metadata overlay."""

    identity = validate_legacy_shovel_asset(source_metadata_path)
    source_path, source = _metadata(source_metadata_path)
    payload, proof = overlay_legacy_contact_metadata(
        source,
        expected_scale=expected_scale,
    )
    converted = copy.deepcopy(payload[CONTACT_KEY])
    target = Path(output_path).expanduser()
    if target.exists() and target.is_symlink():
        raise AssetContractError(f"legacy overlay target must not be a symlink: {target}")
    # Resolve existing parent symlinks even when the final file does not yet
    # exist.  Otherwise a lexical ``run/overlay/...`` path whose parent points
    # into the benchmark checkout could evade the source-path comparison and
    # overwrite the released metadata.
    try:
        target = target.resolve(strict=False)
    except OSError as error:
        raise AssetContractError(
            f"legacy overlay target cannot be resolved: {target}"
        ) from error
    if target == source_path:
        raise AssetContractError(
            "legacy shovel overlay target must be run-local, not the benchmark source"
        )
    if target.exists() and not target.is_file():
        raise AssetContractError(
            f"legacy shovel overlay target is not a regular file: {target}"
        )
    before_hash = sha256_file(target) if target.is_file() else None
    if target != source_path and target.is_file():
        _, existing_target = _metadata(target)
        # The target may have been produced by an earlier identical invocation,
        # but it may not be an unrelated record with the same path.
        target_without_contacts = {
            key: value for key, value in existing_target.items() if key != CONTACT_KEY
        }
        source_without_contacts = {
            key: value for key, value in source.items() if key != CONTACT_KEY
        }
        if target_without_contacts != source_without_contacts:
            raise AssetContractError(
                "existing legacy overlay target differs from pristine shovel metadata"
            )
        target_contacts = existing_target.get(CONTACT_KEY, [])
        if target_contacts not in ([], payload[CONTACT_KEY]):
            raise AssetContractError(
                "existing legacy overlay target has conflicting contact points"
            )
        payload = copy.deepcopy(existing_target)
        # Never trust a merely present target field; always restore the
        # deterministic conversion derived from the pristine source.
        payload[CONTACT_KEY] = converted
    changed = not target.is_file()
    if target.is_file():
        try:
            _, target_value = _metadata(target)
            changed = target_value != payload
        except AssetContractError:
            changed = True
    if changed:
        mode = (target.stat().st_mode & 0o777) if target.exists() else 0o644
        _atomic_write_json(target, payload, mode=mode)
    after_hash = sha256_file(target)
    result = dict(proof)
    result.update(
        {
            "status": "PASSED",
            "asset_identity": identity,
            "source_metadata": str(source_path),
            "source_metadata_sha256": sha256_file(source_path),
            "target_metadata": str(target.resolve()),
            "target_metadata_before_sha256": before_hash,
            "target_metadata_sha256": after_hash,
            "changed": bool(changed),
            "idempotent": not changed,
            "benchmark_asset_source_modified": False,
            "mutation_scope": "run_artifact_only",
        }
    )
    return result


def _mesh_inventory(directory: str | Path, *, label: str) -> list[dict[str, Any]]:
    root = Path(directory).expanduser()
    if root.is_symlink():
        raise AssetContractError(f"{label} object directory must not be a symlink: {root}")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise AssetContractError(f"{label} object directory is unavailable: {root}: {error}") from error
    if not root.is_dir():
        raise AssetContractError(f"{label} object path is not a directory: {root}")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*.glb")):
        if candidate.is_symlink():
            raise AssetContractError(f"{label} mesh must not be a symlink: {candidate}")
        if not candidate.is_file():
            raise AssetContractError(f"{label} mesh is not a regular file: {candidate}")
        relative = candidate.relative_to(root).as_posix()
        rows.append(
            {
                "relative_path": relative,
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise AssetContractError(f"{label} object has no GLB mesh files: {root}")
    return rows


def _compare_meshes(
    small_directory: str | Path, large_directory: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    small_meshes = _mesh_inventory(small_directory, label="small")
    large_meshes = _mesh_inventory(large_directory, label="large")
    small_by_path = {row["relative_path"]: row for row in small_meshes}
    large_by_path = {row["relative_path"]: row for row in large_meshes}
    if set(small_by_path) != set(large_by_path):
        raise AssetContractError(
            "small/large GLB inventories differ: "
            f"small={sorted(small_by_path)}, large={sorted(large_by_path)}"
        )
    mismatches = [
        relative
        for relative in sorted(small_by_path)
        if small_by_path[relative]["sha256"] != large_by_path[relative]["sha256"]
    ]
    if mismatches:
        raise AssetContractError(f"small/large GLB hash mismatch: {mismatches}")
    return small_meshes, large_meshes


def _invariant_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in {CONTACT_KEY, SCALE_KEY}
    }


def validate_asset_pair(
    small_directory: str | Path,
    large_directory: str | Path,
    *,
    expected_small_scale: Sequence[float] = DEFAULT_SMALL_SCALE,
    expected_large_scale: Sequence[float] = DEFAULT_LARGE_SCALE,
    require_canonical_names: bool = False,
) -> dict[str, Any]:
    """Validate the two object records and return immutable provenance.

    The returned mapping is JSON serialisable and deliberately includes both
    mesh inventories and metadata hashes, so a caller can embed it in a run
    receipt.  No file is changed by this function.
    """

    small_root = Path(small_directory).expanduser().resolve(strict=True)
    large_root = Path(large_directory).expanduser().resolve(strict=True)
    if require_canonical_names and small_root.name != SMALL_OBJECT_NAME:
        raise AssetContractError(
            f"small object directory must be named {SMALL_OBJECT_NAME}: {small_root}"
        )
    if require_canonical_names and large_root.name != LARGE_OBJECT_NAME:
        raise AssetContractError(
            f"large object directory must be named {LARGE_OBJECT_NAME}: {large_root}"
        )
    small_metadata, small = _metadata(small_root / MODEL_METADATA_NAME)
    large_metadata, large = _metadata(large_root / MODEL_METADATA_NAME)
    small_meshes, large_meshes = _compare_meshes(small_root, large_root)

    small_scale, large_scale, small_contacts, large_contacts = _validate_metadata_pair(
        small,
        large,
        expected_small_scale=expected_small_scale,
        expected_large_scale=expected_large_scale,
    )
    small_invariants = _invariant_metadata(small)

    return {
        "schema": SCHEMA,
        "status": "PASSED",
        "overlay": "contact_points_pose_only",
        "small_object": SMALL_OBJECT_NAME,
        "large_object": LARGE_OBJECT_NAME,
        "small_directory": str(small_root),
        "large_directory": str(large_root),
        "small_metadata": str(small_metadata),
        "large_metadata": str(large_metadata),
        "small_metadata_sha256": sha256_file(small_metadata),
        "large_metadata_sha256": sha256_file(large_metadata),
        "small_scale": small_scale,
        "large_scale": large_scale,
        "small_contact_points_pose_count": len(small_contacts),
        "large_contact_points_pose_count": len(large_contacts),
        "small_contact_points_pose_sha256": canonical_json_sha256(small_contacts),
        "large_contact_points_pose_sha256": canonical_json_sha256(large_contacts),
        "mesh_hashes_equal": True,
        "small_meshes": small_meshes,
        "large_meshes": large_meshes,
        "invariant_metadata_sha256": canonical_json_sha256(small_invariants),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, mode: int | None = None) -> None:
    """Atomically replace *path* and fsync both the file and parent directory."""

    if path.exists() and path.is_symlink():
        raise AssetContractError(f"overlay target must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode is None:
        mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # fdopen owns the descriptor after entering the context manager.
            raise
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The replacement itself is atomic; some filesystems do not permit
            # directory fsync.  Do not turn a successful write into a false
            # failure, but retain the file-level fsync guarantee.
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def apply_contact_points_overlay(
    small_metadata_path: str | Path,
    large_metadata_path: str | Path,
    *,
    output_path: str | Path | None = None,
    small_directory: str | Path | None = None,
    large_directory: str | Path | None = None,
    require_canonical_names: bool = False,
) -> dict[str, Any]:
    """Apply the reference contact poses to small metadata, atomically.

    ``output_path`` defaults to ``small_metadata_path``.  Supplying an output
    path is useful for an explicit overlay directory and leaves the benchmark
    checkout untouched.  If the target already contains the exact reference
    list, no write occurs and ``changed`` is false.
    """

    small_path, small = _metadata(small_metadata_path)
    large_path, large = _metadata(large_metadata_path)
    if small_directory is None:
        small_directory = small_path.parent
    if large_directory is None:
        large_directory = large_path.parent

    # Validate the pair before considering a write.  This also checks mesh
    # equality and expected scales when callers pass object directories.
    provenance = validate_asset_pair(
        small_directory,
        large_directory,
        require_canonical_names=require_canonical_names,
    )
    if small_path != Path(small_metadata_path).expanduser().resolve(strict=True):
        raise AssetContractError("unexpected small metadata path resolution")
    if large_path != Path(large_metadata_path).expanduser().resolve(strict=True):
        raise AssetContractError("unexpected large metadata path resolution")

    contacts = copy.deepcopy(large[CONTACT_KEY])
    existing = small.get(CONTACT_KEY)
    if not isinstance(existing, list):
        raise AssetContractError("small contact_points_pose must be a list")
    if existing and existing != contacts:
        raise AssetContractError("refusing to replace conflicting small contact_points_pose")

    target = Path(output_path).expanduser() if output_path is not None else small_path
    if target.exists() and target.is_symlink():
        raise AssetContractError(f"overlay target must not be a symlink: {target}")
    target = target.resolve() if target.exists() else target.absolute()
    before_hash = sha256_file(target) if target.is_file() else None

    payload = copy.deepcopy(small)
    payload[CONTACT_KEY] = contacts
    target_contacts_before = existing
    # The target can be a separate file.  When it exists, preserve its own
    # metadata only if it is byte-for-byte the current small record; otherwise
    # fail closed rather than overwriting an unrelated asset.
    if target != small_path and target.exists():
        target_path, target_value = _metadata(target)
        if _invariant_metadata(target_value) != _invariant_metadata(small):
            raise AssetContractError("existing overlay target metadata differs from small asset")
        target_existing = target_value.get(CONTACT_KEY)
        if target_existing not in ([], contacts):
            raise AssetContractError("existing overlay target has conflicting contact poses")
        target_contacts_before = target_existing
        payload = copy.deepcopy(target_value)
        payload[CONTACT_KEY] = contacts

    changed = not target.is_file()
    # Compare parsed semantic content for idempotence and avoid rewriting a
    # file which already contains the desired values.
    if target.is_file():
        try:
            _, target_value = _metadata(target)
            changed = target_value != payload
        except AssetContractError:
            changed = True
    if changed:
        mode = (target.stat().st_mode & 0o777) if target.exists() else 0o644
        _atomic_write_json(target, payload, mode=mode)
    after_hash = sha256_file(target)

    result = dict(provenance)
    result.update(
        {
            "target_metadata": str(target.resolve()),
            "target_metadata_before_sha256": before_hash,
            "target_metadata_sha256": after_hash,
            "target_contact_points_pose_before_sha256": canonical_json_sha256(
                target_contacts_before
            ),
            "target_contact_points_pose_sha256": canonical_json_sha256(contacts),
            "contact_points_pose_count": len(contacts),
            "small_scale_preserved": payload.get(SCALE_KEY) == small.get(SCALE_KEY),
            "changed": bool(changed),
            "idempotent": not changed and target_contacts_before == contacts,
        }
    )
    return result


# Friendly aliases for callers which prefer an imperative name.
ensure_plate_contact_overlay = apply_contact_points_overlay
validate_plate_asset_pair = validate_asset_pair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--small-name", default=SMALL_OBJECT_NAME)
    parser.add_argument("--large-name", default=LARGE_OBJECT_NAME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    small = args.assets_root / args.small_name
    large = args.assets_root / args.large_name
    result = apply_contact_points_overlay(
        small / MODEL_METADATA_NAME,
        large / MODEL_METADATA_NAME,
        output_path=args.output,
        small_directory=small,
        large_directory=large,
        require_canonical_names=True,
    )
    if args.receipt:
        _atomic_write_json(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "AssetContractError",
    "CONTACT_KEY",
    "DEFAULT_LARGE_SCALE",
    "DEFAULT_SMALL_SCALE",
    "DEFAULT_SHOVEL_SCALE",
    "LARGE_OBJECT_NAME",
    "LEGACY_CONTACT_KEY",
    "LEGACY_CONTACT_SCHEMA",
    "LEGACY_TRANSFORM_KEY",
    "MODEL_METADATA_NAME",
    "PRISTINE_SHOVEL_METADATA_SHA256",
    "SCHEMA",
    "SHOVEL_COLLISION_SHA256",
    "SHOVEL_COLLISION_BYTES",
    "SHOVEL_CONTACT_POINTS_POSE_SHA256",
    "SHOVEL_METADATA_NAME",
    "SHOVEL_MODEL_ID",
    "SHOVEL_OBJECT_NAME",
    "SHOVEL_OVERLAY_METADATA_CANONICAL_SHA256",
    "SHOVEL_VISUAL_SHA256",
    "SHOVEL_VISUAL_BYTES",
    "SMALL_OBJECT_NAME",
    "apply_contact_points_overlay",
    "apply_legacy_contact_overlay",
    "canonical_json_sha256",
    "ensure_plate_contact_overlay",
    "legacy_contact_points_pose",
    "overlay_legacy_contact_metadata",
    "main",
    "sha256_file",
    "validate_legacy_shovel_asset",
    "validate_asset_pair",
    "validate_plate_asset_pair",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
