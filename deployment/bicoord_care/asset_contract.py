"""Fail-closed contract and compatibility overlay for the BiCoord plate asset.

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
) -> dict[str, Any]:
    """Validate the two object records and return immutable provenance.

    The returned mapping is JSON serialisable and deliberately includes both
    mesh inventories and metadata hashes, so a caller can embed it in a run
    receipt.  No file is changed by this function.
    """

    small_root = Path(small_directory).expanduser().resolve(strict=True)
    large_root = Path(large_directory).expanduser().resolve(strict=True)
    if small_root.name != SMALL_OBJECT_NAME:
        raise AssetContractError(
            f"small object directory must be named {SMALL_OBJECT_NAME}: {small_root}"
        )
    if large_root.name != LARGE_OBJECT_NAME:
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
    provenance = validate_asset_pair(small_directory, large_directory)
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
                existing
            ),
            "target_contact_points_pose_sha256": canonical_json_sha256(contacts),
            "contact_points_pose_count": len(contacts),
            "small_scale_preserved": payload.get(SCALE_KEY) == small.get(SCALE_KEY),
            "changed": bool(changed),
            "idempotent": not changed and existing == contacts,
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
    "LARGE_OBJECT_NAME",
    "MODEL_METADATA_NAME",
    "SCHEMA",
    "SMALL_OBJECT_NAME",
    "apply_contact_points_overlay",
    "canonical_json_sha256",
    "ensure_plate_contact_overlay",
    "main",
    "sha256_file",
    "validate_asset_pair",
    "validate_plate_asset_pair",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
