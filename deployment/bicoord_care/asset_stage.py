"""Install and audit the official BiCoord asset compatibility contracts.

BiCoord publishes its demonstrations and its six benchmark-specific object
overrides in the same immutable Hugging Face snapshot.  RoboTwin provides the
base object library separately.  The released ``place_plate_and_cup`` task
uses the base ``003_plate`` at scale 0.025 but indexes contact point two, while
that base metadata contains no contact poses.  BiCoord's supplemental
``003_plate_large`` is the byte-identical mesh with the missing poses.

This stage restores that released data contract without changing task source,
planner code, policy code, action/state ranges, or model capacity.  It first
binds both official archives, installs the supplemental archive safely, then
copies exactly ``contact_points_pose`` into the small-plate metadata.

The released ``sweep_block`` task also pins ``082_smallshovel`` model 3, whose
metadata still uses RoboTwin-1.0's ``contact_pose`` plus ``trans_matrix``
schema.  A second run-local overlay derives the equivalent current-format
``contact_points_pose`` while retaining model 3, its meshes, scale, and every
legacy field.  Every input and output is recorded in a hashed stage receipt
before dataset audit or training may start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile
from typing import Any, Mapping, Sequence

from .asset_contract import (
    CONTACT_KEY,
    LEGACY_CONTACT_KEY,
    LEGACY_TRANSFORM_KEY,
    LARGE_OBJECT_NAME,
    MODEL_METADATA_NAME,
    PRISTINE_SHOVEL_METADATA_SHA256,
    SHOVEL_CONTACT_POINTS_POSE_SHA256,
    SHOVEL_METADATA_NAME,
    SHOVEL_MODEL_ID,
    SHOVEL_OBJECT_NAME,
    SHOVEL_OVERLAY_METADATA_CANONICAL_SHA256,
    SMALL_OBJECT_NAME,
    apply_contact_points_overlay,
    apply_legacy_contact_overlay,
    canonical_json_sha256,
    sha256_file,
)
from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    TASKS,
    TASK_ASSET_DYNAMIC_INVENTORY_SHA256,
    TASK_ASSET_DYNAMIC_ITEM_COUNT,
    TASK_ASSET_UNRESOLVED_INTERACTION_COUNT,
    TASK_ASSET_UNRESOLVED_INTERACTION_INVENTORY_SHA256,
)
from .stage_common import (
    artifact,
    assert_common_paths,
    atomic_json,
    common_parser,
    publish_result,
    read_json,
    require_stage_result,
)


ASSET_STAGE_SCHEMA = "before-we-act.bicoord.asset-contract/1"
BICOORD_OBJECTS_ARCHIVE = "objects.zip"
BICOORD_OBJECTS_BYTES = 212_630_937
BICOORD_OBJECTS_SHA256 = (
    "0ba5d5f7175c479ae9a45bfeb3e8daaa58ac445af8a3d3f6fdd704e977321901"
)
BICOORD_OBJECTS_BLOB_ID = "d645b45530bed79b883654af49d3ccc932aa4c9d"
BICOORD_OBJECTS_XET_HASH = (
    "d73002e9d7ea26a2390c4ac3fa8c8ed519c08a68a507fe25ddb85062a9e2e9e3"
)
BICOORD_OBJECT_ROOTS = (
    "003_plate_large",
    "005_french-fries_small",
    "006_hamburg_small",
    "044_microwave_big",
    "059_pencup_jlk",
    "063_tabletrashbin_jlk",
)
BICOORD_OBJECT_MEMBERS = 316
BICOORD_OBJECT_FILES = 287

ROBOTWIN_ASSET_REPO_ID = "TianxingChen/RoboTwin2.0"
ROBOTWIN_ASSET_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
ROBOTWIN_OBJECTS_BYTES = 3_737_778_549
ROBOTWIN_OBJECTS_SHA256 = (
    "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27"
)

PRISTINE_SMALL_METADATA_SHA256 = (
    "0361bfd713327f782e3b026c3e0be5526a2a6e436dea075bf052a45137f50c87"
)
DONOR_METADATA_SHA256 = (
    "d4df83e8478bebcafa0eea03872fd9333b379d87da48bfda067675365d65b912"
)
PLATE_COLLISION_SHA256 = (
    "2d2682b1294d70ebd997bb174e7cef34e8f86f2da496621445eefd4aab621266"
)
PLATE_VISUAL_SHA256 = (
    "557429b6d585ff8b511428cb5726332a45bfc54f50537c9b5fa6ba60b08a39e0"
)


class AssetStageError(RuntimeError):
    """Raised when published assets do not match the frozen run contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise AssetStageError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AssetStageError(f"{label} is missing: {path}") from error
    if not resolved.is_file():
        raise AssetStageError(f"{label} is not a regular file: {path}")
    return resolved


def _exact_file(
    path: Path, *, label: str, expected_bytes: int, expected_sha256: str
) -> dict[str, Any]:
    source = _regular_file(path, label=label)
    size = source.stat().st_size
    if size != int(expected_bytes):
        raise AssetStageError(
            f"{label} byte size drift: {size} != {expected_bytes}: {source}"
        )
    digest = sha256_file(source)
    if digest != expected_sha256:
        raise AssetStageError(
            f"{label} SHA-256 drift: {digest} != {expected_sha256}: {source}"
        )
    return {"path": str(source), "bytes": size, "sha256": digest}


def _dataset_archive(dataset: Path) -> dict[str, Any]:
    """Bind ``objects.zip`` to the pinned Hub tree and metadata sidecar."""

    dataset = dataset.resolve(strict=True)
    archive = _exact_file(
        dataset / BICOORD_OBJECTS_ARCHIVE,
        label="BiCoord supplemental objects archive",
        expected_bytes=BICOORD_OBJECTS_BYTES,
        expected_sha256=BICOORD_OBJECTS_SHA256,
    )
    tree_path = _regular_file(
        dataset
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{DATASET_REVISION}.json",
        label="BiCoord Hugging Face tree manifest",
    )
    tree = read_json(tree_path)
    files = tree.get("files")
    row = files.get(BICOORD_OBJECTS_ARCHIVE) if isinstance(files, Mapping) else None
    expected_row = {
        "size": BICOORD_OBJECTS_BYTES,
        "blob_id": BICOORD_OBJECTS_BLOB_ID,
        "lfs_sha256": BICOORD_OBJECTS_SHA256,
        "lfs_size": BICOORD_OBJECTS_BYTES,
        "xet_hash": BICOORD_OBJECTS_XET_HASH,
    }
    if not isinstance(row, Mapping) or dict(row) != expected_row:
        raise AssetStageError("BiCoord objects.zip Hub-tree identity drift")
    sidecar_path = _regular_file(
        dataset
        / ".cache"
        / "huggingface"
        / "download"
        / f"{BICOORD_OBJECTS_ARCHIVE}.metadata",
        label="BiCoord objects.zip Hugging Face sidecar",
    )
    try:
        with sidecar_path.open("r", encoding="utf-8", errors="strict") as stream:
            revision = stream.readline(256).rstrip("\r\n")
            object_id = stream.readline(256).rstrip("\r\n")
    except (OSError, UnicodeError) as error:
        raise AssetStageError("cannot read BiCoord objects.zip sidecar") from error
    if revision != DATASET_REVISION or object_id != BICOORD_OBJECTS_SHA256:
        raise AssetStageError("BiCoord objects.zip sidecar identity drift")
    return {
        **archive,
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "blob_id": BICOORD_OBJECTS_BLOB_ID,
        "xet_hash": BICOORD_OBJECTS_XET_HASH,
        "tree_manifest": str(tree_path),
        "tree_manifest_sha256": sha256_file(tree_path),
        "metadata_sidecar": str(sidecar_path),
        "metadata_sidecar_sha256": sha256_file(sidecar_path),
    }


def _base_receipt_path() -> Path:
    configured = os.environ.get("BICOORD_ASSETS_RECEIPT")
    return Path(
        configured or "/workspace/manifests/bicoord-base-assets.json"
    ).expanduser()


def _base_archive(benchmark_repo: Path, *, receipt_path: Path | None = None) -> dict[str, Any]:
    archive_path = benchmark_repo / "assets" / ".archives" / "objects.zip"
    archive = _exact_file(
        archive_path,
        label="RoboTwin base objects archive",
        expected_bytes=ROBOTWIN_OBJECTS_BYTES,
        expected_sha256=ROBOTWIN_OBJECTS_SHA256,
    )
    receipt_path = _regular_file(
        receipt_path or _base_receipt_path(), label="RoboTwin base-assets receipt"
    )
    receipt = read_json(receipt_path)
    expected = {
        "schema": "before-we-act.bicoord-base-assets/1",
        "state": "complete",
        "repo_id": ROBOTWIN_ASSET_REPO_ID,
        "revision": ROBOTWIN_ASSET_REVISION,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise AssetStageError(f"RoboTwin base-assets receipt drift at {key}")
    rows = receipt.get("archives")
    matches = [
        row
        for row in rows if isinstance(row, Mapping) and row.get("archive") == "objects.zip"
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise AssetStageError("RoboTwin receipt does not name exactly one objects.zip")
    row = matches[0]
    if (
        int(row.get("bytes", -1)) != ROBOTWIN_OBJECTS_BYTES
        or row.get("sha256") != ROBOTWIN_OBJECTS_SHA256
        or Path(str(row.get("path", ""))).resolve() != Path(archive["path"])
    ):
        raise AssetStageError("RoboTwin receipt objects.zip identity drift")
    return {
        **archive,
        "repo_id": ROBOTWIN_ASSET_REPO_ID,
        "revision": ROBOTWIN_ASSET_REVISION,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\\" in name:
        raise AssetStageError(f"unsafe BiCoord ZIP member: {name!r}")
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part != ".")
    if path.is_absolute() or not parts or ".." in parts:
        raise AssetStageError(f"unsafe BiCoord ZIP member: {name!r}")
    if parts[0] not in BICOORD_OBJECT_ROOTS:
        raise AssetStageError(f"unexpected BiCoord object root in ZIP: {parts[0]}")
    return parts


def _safe_target(root: Path, parts: Sequence[str]) -> Path:
    cursor = root
    for component in parts[:-1]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise AssetStageError(f"supplemental asset parent is a symlink: {cursor}")
        cursor.mkdir(exist_ok=True)
        if not cursor.is_dir():
            raise AssetStageError(f"supplemental asset parent is not a directory: {cursor}")
    target = root.joinpath(*parts)
    try:
        target.absolute().relative_to(root.absolute())
    except ValueError as error:  # defensive; parts were already POSIX-validated
        raise AssetStageError(f"supplemental asset escapes object root: {target}") from error
    if target.is_symlink():
        raise AssetStageError(f"supplemental asset target is a symlink: {target}")
    return target


def _stream_digest(stream: Any) -> tuple[str, bytes]:
    digest = hashlib.sha256()
    blocks: list[bytes] = []
    while block := stream.read(16 * 1024 * 1024):
        digest.update(block)
        blocks.append(block)
    return digest.hexdigest(), b"".join(blocks)


def install_supplemental_archive(archive_path: Path, objects_root: Path) -> dict[str, Any]:
    """Safely and idempotently install every official supplemental file."""

    archive_path = _regular_file(archive_path, label="BiCoord supplemental objects archive")
    if objects_root.is_symlink():
        raise AssetStageError(f"BiCoord objects root must not be a symlink: {objects_root}")
    objects_root.mkdir(parents=True, exist_ok=True)
    objects_root = objects_root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    changed: list[str] = []
    roots: set[str] = set()
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise AssetStageError(f"cannot open BiCoord objects archive: {archive_path}") from error
    with archive:
        members = archive.infolist()
        if len(members) != BICOORD_OBJECT_MEMBERS:
            raise AssetStageError(
                f"BiCoord objects ZIP member count drift: {len(members)}"
            )
        if archive.testzip() is not None:
            raise AssetStageError("BiCoord objects ZIP CRC verification failed")
        for member in members:
            parts = _safe_member_parts(member.filename)
            roots.add(parts[0])
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise AssetStageError(f"BiCoord ZIP contains symlink: {member.filename}")
            target = _safe_target(objects_root, parts)
            if member.is_dir():
                target.mkdir(exist_ok=True)
                continue
            with archive.open(member) as source:
                member_sha, payload = _stream_digest(source)
            existing_sha = (
                sha256_file(target)
                if target.is_file() and target.stat().st_size == member.file_size
                else None
            )
            was_changed = existing_sha != member_sha
            if was_changed:
                temporary = target.with_name(
                    f".{target.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
                )
                try:
                    with temporary.open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.chmod(temporary, 0o644)
                    os.replace(temporary, target)
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                changed.append(member.filename)
            if target.stat().st_size != member.file_size or sha256_file(target) != member_sha:
                raise AssetStageError(f"installed supplemental member differs: {target}")
            rows.append(
                {
                    "member": member.filename,
                    "bytes": member.file_size,
                    "crc32": f"{member.CRC:08x}",
                    "sha256": member_sha,
                    "target": str(target),
                    "changed": was_changed,
                }
            )
    if tuple(sorted(roots)) != tuple(sorted(BICOORD_OBJECT_ROOTS)):
        raise AssetStageError(f"BiCoord supplemental object roots drift: {sorted(roots)}")
    if len(rows) != BICOORD_OBJECT_FILES:
        raise AssetStageError(f"BiCoord supplemental file count drift: {len(rows)}")
    return {
        "objects_root": str(objects_root),
        "roots": sorted(roots),
        "archive_members": BICOORD_OBJECT_MEMBERS,
        "files_verified": len(rows),
        "files_changed": len(changed),
        "changed_members": changed,
        "files": rows,
    }


def _git_revision(path: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise AssetStageError(f"cannot inspect Git revision: {path}") from error
    if len(value) != 40:
        raise AssetStageError(f"invalid Git revision: {value!r}")
    return value


def _tracked_source_status(path: Path) -> dict[str, Any]:
    """Prove that asset installation did not edit Git-tracked benchmark files."""

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--ignore-submodules=untracked",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AssetStageError(
            f"cannot inspect tracked benchmark source: {path}"
        ) from error
    changes = [row for row in completed.stdout.splitlines() if row]
    if changes:
        raise AssetStageError(
            "BiCoord tracked source is dirty during asset installation: "
            f"{changes[:20]!r}"
        )
    return {
        "status": "CLEAN",
        "scope": "git_index_and_worktree_tracked_files",
        "tracked_changes": [],
        "untracked_supplemental_assets_allowed": True,
    }


def _plate_overlay(benchmark_repo: Path, overlay_root: Path) -> dict[str, Any]:
    objects = benchmark_repo / "assets" / "objects"
    small = objects / SMALL_OBJECT_NAME
    donor = objects / LARGE_OBJECT_NAME
    small_metadata = _regular_file(
        small / MODEL_METADATA_NAME, label="small plate metadata"
    )
    donor_metadata = _regular_file(
        donor / MODEL_METADATA_NAME, label="plate contact donor metadata"
    )
    donor_sha = sha256_file(donor_metadata)
    if donor_sha != DONOR_METADATA_SHA256:
        raise AssetStageError(
            f"plate donor metadata drift: {donor_sha} != {DONOR_METADATA_SHA256}"
        )
    try:
        small_value = json.loads(small_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetStageError("small plate metadata is invalid") from error
    small_before = sha256_file(small_metadata)
    # The source checkout is part of the immutable benchmark contract.  Do
    # not accept an already-overlaid (or otherwise edited) plate record just
    # because it happens to make contact index 2 valid: that would erase the
    # proof that this run started from the released upstream defect and would
    # make a later receipt claim ``benchmark_asset_source_modified=False``
    # falsely.  The run-local overlay below is intentionally the only place
    # where the contact field may change.
    if small_before != PRISTINE_SMALL_METADATA_SHA256:
        raise AssetStageError(
            "small plate metadata is not the pristine released record before compatibility overlay"
        )
    contacts = small_value.get(CONTACT_KEY) if isinstance(small_value, Mapping) else None
    if contacts != []:
        raise AssetStageError(
            "pristine small plate contact_points_pose must be empty before compatibility overlay"
        )
    overlay_path = overlay_root / SMALL_OBJECT_NAME / MODEL_METADATA_NAME
    overlay = apply_contact_points_overlay(
        small_metadata,
        donor_metadata,
        output_path=overlay_path,
        small_directory=small,
        large_directory=donor,
        require_canonical_names=True,
    )
    expected_meshes = {
        "collision/base0.glb": PLATE_COLLISION_SHA256,
        "visual/base0.glb": PLATE_VISUAL_SHA256,
    }
    observed_meshes = {
        str(row["relative_path"]): str(row["sha256"])
        for row in overlay.get("small_meshes", [])
        if isinstance(row, Mapping)
    }
    if observed_meshes != expected_meshes:
        raise AssetStageError(f"small plate mesh identity drift: {observed_meshes}")
    if overlay.get("target_contact_points_pose_sha256") != overlay.get(
        "large_contact_points_pose_sha256"
    ):
        raise AssetStageError("plate overlay did not copy the donor contact field exactly")
    return {
        **overlay,
        "source_small_metadata": str(small_metadata),
        "source_small_metadata_sha256": small_before,
        "overlay_metadata": str(Path(overlay["target_metadata"]).resolve()),
        "pristine_small_metadata_sha256": PRISTINE_SMALL_METADATA_SHA256,
        "donor_metadata_expected_sha256": DONOR_METADATA_SHA256,
        "donor_metadata_sha256": donor_sha,
        "copied_fields": [CONTACT_KEY],
        "preserved_fields": "all_except_contact_points_pose",
        "task_source_modified": False,
        "planner_modified": False,
        "model_modified": False,
        "normalization_modified": False,
        "benchmark_asset_source_modified": False,
        "mutation_scope": "run_artifact_and_actor_config_in_memory_only",
    }


def _shovel_overlay(benchmark_repo: Path, overlay_root: Path) -> dict[str, Any]:
    """Build the model-3 legacy-contact adapter without touching the checkout."""

    source = _regular_file(
        benchmark_repo
        / "assets"
        / "objects"
        / SHOVEL_OBJECT_NAME
        / SHOVEL_METADATA_NAME,
        label="legacy small-shovel model-3 metadata",
    )
    source_sha256 = sha256_file(source)
    if source_sha256 != PRISTINE_SHOVEL_METADATA_SHA256:
        raise AssetStageError(
            "small-shovel model-3 metadata is not the pristine released record"
        )
    overlay_path = overlay_root / SHOVEL_OBJECT_NAME / SHOVEL_METADATA_NAME
    try:
        overlay = apply_legacy_contact_overlay(source, output_path=overlay_path)
    except Exception as error:
        # Keep this stage's public error type stable while retaining the
        # low-level contract failure as the causal exception.
        raise AssetStageError("legacy small-shovel overlay failed") from error
    if overlay.get("source_metadata_sha256") != source_sha256:
        raise AssetStageError("legacy small-shovel source identity changed")
    if overlay.get("contact_points_pose_count") != 1:
        raise AssetStageError("legacy small-shovel contact count differs")
    if (
        overlay.get("contact_points_pose_sha256")
        != SHOVEL_CONTACT_POINTS_POSE_SHA256
    ):
        raise AssetStageError("legacy small-shovel derived contact hash differs")
    if overlay.get("added_fields") != [CONTACT_KEY]:
        raise AssetStageError("legacy small-shovel overlay added unexpected fields")
    if overlay.get("derived_fields") != [CONTACT_KEY]:
        raise AssetStageError("legacy small-shovel overlay derivation differs")
    if overlay.get("source_fields") != [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY]:
        raise AssetStageError("legacy small-shovel source fields differ")
    if float(overlay.get("max_scale_equivalence_error", float("inf"))) > 1e-12:
        raise AssetStageError("legacy small-shovel pose conversion is not equivalent")
    try:
        overlay_value = json.loads(
            Path(overlay["target_metadata"]).read_text(encoding="utf-8")
        )
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetStageError("legacy small-shovel overlay cannot be audited") from error
    if (
        canonical_json_sha256(overlay_value)
        != SHOVEL_OVERLAY_METADATA_CANONICAL_SHA256
    ):
        raise AssetStageError("legacy small-shovel overlay metadata hash differs")
    return {
        **overlay,
        "overlay_metadata": str(Path(overlay["target_metadata"]).resolve()),
        "pristine_source_metadata_sha256": PRISTINE_SHOVEL_METADATA_SHA256,
        "overlay_metadata_canonical_sha256": (
            SHOVEL_OVERLAY_METADATA_CANONICAL_SHA256
        ),
        "modelname": SHOVEL_OBJECT_NAME,
        "model_id": SHOVEL_MODEL_ID,
        "task_source_modified": False,
        "planner_modified": False,
        "model_modified": False,
        "normalization_modified": False,
        "benchmark_asset_source_modified": False,
        "mutation_scope": "run_artifact_and_actor_config_in_memory_only",
    }


def _validate_task_asset_audit(
    task_audit: Mapping[str, Any], *, plate_overlay_metadata: Path,
    shovel_overlay_metadata: Path,
) -> None:
    """Bind a nominal audit PASS to the complete pinned 18-task evidence."""

    from .task_asset_audit import SCHEMA as TASK_AUDIT_SCHEMA

    expected_head = {
        "schema": TASK_AUDIT_SCHEMA,
        "status": "PASSED",
        "tasks": list(TASKS),
        "task_count": len(TASKS),
        "actor_reference_count": 21,
        "interaction_reference_count": 95,
        "references_checked": {"tasks": len(TASKS), "actors": 21, "interactions": 95},
        "dynamic_item_count": TASK_ASSET_DYNAMIC_ITEM_COUNT,
        "dynamic_inventory_sha256": TASK_ASSET_DYNAMIC_INVENTORY_SHA256,
        "unresolved_interaction_count": TASK_ASSET_UNRESOLVED_INTERACTION_COUNT,
        "unresolved_interaction_inventory_sha256": (
            TASK_ASSET_UNRESOLVED_INTERACTION_INVENTORY_SHA256
        ),
        "violations": [],
        "expected_pristine_defect_count": 0,
        "unexpected_violation_count": 0,
        "metadata_override_count": 2,
        "metadata_override_keys": [
            {"modelname": SMALL_OBJECT_NAME, "model_id": 0},
            {"modelname": SHOVEL_OBJECT_NAME, "model_id": SHOVEL_MODEL_ID},
        ],
        "read_only_benchmark": True,
        "benchmark_files_written": False,
    }
    for key, expected in expected_head.items():
        if task_audit.get(key) != expected:
            raise AssetStageError(
                f"18-task object point-index audit differs at {key}: "
                f"{task_audit.get(key)!r} != {expected!r}"
            )
    reports = task_audit.get("task_reports")
    if (
        not isinstance(reports, list)
        or [row.get("task") for row in reports if isinstance(row, Mapping)]
        != list(TASKS)
        or any(
            not isinstance(row, Mapping)
            or row.get("status") != "PASSED"
            or row.get("violations") != []
            for row in reports
        )
    ):
        raise AssetStageError("18-task object point-index report coverage differs")
    override_rows = task_audit.get("metadata_overrides")
    if not isinstance(override_rows, list) or len(override_rows) != 2:
        raise AssetStageError("18-task asset audit overlay provenance is missing")
    overrides = {
        (str(row.get("key", {}).get("modelname")), row.get("key", {}).get("model_id")): row
        for row in override_rows
        if isinstance(row, Mapping) and isinstance(row.get("key"), Mapping)
    }
    expected_overrides = {
        (SMALL_OBJECT_NAME, 0): {
            "key": {"modelname": SMALL_OBJECT_NAME, "model_id": 0},
            "status": "USED",
            "source_type": "file",
            "source_path": str(plate_overlay_metadata.resolve()),
            "source_sha256": sha256_file(plate_overlay_metadata),
            "pristine_source_sha256": PRISTINE_SMALL_METADATA_SHA256,
            "contract_status": "PASSED",
            "error": None,
            "used_by_actor_count": 2,
            "used_by_interaction_count": 4,
        },
        (SHOVEL_OBJECT_NAME, SHOVEL_MODEL_ID): {
            "key": {
                "modelname": SHOVEL_OBJECT_NAME,
                "model_id": SHOVEL_MODEL_ID,
            },
            "status": "USED",
            "source_type": "file",
            "source_path": str(shovel_overlay_metadata.resolve()),
            "source_sha256": sha256_file(shovel_overlay_metadata),
            "pristine_source_sha256": PRISTINE_SHOVEL_METADATA_SHA256,
            "contract_status": "PASSED",
            "error": None,
            "used_by_actor_count": 1,
            "used_by_interaction_count": 1,
        },
    }
    if set(overrides) != set(expected_overrides):
        raise AssetStageError("18-task asset audit overlay keys differ")
    for override_key, expected_override in expected_overrides.items():
        override = overrides[override_key]
        for key, expected in expected_override.items():
            if override.get(key) != expected:
                raise AssetStageError(
                    f"18-task asset audit overlay {override_key!r} differs at {key}: "
                    f"{override.get(key)!r} != {expected!r}"
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    require_stage_result(
        args.run, "dataset_download", config_sha256=args.config_sha256
    )
    tracked_source_before = _tracked_source_status(args.benchmark_repo)
    dataset_source = _dataset_archive(args.dataset)
    base_source = _base_archive(args.benchmark_repo)
    supplemental = install_supplemental_archive(
        Path(dataset_source["path"]),
        args.benchmark_repo / "assets" / "objects",
    )
    tracked_source_after = _tracked_source_status(args.benchmark_repo)
    plate = _plate_overlay(
        args.benchmark_repo,
        args.run / "artifacts" / "asset_contract" / "overlay",
    )
    shovel = _shovel_overlay(
        args.benchmark_repo,
        args.run / "artifacts" / "asset_contract" / "overlay",
    )

    # Import after installation/overlay so the task audit sees the exact
    # metadata that the official simulator will load in the next stage.
    from .task_asset_audit import audit_task_assets

    task_audit = audit_task_assets(
        args.benchmark_repo,
        args.benchmark_repo / "assets",
        tasks=TASKS,
        metadata_overrides={
            (SMALL_OBJECT_NAME, 0): Path(plate["overlay_metadata"]),
            (SHOVEL_OBJECT_NAME, SHOVEL_MODEL_ID): Path(
                shovel["overlay_metadata"]
            ),
        },
    )
    _validate_task_asset_audit(
        task_audit,
        plate_overlay_metadata=Path(plate["overlay_metadata"]),
        shovel_overlay_metadata=Path(shovel["overlay_metadata"]),
    )
    value = {
        "schema": ASSET_STAGE_SCHEMA,
        "status": "PASSED",
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "benchmark_revision": _git_revision(args.benchmark_repo),
        "dataset_archive": dataset_source,
        "base_archive": base_source,
        "supplemental_install": supplemental,
        "supplemental_assets_installed": True,
        "supplemental_assets_source": "official_pinned_BiCoord_objects.zip",
        "supplemental_assets_files_verified": supplemental["files_verified"],
        "supplemental_assets_files_changed": supplemental["files_changed"],
        "benchmark_tracked_source_modified": False,
        "benchmark_tracked_source_before_install": tracked_source_before,
        "benchmark_tracked_source_after_install": tracked_source_after,
        "plate_overlay": plate,
        "shovel_overlay": shovel,
        "task_asset_audit": task_audit,
        "tasks": list(TASKS),
        "task_source_modified": False,
        "upstream_model_modified": False,
        "normalization_modified": False,
        "completed_at": _utc_now(),
    }
    receipt = args.run / "artifacts" / "asset_contract" / "asset_contract.json"
    atomic_json(receipt, value)
    return publish_result(
        args,
        stage="asset_contract",
        artifacts=[artifact(receipt, kind="asset_contract")],
        asset_contract=str(receipt.resolve()),
        asset_contract_sha256=sha256_file(receipt),
        dataset_archive_sha256=BICOORD_OBJECTS_SHA256,
        base_archive_sha256=ROBOTWIN_OBJECTS_SHA256,
        plate_metadata_sha256=plate["target_metadata_sha256"],
        shovel_metadata_sha256=shovel["target_metadata_sha256"],
        shovel_contact_points_pose_sha256=shovel[
            "contact_points_pose_sha256"
        ],
        shovel_contact_points_pose_count=shovel["contact_points_pose_count"],
        contact_points_pose_count=plate["contact_points_pose_count"],
        copied_fields=[CONTACT_KEY],
        task_asset_references_checked=task_audit["references_checked"],
        task_asset_task_count=task_audit["task_count"],
        task_asset_actor_reference_count=task_audit["actor_reference_count"],
        task_asset_interaction_reference_count=(
            task_audit["interaction_reference_count"]
        ),
        task_asset_dynamic_inventory_sha256=(
            task_audit["dynamic_inventory_sha256"]
        ),
        task_asset_unresolved_inventory_sha256=(
            task_audit["unresolved_interaction_inventory_sha256"]
        ),
        task_source_modified=False,
        supplemental_assets_installed=True,
        supplemental_assets_files_verified=supplemental["files_verified"],
        supplemental_assets_files_changed=supplemental["files_changed"],
        benchmark_tracked_source_modified=False,
        upstream_model_modified=False,
        normalization_modified=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = common_parser(__doc__, ("verify-and-overlay",))
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ASSET_STAGE_SCHEMA",
    "AssetStageError",
    "BICOORD_OBJECTS_SHA256",
    "ROBOTWIN_OBJECTS_SHA256",
    "install_supplemental_archive",
    "main",
    "run",
]
