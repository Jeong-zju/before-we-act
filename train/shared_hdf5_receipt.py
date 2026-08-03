"""Auditable reuse of an already accepted HDF5 integrity verification.

The expensive content hashes remain anchored in an exact accepted checkpoint.
For later runs, a receipt binds those hashes to the current manifests and to
the filesystem identity of every episode file.  A changed manifest, path,
size, inode, device, or modification time invalidates the receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable

import torch


RECEIPT_FORMAT = "wam.shared_hdf5_verification_receipt/1"
PROOF_CHECKPOINT_FORMAT = "wam.robofactory.s3_r6.world_action_flow.checkpoint/1"
EXPECTED_TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _manifest_records(manifest: Path) -> tuple[str, list[dict[str, object]]]:
    manifest_bytes = manifest.read_bytes()
    try:
        raw = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset manifest: {manifest}") from exc
    root = _mapping(raw, "manifest")
    task = _mapping(root.get("task"), "manifest.task")
    task_id = str(task.get("id", ""))
    episodes = root.get("episodes")
    if task_id not in EXPECTED_TASKS or not isinstance(episodes, list):
        raise ValueError(f"unsupported task/episode contract in {manifest}")
    records: list[dict[str, object]] = []
    for position, value in enumerate(episodes):
        episode = _mapping(value, f"episodes[{position}]")
        relative = episode.get("hdf5_path")
        expected_sha256 = str(episode.get("hdf5_sha256", ""))
        expected_size = episode.get("hdf5_size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or SHA256_PATTERN.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise ValueError(f"invalid HDF5 identity in {manifest} episode {position}")
        path = (manifest.parent / relative).resolve(strict=True)
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            raise ValueError(f"HDF5 size/type differs from manifest: {path}")
        records.append(
            {
                "episode_index": int(episode.get("episode_index", position)),
                "path": str(path),
                "hdf5_sha256": expected_sha256,
                "size_bytes": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            }
        )
    return task_id, records


def create_shared_hdf5_receipt(
    manifests: Sequence[str | Path],
    *,
    proof_checkpoint: str | Path,
    expected_proof_sha256: str,
    output: str | Path,
    verify_imported_content_if_newer: bool = False,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Create a stat-bound receipt from the exact accepted R6L-P1 proof.

    Files older than the proof reuse its fail-closed content verification.  A
    freshly downloaded/copied file has a newer mtime and is accepted only when
    the caller explicitly requests a one-time SHA256 comparison with the exact
    manifest.  The resulting receipt then binds that verified content to its
    current filesystem identity.
    """

    if SHA256_PATTERN.fullmatch(expected_proof_sha256) is None:
        raise ValueError("expected proof checkpoint SHA256 is invalid")
    proof = Path(proof_checkpoint).resolve(strict=True)
    observed_proof_sha256 = file_sha256(proof)
    if observed_proof_sha256 != expected_proof_sha256:
        raise ValueError("proof checkpoint SHA256 differs from the accepted identity")
    proof_stat = proof.stat()
    checkpoint = torch.load(proof, map_location="cpu", weights_only=False, mmap=True)
    checkpoint = _mapping(checkpoint, "proof checkpoint")
    method = _mapping(checkpoint.get("method"), "proof checkpoint.method")
    if (
        checkpoint.get("format_version") != PROOF_CHECKPOINT_FORMAT
        or method.get("micro_round") != "R6L"
        or method.get("candidate_id") != "P1"
        or method.get("model_kind") != "s3_r6l_protected_local_gated"
    ):
        raise ValueError("proof checkpoint is not the accepted R6L-P1 policy")
    proof_data = _mapping(checkpoint.get("data"), "proof checkpoint.data")
    proof_manifests_raw = proof_data.get("manifests")
    if not isinstance(proof_manifests_raw, list):
        raise ValueError("proof checkpoint does not declare dataset manifests")
    proof_manifests = {
        str(_mapping(value, "proof manifest").get("task_id")): str(
            _mapping(value, "proof manifest").get("sha256", "")
        )
        for value in proof_manifests_raw
    }
    if set(proof_manifests) != set(EXPECTED_TASKS) or any(
        SHA256_PATTERN.fullmatch(value) is None for value in proof_manifests.values()
    ):
        raise ValueError("proof checkpoint five-task manifest identities are invalid")

    manifest_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    seen_tasks: set[str] = set()
    for value in manifests:
        manifest = Path(value).resolve(strict=True)
        manifest_sha256 = file_sha256(manifest)
        task_id, records = _manifest_records(manifest)
        if task_id in seen_tasks:
            raise ValueError(f"duplicate receipt task {task_id}")
        seen_tasks.add(task_id)
        if proof_manifests.get(task_id) != manifest_sha256:
            raise ValueError(f"{task_id} manifest differs from the accepted proof")
        manifest_rows.append(
            {
                "task_id": task_id,
                "path": str(manifest),
                "sha256": manifest_sha256,
                "episodes": len(records),
            }
        )
        file_rows.extend(records)
    if seen_tasks != set(EXPECTED_TASKS) or len(file_rows) != 750:
        raise ValueError("receipt requires the exact five-task/750-episode dataset")
    manifest_rows.sort(key=lambda value: EXPECTED_TASKS.index(str(value["task_id"])))
    file_rows.sort(key=lambda value: str(value["path"]))
    imported_rows = [
        record
        for record in file_rows
        if int(record["mtime_ns"]) > proof_stat.st_mtime_ns
    ]
    if imported_rows and not verify_imported_content_if_newer:
        raise ValueError(
            "HDF5 content newer than the accepted proof requires explicit "
            "one-time manifest SHA256 verification"
        )
    imported_bytes = sum(int(record["size_bytes"]) for record in imported_rows)
    verified_bytes = 0
    for index, record in enumerate(imported_rows, start=1):
        episode = Path(str(record["path"]))
        observed_sha256 = file_sha256(episode)
        if observed_sha256 != record["hdf5_sha256"]:
            raise ValueError(f"imported HDF5 SHA256 differs from manifest: {episode}")
        verified_bytes += int(record["size_bytes"])
        if progress is not None:
            progress(
                {
                    "event": "shared_hdf5_import_sha256_progress",
                    "verified_files": index,
                    "total_files": len(imported_rows),
                    "verified_bytes": verified_bytes,
                    "total_bytes": imported_bytes,
                    "path": str(episode),
                }
            )
    verification_semantics = (
        "accepted checkpoint was built after fail-closed manifest HDF5 SHA256 "
        "verification; current files predating that proof reuse it"
    )
    if imported_rows:
        verification_semantics += (
            "; files newer than the proof were rehashed against the exact "
            "accepted manifests before this stat-bound receipt was written"
        )
    payload: dict[str, object] = {
        "format_version": RECEIPT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verification_semantics": verification_semantics,
        "content_verification": {
            "mode": (
                "accepted_proof_plus_import_manifest_sha256"
                if imported_rows
                else "accepted_proof_mtime_reuse"
            ),
            "imported_files_sha256_verified": len(imported_rows),
            "imported_bytes_sha256_verified": verified_bytes,
            "all_750_files_content_anchored": True,
        },
        "proof": {
            "path": str(proof),
            "sha256": observed_proof_sha256,
            "size_bytes": proof_stat.st_size,
            "mtime_ns": proof_stat.st_mtime_ns,
            "format_version": checkpoint["format_version"],
            "source_git_commit": _mapping(
                checkpoint.get("source"), "proof checkpoint.source"
            ).get("git_commit"),
        },
        "manifests": manifest_rows,
        "files": file_rows,
    }
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite receipt {destination}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(destination)
    return payload


def validate_shared_hdf5_receipt(
    receipt: str | Path,
    manifests: Sequence[str | Path],
    *,
    expected_proof_sha256: str,
    expected_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Validate a receipt without reading the 707-GiB HDF5 payload again."""

    path = Path(receipt).resolve(strict=True)
    if expected_receipt_sha256 is not None:
        if (
            SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None
            or file_sha256(path) != expected_receipt_sha256
        ):
            raise ValueError("shared HDF5 receipt SHA256 differs from runner identity")
    try:
        raw = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid shared HDF5 receipt: {path}") from exc
    payload = _mapping(raw, "receipt")
    if payload.get("format_version") != RECEIPT_FORMAT:
        raise ValueError("unsupported shared HDF5 receipt format")
    proof = _mapping(payload.get("proof"), "receipt.proof")
    if proof.get("sha256") != expected_proof_sha256:
        raise ValueError("receipt proof differs from the configured accepted policy")
    proof_path = Path(str(proof.get("path", ""))).resolve(strict=True)
    proof_stat = proof_path.stat()
    if (
        proof_stat.st_size != proof.get("size_bytes")
        or proof_stat.st_mtime_ns != proof.get("mtime_ns")
    ):
        raise ValueError("accepted proof checkpoint changed after receipt creation")

    rows = payload.get("manifests")
    files = payload.get("files")
    if not isinstance(rows, list) or not isinstance(files, list) or len(files) != 750:
        raise ValueError("receipt manifest/file inventory is invalid")
    expected_manifest_paths = {Path(value).resolve(strict=True) for value in manifests}
    observed_manifest_paths: set[Path] = set()
    for value in rows:
        row = _mapping(value, "receipt manifest")
        manifest = Path(str(row.get("path", ""))).resolve(strict=True)
        observed_manifest_paths.add(manifest)
        if file_sha256(manifest) != row.get("sha256"):
            raise ValueError(f"dataset manifest changed after receipt creation: {manifest}")
    if observed_manifest_paths != expected_manifest_paths:
        raise ValueError("receipt manifests differ from the configured dataset")

    observed_files: set[Path] = set()
    for value in files:
        row = _mapping(value, "receipt file")
        episode = Path(str(row.get("path", ""))).resolve(strict=True)
        if episode in observed_files:
            raise ValueError(f"receipt duplicates HDF5 path: {episode}")
        observed_files.add(episode)
        metadata = episode.stat()
        expected = (
            row.get("device"),
            row.get("inode"),
            row.get("size_bytes"),
            row.get("mtime_ns"),
        )
        actual = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        if actual != expected or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"HDF5 identity changed after receipt creation: {episode}")
    return dict(payload)
