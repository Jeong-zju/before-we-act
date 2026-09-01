"""Revision-pinned, resumable downloader for the official BiCoord dataset.

Authentication is read exclusively from ``HF_TOKEN`` or
``HUGGINGFACE_HUB_TOKEN``.  The token is passed through the Hub Python API and
never appears in argv, logs, or receipts.  A successful stage proves the
resolved commit and all 1,800 HDF5 paths; content/schema hashes are handled by
the following audit stage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Sequence

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    TASKS,
    TOTAL_EPISODES,
)
from .hdf5_data import discover_all_episode_files
from .stage_common import artifact, assert_common_paths, atomic_json, common_parser, publish_result, read_json, sha256_file, utc_now


TREE_FORMAT_VERSION = 1
EXPECTED_SNAPSHOT_FILES = 9_057
DOWNLOAD_INTENT_SCHEMA = "before-we-act.bicoord-download-intent/1"
DOWNLOAD_RECEIPT_SCHEMA = "before-we-act.bicoord-dataset-download/1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")


def _tree_path(dataset: Path) -> Path:
    return (
        dataset
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{DATASET_REVISION}.json"
    )


def _hub_tree_rows(
    api: Any,
) -> dict[str, dict[str, Any]]:
    """Read the immutable file table for the pinned Hub revision.

    ``snapshot_download(local_dir=...)`` writes per-file ``*.metadata``
    sidecars, but it does not expose the complete path/object table that the
    offline verifier needs.  Build that table from ``HfApi.list_repo_tree``
    when a local snapshot does not already carry our versioned manifest.  The
    caller supplies an API object so this helper stays straightforward to
    mock in tests and never handles credentials itself.
    """

    try:
        # ``expand=False`` keeps the recursive listing on the inexpensive tree
        # endpoint.  RepoFile still carries size/blob_id/LFS/Xet identities;
        # this was verified against the pinned BiCoord revision with
        # huggingface_hub 1.28.0.  ``expand=True`` asks the Hub for per-entry
        # commit metadata that this manifest neither records nor trusts and is
        # disproportionately slow for the 9,057-file snapshot.
        entries = api.list_repo_tree(
            DATASET_REPO_ID,
            path_in_repo="",
            recursive=True,
            expand=False,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )
    except TypeError:
        # Older huggingface_hub releases do not accept ``expand``.  The
        # immutable blob/LFS identity is still available in their returned
        # records, so retry with the compatible signature.
        entries = api.list_repo_tree(
            DATASET_REPO_ID,
            path_in_repo="",
            recursive=True,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )
    rows: dict[str, dict[str, Any]] = {}
    for entry in entries:
        # RepoFolder objects have no ``size``/``blob_id`` and must not enter
        # the file manifest.
        relative = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        blob_id = getattr(entry, "blob_id", None)
        if relative is None and isinstance(entry, dict):
            relative = entry.get("path")
            size = entry.get("size")
            blob_id = entry.get("blob_id", entry.get("oid"))
        if not isinstance(relative, str) or size is None or blob_id is None:
            continue
        row: dict[str, Any] = {"size": int(size), "blob_id": str(blob_id)}
        lfs = getattr(entry, "lfs", None)
        xet_hash = getattr(entry, "xet_hash", None)
        if isinstance(entry, dict):
            lfs = entry.get("lfs", lfs)
            xet_hash = entry.get("xet_hash", entry.get("xetHash", xet_hash))
        if lfs is not None:
            lfs_size = getattr(lfs, "size", None)
            lfs_sha = getattr(lfs, "sha256", None)
            if isinstance(lfs, dict):
                lfs_size = lfs.get("size", lfs_size)
                lfs_sha = lfs.get("sha256", lfs.get("oid", lfs_sha))
            if lfs_size is not None and lfs_sha is not None:
                row.update(lfs_size=int(lfs_size), lfs_sha256=str(lfs_sha))
        if xet_hash is not None:
            row["xet_hash"] = str(xet_hash)
        if relative in rows:
            raise RuntimeError(f"Hugging Face tree contains duplicate path: {relative}")
        # Reuse the same strict path/identity checks as the offline verifier.
        _safe_manifest_parts(relative)
        _tree_object_identity(row, relative)
        rows[relative] = row
    if not rows:
        raise RuntimeError("Hugging Face API returned an empty file tree")
    if len(rows) != EXPECTED_SNAPSHOT_FILES:
        raise RuntimeError(
            "pinned Hugging Face API tree coverage differs: "
            f"{len(rows)} != {EXPECTED_SNAPSHOT_FILES}"
        )
    hdf5 = {relative for relative in rows if relative.lower().endswith(".hdf5")}
    expected_hdf5 = {
        f"{task}/demo_clean/data/episode{episode}.hdf5"
        for task in TASKS
        for episode in range(100)
    }
    if hdf5 != expected_hdf5:
        raise RuntimeError("pinned Hugging Face API tree has unexpected HDF5 coverage")
    return rows


def _ensure_tree_manifest(dataset: Path, api: Any) -> Path:
    """Create the verifier manifest when Hub did not provide it locally."""

    dataset = dataset.expanduser().resolve(strict=True)
    path = _tree_path(dataset)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Hugging Face tree manifest is not a regular file: {path}")
        return path
    rows = _hub_tree_rows(api)
    # ``atomic_json`` creates parent directories and fsyncs the replacement.
    # Dataset download is a singleton CPU stage under the supervisor.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        atomic_json(path, {"format_version": TREE_FORMAT_VERSION, "files": rows})
    return path


def _safe_manifest_parts(relative: str) -> tuple[str, ...]:
    """Return path components which are safe to join below a snapshot root.

    Hugging Face tree entries use POSIX paths even on hosts where ``Path``
    uses another separator.  Checking the POSIX representation first avoids
    accepting ``.``/``..`` aliases, and callers additionally verify every
    resolved component below the intended root (including parent symlinks).
    """

    if not isinstance(relative, str):
        raise RuntimeError(f"unsafe path in Hugging Face tree manifest: {relative!r}")
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or ".." in posix.parts
        or "." in posix.parts
        or str(posix) != relative
        or any(not part for part in posix.parts)
    ):
        raise RuntimeError(f"unsafe path in Hugging Face tree manifest: {relative!r}")
    return tuple(posix.parts)


def _tree_object_identity(entry: dict[str, Any], relative: str) -> tuple[str, str]:
    """Return ``(kind, object id)`` from one immutable Hub-tree row."""

    lfs = entry.get("lfs_sha256")
    blob = entry.get("blob_id")
    if lfs is not None:
        if not isinstance(lfs, str) or _SHA256_RE.fullmatch(lfs) is None:
            raise RuntimeError(f"snapshot tree has invalid LFS identity: {relative}")
        lfs_size = entry.get("lfs_size")
        if lfs_size is not None and (
            isinstance(lfs_size, bool)
            or not isinstance(lfs_size, int)
            or lfs_size != entry.get("size")
        ):
            raise RuntimeError(f"snapshot tree has inconsistent LFS size: {relative}")
        return "lfs_sha256", lfs
    if not isinstance(blob, str) or _GIT_BLOB_RE.fullmatch(blob) is None:
        raise RuntimeError(f"snapshot tree lacks a valid object identity: {relative}")
    return "git_blob_sha1", blob


def _assert_no_symlink_components(root: Path, target: Path) -> Path:
    """Resolve ``target`` and reject symlinked components under ``root``.

    Checking only ``target.is_symlink()`` is insufficient: a symlinked parent
    (for example ``task -> /tmp/other``) can redirect an otherwise ordinary
    file outside the pinned snapshot.  Resolve and inspect each component so
    both the tree files and metadata sidecars are confined to their roots.
    """

    try:
        root = root.expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"snapshot path root is missing: {root}") from error
    target = target.expanduser()
    if not target.is_absolute():
        target = root / target
    try:
        lexical = target.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"snapshot path escapes root: {target}") from error
    cursor = root
    for component in lexical.parts:
        cursor = cursor / component
        # Inspect the lexical path.  Inspecting only its resolved spelling can
        # hide an in-root symlink such as ``task -> real_task``.
        if cursor.is_symlink():
            raise RuntimeError(f"snapshot path contains symbolic component: {cursor}")
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError(f"snapshot path escapes or is missing: {target}") from error
    return resolved


def _safe_snapshot_path(dataset: Path, relative: str) -> Path:
    """Resolve one Hub-tree path without permitting traversal or symlinks."""

    parts = _safe_manifest_parts(relative)
    target = _assert_no_symlink_components(dataset, dataset.joinpath(*parts))
    if not target.is_file():
        raise RuntimeError(f"snapshot file is missing or symbolic: {target}")
    return target


def _verify_local_snapshot(
    dataset: Path, *, expected_files: int | None = None
) -> dict[str, Any]:
    """Prove a complete pinned ``snapshot_download(local_dir=...)`` tree.

    The Hub tree manifest supplies every path and byte size.  Its per-file
    metadata sidecars bind those local files to the requested immutable
    revision and object identity.  This is deliberately performed before any
    token lookup or network call, so a complete cached snapshot is reusable
    without exposing credentials to the long-running supervisor.
    """

    dataset_input = dataset.expanduser()
    # The root itself may be a user-selected mount/symlink, but after this
    # point every manifest path is required to resolve beneath this canonical
    # directory.  Keeping one canonical root also makes receipt paths stable.
    dataset = dataset_input.resolve(strict=True)
    if not dataset.is_dir():
        raise RuntimeError(f"dataset snapshot root is not a directory: {dataset}")
    expected_files = (
        EXPECTED_SNAPSHOT_FILES if expected_files is None else int(expected_files)
    )
    tree_path = _assert_no_symlink_components(dataset, _tree_path(dataset))
    if not tree_path.is_file():
        raise RuntimeError(f"Hugging Face tree manifest is not a file: {tree_path}")
    tree = read_json(tree_path)
    if int(tree.get("format_version", -1)) != TREE_FORMAT_VERSION:
        raise RuntimeError("unsupported Hugging Face tree-manifest format")
    files = tree.get("files")
    if not isinstance(files, dict) or len(files) != int(expected_files):
        raise RuntimeError(
            "pinned Hugging Face tree coverage differs: "
            f"{len(files) if isinstance(files, dict) else 'invalid'} != {expected_files}"
        )

    metadata_root = dataset / ".cache" / "huggingface" / "download"
    # The metadata root itself is part of the snapshot trust boundary.  If a
    # mount accidentally leaves it as a symlink to another cache, checking
    # only each final ``*.metadata`` file would still authorize foreign
    # provenance.
    metadata_root = _assert_no_symlink_components(dataset, metadata_root)
    if not metadata_root.is_dir():
        raise RuntimeError(f"snapshot metadata root is missing: {metadata_root}")
    hdf5_from_tree: set[str] = set()
    total_bytes = 0
    for relative, entry in files.items():
        if not isinstance(relative, str) or not isinstance(entry, dict):
            raise RuntimeError("Hugging Face tree contains a malformed file row")
        target = _safe_snapshot_path(dataset, relative)
        expected_size = entry.get("size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or target.stat().st_size != expected_size
        ):
            raise RuntimeError(f"snapshot byte size differs: {target}")
        total_bytes += expected_size

        _identity_kind, object_id = _tree_object_identity(entry, relative)
        parts = _safe_manifest_parts(relative)
        metadata = metadata_root.joinpath(*parts[:-1]) / (parts[-1] + ".metadata")
        try:
            metadata = _assert_no_symlink_components(metadata_root, metadata)
        except RuntimeError as error:
            raise RuntimeError(f"snapshot metadata is missing or symbolic: {metadata}") from error
        if not metadata.is_file():
            raise RuntimeError(f"snapshot metadata sidecar is missing: {metadata}")
        with metadata.open("r", encoding="utf-8", errors="strict") as stream:
            revision = stream.readline(256).rstrip("\r\n")
            recorded_object = stream.readline(256).rstrip("\r\n")
        if revision != DATASET_REVISION or recorded_object != object_id:
            raise RuntimeError(f"snapshot metadata provenance differs: {metadata}")
        if relative.lower().endswith(".hdf5"):
            hdf5_from_tree.add(relative)

    discovered = discover_all_episode_files(dataset, require_complete=True)
    hdf5_paths = [path.resolve(strict=True) for task in TASKS for path in discovered[task]]
    try:
        relative_hdf5 = {path.relative_to(dataset).as_posix() for path in hdf5_paths}
    except ValueError as error:
        raise RuntimeError("discovered HDF5 path escapes the pinned dataset root") from error
    if (
        len(hdf5_paths) != TOTAL_EPISODES
        or len(relative_hdf5) != TOTAL_EPISODES
        or relative_hdf5 != hdf5_from_tree
    ):
        raise RuntimeError("Hugging Face tree and 1,800 HDF5 episode paths differ")
    counts = {task: len(discovered[task]) for task in TASKS}
    return {
        "tree_manifest": str(tree_path.resolve()),
        "tree_manifest_sha256": sha256_file(tree_path),
        "tree_format_version": TREE_FORMAT_VERSION,
        "snapshot_files": len(files),
        "snapshot_bytes": total_bytes,
        "metadata_sidecars_verified": len(files),
        "hdf5_paths": [str(path) for path in hdf5_paths],
        "episodes": len(hdf5_paths),
        "episodes_per_task": counts,
    }


def _valid_partial_tree_identity(dataset: Path) -> bool:
    """Validate the immutable identity portion of an incomplete Hub tree.

    A full tree manifest is written before all payloads finish downloading.
    It is therefore a safe resume marker only when its complete path/object
    table matches the frozen BiCoord coverage.  A merely parseable ``files``
    dictionary is not enough to authorize mixing into a non-empty directory.
    """

    try:
        dataset = dataset.expanduser().resolve(strict=True)
        tree_path = _assert_no_symlink_components(dataset, _tree_path(dataset))
        tree = read_json(tree_path)
        files = tree.get("files")
        if (
            int(tree.get("format_version", -1)) != TREE_FORMAT_VERSION
            or not isinstance(files, dict)
            or len(files) != EXPECTED_SNAPSHOT_FILES
        ):
            return False
        hdf5: set[str] = set()
        for relative, entry in files.items():
            if not isinstance(relative, str) or not isinstance(entry, dict):
                return False
            _safe_manifest_parts(relative)
            size = entry.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                return False
            _tree_object_identity(entry, relative)
            if relative.lower().endswith(".hdf5"):
                hdf5.add(relative)
        expected_hdf5 = {
            f"{task}/demo_clean/data/episode{episode}.hdf5"
            for task in TASKS
            for episode in range(100)
        }
        return hdf5 == expected_hdf5
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _intent(evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": DOWNLOAD_INTENT_SCHEMA,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
    }
    if evidence is not None:
        value.update(
            {
                "state": "VERIFIED",
                "tree_manifest_sha256": evidence["tree_manifest_sha256"],
                "snapshot_files": evidence["snapshot_files"],
            }
        )
    else:
        value["state"] = "DOWNLOADING"
    return value


def _valid_existing_intent(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return False
    head = {
        "schema": DOWNLOAD_INTENT_SCHEMA,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
    }
    if any(value.get(key) != expected for key, expected in head.items()):
        return False
    state = value.get("state")
    if state == "DOWNLOADING":
        return value == _intent()
    if state == "VERIFIED":
        return (
            set(value) == {
                *head,
                "state",
                "tree_manifest_sha256",
                "snapshot_files",
            }
            and isinstance(value.get("tree_manifest_sha256"), str)
            and _SHA256_RE.fullmatch(str(value["tree_manifest_sha256"])) is not None
            and isinstance(value.get("snapshot_files"), int)
            and not isinstance(value.get("snapshot_files"), bool)
            and int(value["snapshot_files"]) > 0
        )
    return False


def _read_existing_intent(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or not _valid_existing_intent(path):
        raise RuntimeError(f"invalid pinned download intent: {path}")
    return read_json(path)


def _finalize_intent(path: Path, evidence: dict[str, Any]) -> None:
    expected = _intent(evidence)
    if path.is_symlink():
        raise RuntimeError(f"download intent must not be symbolic: {path}")
    if not path.is_file():
        # Even a pre-populated offline snapshot gets the same monotonic state
        # machine as a fresh network transfer.  Never materialize VERIFIED
        # without first declaring the exact pinned download intent.
        atomic_json(path, _intent())
    current = read_json(path)
    if not _valid_existing_intent(path):
        raise RuntimeError("download intent changed while verifying snapshot")
    state = current.get("state")
    if state == "DOWNLOADING":
        # The only permitted mutation is the monotonic transition from a
        # pinned partial-download declaration to the verified tree identity.
        atomic_json(path, expected)
    elif state != "VERIFIED" or current != expected:
        raise RuntimeError("verified download intent differs from local snapshot")


def _validate_existing_receipt(
    receipt: dict[str, Any],
    *,
    dataset: Path,
    evidence: dict[str, Any],
    intent_sha256: str,
) -> None:
    """Bind an existing PASSED receipt to the current tree and intent.

    A PASSED file is immutable evidence.  It may be reused, but it must never
    be silently rewritten to bless a changed snapshot.  This checks every
    stable provenance/coverage field while intentionally allowing its
    original timestamp and credential-source label to remain unchanged.
    """

    expected = {
        "schema": DOWNLOAD_RECEIPT_SCHEMA,
        "status": "PASSED",
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "resolved_revision": DATASET_REVISION,
        "snapshot_path": str(dataset.resolve()),
        "local_dir": str(dataset.resolve()),
        "tasks": list(TASKS),
        "episodes_per_task": dict(evidence["episodes_per_task"]),
        "episodes": int(evidence["episodes"]),
        "hdf5_paths": list(evidence["hdf5_paths"]),
        "token_embedded": False,
        "download_intent_sha256": intent_sha256,
        "tree_manifest": evidence["tree_manifest"],
        "tree_manifest_sha256": evidence["tree_manifest_sha256"],
        "snapshot_files": int(evidence["snapshot_files"]),
        "snapshot_bytes": int(evidence["snapshot_bytes"]),
        "metadata_sidecars_verified": int(evidence["metadata_sidecars_verified"]),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"existing PASSED dataset receipt provenance differs at {key}"
            )
    allowed_sources = {
        "HF_TOKEN_environment",
        "HUGGINGFACE_HUB_TOKEN_environment",
        "local_snapshot_verified_without_token",
    }
    if receipt.get("token_source") not in allowed_sources:
        raise RuntimeError("existing PASSED dataset receipt has invalid token provenance")
    if not isinstance(receipt.get("downloaded_at"), str) or not receipt["downloaded_at"]:
        raise RuntimeError("existing PASSED dataset receipt lacks completion time")


def _token() -> tuple[str, str]:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "HF_TOKEN_environment"
    if os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return os.environ["HUGGINGFACE_HUB_TOKEN"], "HUGGINGFACE_HUB_TOKEN_environment"
    raise RuntimeError(
        "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required; refusing anonymous formal download"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args)
    args.dataset.mkdir(parents=True, exist_ok=True)
    args.dataset = args.dataset.expanduser().resolve(strict=True)
    dataset_receipt = args.dataset / "dataset_receipt.json"
    intent_path = args.dataset / ".bicoord_download_intent.json"
    snapshot: str = str(args.dataset)

    # A completed local HF snapshot is a first-class offline input.  In
    # particular, do not even inspect HF_TOKEN on this path: credentials must
    # not be needed merely to resume a verified immutable tree.
    try:
        evidence = _verify_local_snapshot(args.dataset)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as offline_error:
        evidence = None
        offline_reason = str(offline_error)
    else:
        offline_reason = None

    existing_intent = _read_existing_intent(intent_path)
    if evidence is None and existing_intent is not None and existing_intent["state"] == "VERIFIED":
        # VERIFIED is terminal.  Never mutate/download into a snapshot which
        # no longer satisfies the exact tree named by that state.
        raise RuntimeError(
            "VERIFIED download intent no longer matches the local snapshot: "
            f"{offline_reason}"
        )

    prior: dict[str, Any] | None = None
    if dataset_receipt.exists():
        if dataset_receipt.is_symlink() or not dataset_receipt.is_file():
            raise RuntimeError(f"dataset receipt must be a regular file: {dataset_receipt}")
        prior = read_json(dataset_receipt)
        if prior.get("status") != "PASSED":
            # A previous failed/partial attempt is evidence of failure, not a
            # license to manufacture a successful receipt from a directory.
            raise RuntimeError(
                f"existing dataset receipt is not PASSED; refusing to overwrite: {dataset_receipt}"
            )
        if (
            prior.get("dataset_repo_id") != DATASET_REPO_ID
            or prior.get("dataset_revision") != DATASET_REVISION
        ):
            raise RuntimeError("existing dataset receipt belongs to another revision")
        if evidence is None:
            raise RuntimeError(
                "existing PASSED dataset receipt no longer matches local snapshot: "
                f"{offline_reason}"
            )
        expected_intent = _intent(evidence)
        if existing_intent != expected_intent:
            raise RuntimeError(
                "existing PASSED dataset receipt lacks its exact VERIFIED download intent"
            )
        _validate_existing_receipt(
            prior,
            dataset=args.dataset,
            evidence=evidence,
            intent_sha256=sha256_file(intent_path),
        )

    reused = evidence is not None
    token_source = "local_snapshot_verified_without_token"
    resolved = DATASET_REVISION
    if not reused:
        # Only a real download needs credentials or network metadata.  A
        # pre-existing non-empty directory is accepted for resume only when a
        # pinned intent (or a matching Hub tree manifest) proves its identity.
        if existing_intent is not None:
            if existing_intent["state"] != "DOWNLOADING":
                raise RuntimeError("only a DOWNLOADING intent may resume network transfer")
        else:
            tree_identity_ok = _valid_partial_tree_identity(args.dataset)
            non_hidden_files = any(
                path.is_file() and not path.name.startswith(".")
                for path in args.dataset.rglob("*")
            )
            if non_hidden_files and not tree_identity_ok:
                raise RuntimeError(
                    "dataset directory is non-empty without a pinned intent/tree; "
                    f"refusing to mix snapshots: {args.dataset}"
                )
            atomic_json(intent_path, _intent())
            existing_intent = _intent()

        token, token_source = _token()
        try:
            from huggingface_hub import HfApi, snapshot_download
        except Exception as error:  # pragma: no cover - host dependent
            raise RuntimeError("huggingface_hub is required for BiCoord download") from error
        api = HfApi(token=token)
        info = api.dataset_info(
            DATASET_REPO_ID,
            revision=DATASET_REVISION,
            files_metadata=False,
        )
        resolved = str(getattr(info, "sha", ""))
        if resolved != DATASET_REVISION:
            raise RuntimeError(
                f"Hugging Face revision drift: requested {DATASET_REVISION}, resolved {resolved}"
            )
        _ensure_tree_manifest(args.dataset, api)
        snapshot = str(
            snapshot_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=str(args.dataset),
                token=token,
                max_workers=int(os.environ.get("BICOORD_HF_WORKERS", "16")),
            )
        )
        downloaded_root = Path(snapshot).expanduser().resolve(strict=True)
        if downloaded_root != args.dataset:
            raise RuntimeError(
                "snapshot_download returned a path outside the pinned local_dir: "
                f"{downloaded_root} != {args.dataset}"
            )
        evidence = _verify_local_snapshot(args.dataset)
        reused = False

    assert evidence is not None
    paths = list(evidence["hdf5_paths"])
    counts = dict(evidence["episodes_per_task"])
    total = int(evidence["episodes"])
    # Pin the intent after verification.  The receipt itself references this
    # hash; changing either file makes the next invocation fail closed.
    _finalize_intent(intent_path, evidence)
    intent_sha = sha256_file(intent_path)
    new_receipt = {
        "schema": DOWNLOAD_RECEIPT_SCHEMA,
        "status": "PASSED",
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "resolved_revision": resolved,
        "snapshot_path": str(Path(snapshot).resolve()),
        "local_dir": str(args.dataset.resolve()),
        "tasks": list(TASKS),
        "episodes_per_task": counts,
        "episodes": total,
        "hdf5_paths": paths,
        "token_source": token_source,
        "token_embedded": False,
        "download_intent_sha256": intent_sha,
        "tree_manifest": evidence["tree_manifest"],
        "tree_manifest_sha256": evidence["tree_manifest_sha256"],
        "snapshot_files": evidence["snapshot_files"],
        "snapshot_bytes": evidence["snapshot_bytes"],
        "metadata_sidecars_verified": evidence["metadata_sidecars_verified"],
        "downloaded_at": utc_now(),
    }
    # Keep a copy with the immutable data snapshot and another under the run.
    # The data copy lets a restarted audit verify provenance even if the stage
    # result directory was moved.
    # A previously PASSED receipt is immutable.  Reusing its bytes preserves
    # every downstream artifact hash; a provenance mismatch was rejected
    # above instead of being overwritten with a newly blessed receipt.
    receipt = prior if prior is not None else new_receipt
    if prior is None:
        atomic_json(dataset_receipt, receipt)
    run_receipt = args.run / "artifacts" / "dataset_download" / "dataset_receipt.json"
    atomic_json(run_receipt, receipt)
    return publish_result(
        args,
        stage="dataset_download",
        artifacts=[
            artifact(dataset_receipt, kind="dataset_download_receipt"),
            artifact(run_receipt, kind="dataset_download_receipt_copy"),
        ],
        dataset=str(args.dataset.resolve()),
        dataset_revision=DATASET_REVISION,
        episodes=total,
        episodes_per_task=counts,
        token_source=token_source,
        reused_existing_snapshot=reused,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = common_parser(__doc__, ("download",))
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run"]
