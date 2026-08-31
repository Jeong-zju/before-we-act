"""Revision-pinned, resumable RoboTwin asset installer for BiCoord.

BiCoord contains only its benchmark-specific object overrides.  The simulator
also needs RoboTwin's official background textures, embodiments and base
object library.  This helper downloads the three immutable archives, checks
their ZIP integrity, extracts missing paths without overwriting BiCoord's
overrides, and publishes an atomic receipt.  Credentials are read only from
the environment.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "TianxingChen/RoboTwin2.0"
REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
ARCHIVES = (
    "background_texture.zip",
    "embodiments.zip",
    "objects.zip",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _safe_members(archive: zipfile.ZipFile, destination: Path):
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(f"unsafe ZIP member: {member.filename}") from error
        yield member, target


def _extract_missing(source: Path, destination: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(source) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member in {source}: {bad}")
        for member, target in _safe_members(archive, destination):
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            # BiCoord's objects.zip is applied first and owns its overrides.
            # Never replace an existing benchmark file with a base asset.
            if target.is_file() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            with archive.open(member) as reader, temporary.open("wb") as writer:
                while block := reader.read(16 * 1024 * 1024):
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, target)
            extracted += 1
    return extracted


def main() -> None:
    assets = Path(
        os.environ.get(
            "BICOORD_ASSETS_ROOT", "/workspace/repos/bicoord-bench/assets"
        )
    ).resolve()
    receipt_path = Path(
        os.environ.get(
            "BICOORD_ASSETS_RECEIPT",
            "/workspace/manifests/bicoord-base-assets.json",
        )
    ).resolve()
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError("Hugging Face token is not configured")
    info = HfApi(token=token).dataset_info(REPO_ID, revision=REVISION)
    if str(info.sha) != REVISION:
        raise RuntimeError(f"RoboTwin asset revision drift: {info.sha}")
    assets.mkdir(parents=True, exist_ok=True)
    download_root = assets / ".archives"
    download_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ARCHIVES:
        print(json.dumps({"event": "asset_download", "archive": name}), flush=True)
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=name,
                repo_type="dataset",
                revision=REVISION,
                local_dir=download_root,
                token=token,
            )
        ).resolve()
        print(json.dumps({"event": "asset_extract", "archive": name}), flush=True)
        count = _extract_missing(path, assets)
        rows.append(
            {
                "archive": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "files_extracted": count,
            }
        )
        _atomic_json(
            receipt_path,
            {
                "schema": "before-we-act.bicoord-base-assets/1",
                "state": "running",
                "repo_id": REPO_ID,
                "revision": REVISION,
                "assets_root": str(assets),
                "archives": rows,
                "updated_at": _now(),
            },
        )
    required = (
        assets / "embodiments" / "aloha-agilex" / "config.yml",
        assets / "objects",
        assets / "background_texture",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError(f"RoboTwin assets incomplete: {[str(x) for x in required]}")
    _atomic_json(
        receipt_path,
        {
            "schema": "before-we-act.bicoord-base-assets/1",
            "state": "complete",
            "repo_id": REPO_ID,
            "revision": REVISION,
            "assets_root": str(assets),
            "archives": rows,
            "required_paths": [str(path) for path in required],
            "completed_at": _now(),
        },
    )


if __name__ == "__main__":
    main()
