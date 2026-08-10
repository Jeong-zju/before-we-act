"""Fail-closed verification for R11 read-only official source checkouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed for {root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _normal_url(value: str) -> str:
    return value.removesuffix(".git").rstrip("/")


def read_source_receipt(path: str | Path) -> dict:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("format_version") != "before-we-act.r11.source_receipt/1":
        raise ValueError(f"unsupported source receipt: {path}")
    if not receipt.get("upstream_url") or len(receipt.get("upstream_commit", "")) != 40:
        raise ValueError(f"incomplete source identity in {path}")
    if not receipt.get("files"):
        raise ValueError(f"source receipt has no files: {path}")
    return receipt


def verify_vendor_checkout(
    receipt_path: str | Path,
    checkout: str | Path,
    *,
    require_clean: bool = True,
) -> dict:
    """Verify commit, origin and every source file before adding it to imports."""

    receipt = read_source_receipt(receipt_path)
    root = Path(checkout).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    actual_commit = _git(root, "rev-parse", "HEAD")
    if actual_commit != receipt["upstream_commit"]:
        raise ValueError(
            f"vendor commit mismatch: expected {receipt['upstream_commit']}, got {actual_commit}"
        )
    origin = _git(root, "remote", "get-url", "origin")
    if _normal_url(origin) != _normal_url(receipt["upstream_url"]):
        raise ValueError(
            f"vendor origin mismatch: expected {receipt['upstream_url']}, got {origin}"
        )
    if require_clean:
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise ValueError(f"vendor checkout contains tracked changes: {dirty.splitlines()[0]}")

    verified = []
    for entry in receipt["files"]:
        relative = entry["upstream_path"]
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"source path escapes checkout: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(
                f"source hash mismatch for {relative}: expected {entry['sha256']}, "
                f"got {actual_sha256}"
            )
        verified.append(
            {
                "upstream_path": relative,
                "sha256": actual_sha256,
                "purpose": entry.get("purpose", "runtime"),
            }
        )
    return {
        "status": "PASSED",
        "upstream_url": receipt["upstream_url"],
        "upstream_commit": actual_commit,
        "checkout": str(root),
        "files": verified,
    }


def validate_asset_receipt(path: str | Path, expected: Mapping[str, str]) -> dict:
    """Validate public foundation files without ever reading authentication state."""

    receipt_path = Path(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASSED":
        raise ValueError(f"foundation receipt is not PASSED: {receipt_path}")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"foundation receipt mismatch at {key}")
    asset = Path(receipt["local_path"])
    actual = sha256_file(asset)
    if actual != receipt.get("sha256"):
        raise ValueError(f"foundation file hash mismatch: {asset}")
    return receipt
