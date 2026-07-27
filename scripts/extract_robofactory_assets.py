#!/usr/bin/env python3
"""Safely extract a pinned RoboFactory asset archive and verify sentinels."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import stat
from zipfile import ZipFile, ZipInfo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require", action="append", default=[])
    return parser


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    required: Sequence[str] = (),
) -> None:
    archive_path = archive_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            _validate_member(member, output_dir=output_dir)
        corrupted = archive.testzip()
        if corrupted is not None:
            raise RuntimeError(f"corrupt RoboFactory asset archive member: {corrupted}")
        archive.extractall(output_dir)
    for relative in required:
        sentinel = _required_path(output_dir, relative)
        if not sentinel.is_file() or sentinel.stat().st_size <= 0:
            raise RuntimeError(f"RoboFactory asset sentinel is missing: {sentinel}")


def _validate_member(member: ZipInfo, *, output_dir: Path) -> None:
    destination = (output_dir / member.filename).resolve()
    if destination != output_dir and output_dir not in destination.parents:
        raise RuntimeError(f"unsafe RoboFactory asset archive path: {member.filename}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RuntimeError(f"RoboFactory asset archive contains symlink: {member.filename}")


def _required_path(output_dir: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"required asset path must be relative: {relative}")
    return output_dir / value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extract_archive(args.archive, args.output_dir, required=args.require)
    print(f"RoboFactory assets ready: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
