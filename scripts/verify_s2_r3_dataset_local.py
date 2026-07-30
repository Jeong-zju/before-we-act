#!/usr/bin/env python3
"""Quickly prove that one locally downloaded S2-R3 task is complete."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


FORMAT_VERSION = "wam.multimodal.trajectory.training_manifest/1"
DATASET_PROTOCOL = "generic_multimodal_trajectory"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument("--expected-episodes", type=int, default=150)
    return parser


def quick_validate_dataset(
    manifest_path: str | Path,
    *,
    expected_task: str,
    expected_episodes: int,
) -> dict[str, Any]:
    if not expected_task:
        raise ValueError("expected task cannot be empty")
    if expected_episodes <= 0:
        raise ValueError("expected episodes must be positive")
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if not manifest.is_file():
        raise ValueError(f"manifest is not a file: {manifest}")
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {manifest}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("training manifest root must be an object")
    if raw.get("format_version") != FORMAT_VERSION:
        raise ValueError("unexpected training manifest format")
    if raw.get("dataset_protocol") != DATASET_PROTOCOL:
        raise ValueError("unexpected dataset protocol")

    episodes = raw.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        observed = len(episodes) if isinstance(episodes, list) else 0
        raise ValueError(
            f"expected {expected_episodes} episodes, manifest declares {observed}"
        )
    episode_paths: set[Path] = set()
    episode_bytes = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"episodes[{index}] must be an object")
        if str(episode.get("task_id", "")) != expected_task:
            raise ValueError(f"episodes[{index}] has the wrong task id")
        episode_path = _required_local_file(
            manifest.parent,
            episode.get("hdf5_path"),
            field=f"episodes[{index}].hdf5_path",
        )
        if episode_path in episode_paths:
            raise ValueError(f"duplicate episode path: {episode_path}")
        episode_paths.add(episode_path)
        episode_bytes += episode_path.stat().st_size

    normalization = _mapping(raw, "normalization")
    normalization_path = _required_local_file(
        manifest.parent,
        normalization.get("path"),
        field="normalization.path",
    )
    source = _mapping(raw, "source")
    conversion_path = _required_local_file(
        manifest.parent,
        source.get("conversion_manifest_path"),
        field="source.conversion_manifest_path",
    )
    return {
        "complete": True,
        "task_id": expected_task,
        "episodes": len(episode_paths),
        "episode_bytes": episode_bytes,
        "manifest": str(manifest),
        "normalization": str(normalization_path),
        "conversion_manifest": str(conversion_path),
    }


def _required_local_file(root: Path, value: object, *, field: str) -> Path:
    relative = str(value or "")
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{field} must be a safe relative path")
    target = root.joinpath(*pure.parts).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"{field} is missing locally: {target}")
    if target.stat().st_size <= 0:
        raise ValueError(f"{field} is empty locally: {target}")
    return target


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be an object")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = quick_validate_dataset(
            args.manifest,
            expected_task=args.expected_task,
            expected_episodes=args.expected_episodes,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "complete": False,
                    "manifest": str(args.manifest),
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
