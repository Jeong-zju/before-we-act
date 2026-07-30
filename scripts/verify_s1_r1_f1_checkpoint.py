#!/usr/bin/env python3
"""Fail closed unless a checkpoint is the frozen, promoted S1-R1 F1 recipe."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.static_rgb_act import StaticRGBMoEACTConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_flow/s1_r1_f1_flow_cold.yaml",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Root used to resolve the config's repository-relative manifests.",
    )
    return parser


def verify_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    *,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    checkpoint = checkpoint_path.expanduser().resolve(strict=True)
    config = config_path.expanduser().resolve(strict=True)
    root = repo_root.expanduser().resolve(strict=True)
    raw = _load_yaml(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("S1-R1 F1 checkpoint root must be a mapping")

    _expect(
        payload.get("format_version")
        == "wam.robofactory.agent_factorized_flow.checkpoint/1",
        "checkpoint format",
    )
    training = _mapping(raw, "training")
    _expect(
        payload.get("update") == int(training["updates"]) == 80_000,
        "completed update count",
    )
    method = _mapping(payload, "method")
    expected_method = {
        "round_id": "s1-r1",
        "candidate_id": "F1",
        "action_generator": "rectified_flow_cold",
        "future_path": False,
        "active_agent_loss_weighting": False,
    }
    for key, expected in expected_method.items():
        _expect(method.get(key) == expected, f"method.{key}")

    generation = _mapping(raw, "generation")
    _expect(
        dict(_mapping(payload, "generation")) == dict(generation),
        "cold Gaussian four-step Euler generation contract",
    )
    _expect(
        dict(_mapping(payload, "training")) == dict(training),
        "frozen S1-R1 F1 training contract",
    )
    _expect(
        dict(_mapping(payload, "vision")) == dict(_mapping(raw, "vision")),
        "frozen DINOv3 contract",
    )
    expected_model = StaticRGBMoEACTConfig.from_dict(
        _mapping(raw, "model")
    ).to_dict()
    _expect(
        dict(_mapping(payload, "model_config")) == expected_model,
        "frozen model architecture",
    )
    _expect(
        isinstance(payload.get("model"), Mapping) and bool(payload["model"]),
        "non-empty model state",
    )

    source = _mapping(payload, "source")
    _expect(
        source.get("config_sha256") == _sha256(config),
        "S1-R1 F1 config hash",
    )
    expected_manifests = []
    for relative in _mapping(raw, "data")["manifests"]:
        manifest = (root / str(relative)).resolve(strict=True)
        expected_manifests.append(
            {
                "task_id": _manifest_task_id(manifest),
                "sha256": _sha256(manifest),
            }
        )
    observed_manifests = []
    for value in _mapping(payload, "data").get("manifests", []):
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint data.manifests entries must be mappings")
        observed_manifests.append(
            {
                "task_id": value.get("task_id"),
                "sha256": value.get("sha256"),
            }
        )
    _expect(
        observed_manifests == expected_manifests,
        "ordered training manifest identities",
    )

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(config),
        "config_sha256": _sha256(config),
        "update": payload["update"],
        "method": dict(method),
        "manifests": expected_manifests,
    }
    return result


def _manifest_task_id(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest root must be a mapping: {path}")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"manifest has no episodes: {path}")
    task_ids = {
        episode.get("task_id")
        for episode in episodes
        if isinstance(episode, Mapping)
    }
    if len(task_ids) != 1 or not all(
        isinstance(task_id, str) and task_id for task_id in task_ids
    ):
        raise ValueError(f"manifest must contain exactly one task id: {path}")
    return next(iter(task_ids))


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("config root must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(f"S1-R1 F1 checkpoint violates {label}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_checkpoint(
        args.checkpoint,
        args.config,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
