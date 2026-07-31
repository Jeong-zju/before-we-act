#!/usr/bin/env python3
"""Fail-closed verification for the S2-R3 W1 initialization checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import LocalFuturePredictorConfig  # noqa: E402
from scripts.train_static_rgb_act_moe import _load_yaml, _mapping  # noqa: E402
from train.s2_future_prediction import file_sha256  # noqa: E402


FORMAT_VERSION = "wam.robofactory.s2_r3.local_future_predictor.checkpoint/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_flow/s2_r3_local_future.yaml",
    )
    return parser


def verify_checkpoint(
    checkpoint: Path,
    config: Path,
) -> dict[str, object]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    config = config.expanduser().resolve(strict=True)
    raw = _load_yaml(config)
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or value.get(
        "format_version"
    ) != FORMAT_VERSION:
        raise ValueError("checkpoint is not an S2-R3 predictor")
    method = _mapping(value, "method")
    expected_method = {
        "round_id": "s2-r3",
        "candidate_id": "W1",
        "model_kind": "s2_r3_local_action_conditioned",
        "future_scope": "local",
        "action_conditioning": True,
        "world_predictor_path": "strictly_off_path",
        "future_target_input": False,
    }
    for key, expected in expected_method.items():
        if method.get(key) != expected:
            raise ValueError(f"S2-R3 W1 method.{key} identity drifted")
    if int(value.get("update", -1)) != 10000:
        raise ValueError("S2-R3 W1 parent must complete exactly 10,000 updates")
    observed_model = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(value, "model_config"))
    )
    configured_model = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "model"))
    )
    if observed_model != configured_model:
        raise ValueError("S2-R3 W1 model config differs from the frozen recipe")
    model = value.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("S2-R3 W1 checkpoint has no predictor state")
    forbidden = ("flow", "dinov3", "vision_encoder")
    if any(
        token in str(name).lower()
        for name in model
        for token in forbidden
    ):
        raise ValueError("S2-R3 W1 predictor embeds a frozen Flow/DINO state")
    artifact_path = (
        ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
    ).resolve(strict=True)
    if value.get("future_artifacts_sha256") != file_sha256(artifact_path):
        raise ValueError("S2-R3 W1 PCA/statistics identity changed")
    flow_path = (
        ROOT / str(_mapping(raw, "parent")["flow_checkpoint"])
    ).resolve(strict=True)
    frozen = _mapping(value, "frozen_parent")
    if frozen.get("flow_checkpoint_sha256") != file_sha256(flow_path):
        raise ValueError("S2-R3 W1 frozen Flow identity changed")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "update": int(value["update"]),
        "model_kind": method["model_kind"],
        "pca_statistics_sha256": value["future_artifacts_sha256"],
        "flow_checkpoint_sha256": frozen["flow_checkpoint_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_checkpoint(args.checkpoint, args.config)
    import json

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
