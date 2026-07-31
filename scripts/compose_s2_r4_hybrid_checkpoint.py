#!/usr/bin/env python3
"""Write a lightweight, evaluate-only S2-R4 hybrid source manifest."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_s2_r4_future_predictor import CHECKPOINT_FORMAT  # noqa: E402
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _git_commit,
    _load_yaml,
    _mapping,
    _sha256,
)
from train.s2_future_prediction import file_sha256  # noqa: E402
from train.s2_model_registry import (  # noqa: E402
    validate_s2_r4_hybrid_diagnostic,
)


FORMAT_VERSION = "wam.robofactory.s2_r4.protected_hybrid_manifest/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--own-source", type=Path, required=True)
    parser.add_argument("--team-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    own_path = args.own_source.expanduser().resolve(strict=True)
    team_path = args.team_source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if own_path == team_path:
        raise ValueError("hybrid own and team sources must be distinct")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite hybrid manifest {output}")

    config = _load_yaml(config_path)
    model_kind = validate_s2_r4_hybrid_diagnostic(_mapping(config, "round"))
    own = _checkpoint(own_path, candidate="P0", team_shared=False)
    team = _checkpoint(team_path, candidate="P1", team_shared=True)
    compatibility = _compatibility(own, team)
    if not all(compatibility.values()):
        failed = sorted(key for key, passed in compatibility.items() if not passed)
        raise ValueError(f"hybrid source compatibility failed: {failed}")

    prefixes = tuple(str(value) for value in _mapping(config, "sources")[
        "discard_team_prefixes"
    ])
    team_keys = tuple(str(key) for key in _mapping(team, "model"))
    discarded = tuple(
        key for key in team_keys if any(key.startswith(prefix) for prefix in prefixes)
    )
    retained = tuple(key for key in team_keys if key not in discarded)
    if not discarded or not retained:
        raise ValueError("hybrid must discard P1 own/local and retain team keys")
    if any(key.startswith("local_predictor.") for key in retained):
        raise ValueError("P1 local weights cannot survive hybrid composition")

    payload = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_id": "s2-r4-hybrid",
        "candidate_id": "HYBRID",
        "model_kind": model_kind,
        "mode": "evaluate_only",
        "training_performed": False,
        "optimizer_created": False,
        "statistics_fitted": False,
        "sources": {
            "protected_own": _source_identity(own_path, own),
            "team": _source_identity(team_path, team),
        },
        "composition": {
            "own_output": "protected_p0_direct",
            "team_input_projection": "protected_p0_clone_detached",
            "team_modules": "old_p1_peer_shared_only",
            "discarded_team_source_keys": list(discarded),
            "retained_team_source_keys": list(retained),
        },
        "compatibility": compatibility,
        "artifacts": {
            "future_artifacts_sha256": own["future_artifacts_sha256"],
            "flow_checkpoint_sha256": _mapping(own, "frozen_parent")[
                "flow_checkpoint_sha256"
            ],
            "dinov3_weights_sha256": _mapping(own, "frozen_parent")[
                "dinov3_weights_sha256"
            ],
            "dinov3_config_sha256": _mapping(own, "frozen_parent")[
                "dinov3_config_sha256"
            ],
            "manifests": _mapping(own, "data")["manifests"],
        },
        "source": {
            "git_commit": _git_commit(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "hybrid_manifest": str(output),
                "own_sha256": payload["sources"]["protected_own"]["sha256"],
                "team_sha256": payload["sources"]["team"]["sha256"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _checkpoint(path: Path, *, candidate: str, team_shared: bool) -> Mapping[str, object]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"source is not an S2-R4 checkpoint: {path}")
    method = _mapping(saved, "method")
    expected_kind = (
        "s2_r4_team_shared_action_conditioned"
        if team_shared
        else "s2_r4_local_action_conditioned"
    )
    if (
        method.get("candidate_id") != candidate
        or method.get("model_kind") != expected_kind
        or method.get("team_shared") is not team_shared
        or method.get("action_conditioning") is not True
        or method.get("world_predictor_path") != "strictly_off_path"
    ):
        raise ValueError(f"source checkpoint identity mismatch: {path}")
    model = _mapping(saved, "model")
    if any(str(key).startswith(("flow.", "vision.", "dinov3.")) for key in model):
        raise ValueError(f"source checkpoint illegally embeds frozen parents: {path}")
    return saved


def _compatibility(
    own: Mapping[str, object],
    team: Mapping[str, object],
) -> dict[str, bool]:
    own_parent = _mapping(own, "frozen_parent")
    team_parent = _mapping(team, "frozen_parent")
    own_init = _mapping(own, "initialization_parent")
    team_init = _mapping(team, "initialization_parent")
    return {
        "local_model_config_exact": own.get("model_config") == team.get("model_config"),
        "future_artifact_exact": own.get("future_artifacts_sha256")
        == team.get("future_artifacts_sha256"),
        "flow_parent_exact": own_parent.get("flow_checkpoint_sha256")
        == team_parent.get("flow_checkpoint_sha256"),
        "dinov3_weights_exact": own_parent.get("dinov3_weights_sha256")
        == team_parent.get("dinov3_weights_sha256"),
        "dinov3_config_exact": own_parent.get("dinov3_config_sha256")
        == team_parent.get("dinov3_config_sha256"),
        "r3_parent_exact": own_init.get("r3_w1_checkpoint_sha256")
        == team_init.get("r3_w1_checkpoint_sha256"),
        "training_manifest_exact": _mapping(own, "data").get("manifests")
        == _mapping(team, "data").get("manifests"),
    }


def _source_identity(path: Path, checkpoint: Mapping[str, object]) -> dict[str, object]:
    method = _mapping(checkpoint, "method")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "format_version": checkpoint["format_version"],
        "model_kind": method["model_kind"],
        "candidate_id": method["candidate_id"],
        "git_commit": _mapping(checkpoint, "source").get("git_commit"),
        "update": checkpoint.get("update"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
