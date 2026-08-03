#!/usr/bin/env python3
"""Fail closed on S4-R7 config drift and paired 200-step preflight drift."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_static_rgb_act_moe import _load_yaml, _mapping  # noqa: E402
from train.s4_model_registry import validate_s4_r7_pair  # noqa: E402


FORMAT_VERSION = "wam.robofactory.s4_r7.pair_exact/1"
PREFLIGHT_FORMAT = "wam.robofactory.s4_r7.preflight/1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-config", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--p0-preflight", type=Path)
    parser.add_argument("--p1-preflight", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p0 = _load_yaml(args.p0_config.expanduser().resolve(strict=True))
    p1 = _load_yaml(args.p1_config.expanduser().resolve(strict=True))
    checks = validate_configs(p0, p1)
    preflight: dict[str, Any] | None = None
    if not args.config_only:
        if args.p0_preflight is None or args.p1_preflight is None:
            raise ValueError("full validation requires both preflight reports")
        preflight, preflight_checks = validate_preflights(
            _read_json(args.p0_preflight), _read_json(args.p1_preflight), p0, p1
        )
        checks.update(preflight_checks)
    passed = all(checks.values())
    result = {
        "format_version": FORMAT_VERSION,
        "round_id": "s4-r7",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "config_only" if args.config_only else "paired_200_step_preflight",
        "checks": checks,
        "passed": passed,
        "candidate_axis": {
            "name": "utility_coupling_weight",
            "P0": 0.0,
            "P1": 0.05,
        },
        "preflight": preflight,
    }
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


def validate_configs(
    p0: Mapping[str, Any], p1: Mapping[str, Any]
) -> dict[str, bool]:
    validate_s4_r7_pair(p0, p1)
    observed = []
    for raw in (p0, p1):
        training = _mapping(raw, "training")
        model = _mapping(raw, "model")
        evaluation = _mapping(raw, "evaluation")
        data = _mapping(raw, "data")
        vision = _mapping(raw, "vision")
        manifests = data.get("manifests")
        tasks = tuple(
            Path(str(path)).parent.name for path in manifests
        ) if isinstance(manifests, list) else ()
        recipe = {
            "budget_mode": training.get("budget_mode"),
            "updates": int(training.get("updates", 0)),
            "micro": int(training.get("micro_team_batch", 0)),
            "accum": int(training.get("gradient_accumulation", 0)),
            "effective": int(training.get("effective_team_batch", 0)),
            "counterfactual_every": int(training.get("counterfactual_every", 0)),
            "unfreeze": int(training.get("flow_unfreeze_update", 0)),
            "optimizer": training.get("optimizer"),
            "prefetch_factor": int(training.get("prefetch_factor", 0)),
            "vision_inference_batch": int(vision.get("inference_batch_size", 0)),
            "min_updates_per_second": float(
                training.get("min_updates_per_second", 0.0)
            ),
            "flow_lr": float(training.get("flow_learning_rate", -1)),
            "future_body_lr": float(training.get("future_body_learning_rate", -1)),
            "future_head_lr": float(training.get("future_head_learning_rate", -1)),
            "legacy_adapter_lr": float(
                training.get("legacy_adapter_learning_rate", -1)
            ),
            "evidence_lr": float(training.get("evidence_adapter_learning_rate", -1)),
            "router_lr": float(training.get("router_learning_rate", -1)),
            "warmup": int(training.get("warmup_updates", 0)),
            "flow_warmup": int(training.get("flow_warmup_updates", 0)),
            "weight_decay": float(training.get("weight_decay", -1)),
            "gradient_clip": float(training.get("gradient_clip_norm", -1)),
            "tasks": tasks,
            "evidence_sources": tuple(model.get("evidence_sources", ())),
            "evidence_horizons": tuple(model.get("evidence_horizons", ())),
            "rank": int(model.get("evidence_rank", 0)),
            "route_mode": model.get("route_mode"),
            "gate_max": float(model.get("new_gate_max", -1)),
            "gate_episodes": int(evaluation.get("episodes", 0)),
            "gate_seed_start": int(evaluation.get("seed_start", -1)),
            "gate_tasks": tuple(evaluation.get("tasks", ())),
            "conditions": tuple(evaluation.get("conditions", ())),
        }
        observed.append(recipe)
    checks = {
        "registered_candidate_axis_only": True,
        "same_recipe_after_axis_normalization": observed[0] == observed[1],
        "five_task_order_exact": observed[0]["tasks"] == TASKS
        and observed[0]["gate_tasks"] == TASKS,
        "fast_selection_30000_effective_team_batch_12": observed[0][
            "budget_mode"
        ]
        == "fast_selection_30k"
        and observed[0]["updates"] == 30_000
        and observed[0]["effective"] == 12
        and observed[0]["micro"] * observed[0]["accum"] == 12,
        "supported_micro_accum_pair": (observed[0]["micro"], observed[0]["accum"])
        in {(4, 3), (2, 6), (1, 12)},
        "fast_vision_and_optimizer_recipe": observed[0]["vision_inference_batch"]
        == 16
        and observed[0]["optimizer"] == "adamw_fused"
        and observed[0]["prefetch_factor"] == 4
        and observed[0]["min_updates_per_second"] == 0.75,
        "forced_audit_every_4": observed[0]["counterfactual_every"] == 4,
        "flow_unfreezes_at_update_6400": observed[0]["unfreeze"] == 6_400,
        "fixed_learning_rates": (
            observed[0]["flow_lr"],
            observed[0]["future_body_lr"],
            observed[0]["future_head_lr"],
            observed[0]["legacy_adapter_lr"],
            observed[0]["evidence_lr"],
            observed[0]["router_lr"],
        ) == (2e-5, 5e-5, 1e-4, 1e-4, 2e-4, 3e-4),
        "fixed_schedule_and_clip": observed[0]["warmup"] == 500
        and observed[0]["flow_warmup"] == 500
        and observed[0]["weight_decay"] == 1e-4
        and observed[0]["gradient_clip"] == 1.0,
        "token_preserving_contract": observed[0]["evidence_sources"]
        == ("own", "peer", "shared")
        and observed[0]["evidence_horizons"] == (1, 25, 50, 100)
        and observed[0]["rank"] == 32
        and observed[0]["route_mode"] == "dense"
        and observed[0]["gate_max"] == 0.25,
        "gate20_seed_900_919": observed[0]["gate_episodes"] == 20
        and observed[0]["gate_seed_start"] == 900,
        "normal_then_four_core_conditions_first": observed[0]["conditions"][:4]
        == (
            "normal",
            "legacy_reference",
            "world_evidence_gate_zero",
            "shuffle_all",
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"S4-R7 config pair failed: {failed}")
    return checks


def validate_preflights(
    p0: Mapping[str, Any],
    p1: Mapping[str, Any],
    p0_config: Mapping[str, Any],
    p1_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    rows = {"P0": p0, "P1": p1}
    expected_kinds = {
        "P0": "s4_r7_token_preserving",
        "P1": "s4_r7_world_utility_coupling",
    }
    for candidate, value in rows.items():
        identity = _mapping(value, "identity")
        if (
            value.get("format_version") != PREFLIGHT_FORMAT
            or identity.get("round_id") != "s4-r7"
            or identity.get("candidate_id") != candidate
            or identity.get("model_kind") != expected_kinds[candidate]
            or int(value.get("updates", 0)) != 200
        ):
            raise ValueError(f"{candidate} preflight identity is invalid")
    p0_training = _mapping(p0_config, "training")
    p1_training = _mapping(p1_config, "training")
    expected_micro = int(p0_training["micro_team_batch"])
    expected_accum = int(p0_training["gradient_accumulation"])
    memory = [float(value.get("peak_memory_bytes", 0.0)) for value in rows.values()]
    total = [float(value.get("gpu_total_memory_bytes", 0.0)) for value in rows.values()]
    headroom = [capacity - used for capacity, used in zip(total, memory, strict=True)]
    normalized_hash_keys = (
        "dataset_index_sequence_sha256",
        "agent_count_histogram",
        "update_1_trainable_name_sha256",
        "flow_unfreeze_trainable_name_sha256",
        "vision_inference_batch_size",
        "shared_hdf5_receipt_sha256",
        "future_feature_cache_sha256",
        "learning_rate_curve_sha256",
    )
    checks = {
        "both_completed_200_updates": all(
            value.get("completed") is True for value in rows.values()
        ),
        "same_dataset_index_sequence": p0.get("dataset_index_sequence_sha256")
        == p1.get("dataset_index_sequence_sha256"),
        "same_agent_count_histogram": p0.get("agent_count_histogram")
        == p1.get("agent_count_histogram"),
        "same_update_1_trainable_names": p0.get("update_1_trainable_name_sha256")
        == p1.get("update_1_trainable_name_sha256"),
        "same_flow_unfreeze_update": p0.get("flow_unfreeze_update")
        == p1.get("flow_unfreeze_update")
        == 6_400,
        "same_flow_unfreeze_trainable_names": p0.get(
            "flow_unfreeze_trainable_name_sha256"
        )
        == p1.get("flow_unfreeze_trainable_name_sha256"),
        "same_vision_inference_batch_size": p0.get(
            "vision_inference_batch_size"
        )
        == p1.get("vision_inference_batch_size")
        == int(_mapping(p0_config, "vision")["inference_batch_size"])
        == int(_mapping(p1_config, "vision")["inference_batch_size"]),
        "same_shared_hdf5_receipt": p0.get("shared_hdf5_receipt_sha256")
        == p1.get("shared_hdf5_receipt_sha256")
        and SHA256_PATTERN.fullmatch(
            str(p0.get("shared_hdf5_receipt_sha256", ""))
        )
        is not None,
        "same_future_feature_cache": p0.get("future_feature_cache_sha256")
        == p1.get("future_feature_cache_sha256")
        and SHA256_PATTERN.fullmatch(
            str(p0.get("future_feature_cache_sha256", ""))
        )
        is not None,
        "same_learning_rate_curves": p0.get("learning_rate_curve_sha256")
        == p1.get("learning_rate_curve_sha256"),
        "same_micro_accum_effective_batch": all(
            int(value.get("micro_team_batch", 0)) == expected_micro
            and int(value.get("gradient_accumulation", 0)) == expected_accum
            and int(value.get("effective_team_batch", 0)) == 12
            for value in rows.values()
        ) and _mapping(p1_config, "training") == p1_training,
        "resume_next_batch_exact": all(
            value.get("resume_next_batch_exact") is True for value in rows.values()
        ),
        "forced_audit_measured": all(
            float(value.get("forced_audit_seconds", 0.0)) > 0.0
            for value in rows.values()
        ),
        "throughput_measured": all(
            float(value.get("updates_per_second", 0.0)) > 0.0
            for value in rows.values()
        ),
        "throughput_meets_12h_training_slo": all(
            float(value.get("updates_per_second", 0.0))
            >= float(_mapping(config, "training")["min_updates_per_second"])
            for value, config in zip(
                rows.values(), (p0_config, p1_config), strict=True
            )
        ),
        "memory_headroom_at_least_2gib": all(
            value >= 2 * 1024**3 for value in headroom
        ),
        "no_oom": all(value.get("oom") is False for value in rows.values()),
    }
    memory_failed = (
        not checks["memory_headroom_at_least_2gib"] or not checks["no_oom"]
    )
    vision_batch = int(_mapping(p0_config, "vision")["inference_batch_size"])
    required_fallback = None
    if memory_failed and vision_batch == 16:
        required_fallback = "dino8_micro4_accum3"
    elif memory_failed and (expected_micro, expected_accum) == (4, 3):
        required_fallback = "micro2_accum6"
    elif memory_failed and (expected_micro, expected_accum) == (2, 6):
        required_fallback = "micro1_accum12"

    # Make the equality surface explicit in the report for later audit.
    detail = {
        "P0": dict(p0),
        "P1": dict(p1),
        "compared_keys": list(normalized_hash_keys),
        "memory_headroom_bytes": {"P0": headroom[0], "P1": headroom[1]},
        "estimated_training_hours": {
            candidate: 30_000
            / max(float(value.get("updates_per_second", 0.0)), 1e-12)
            / 3600
            for candidate, value in rows.items()
        },
        "required_fallback": required_fallback,
    }
    return detail, checks


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
