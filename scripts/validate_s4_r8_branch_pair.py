#!/usr/bin/env python3
"""Validate the R8 pair, paired preflight, and exact step-0 equality."""

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

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.s4_r8_model_io import build_s4_r8_model  # noqa: E402
from scripts.train_static_rgb_act_moe import _load_yaml, _mapping  # noqa: E402
from train.s4_model_registry import validate_s4_r8_pair  # noqa: E402


FORMAT_VERSION = "wam.robofactory.s4_r8.pair_exact/1"
PREFLIGHT_FORMAT = "wam.robofactory.s4_r8.preflight/1"
STEP0_FORMAT = "wam.robofactory.s4_r8.p0_p1_step0_exact/1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)
EXPECTED_KINDS = {
    "P0": "s4_r8_horizon_prefix_mean",
    "P1": "s4_r8_causal_prefix_attention",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-config", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--p0-preflight", type=Path)
    parser.add_argument("--p1-preflight", type=Path)
    parser.add_argument("--step0-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p0 = _load_yaml(args.p0_config.expanduser().resolve(strict=True))
    p1 = _load_yaml(args.p1_config.expanduser().resolve(strict=True))
    checks = validate_configs(p0, p1)
    preflight = None
    step0_reference = None
    if not args.config_only:
        if (
            args.p0_preflight is None
            or args.p1_preflight is None
            or args.step0_output is None
        ):
            raise ValueError(
                "full R8 pair validation requires both preflights and step0 output"
            )
        preflight, preflight_checks = validate_preflights(
            _read_json(args.p0_preflight),
            _read_json(args.p1_preflight),
            p0,
            p1,
        )
        checks.update(preflight_checks)
        step0 = validate_step0(p0, p1)
        _atomic_json(args.step0_output.expanduser().resolve(), step0)
        checks["p0_p1_fp32_eval_step0_elementwise_exact"] = step0["passed"] is True
        step0_reference = {
            "path": str(args.step0_output.expanduser().resolve(strict=True)),
            "format_version": STEP0_FORMAT,
        }
    passed = all(checks.values())
    result = {
        "format_version": FORMAT_VERSION,
        "round_id": "s4-r8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "config_only" if args.config_only else "paired_200_step_preflight",
        "checks": checks,
        "passed": passed,
        "candidate_axis": {
            "name": "action_prefix_aggregator",
            "P0": "prefix_mean",
            "P1": "causal_prefix_attention",
            "action_prefix_rank": 32,
            "utility_coupling_weight_fixed": 0.0,
        },
        "step0_exact": step0_reference,
        "preflight": preflight,
    }
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


def validate_configs(p0: Mapping[str, Any], p1: Mapping[str, Any]) -> dict[str, bool]:
    validate_s4_r8_pair(p0, p1)
    training = _mapping(p0, "training")
    data = _mapping(p0, "data")
    model = _mapping(p0, "model")
    vision = _mapping(p0, "vision")
    evaluation = _mapping(p0, "evaluation")
    manifests = data.get("manifests")
    tasks = (
        tuple(Path(str(path)).parent.name for path in manifests)
        if isinstance(manifests, list)
        else ()
    )
    checks = {
        "registered_candidate_axis_only": True,
        "parallel_r8_has_no_r7_checkpoint_dependency": float(
            training["utility_coupling_weight"]
        )
        == 0.0,
        "strict_horizons_and_rank": tuple(model["future_horizons"]) == (1, 25, 50, 100)
        and int(model["action_prefix_rank"]) == 32,
        "all_750_episodes_used_for_training": data.get("training_split") == "all",
        "five_task_order_exact": tasks == TASKS
        and tuple(evaluation.get("tasks", ())) == TASKS,
        "fast_selection_30000_effective_team_batch_12": training.get("budget_mode")
        == "fast_selection_30k"
        and int(training["updates"]) == 30_000
        and int(training["flow_unfreeze_update"]) == 6_400
        and int(training["effective_team_batch"]) == 12
        and int(training["micro_team_batch"]) * int(training["gradient_accumulation"])
        == 12,
        "shared_float32_future_cache": data.get("future_feature_cache_mode")
        == "shared_float32_projected_next_view",
        "fast_5090_recipe": int(vision["inference_batch_size"]) == 16
        and training.get("optimizer") == "adamw_fused"
        and int(training["num_workers"]) == 8
        and int(training["prefetch_factor"]) == 4
        and float(training["min_updates_per_second"]) == 0.75,
        "gate20_seed_900_919": int(evaluation["episodes"]) == 20
        and int(evaluation["seed_start"]) == 900,
        "special_prefix_bootstrap_registered": int(
            evaluation.get("prefix_windows_per_episode", 0)
        )
        > 0
        and int(evaluation.get("prefix_batch_size", 0)) >= 2
        and int(evaluation["bootstrap_samples"]) == 10_000,
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise ValueError(f"S4-R8 config pair failed: {failed}")
    return checks


def validate_preflights(
    p0: Mapping[str, Any],
    p1: Mapping[str, Any],
    p0_config: Mapping[str, Any],
    p1_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    rows = {"P0": p0, "P1": p1}
    for candidate, value in rows.items():
        identity = _mapping(value, "identity")
        if (
            value.get("format_version") != PREFLIGHT_FORMAT
            or identity.get("round_id") != "s4-r8"
            or identity.get("candidate_id") != candidate
            or identity.get("model_kind") != EXPECTED_KINDS[candidate]
            or int(value.get("updates", 0)) != 200
        ):
            raise ValueError(f"{candidate} R8 preflight identity is invalid")
    memory = [float(value.get("peak_memory_bytes", 0)) for value in rows.values()]
    total = [float(value.get("gpu_total_memory_bytes", 0)) for value in rows.values()]
    headroom = [capacity - used for capacity, used in zip(total, memory, strict=True)]
    checks = {
        "both_completed_200_updates": all(
            value.get("completed") is True for value in rows.values()
        ),
        "same_dataset_index_sequence": p0.get("dataset_index_sequence_sha256")
        == p1.get("dataset_index_sequence_sha256"),
        "same_agent_count_histogram": p0.get("agent_count_histogram")
        == p1.get("agent_count_histogram"),
        "same_shared_hdf5_receipt": p0.get("shared_hdf5_receipt_sha256")
        == p1.get("shared_hdf5_receipt_sha256")
        and SHA256_PATTERN.fullmatch(str(p0.get("shared_hdf5_receipt_sha256", "")))
        is not None,
        "same_future_feature_cache": p0.get("future_feature_cache_sha256")
        == p1.get("future_feature_cache_sha256")
        and SHA256_PATTERN.fullmatch(str(p0.get("future_feature_cache_sha256", "")))
        is not None,
        "same_learning_rate_curves": p0.get("learning_rate_curve_sha256")
        == p1.get("learning_rate_curve_sha256"),
        "same_batch_recipe": all(
            int(value.get("effective_team_batch", 0)) == 12 for value in rows.values()
        ),
        "resume_next_batch_exact": all(
            value.get("resume_next_batch_exact") is True for value in rows.values()
        ),
        "throughput_meets_slo": all(
            float(value.get("updates_per_second", 0.0))
            >= float(_mapping(config, "training")["min_updates_per_second"])
            for value, config in zip(rows.values(), (p0_config, p1_config), strict=True)
        ),
        "memory_headroom_at_least_2gib": all(
            value >= 2 * 1024**3 for value in headroom
        ),
        "no_oom": all(value.get("oom") is False for value in rows.values()),
    }
    return {
        "P0": dict(p0),
        "P1": dict(p1),
        "memory_headroom_bytes": {"P0": headroom[0], "P1": headroom[1]},
    }, checks


def validate_step0(
    p0_config: Mapping[str, Any], p1_config: Mapping[str, Any]
) -> dict[str, Any]:
    p0, legacy0, identity0 = build_s4_r8_model(p0_config, device=torch.device("cpu"))
    p1, legacy1, identity1 = build_s4_r8_model(p1_config, device=torch.device("cpu"))
    del legacy0, legacy1
    p0.eval()
    p1.eval()
    left = p0.state_dict()
    right = p1.state_dict()
    extra = sorted(set(right) - set(left))
    missing = sorted(set(left) - set(right))
    common_exact = all(
        torch.equal(left[name], right[name]) for name in left if name in right
    )
    expected_extra_suffixes = {
        "query_weight",
        "key_weight",
        "value_weight",
        "output_weight",
        "output_bias",
    }
    extra_expected = (
        len(extra) == 5
        and {name.rsplit(".", 1)[-1] for name in extra} == expected_extra_suffixes
    )
    output_zero = bool(
        torch.count_nonzero(
            p1.active_parent.future_predictor.action_prefix_aggregator.output_weight
        ).item()
        == 0
        and torch.count_nonzero(
            p1.active_parent.future_predictor.action_prefix_aggregator.output_bias
        ).item()
        == 0
    )
    generator = torch.Generator().manual_seed(80808)
    state = torch.randn(2, 4, 18, generator=generator)
    visual = torch.randn(2, 4, 4, 256, generator=generator)
    shared = torch.randn(2, 4, 256, generator=generator)
    actions = torch.randn(2, 4, 100, 8, generator=generator)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    with torch.inference_mode():
        pred0 = p0.active_parent.future_predictor(state, visual, shared, actions, valid)
        pred1 = p1.active_parent.future_predictor(state, visual, shared, actions, valid)
    field_diffs = {
        name: float((getattr(pred0, name) - getattr(pred1, name)).abs().max())
        for name in (
            "own_state",
            "own_visual",
            "peer_state",
            "peer_visual",
            "shared_visual",
        )
    }
    checks = {
        "common_state_dict_elementwise_exact": common_exact,
        "p0_has_no_missing_common_state": not missing,
        "p1_extra_parameters_are_exact_rank32_attention_axis": extra_expected,
        "p1_output_projection_exact_zero": output_zero,
        "fp32_eval_predictions_elementwise_exact": max(field_diffs.values()) == 0.0,
        "no_r7_candidate_checkpoint_consumed": identity0.get(
            "r7_candidate_checkpoint_consumed"
        )
        is False
        and identity1.get("r7_candidate_checkpoint_consumed") is False,
    }
    return {
        "format_version": STEP0_FORMAT,
        "round_id": "s4-r8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(checks.values()),
        "p1_extra_state_names": extra,
        "maximum_absolute_difference_by_output": field_diffs,
        "fp32_eval": True,
    }


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
