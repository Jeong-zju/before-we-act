#!/usr/bin/env python3
"""Apply the preregistered S4-R8 causal-prefix acceptance rules."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.accept_s4_r7 import (  # noqa: E402
    CONDITIONS,
    _summarize_condition,
    _valid_report_reference,
)


CANDIDATE_REPORT_FORMAT = "wam.robofactory.s4_r8.causal_candidate_report/1"
CHECKPOINT_FORMAT = "wam.robofactory.s4_r8.horizon_causal.checkpoint/1"
PAIR_EXACT_FORMAT = "wam.robofactory.s4_r8.pair_exact/1"
PREFIX_SUFFIX_FORMAT = "wam.robofactory.s4_r8.prefix_suffix_exact/1"
PREFIX_SHUFFLE_FORMAT = "wam.robofactory.s4_r8.prefix_shuffle_by_source_horizon/1"
FORMAT_VERSION = "wam.robofactory.s4_r8.acceptance/1"
EXPECTED_KINDS = {
    "P0": "s4_r8_horizon_prefix_mean",
    "P1": "s4_r8_causal_prefix_attention",
}
EXPECTED_AGGREGATORS = {
    "P0": "prefix_mean",
    "P1": "causal_prefix_attention",
}
STRUCTURAL_GATES = (
    "token_contract_exact",
    "dense_router_train_inference_identical",
    "legacy_reference_elementwise_exact_not_required",
    "legacy_reference_file_unchanged",
    "active_gate_zero_elementwise_exact",
    "active_gate_zero_without_provider_elementwise_exact",
    "dino_optimizer_excluded",
    "legacy_reference_optimizer_excluded",
    "auxiliary_weights_zero",
    "no_depth_or_wrist_input",
    "no_ground_truth_future_input",
    "r7_candidate_checkpoint_not_consumed",
    "strict_horizon_prefix_mask",
    "p1_output_projection_zero_initialized_or_p0",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-exact", type=Path, required=True)
    for candidate in ("p0", "p1"):
        parser.add_argument(f"--{candidate}-report", type=Path, required=True)
        parser.add_argument(f"--{candidate}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{candidate}-prefix-suffix", type=Path, required=True)
        parser.add_argument(f"--{candidate}-prefix-shuffle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_acceptance(
        _read_json(args.pair_exact),
        {
            "P0": (
                _read_json(args.p0_report),
                _read_checkpoint(args.p0_checkpoint),
                _read_json(args.p0_prefix_suffix),
                _read_json(args.p0_prefix_shuffle),
            ),
            "P1": (
                _read_json(args.p1_report),
                _read_checkpoint(args.p1_checkpoint),
                _read_json(args.p1_prefix_suffix),
                _read_json(args.p1_prefix_shuffle),
            ),
        },
    )
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_acceptance(
    pair_exact: Mapping[str, Any],
    inputs: Mapping[
        str,
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ],
    ],
) -> dict[str, Any]:
    if (
        pair_exact.get("format_version") != PAIR_EXACT_FORMAT
        or pair_exact.get("round_id") != "s4-r8"
        or pair_exact.get("passed") is not True
    ):
        raise ValueError("S4-R8 pair exactness did not pass")
    checks = _mapping(pair_exact, "checks")
    if checks.get("p0_p1_fp32_eval_step0_elementwise_exact") is not True:
        raise ValueError("S4-R8 P0/P1 step-0 exactness is missing")
    candidates = {
        candidate: _evaluate_candidate(candidate, *inputs[candidate])
        for candidate in ("P0", "P1")
    }
    if candidates["P0"]["seed_protocol"] != candidates["P1"]["seed_protocol"]:
        raise ValueError("R8 candidate Gate20 seed protocols differ")
    eligible = [name for name, result in candidates.items() if result["passed"]]
    if not eligible:
        winner = "retain_r7_winner_method"
        decision = "fail_both_no_r8_method_added"
    elif len(eligible) == 1:
        winner = eligible[0]
        decision = f"pass_{winner.lower()}"
    else:
        p0_macro = float(candidates["P0"]["macro_success_rate"])
        p1_macro = float(candidates["P1"]["macro_success_rate"])
        winner = "P1" if p1_macro > p0_macro else "P0"
        decision = f"pass_{winner.lower()}"
    return {
        "format_version": FORMAT_VERSION,
        "round_id": "s4-r8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": (
            "eliminate candidates failing common structural/causal Gate20 or "
            "R8 suffix-exact/prefix-bootstrap gates; select larger normal "
            "Gate20 macro; exact tie defaults P0"
        ),
        "pair_exact": dict(pair_exact),
        "candidates": candidates,
        "eligible_candidates": eligible,
        "winner": winner,
        "passed": winner in {"P0", "P1"},
        "decision": decision,
        "tie_break": "P0",
        "method_combination_not_weight_fusion": True,
        "r7_candidate_checkpoint_consumed": False,
    }


def _evaluate_candidate(
    candidate: str,
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    suffix: Mapping[str, Any],
    shuffle: Mapping[str, Any],
) -> dict[str, Any]:
    kind = EXPECTED_KINDS[candidate]
    aggregator = EXPECTED_AGGREGATORS[candidate]
    identity = _mapping(report, "identity")
    method = _mapping(checkpoint, "method")
    if (
        report.get("format_version") != CANDIDATE_REPORT_FORMAT
        or identity.get("round_id") != "s4-r8"
        or identity.get("candidate_id") != candidate
        or identity.get("model_kind") != kind
        or checkpoint.get("format_version") != CHECKPOINT_FORMAT
        or checkpoint.get("update") != 30_000
        or method.get("round_id") != "s4-r8"
        or method.get("candidate_id") != candidate
        or method.get("model_kind") != kind
        or method.get("action_prefix_aggregator") != aggregator
        or float(method.get("utility_coupling_weight", -1.0)) != 0.0
    ):
        raise ValueError(f"{candidate} R8 report/checkpoint identity differs")
    checkpoint_sha = str(checkpoint["file_sha256"])
    if report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{candidate} report/checkpoint SHA256 differs")
    for special, expected in (
        (suffix, PREFIX_SUFFIX_FORMAT),
        (shuffle, PREFIX_SHUFFLE_FORMAT),
    ):
        if (
            special.get("format_version") != expected
            or special.get("candidate_id") != candidate
            or special.get("checkpoint_sha256") != checkpoint_sha
            or special.get("action_prefix_aggregator") != aggregator
        ):
            raise ValueError(f"{candidate} R8 special report identity differs")
    structural = _mapping(report, "structural_invariants")
    structural_results = {key: structural.get(key) is True for key in STRUCTURAL_GATES}
    training = _mapping(report, "training_audits")
    training_results = {
        "checkpoint_update_30000": training.get("checkpoint_update_30000") is True,
        "parameter_gradient_audit_passed": training.get(
            "parameter_gradient_audit_passed"
        )
        is True,
        "module_exposure_passed": training.get("module_exposure_passed") is True,
        "formal_budget_complete": training.get("formal_budget_complete") is True,
    }
    required_reports = {
        key: _valid_report_reference(value)
        for key, value in _mapping(report, "reports").items()
    }
    conditions = _mapping(report, "gate20")
    summarized = {
        name: _summarize_condition(name, _mapping(conditions, name))
        for name in CONDITIONS
    }
    seed_protocol = summarized["normal"]["seed_protocol"]
    if any(value["seed_protocol"] != seed_protocol for value in summarized.values()):
        raise ValueError(f"{candidate} R8 Gate20 seeds are not paired")
    normal = float(summarized["normal"]["macro_success_rate"])
    causal = {
        "normal_not_below_legacy": normal
        >= float(summarized["legacy_reference"]["macro_success_rate"]),
        "normal_strictly_above_world_evidence_gate_zero": normal
        > float(summarized["world_evidence_gate_zero"]["macro_success_rate"]),
        "normal_strictly_above_shuffle_all": normal
        > float(summarized["shuffle_all"]["macro_success_rate"]),
    }
    utility = _mapping(report, "utility_calibration")
    wuc_disabled = {
        "forced_evidence_audit_present": utility.get("forced_evidence_audit_present")
        is True,
        "utility_weight_exact_zero": float(
            utility.get("utility_coupling_weight", float("nan"))
        )
        == 0.0,
        "wuc_backward_disabled": utility.get("wuc_backward_disabled") is True,
    }
    suffix_groups = _mapping(suffix, "groups")
    shuffle_groups = _mapping(shuffle, "groups")
    expected_groups = {
        f"{source}@{horizon}"
        for source in ("own", "peer", "shared")
        for horizon in (1, 25, 50, 100)
    }
    special = {
        "all_12_suffix_groups_present": set(suffix_groups) == expected_groups,
        "all_12_shuffle_groups_present": set(shuffle_groups) == expected_groups,
        "suffix_report_passed": suffix.get("passed") is True,
        "prefix_shuffle_report_passed": shuffle.get("passed") is True,
        "every_suffix_exact_and_prefix_sensitive": all(
            isinstance(value, Mapping)
            and value.get("suffix_elementwise_exact") is True
            and value.get("legal_prefix_changes_output") is True
            for value in suffix_groups.values()
        ),
        "every_prefix_shuffle_bootstrap_lower_positive": all(
            isinstance(value, Mapping)
            and float(value.get("episode_bootstrap_95_lower", 0.0)) > 0.0
            for value in shuffle_groups.values()
        ),
    }
    passed = all(
        all(group.values())
        for group in (
            structural_results,
            training_results,
            required_reports,
            causal,
            wuc_disabled,
            special,
        )
    )
    return {
        "candidate_id": candidate,
        "model_kind": kind,
        "action_prefix_aggregator": aggregator,
        "structural_invariants": structural_results,
        "training_audits": training_results,
        "required_reports": required_reports,
        "causal_gates": causal,
        "wuc_fixed_off": wuc_disabled,
        "r8_special_gates": special,
        "gate20": summarized,
        "seed_protocol": seed_protocol,
        "macro_success_rate": normal,
        "heldout_flow_error": float(report.get("heldout_flow_error", float("inf"))),
        "passed": passed,
    }


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    import hashlib

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = dict(value)
    result["file_sha256"] = digest.hexdigest()
    return result


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
