#!/usr/bin/env python3
"""Apply the preregistered S4-R7 structural, causal, and Gate20 rules."""

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


CANDIDATE_REPORT_FORMAT = "wam.robofactory.s4_r7.causal_candidate_report/1"
CHECKPOINT_FORMAT = "wam.robofactory.s4_r7.world_utility.checkpoint/1"
PAIR_EXACT_FORMAT = "wam.robofactory.s4_r7.pair_exact/1"
FORMAT_VERSION = "wam.robofactory.s4_r7.acceptance/1"
TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)
EXPECTED_KINDS = {
    "P0": "s4_r7_token_preserving",
    "P1": "s4_r7_world_utility_coupling",
}
CONDITIONS = (
    "legacy_reference",
    "normal",
    "world_evidence_gate_zero",
    "all_world_gates_zero",
    "shuffle_all",
    "shuffle_own",
    "shuffle_peer",
    "shuffle_shared",
)
STRUCTURAL_GATES = (
    "token_contract_exact",
    "dense_router_train_inference_identical",
    "legacy_reference_elementwise_exact",
    "legacy_reference_file_unchanged",
    "active_gate_zero_elementwise_exact",
    "active_gate_zero_without_provider_elementwise_exact",
    "dino_optimizer_excluded",
    "legacy_reference_optimizer_excluded",
    "auxiliary_weights_zero",
    "no_depth_or_wrist_input",
    "no_ground_truth_future_input",
)
REQUIRED_REPORT_KEYS = (
    "parameter_gradient_audit",
    "module_exposure",
    "forced_evidence_errors",
    "router_utility_spearman",
    "source_shuffle_gate20",
    "legacy_scaled_zero_shuffle_gate20",
    "artifact_hashes",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-exact", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--p1-report", type=Path, required=True)
    parser.add_argument("--p0-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_acceptance(
        _read_json(args.pair_exact),
        _read_json(args.p0_report),
        _read_json(args.p1_report),
        _read_checkpoint(args.p0_checkpoint),
        _read_checkpoint(args.p1_checkpoint),
    )
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_acceptance(
    pair_exact: Mapping[str, Any],
    p0_report: Mapping[str, Any],
    p1_report: Mapping[str, Any],
    p0_checkpoint: Mapping[str, Any],
    p1_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_pair_exact(pair_exact)
    candidates = {
        "P0": _evaluate_candidate("P0", p0_report, p0_checkpoint),
        "P1": _evaluate_candidate("P1", p1_report, p1_checkpoint),
    }
    if candidates["P0"]["seed_protocol"] != candidates["P1"]["seed_protocol"]:
        raise ValueError("P0/P1 Gate20 seed protocols differ")
    p0_parent = _mapping(p0_checkpoint, "parent_identity")
    p1_parent = _mapping(p1_checkpoint, "parent_identity")
    for key in (
        "legacy_r6l_policy_sha256",
        "active_flow_checkpoint_sha256",
        "local_future_checkpoint_sha256",
        "team_future_checkpoint_sha256",
        "pca_artifact_sha256",
    ):
        if p0_parent.get(key) != p1_parent.get(key):
            raise ValueError(f"P0/P1 parent identity differs for {key}")

    eligible = [name for name, row in candidates.items() if row["passed"]]
    if not eligible:
        winner = "legacy_reference"
        decision = "fail_both_retain_r6l_p1"
    elif len(eligible) == 1:
        winner = eligible[0]
        decision = f"pass_{winner.lower()}"
    else:
        p0_macro = float(candidates["P0"]["macro_success_rate"])
        p1_macro = float(candidates["P1"]["macro_success_rate"])
        if p1_macro > p0_macro:
            winner = "P1"
        elif p0_macro > p1_macro:
            winner = "P0"
        else:
            p1_utility = bool(candidates["P1"]["utility_calibration_passed"])
            p1_flow = float(candidates["P1"]["heldout_flow_error"])
            p0_flow = float(candidates["P0"]["heldout_flow_error"])
            winner = "P1" if p1_utility and p1_flow < p0_flow else "P0"
        decision = f"pass_{winner.lower()}"
    return {
        "format_version": FORMAT_VERSION,
        "round_id": "s4-r7",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": (
            "eliminate candidates failing structural/causal gates; require normal "
            ">= legacy and normal > active-gate-zero,shuffle-all; select larger "
            "normal Gate20 macro; exact tie defaults P0 unless P1 utility passes "
            "and has lower held-out Flow error"
        ),
        "pair_exact": dict(pair_exact),
        "candidates": candidates,
        "eligible_candidates": eligible,
        "winner": winner,
        "passed": winner in {"P0", "P1"},
        "decision": decision,
        "all_world_gates_zero_is_report_only": True,
        "r8_may_start": winner in {"P0", "P1"},
    }


def _validate_pair_exact(value: Mapping[str, Any]) -> None:
    if value.get("format_version") != PAIR_EXACT_FORMAT:
        raise ValueError("pair_exact has an unsupported format")
    if value.get("round_id") != "s4-r7" or value.get("passed") is not True:
        raise ValueError("S4-R7 cannot accept a pair that failed exactness")
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("pair_exact must contain named checks")
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise ValueError(f"pair_exact contains failed checks: {failed}")


def _evaluate_candidate(
    candidate_id: str,
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    expected_kind = EXPECTED_KINDS[candidate_id]
    if report.get("format_version") != CANDIDATE_REPORT_FORMAT:
        raise ValueError(f"{candidate_id} candidate report format is unsupported")
    identity = _mapping(report, "identity")
    if (
        identity.get("round_id") != "s4-r7"
        or identity.get("candidate_id") != candidate_id
        or identity.get("model_kind") != expected_kind
    ):
        raise ValueError(f"{candidate_id} report identity is outside the registry")
    method = _mapping(checkpoint, "method")
    if (
        checkpoint.get("format_version") != CHECKPOINT_FORMAT
        or method.get("round_id") != "s4-r7"
        or method.get("candidate_id") != candidate_id
        or method.get("model_kind") != expected_kind
    ):
        raise ValueError(f"{candidate_id} checkpoint identity is outside the registry")
    if report.get("checkpoint_sha256") != _sha256_from_payload(checkpoint):
        # Checkpoints written by the trainer carry their own file hash in the
        # report. Synthetic unit fixtures may instead provide payload_sha256.
        if report.get("checkpoint_sha256") != checkpoint.get("payload_sha256"):
            raise ValueError(f"{candidate_id} report/checkpoint hash identity differs")

    structural = _mapping(report, "structural_invariants")
    structural_results = {
        key: structural.get(key) is True for key in STRUCTURAL_GATES
    }
    training = _mapping(report, "training_audits")
    training_results = {
        "checkpoint_update_30000": checkpoint.get("update") == 30_000
        and training.get("checkpoint_update_30000") is True,
        "parameter_gradient_audit_passed": training.get(
            "parameter_gradient_audit_passed"
        )
        is True,
        "module_exposure_passed": training.get("module_exposure_passed") is True,
        "formal_budget_complete": training.get("formal_budget_complete") is True,
    }
    report_paths = _mapping(report, "reports")
    required_reports = {
        key: _valid_report_reference(report_paths.get(key))
        for key in REQUIRED_REPORT_KEYS
    }
    conditions = _mapping(report, "gate20")
    summarized = {
        name: _summarize_condition(name, _mapping(conditions, name))
        for name in CONDITIONS
    }
    seed_protocol = summarized["normal"]["seed_protocol"]
    for name, value in summarized.items():
        if value["seed_protocol"] != seed_protocol:
            raise ValueError(f"{candidate_id} {name} uses a different seed protocol")
    normal = float(summarized["normal"]["macro_success_rate"])
    legacy = float(summarized["legacy_reference"]["macro_success_rate"])
    gate_zero = float(
        summarized["world_evidence_gate_zero"]["macro_success_rate"]
    )
    shuffled = float(summarized["shuffle_all"]["macro_success_rate"])
    causal = {
        "normal_not_below_legacy": normal >= legacy,
        "normal_strictly_above_world_evidence_gate_zero": normal > gate_zero,
        "normal_strictly_above_shuffle_all": normal > shuffled,
    }

    utility = _mapping(report, "utility_calibration")
    forced_audit_present = utility.get("forced_evidence_audit_present") is True
    if candidate_id == "P1":
        utility_checks = {
            "forced_evidence_audit_present": forced_audit_present,
            "spearman_positive": float(utility.get("spearman", 0.0)) > 0.0,
            "episode_bootstrap_95_lower_positive": float(
                utility.get("episode_bootstrap_95_lower", 0.0)
            ) > 0.0,
            "wuc_router_gradient_nonzero": float(
                utility.get("wuc_router_gradient_norm", 0.0)
            ) > 0.0,
            "wuc_forbidden_gradient_exact_zero": float(
                utility.get("wuc_forbidden_gradient_norm", float("inf"))
            ) == 0.0,
        }
    else:
        utility_checks = {
            "forced_evidence_audit_present": forced_audit_present,
            "utility_weight_exact_zero": float(
                utility.get("utility_coupling_weight", float("nan"))
            ) == 0.0,
            "wuc_backward_disabled": utility.get("wuc_backward_disabled") is True,
        }
    utility_passed = all(utility_checks.values())
    passed = (
        all(structural_results.values())
        and all(training_results.values())
        and all(required_reports.values())
        and all(causal.values())
        and utility_passed
    )
    source_gaps = {
        source: normal - float(summarized[f"shuffle_{source}"]["macro_success_rate"])
        for source in ("own", "peer", "shared")
    }
    return {
        "candidate_id": candidate_id,
        "model_kind": expected_kind,
        "structural_invariants": structural_results,
        "training_audits": training_results,
        "required_reports": required_reports,
        "gate20": summarized,
        "seed_protocol": seed_protocol,
        "macro_success_rate": normal,
        "legacy_macro_success_rate": legacy,
        "causal_gates": causal,
        "source_shuffle_gaps": source_gaps,
        "positive_source_claims": sorted(
            source for source, gap in source_gaps.items() if gap > 0.0
        ),
        "all_world_gates_zero_report_only": summarized["all_world_gates_zero"],
        "utility_checks": utility_checks,
        "utility_calibration_passed": utility_passed,
        "heldout_flow_error": float(report.get("heldout_flow_error", float("inf"))),
        "passed": passed,
    }


def _summarize_condition(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if tuple(value.get("task_order", ())) != TASKS:
        raise ValueError(f"{name} must contain the exact five-task order")
    tasks = _mapping(value, "tasks")
    task_rows: dict[str, Any] = {}
    expected_seeds = list(range(900, 920))
    for task in TASKS:
        row = _mapping(tasks, task)
        episodes = row.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 20:
            raise ValueError(f"{name}/{task} must contain exactly 20 episodes")
        seeds: list[int] = []
        successes = 0
        for episode in episodes:
            if not isinstance(episode, Mapping):
                raise ValueError(f"{name}/{task} episode rows must be mappings")
            seeds.append(int(episode.get("seed", -1)))
            successes += int(episode.get("success") is True)
        if seeds != expected_seeds:
            raise ValueError(f"{name}/{task} must use paired seeds 900..919")
        task_rows[task] = {
            "successes": successes,
            "episodes": 20,
            "success_rate": successes / 20.0,
        }
    macro = sum(row["success_rate"] for row in task_rows.values()) / len(TASKS)
    return {
        "task_order": list(TASKS),
        "seed_protocol": {"seed_start": 900, "episodes_per_task": 20},
        "tasks": task_rows,
        "macro_success_rate": macro,
    }


def _valid_report_reference(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    if not isinstance(value, Mapping):
        return False
    digest = value.get("sha256")
    return isinstance(digest, str) and len(digest) == 64


def _sha256_from_payload(value: Mapping[str, Any]) -> str | None:
    digest = value.get("file_sha256")
    return str(digest) if isinstance(digest, str) else None


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    value = torch.load(
        path.expanduser().resolve(strict=True), map_location="cpu", weights_only=False
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    # The path hash cannot safely be embedded into the file itself. Attach it
    # transiently for report/checkpoint identity verification.
    result = dict(value)
    result["file_sha256"] = _file_sha256(path)
    return result


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
