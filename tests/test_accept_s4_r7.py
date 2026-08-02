from __future__ import annotations

from copy import deepcopy

from scripts.accept_s4_r7 import (
    CANDIDATE_REPORT_FORMAT,
    CHECKPOINT_FORMAT,
    CONDITIONS,
    PAIR_EXACT_FORMAT,
    REQUIRED_REPORT_KEYS,
    STRUCTURAL_GATES,
    TASKS,
    build_acceptance,
)


def _episodes(successes: int) -> list[dict[str, object]]:
    return [
        {"seed": seed, "success": offset < successes}
        for offset, seed in enumerate(range(900, 920))
    ]


def _condition(successes: int) -> dict[str, object]:
    return {
        "task_order": list(TASKS),
        "tasks": {
            task: {"episodes": _episodes(successes)} for task in TASKS
        },
    }


def _report(candidate_id: str, *, normal: int) -> dict[str, object]:
    kind = (
        "s4_r7_token_preserving"
        if candidate_id == "P0"
        else "s4_r7_world_utility_coupling"
    )
    successes = {
        "legacy_reference": 8,
        "normal": normal,
        "world_evidence_gate_zero": normal - 1,
        "all_world_gates_zero": 7,
        "shuffle_all": normal - 2,
        "shuffle_own": normal - 1,
        "shuffle_peer": normal,
        "shuffle_shared": normal - 1,
    }
    return {
        "format_version": CANDIDATE_REPORT_FORMAT,
        "identity": {
            "round_id": "s4-r7",
            "candidate_id": candidate_id,
            "model_kind": kind,
        },
        "checkpoint_sha256": f"checkpoint-{candidate_id}",
        "structural_invariants": {name: True for name in STRUCTURAL_GATES},
        "training_audits": {
            "checkpoint_update_125000": True,
            "parameter_gradient_audit_passed": True,
            "module_exposure_passed": True,
            "formal_budget_complete": True,
        },
        "reports": {
            name: {"sha256": "a" * 64} for name in REQUIRED_REPORT_KEYS
        },
        "gate20": {
            name: _condition(successes[name]) for name in CONDITIONS
        },
        "utility_calibration": (
            {
                "forced_evidence_audit_present": True,
                "utility_coupling_weight": 0.0,
                "wuc_backward_disabled": True,
            }
            if candidate_id == "P0"
            else {
                "forced_evidence_audit_present": True,
                "spearman": 0.2,
                "episode_bootstrap_95_lower": 0.01,
                "wuc_router_gradient_norm": 0.3,
                "wuc_forbidden_gradient_norm": 0.0,
            }
        ),
        "heldout_flow_error": 0.4 if candidate_id == "P0" else 0.3,
    }


def _checkpoint(candidate_id: str) -> dict[str, object]:
    kind = (
        "s4_r7_token_preserving"
        if candidate_id == "P0"
        else "s4_r7_world_utility_coupling"
    )
    return {
        "format_version": CHECKPOINT_FORMAT,
        "file_sha256": f"checkpoint-{candidate_id}",
        "update": 125_000,
        "method": {
            "round_id": "s4-r7",
            "candidate_id": candidate_id,
            "model_kind": kind,
        },
        "parent_identity": {
            "legacy_r6l_policy_sha256": "1" * 64,
            "active_flow_checkpoint_sha256": "2" * 64,
            "local_future_checkpoint_sha256": "3" * 64,
            "team_future_checkpoint_sha256": "4" * 64,
            "pca_artifact_sha256": "5" * 64,
        },
    }


def _pair_exact() -> dict[str, object]:
    return {
        "format_version": PAIR_EXACT_FORMAT,
        "round_id": "s4-r7",
        "checks": {"dataset": True, "resume": True},
        "passed": True,
    }


def test_acceptance_selects_higher_passing_normal_macro() -> None:
    result = build_acceptance(
        _pair_exact(),
        _report("P0", normal=10),
        _report("P1", normal=11),
        _checkpoint("P0"),
        _checkpoint("P1"),
    )
    assert result["winner"] == "P1"
    assert result["passed"] is True
    assert result["candidates"]["P1"]["utility_calibration_passed"] is True


def test_acceptance_tie_uses_p1_only_with_utility_and_lower_flow_error() -> None:
    p0 = _report("P0", normal=10)
    p1 = _report("P1", normal=10)
    result = build_acceptance(
        _pair_exact(), p0, p1, _checkpoint("P0"), _checkpoint("P1")
    )
    assert result["winner"] == "P1"

    failed_utility = deepcopy(p1)
    failed_utility["utility_calibration"]["spearman"] = 0.0
    result = build_acceptance(
        _pair_exact(), p0, failed_utility, _checkpoint("P0"), _checkpoint("P1")
    )
    assert result["winner"] == "P0"


def test_acceptance_rejects_normal_without_strict_causal_gap() -> None:
    p0 = _report("P0", normal=10)
    p1 = _report("P1", normal=11)
    p1["gate20"]["shuffle_all"] = _condition(11)
    result = build_acceptance(
        _pair_exact(), p0, p1, _checkpoint("P0"), _checkpoint("P1")
    )
    assert result["candidates"]["P1"]["passed"] is False
    assert result["winner"] == "P0"


def test_all_world_zero_is_report_only() -> None:
    p0 = _report("P0", normal=10)
    p1 = _report("P1", normal=11)
    p0["gate20"]["all_world_gates_zero"] = _condition(20)
    result = build_acceptance(
        _pair_exact(), p0, p1, _checkpoint("P0"), _checkpoint("P1")
    )
    assert result["candidates"]["P0"]["passed"] is True
    assert result["all_world_gates_zero_is_report_only"] is True


def test_training_audit_failure_eliminates_candidate() -> None:
    p0 = _report("P0", normal=10)
    p1 = _report("P1", normal=11)
    p1["training_audits"]["parameter_gradient_audit_passed"] = False
    result = build_acceptance(
        _pair_exact(), p0, p1, _checkpoint("P0"), _checkpoint("P1")
    )
    assert result["candidates"]["P1"]["passed"] is False
    assert result["winner"] == "P0"
