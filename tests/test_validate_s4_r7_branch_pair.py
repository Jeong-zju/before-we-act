from __future__ import annotations

from copy import deepcopy

from scripts.validate_s4_r7_branch_pair import validate_preflights


def _config(*, micro: int = 2, accumulation: int = 6) -> dict[str, object]:
    return {
        "training": {
            "micro_team_batch": micro,
            "gradient_accumulation": accumulation,
        }
    }


def _preflight(candidate: str) -> dict[str, object]:
    return {
        "format_version": "wam.robofactory.s4_r7.preflight/1",
        "identity": {
            "round_id": "s4-r7",
            "candidate_id": candidate,
            "model_kind": (
                "s4_r7_token_preserving"
                if candidate == "P0"
                else "s4_r7_world_utility_coupling"
            ),
        },
        "updates": 200,
        "completed": True,
        "oom": False,
        "micro_team_batch": 2,
        "gradient_accumulation": 6,
        "effective_team_batch": 12,
        "dataset_index_sequence_sha256": "a" * 64,
        "agent_count_histogram": {"1": 10, "2": 20},
        "update_1_trainable_name_sha256": "b" * 64,
        "update_26668_trainable_name_sha256": "c" * 64,
        "learning_rate_curve_sha256": "d" * 64,
        "resume_next_batch_exact": True,
        "forced_audit_seconds": 1.0,
        "updates_per_second": 2.0,
        "peak_memory_bytes": 20 * 1024**3,
        "gpu_total_memory_bytes": 32 * 1024**3,
    }


def test_any_oom_requests_paired_micro1_accum12_even_with_large_headroom() -> None:
    p0 = _preflight("P0")
    p1 = _preflight("P1")
    p1["completed"] = False
    p1["oom"] = True
    detail, checks = validate_preflights(p0, p1, _config(), _config())
    assert checks["memory_headroom_at_least_2gib"] is True
    assert checks["no_oom"] is False
    assert detail["required_fallback"] == "micro1_accum12"


def test_any_low_headroom_requests_paired_micro1_accum12_without_oom() -> None:
    p0 = _preflight("P0")
    p1 = _preflight("P1")
    p0["peak_memory_bytes"] = 31 * 1024**3 + 1
    detail, checks = validate_preflights(p0, p1, _config(), _config())
    assert checks["no_oom"] is True
    assert checks["memory_headroom_at_least_2gib"] is False
    assert detail["required_fallback"] == "micro1_accum12"


def test_micro1_accum12_failure_does_not_invent_another_fallback() -> None:
    p0 = _preflight("P0")
    p1 = _preflight("P1")
    for report in (p0, p1):
        report["micro_team_batch"] = 1
        report["gradient_accumulation"] = 12
    p0["completed"] = False
    p0["oom"] = True
    detail, checks = validate_preflights(
        p0,
        p1,
        _config(micro=1, accumulation=12),
        _config(micro=1, accumulation=12),
    )
    assert checks["no_oom"] is False
    assert detail["required_fallback"] is None


def test_one_sided_recipe_change_still_fails_pair_exact() -> None:
    p0 = _preflight("P0")
    p1 = deepcopy(_preflight("P1"))
    p1["micro_team_batch"] = 1
    p1["gradient_accumulation"] = 12
    _, checks = validate_preflights(p0, p1, _config(), _config())
    assert checks["same_micro_accum_effective_batch"] is False
