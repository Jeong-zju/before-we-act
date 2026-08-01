"""Fail-closed model-kind registry for S3-R6 training and acceptance."""

from __future__ import annotations

from collections.abc import Mapping


S3_R6_MODEL_KINDS: dict[str, tuple[str, str, str, bool]] = {
    "s3_r6l_protected_local_aux": ("R6L", "P0", "local", False),
    "s3_r6l_protected_local_gated": ("R6L", "P1", "local", True),
    "s3_r6j_protected_team_offpath": ("R6J", "P0", "team_shared", False),
    "s3_r6j_protected_team_gated": ("R6J", "P1", "team_shared", True),
}


def validate_s3_r6_candidate(
    value: Mapping[str, object],
) -> tuple[str, str, str, str, bool]:
    round_id = str(value.get("round_id", ""))
    micro_round = str(value.get("micro_round", ""))
    candidate_id = str(value.get("candidate_id", ""))
    model_kind = str(value.get("model_kind", ""))
    future_scope = str(value.get("future_scope", ""))
    injection = value.get("injection")
    if round_id != "s3-r6" or model_kind not in S3_R6_MODEL_KINDS:
        raise ValueError(f"unregistered S3-R6 model kind: {model_kind!r}")
    expected = S3_R6_MODEL_KINDS[model_kind]
    observed = (micro_round, candidate_id, future_scope, injection)
    if observed != expected:
        raise ValueError(
            f"S3-R6 identity disagrees with allowlist: {observed!r} != {expected!r}"
        )
    return micro_round, candidate_id, model_kind, future_scope, bool(injection)


__all__ = ["S3_R6_MODEL_KINDS", "validate_s3_r6_candidate"]
