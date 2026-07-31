"""Explicit S2-R3 model-kind allowlist used by train/evaluate/accept paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


S2_R3_MODEL_KINDS: dict[str, bool] = {
    "s2_r3_local_action_independent": False,
    "s2_r3_local_action_conditioned": True,
}

S2_R4_MODEL_KINDS: dict[str, bool] = {
    "s2_r4_local_action_conditioned": False,
    "s2_r4_team_shared_action_conditioned": True,
}

S2_R4_HYBRID_MODEL_KINDS = frozenset(
    {"s2_r4_protected_hybrid_diagnostic"}
)


def validate_s2_candidate(value: Mapping[str, Any]) -> tuple[str, str, bool]:
    candidate_id = str(value.get("candidate_id", ""))
    model_kind = str(value.get("model_kind", ""))
    configured = value.get("action_conditioning")
    if candidate_id not in {"W0", "W1"}:
        raise ValueError("S2-R3 candidate_id must be W0 or W1")
    if model_kind not in S2_R3_MODEL_KINDS:
        raise ValueError(
            "S2-R3 model kind is not in the train/evaluation allowlist"
        )
    if not isinstance(configured, bool):
        raise ValueError("round.action_conditioning must be boolean")
    expected = S2_R3_MODEL_KINDS[model_kind]
    if configured is not expected:
        raise ValueError("model kind and action_conditioning disagree")
    if (candidate_id == "W1") is not configured:
        raise ValueError("W0 must be action-independent and W1 action-conditioned")
    return candidate_id, model_kind, configured


def validate_s2_r4_candidate(
    value: Mapping[str, Any],
) -> tuple[str, str, bool]:
    """Fail closed for the two pre-registered R4 future-scope candidates."""

    candidate_id = str(value.get("candidate_id", ""))
    model_kind = str(value.get("model_kind", ""))
    configured = value.get("team_shared")
    if candidate_id not in {"P0", "P1"}:
        raise ValueError("S2-R4 candidate_id must be P0 or P1")
    if model_kind not in S2_R4_MODEL_KINDS:
        raise ValueError(
            "S2-R4 model kind is not in the train/evaluation allowlist"
        )
    if not isinstance(configured, bool):
        raise ValueError("round.team_shared must be boolean")
    expected = S2_R4_MODEL_KINDS[model_kind]
    if configured is not expected:
        raise ValueError("model kind and team_shared disagree")
    if (candidate_id == "P1") is not configured:
        raise ValueError("P0 must be local and P1 must enable team/shared slots")
    return candidate_id, model_kind, configured


def validate_s2_r4_hybrid_diagnostic(
    value: Mapping[str, Any],
) -> str:
    """Accept only the pre-registered evaluate-only protected hybrid."""

    model_kind = str(value.get("model_kind", ""))
    if model_kind not in S2_R4_HYBRID_MODEL_KINDS:
        raise ValueError(
            "S2-R4 hybrid model kind is not in the evaluation allowlist"
        )
    if value.get("mode") != "evaluate_only":
        raise ValueError("S2-R4 protected hybrid must be evaluate_only")
    if value.get("training_allowed") is not False:
        raise ValueError("S2-R4 protected hybrid must forbid training")
    return model_kind


def require_trainable_s2_r4_model_kind(model_kind: str) -> None:
    """Fail closed when the diagnostic kind reaches any trainer."""

    if model_kind in S2_R4_HYBRID_MODEL_KINDS:
        raise ValueError(
            "S2-R4 protected hybrid is evaluate-only and cannot be trained"
        )
    if model_kind not in S2_R4_MODEL_KINDS:
        raise ValueError("S2-R4 model kind is not in the training allowlist")


__all__ = [
    "S2_R3_MODEL_KINDS",
    "S2_R4_HYBRID_MODEL_KINDS",
    "S2_R4_MODEL_KINDS",
    "require_trainable_s2_r4_model_kind",
    "validate_s2_candidate",
    "validate_s2_r4_candidate",
    "validate_s2_r4_hybrid_diagnostic",
]
