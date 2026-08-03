"""Fail-closed S4-R7/R8 model registry and paired-axis validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


S4_R7_MODEL_KINDS: dict[str, tuple[str, float]] = {
    "s4_r7_token_preserving": ("P0", 0.0),
    "s4_r7_world_utility_coupling": ("P1", 0.05),
}
S4_R7_BUDGET_MODES: dict[str, tuple[int, int, int]] = {
    # (optimizer updates, Flow unfreeze update, nominal valid-agent windows)
    "fast_selection_30k": (30_000, 6_400, 1_152_000),
}

S4_R8_MODEL_KINDS: dict[str, tuple[str, str]] = {
    "s4_r8_horizon_prefix_mean": ("P0", "prefix_mean"),
    "s4_r8_causal_prefix_attention": (
        "P1",
        "causal_prefix_attention",
    ),
}

S4_MODEL_KINDS = frozenset(S4_R7_MODEL_KINDS) | frozenset(
    S4_R8_MODEL_KINDS
)
S4_ZERO_AUXILIARY_WEIGHTS = (
    "relation_weight",
    "specialization_weight",
    "anchor_weight",
)
S4_ALLOWED_OBJECTIVE_WEIGHTS = frozenset(
    {
        "flow_loss_weight",
        "future_state_loss_weight",
        "future_visual_loss_weight",
        "utility_coupling_weight",
        *S4_ZERO_AUXILIARY_WEIGHTS,
    }
)

_MISSING = object()


def _field(
    value: Mapping[str, object],
    key: str,
    *,
    sections: Sequence[str],
    default: object = _MISSING,
) -> object:
    observed: list[tuple[str, object]] = []
    if key in value:
        observed.append((key, value[key]))
    for section_name in sections:
        if section_name not in value:
            continue
        raw_section = value[section_name]
        if not isinstance(raw_section, Mapping):
            raise ValueError(f"S4 config section {section_name!r} must be a mapping")
        if key in raw_section:
            observed.append((f"{section_name}.{key}", raw_section[key]))
    if not observed:
        if default is _MISSING:
            raise ValueError(f"S4 config is missing required field {key!r}")
        return default
    expected = observed[0][1]
    if any(candidate != expected for _, candidate in observed[1:]):
        locations = [location for location, _ in observed]
        raise ValueError(
            f"S4 duplicate field {key!r} disagrees across {locations!r}"
        )
    return expected


def _finite_weight(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"S4 {name} must be a finite numeric weight")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"S4 {name} must be a finite numeric weight")
    return normalized


def _validate_disabled_auxiliary_weights(value: Mapping[str, object]) -> None:
    scopes: list[tuple[str, Mapping[object, object]]] = [("<root>", value)]
    for section_name in ("training", "round"):
        raw_section = value.get(section_name)
        if isinstance(raw_section, Mapping):
            scopes.append((section_name, raw_section))
    for scope_name, scope in scopes:
        for raw_key in scope:
            key = str(raw_key)
            if key.endswith("_weight") and key not in S4_ALLOWED_OBJECTIVE_WEIGHTS:
                raise ValueError(
                    f"S4 unknown auxiliary/objective weight is not in the "
                    f"allowlist: {scope_name}.{key}"
                )
    for key in S4_ZERO_AUXILIARY_WEIGHTS:
        configured = _field(
            value,
            key,
            sections=("training", "round"),
            default=0.0,
        )
        if _finite_weight(configured, name=key) != 0.0:
            raise ValueError(
                f"S4 {key} is disabled by the capability-only allowlist and "
                "must be exactly zero"
            )


def validate_s4_r7_candidate(
    value: Mapping[str, object],
) -> tuple[str, str, float]:
    """Validate one R7 candidate against the pre-registered WUC axis."""

    round_id = str(_field(value, "round_id", sections=("round",)))
    candidate_id = str(_field(value, "candidate_id", sections=("round",)))
    model_kind = str(_field(value, "model_kind", sections=("round",)))
    if round_id != "s4-r7" or model_kind not in S4_R7_MODEL_KINDS:
        raise ValueError(
            f"S4-R7 model kind is not in the train/evaluation allowlist: "
            f"{model_kind!r}"
        )
    expected_candidate, expected_utility = S4_R7_MODEL_KINDS[model_kind]
    utility = _finite_weight(
        _field(
            value,
            "utility_coupling_weight",
            sections=("training", "round"),
        ),
        name="utility_coupling_weight",
    )
    if candidate_id != expected_candidate or utility != expected_utility:
        raise ValueError(
            "S4-R7 candidate identity disagrees with the model-kind allowlist"
        )
    budget_mode = str(_field(value, "budget_mode", sections=("training",)))
    if budget_mode not in S4_R7_BUDGET_MODES:
        raise ValueError(
            f"S4-R7 budget mode is not in the training allowlist: {budget_mode!r}"
        )
    expected_updates, expected_unfreeze, expected_agent_windows = (
        S4_R7_BUDGET_MODES[budget_mode]
    )
    observed_budget = (
        int(_field(value, "updates", sections=("training",))),
        int(_field(value, "flow_unfreeze_update", sections=("training",))),
        int(_field(value, "agent_window_budget", sections=("training",))),
    )
    if observed_budget != (
        expected_updates,
        expected_unfreeze,
        expected_agent_windows,
    ):
        raise ValueError(
            "S4-R7 budget fields disagree with the registered fast-selection mode"
        )
    _validate_disabled_auxiliary_weights(value)
    return candidate_id, model_kind, utility


def validate_s4_r8_candidate(
    value: Mapping[str, object],
) -> tuple[str, str, str]:
    """Validate one R8 candidate against the pre-registered aggregator axis."""

    round_id = str(_field(value, "round_id", sections=("round",)))
    candidate_id = str(_field(value, "candidate_id", sections=("round",)))
    model_kind = str(_field(value, "model_kind", sections=("round",)))
    if round_id != "s4-r8" or model_kind not in S4_R8_MODEL_KINDS:
        raise ValueError(
            f"S4-R8 model kind is not in the train/evaluation allowlist: "
            f"{model_kind!r}"
        )
    expected_candidate, expected_aggregator = S4_R8_MODEL_KINDS[model_kind]
    aggregator = str(
        _field(
            value,
            "action_prefix_aggregator",
            sections=("model", "round"),
        )
    )
    if candidate_id != expected_candidate or aggregator != expected_aggregator:
        raise ValueError(
            "S4-R8 candidate identity disagrees with the model-kind allowlist"
        )
    _validate_disabled_auxiliary_weights(value)
    return candidate_id, model_kind, aggregator


def validate_s4_candidate(
    value: Mapping[str, object],
) -> tuple[str, str, object]:
    """Dispatch to the exact registered S4 round; unknown rounds fail closed."""

    round_id = str(_field(value, "round_id", sections=("round",)))
    if round_id == "s4-r7":
        return validate_s4_r7_candidate(value)
    if round_id == "s4-r8":
        return validate_s4_r8_candidate(value)
    raise ValueError(f"S4 round is not in the allowlist: {round_id!r}")


_R7_AXIS_PATHS = frozenset(
    {
        ("candidate_id",),
        ("model_kind",),
        ("utility_coupling_weight",),
        ("round", "candidate_id"),
        ("round", "model_kind"),
        ("round", "utility_coupling_weight"),
        ("training", "utility_coupling_weight"),
    }
)
_R8_AXIS_PATHS = frozenset(
    {
        ("candidate_id",),
        ("model_kind",),
        ("action_prefix_aggregator",),
        ("round", "candidate_id"),
        ("round", "model_kind"),
        ("round", "action_prefix_aggregator"),
        ("model", "action_prefix_aggregator"),
    }
)


def _canonical_without_paths(
    value: object,
    *,
    ignored_paths: frozenset[tuple[str, ...]],
    path: tuple[str, ...] = (),
) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            if child_path in ignored_paths:
                continue
            normalized[key] = _canonical_without_paths(
                child, ignored_paths=ignored_paths, path=child_path
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _canonical_without_paths(
                child, ignored_paths=ignored_paths, path=(*path, str(index))
            )
            for index, child in enumerate(value)
        ]
    return value


def _first_difference(
    left: object, right: object, *, path: str = "<root>"
) -> str:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return f"{path} keys {sorted(left_keys)!r} != {sorted(right_keys)!r}"
        for key in sorted(left_keys):
            child = _first_difference(
                left[key], right[key], path=f"{path}.{key}"
            )
            if child:
                return child
        return ""
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} lengths {len(left)} != {len(right)}"
        for index, (left_child, right_child) in enumerate(
            zip(left, right, strict=True)
        ):
            child = _first_difference(
                left_child, right_child, path=f"{path}[{index}]"
            )
            if child:
                return child
        return ""
    return "" if left == right else f"{path}: {left!r} != {right!r}"


def validate_s4_candidate_pair(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    round_id: str | None = None,
) -> str:
    """Reject a pair if anything beyond its registered candidate axis drifts."""

    left_round = str(_field(left, "round_id", sections=("round",)))
    right_round = str(_field(right, "round_id", sections=("round",)))
    expected_round = left_round if round_id is None else str(round_id)
    if left_round != expected_round or right_round != expected_round:
        raise ValueError("S4 pair candidates do not belong to the same round")
    if expected_round == "s4-r7":
        identities = (
            validate_s4_r7_candidate(left),
            validate_s4_r7_candidate(right),
        )
        ignored_paths = _R7_AXIS_PATHS
    elif expected_round == "s4-r8":
        identities = (
            validate_s4_r8_candidate(left),
            validate_s4_r8_candidate(right),
        )
        ignored_paths = _R8_AXIS_PATHS
    else:
        raise ValueError(f"S4 pair round is not in the allowlist: {expected_round!r}")
    if {identity[0] for identity in identities} != {"P0", "P1"}:
        raise ValueError("S4 pair must contain exactly one registered P0 and P1")

    canonical_left = _canonical_without_paths(
        left, ignored_paths=ignored_paths
    )
    canonical_right = _canonical_without_paths(
        right, ignored_paths=ignored_paths
    )
    difference = _first_difference(canonical_left, canonical_right)
    if difference:
        raise ValueError(
            f"{expected_round} pair drifts beyond its pre-registered axis: "
            f"{difference}"
        )
    return expected_round


def validate_s4_r7_pair(
    p0: Mapping[str, object], p1: Mapping[str, object]
) -> str:
    return validate_s4_candidate_pair(p0, p1, round_id="s4-r7")


def validate_s4_r8_pair(
    p0: Mapping[str, object], p1: Mapping[str, object]
) -> str:
    return validate_s4_candidate_pair(p0, p1, round_id="s4-r8")


# Descriptive aliases used by launcher/acceptance code.
validate_s4_pair = validate_s4_candidate_pair
validate_s4_r7_candidate_pair = validate_s4_r7_pair
validate_s4_r8_candidate_pair = validate_s4_r8_pair


__all__ = [
    "S4_MODEL_KINDS",
    "S4_ALLOWED_OBJECTIVE_WEIGHTS",
    "S4_R7_BUDGET_MODES",
    "S4_R7_MODEL_KINDS",
    "S4_R8_MODEL_KINDS",
    "S4_ZERO_AUXILIARY_WEIGHTS",
    "validate_s4_candidate",
    "validate_s4_candidate_pair",
    "validate_s4_pair",
    "validate_s4_r7_candidate",
    "validate_s4_r7_candidate_pair",
    "validate_s4_r7_pair",
    "validate_s4_r8_candidate",
    "validate_s4_r8_candidate_pair",
    "validate_s4_r8_pair",
]
