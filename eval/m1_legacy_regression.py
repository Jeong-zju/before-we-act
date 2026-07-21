"""Auditable retained-task regression evidence for Phase M1.

The formal protocol evaluates exactly one M1 checkpoint per retained-task seed
using a round-robin assignment over the three training seeds, and evaluates the
immutable legacy direct policy on the identical seed set.  This module keeps
the simulator-facing observer small and makes the fail-closed report builder
independently testable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from envs.runtime import RolloutSummary, SimulationTransition


FORMAT_VERSION = "wam.multimodal.m1.legacy_regression/1"
REQUIRED_SUITES = ("standard", "challenge")
M1_POLICY = "m1_state_vision_future"
LEGACY_POLICY = "legacy_joint_wam_direct"
EXPECTED_M1_ACTION_SOURCE = "m1_state_vision_future"
EXPECTED_LEGACY_ACTION_SOURCE = "legacy_joint_wam_direct"
FORMAL_EPISODES_PER_SUITE = 500
MAXIMUM_SUCCESS_REGRESSION = 0.05
FORMAL_VISUAL_REFRESH_STRIDE = 2

_FORBIDDEN_PATH_PARTS = {
    "braking_agent",
    "braking_time",
    "cue_id",
    "cue_value",
    "cue_variant",
    "event_truth",
    "privileged_state",
    "rendered_cue_variant",
    "task_truth",
}


@dataclass(frozen=True)
class LegacyRegressionEpisode:
    """One policy outcome plus the complete per-episode runtime audit."""

    suite: str
    seed: int
    policy: str
    train_seed: int | None
    success: bool
    failure: bool
    failure_reason: str
    steps: int
    total_reward: float
    action_sources: tuple[str, ...]
    presented_observation_paths: tuple[tuple[str, ...], ...]
    consumed_observation_paths: tuple[tuple[str, ...], ...]
    direct_execution_steps: int
    fallback_steps: int
    fallback_reasons: tuple[str, ...]
    privileged_observation_seen: bool
    actions_finite_and_bounded: bool
    fixed_rgb_presented_steps: int
    fixed_rgb_consumed_steps: int
    visual_frame_indices: tuple[int, ...]
    visual_10hz_pattern_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditedLegacyDirectPolicy:
    """Expose an explicit action-source contract around legacy Joint WAM."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        self.last_diagnostics = {}

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        paths = _leaf_paths(observation)
        forbidden = _forbidden_paths(paths)
        if forbidden:
            raise RuntimeError(
                f"forbidden observation leaked into legacy policy: {list(forbidden)}"
            )
        if set(observation) != {"proprioception"}:
            raise RuntimeError(
                "legacy direct policy must receive proprioception and no RGB/task truth"
            )
        action = np.asarray(self.policy.act(observation), dtype=np.float32)
        base = dict(getattr(self.policy, "last_diagnostics", {}) or {})
        direct = base.get("direct_flow_executed") is True
        reason = str(base.get("fallback_reason", "none"))
        self.last_diagnostics = {
            **base,
            "action_source": EXPECTED_LEGACY_ACTION_SOURCE,
            "fallback_used": not direct,
            "fallback_enabled": False,
            "fallback_reason": reason,
            "presented_observation_paths": paths,
            "consumed_observation_paths": ("proprioception",),
            "privileged_state_seen": False,
        }
        return action


class LegacyRegressionObserver:
    """Collect exact runtime contracts without exposing truth to a policy."""

    def __init__(
        self,
        *,
        suite: str,
        policy_name: str,
        policy: Any,
        train_seed: int | None,
        control_hz: float = 20.0,
        visual_hz: float = 10.0,
    ) -> None:
        if suite not in REQUIRED_SUITES:
            raise ValueError(f"unsupported legacy suite {suite!r}")
        if policy_name not in {M1_POLICY, LEGACY_POLICY}:
            raise ValueError(f"unsupported legacy policy {policy_name!r}")
        if policy_name == M1_POLICY and train_seed is None:
            raise ValueError("M1 retained-task records require a train seed")
        if policy_name == LEGACY_POLICY and train_seed is not None:
            raise ValueError("legacy direct records cannot have a train seed")
        if control_hz <= 0.0 or visual_hz <= 0.0:
            raise ValueError("control and visual rates must be positive")
        ratio = control_hz / visual_hz
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) <= 0:
            raise ValueError("formal visual refresh must divide the control rate")
        self.suite = suite
        self.policy_name = policy_name
        self.policy = policy
        self.train_seed = train_seed
        self.refresh_stride = int(round(ratio))
        self.seed = -1
        self.action_sources: set[str] = set()
        self.presented_paths: set[tuple[str, ...]] = set()
        self.consumed_paths: set[tuple[str, ...]] = set()
        self.direct_steps = 0
        self.fallback_steps = 0
        self.fallback_reasons: list[str] = []
        self.privileged_seen = False
        self.actions_valid = True
        self.fixed_rgb_presented_steps = 0
        self.fixed_rgb_consumed_steps = 0
        self.visual_frame_indices: list[int] = []

    def on_episode_start(self, *, seed: int | None, **_: Any) -> None:
        if seed is None:
            raise ValueError("legacy regression requires explicit episode seeds")
        self.seed = int(seed)

    def on_transition(self, transition: SimulationTransition) -> None:
        diagnostics = dict(getattr(self.policy, "last_diagnostics", {}) or {})
        source = str(diagnostics.get("action_source", ""))
        if source:
            self.action_sources.add(source)
        presented = tuple(
            sorted(str(value) for value in diagnostics.get("presented_observation_paths", ()))
        )
        consumed = tuple(
            sorted(str(value) for value in diagnostics.get("consumed_observation_paths", ()))
        )
        self.presented_paths.add(presented)
        self.consumed_paths.add(consumed)
        direct = diagnostics.get("direct_flow_executed") is True
        self.direct_steps += int(direct)
        fallback = bool(
            diagnostics.get("fallback_used", False)
            or diagnostics.get("fallback_enabled", False)
            or not direct
        )
        self.fallback_steps += int(fallback)
        reason = str(diagnostics.get("fallback_reason", "none"))
        if reason != "none":
            self.fallback_reasons.append(reason)
        forbidden = _forbidden_paths((*presented, *consumed))
        self.privileged_seen = bool(
            self.privileged_seen
            or diagnostics.get("privileged_state_seen", False)
            or forbidden
        )
        action = np.asarray(transition.action, dtype=np.float32)
        self.actions_valid = bool(
            self.actions_valid
            and action.ndim == 1
            and np.isfinite(action).all()
            and np.all(action >= -1.0)
            and np.all(action <= 1.0)
        )
        self.fixed_rgb_presented_steps += int("images.fixed" in presented)
        self.fixed_rgb_consumed_steps += int("images.fixed" in consumed)
        if self.policy_name == M1_POLICY:
            runner_indices = transition.image_frame_indices
            runner_index = (
                runner_indices.get("fixed")
                if isinstance(runner_indices, Mapping)
                else None
            )
            diagnostic_index = diagnostics.get("visual_frame_index")
            if (
                isinstance(runner_index, bool)
                or not isinstance(runner_index, (int, np.integer))
                or isinstance(diagnostic_index, bool)
                or not isinstance(diagnostic_index, (int, np.integer))
                or int(runner_index) != int(diagnostic_index)
            ):
                # Preserve a concrete fail-closed marker in the episode evidence.
                self.visual_frame_indices.append(-1)
            else:
                self.visual_frame_indices.append(int(runner_index))

    def on_episode_end(self, summary: RolloutSummary) -> None:
        del summary

    def finish(self, summary: RolloutSummary) -> LegacyRegressionEpisode:
        return LegacyRegressionEpisode(
            suite=self.suite,
            seed=self.seed,
            policy=self.policy_name,
            train_seed=self.train_seed,
            success=bool(summary.final_info.get("success", False)),
            failure=bool(summary.final_info.get("failure", False)),
            failure_reason=str(summary.final_info.get("failure_reason", "none")),
            steps=int(summary.steps),
            total_reward=float(summary.total_reward),
            action_sources=tuple(sorted(self.action_sources)),
            presented_observation_paths=tuple(sorted(self.presented_paths)),
            consumed_observation_paths=tuple(sorted(self.consumed_paths)),
            direct_execution_steps=self.direct_steps,
            fallback_steps=self.fallback_steps,
            fallback_reasons=tuple(self.fallback_reasons),
            privileged_observation_seen=self.privileged_seen,
            actions_finite_and_bounded=self.actions_valid,
            fixed_rgb_presented_steps=self.fixed_rgb_presented_steps,
            fixed_rgb_consumed_steps=self.fixed_rgb_consumed_steps,
            visual_frame_indices=tuple(self.visual_frame_indices),
            visual_10hz_pattern_valid=_visual_pattern_valid(
                self.visual_frame_indices,
                steps=int(summary.steps),
                stride=self.refresh_stride,
                required=self.policy_name == M1_POLICY,
            ),
        )


def rotating_train_seed(episode_index: int, train_seeds: Sequence[int]) -> int:
    """Return the preregistered round-robin M1 checkpoint assignment."""

    if isinstance(episode_index, bool) or int(episode_index) < 0:
        raise ValueError("episode_index must be a non-negative integer")
    normalized = _three_unique_train_seeds(train_seeds)
    return normalized[int(episode_index) % len(normalized)]


def legacy_regression_report(
    m1_records: Sequence[LegacyRegressionEpisode | Mapping[str, Any]],
    legacy_records: Sequence[LegacyRegressionEpisode | Mapping[str, Any]],
    *,
    suite_seeds: Mapping[str, Sequence[int]],
    train_seeds: Sequence[int],
    formal_protocol: bool,
    source_checkpoint_sha256_before: str,
    source_checkpoint_sha256_after: str,
    expected_source_checkpoint_sha256: str,
    checkpoint_evidence: Mapping[str, Any],
    expected_episodes_per_suite: int = FORMAL_EPISODES_PER_SUITE,
    maximum_regression: float = MAXIMUM_SUCCESS_REGRESSION,
) -> dict[str, Any]:
    """Validate paired retained-task evidence and return acceptance-ready counts."""

    seeds = _three_unique_train_seeds(train_seeds)
    expected_count = int(expected_episodes_per_suite)
    if expected_count <= 0:
        raise ValueError("expected_episodes_per_suite must be positive")
    if not np.isfinite(maximum_regression) or maximum_regression < 0.0:
        raise ValueError("maximum_regression must be finite and non-negative")
    expected_by_suite = _validate_suite_seed_sets(suite_seeds, expected_count)
    m1 = tuple(_coerce_record(value) for value in m1_records)
    legacy = tuple(_coerce_record(value) for value in legacy_records)
    m1_map, m1_duplicates = _record_map(m1)
    legacy_map, legacy_duplicates = _record_map(legacy)

    suite_reports: dict[str, Any] = {}
    exact_matrix = not m1_duplicates and not legacy_duplicates
    action_contract = True
    observation_contract = True
    direct_contract = True
    no_fallback = True
    no_privileged = True
    actions_valid = True
    visual_contract = True
    rotation_contract = True
    for suite in REQUIRED_SUITES:
        expected_seeds = expected_by_suite[suite]
        expected_keys = {(suite, seed) for seed in expected_seeds}
        observed_m1 = {key for key in m1_map if key[0] == suite}
        observed_legacy = {key for key in legacy_map if key[0] == suite}
        suite_exact = observed_m1 == expected_keys and observed_legacy == expected_keys
        exact_matrix = exact_matrix and suite_exact
        pairs: list[dict[str, Any]] = []
        outcomes = Counter()
        train_seed_counts: Counter[int] = Counter()
        m1_successes = 0
        legacy_successes = 0
        for index, seed in enumerate(expected_seeds):
            key = (suite, seed)
            first = m1_map.get(key)
            second = legacy_map.get(key)
            if first is None or second is None:
                continue
            expected_train_seed = rotating_train_seed(index, seeds)
            rotation_ok = first.train_seed == expected_train_seed
            rotation_contract = rotation_contract and rotation_ok
            if first.train_seed is not None:
                train_seed_counts[int(first.train_seed)] += 1
            first_action_ok = first.action_sources == (EXPECTED_M1_ACTION_SOURCE,)
            second_action_ok = second.action_sources == (EXPECTED_LEGACY_ACTION_SOURCE,)
            action_contract = action_contract and first_action_ok and second_action_ok
            m1_paths_ok = _m1_observation_contract(first)
            legacy_paths_ok = _legacy_observation_contract(second)
            observation_contract = observation_contract and m1_paths_ok and legacy_paths_ok
            direct_ok = (
                first.direct_execution_steps == first.steps
                and second.direct_execution_steps == second.steps
            )
            fallback_ok = (
                first.fallback_steps == 0
                and second.fallback_steps == 0
                and not first.fallback_reasons
                and not second.fallback_reasons
            )
            privileged_ok = not (
                first.privileged_observation_seen
                or second.privileged_observation_seen
            )
            bounded_ok = (
                first.actions_finite_and_bounded
                and second.actions_finite_and_bounded
            )
            m1_pattern = _visual_pattern_valid(
                first.visual_frame_indices,
                steps=first.steps,
                stride=FORMAL_VISUAL_REFRESH_STRIDE,
                required=True,
            )
            legacy_pattern = _visual_pattern_valid(
                second.visual_frame_indices,
                steps=second.steps,
                stride=FORMAL_VISUAL_REFRESH_STRIDE,
                required=False,
            )
            rgb_ok = (
                first.fixed_rgb_presented_steps == first.steps
                and first.fixed_rgb_consumed_steps == first.steps
                and first.visual_10hz_pattern_valid is m1_pattern
                and m1_pattern
                and second.fixed_rgb_presented_steps == 0
                and second.fixed_rgb_consumed_steps == 0
                and not second.visual_frame_indices
                and second.visual_10hz_pattern_valid is legacy_pattern
                and legacy_pattern
            )
            direct_contract = direct_contract and direct_ok
            no_fallback = no_fallback and fallback_ok
            no_privileged = no_privileged and privileged_ok
            actions_valid = actions_valid and bounded_ok
            visual_contract = visual_contract and rgb_ok
            m1_successes += int(first.success)
            legacy_successes += int(second.success)
            outcome_key = f"m1_{int(first.success)}_legacy_{int(second.success)}"
            outcomes[outcome_key] += 1
            pairs.append(
                {
                    "seed": seed,
                    "m1_train_seed": first.train_seed,
                    "expected_m1_train_seed": expected_train_seed,
                    "rotation_matches": rotation_ok,
                    "m1_success": first.success,
                    "legacy_success": second.success,
                    "outcome_matches": first.success == second.success,
                    "m1_failure_reason": first.failure_reason,
                    "legacy_failure_reason": second.failure_reason,
                    "action_sources": {
                        "m1": list(first.action_sources),
                        "legacy": list(second.action_sources),
                    },
                }
            )
        paired_count = len(pairs)
        m1_rate = m1_successes / paired_count if paired_count else 0.0
        legacy_rate = legacy_successes / paired_count if paired_count else 0.0
        regression = legacy_rate - m1_rate
        suite_reports[suite] = {
            "m1_successes": m1_successes,
            "m1_episodes": paired_count,
            "legacy_successes": legacy_successes,
            "legacy_episodes": paired_count,
            "m1_success_rate": m1_rate,
            "legacy_success_rate": legacy_rate,
            "legacy_minus_m1": regression,
            "maximum_regression": float(maximum_regression),
            "regression_passed": bool(
                paired_count == expected_count
                and regression <= maximum_regression + 1e-12
            ),
            "exact_seed_pairing": suite_exact and paired_count == expected_count,
            "seed_sha256": _integer_sequence_sha256(expected_seeds),
            "train_seed_counts": {
                str(seed): train_seed_counts.get(seed, 0) for seed in seeds
            },
            "paired_outcomes": dict(sorted(outcomes.items())),
            "pairs": pairs,
        }

    no_unexpected_suites = all(
        record.suite in REQUIRED_SUITES for record in (*m1, *legacy)
    )
    source_immutable = bool(
        _is_sha256(source_checkpoint_sha256_before)
        and source_checkpoint_sha256_before == source_checkpoint_sha256_after
        and source_checkpoint_sha256_before == expected_source_checkpoint_sha256
    )
    checkpoint_contract = _checkpoint_contract(checkpoint_evidence, seeds)
    regression_passed = all(
        suite_reports[suite]["regression_passed"] for suite in REQUIRED_SUITES
    )
    checks = {
        "formal_protocol": formal_protocol is True,
        "exact_500_seed_pairs_per_suite": bool(
            expected_count == FORMAL_EPISODES_PER_SUITE
            and exact_matrix
            and no_unexpected_suites
        ),
        "round_robin_three_train_seeds": rotation_contract,
        "primary_checkpoints_strict_and_bound": checkpoint_contract,
        "action_sources_exact": action_contract,
        "observation_paths_exact": observation_contract,
        "m1_raw_fixed_rgb_10hz_legacy_no_rgb": visual_contract,
        "direct_execution_all_steps": direct_contract,
        "fallback_disabled_and_unused": no_fallback,
        "no_privileged_observations": no_privileged,
        "actions_finite_and_bounded": actions_valid,
        "source_checkpoint_immutable": source_immutable,
        "legacy_regression_at_most_5pp": regression_passed,
    }
    technical = all(
        value for name, value in checks.items() if name != "formal_protocol"
    )
    passed = bool(formal_protocol is True and technical)
    return {
        "format_version": FORMAT_VERSION,
        "formal_protocol": formal_protocol is True,
        "passed": passed,
        "diagnostic_criteria_met": technical,
        "checks": checks,
        "train_seeds": list(seeds),
        "expected_episodes_per_suite": expected_count,
        "source_checkpoint": {
            "expected_tree_sha256": expected_source_checkpoint_sha256,
            "tree_sha256_before": source_checkpoint_sha256_before,
            "tree_sha256_after": source_checkpoint_sha256_after,
            "immutable": source_immutable,
        },
        "checkpoint_evidence": dict(checkpoint_evidence),
        "suites": suite_reports,
        "records": {
            "m1": [value.to_dict() for value in m1],
            "legacy": [value.to_dict() for value in legacy],
        },
    }


def checkpoint_tree_sha256(directory: str | Path) -> str:
    """Hash a checkpoint tree without following symlinks."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    tree: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint tree cannot contain symlinks: {path}")
        if path.is_file():
            tree[str(path.relative_to(root))] = _file_sha256(path)
    if not tree:
        raise ValueError("checkpoint tree is empty")
    serialized = json.dumps(
        dict(sorted(tree.items())), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def module_state_sha256(module: Any) -> str:
    """Hash state tensors for exact embedded-legacy comparisons."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(str(name).encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _coerce_record(
    value: LegacyRegressionEpisode | Mapping[str, Any],
) -> LegacyRegressionEpisode:
    if isinstance(value, LegacyRegressionEpisode):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("legacy regression records must be mappings or dataclasses")
    try:
        return LegacyRegressionEpisode(
            suite=str(value["suite"]),
            seed=int(value["seed"]),
            policy=str(value["policy"]),
            train_seed=(
                None if value.get("train_seed") is None else int(value["train_seed"])
            ),
            success=_strict_bool(value["success"], "success"),
            failure=_strict_bool(value["failure"], "failure"),
            failure_reason=str(value["failure_reason"]),
            steps=int(value["steps"]),
            total_reward=float(value["total_reward"]),
            action_sources=tuple(str(item) for item in value["action_sources"]),
            presented_observation_paths=tuple(
                tuple(str(item) for item in paths)
                for paths in value["presented_observation_paths"]
            ),
            consumed_observation_paths=tuple(
                tuple(str(item) for item in paths)
                for paths in value["consumed_observation_paths"]
            ),
            direct_execution_steps=int(value["direct_execution_steps"]),
            fallback_steps=int(value["fallback_steps"]),
            fallback_reasons=tuple(str(item) for item in value["fallback_reasons"]),
            privileged_observation_seen=_strict_bool(
                value["privileged_observation_seen"], "privileged_observation_seen"
            ),
            actions_finite_and_bounded=_strict_bool(
                value["actions_finite_and_bounded"], "actions_finite_and_bounded"
            ),
            fixed_rgb_presented_steps=int(value["fixed_rgb_presented_steps"]),
            fixed_rgb_consumed_steps=int(value["fixed_rgb_consumed_steps"]),
            visual_frame_indices=tuple(int(item) for item in value["visual_frame_indices"]),
            visual_10hz_pattern_valid=_strict_bool(
                value["visual_10hz_pattern_valid"], "visual_10hz_pattern_valid"
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid legacy regression record: {error}") from error


def _record_map(
    records: Sequence[LegacyRegressionEpisode],
) -> tuple[dict[tuple[str, int], LegacyRegressionEpisode], list[tuple[str, int]]]:
    result: dict[tuple[str, int], LegacyRegressionEpisode] = {}
    duplicates: list[tuple[str, int]] = []
    for record in records:
        key = (record.suite, record.seed)
        expected_policy = M1_POLICY if record.train_seed is not None else LEGACY_POLICY
        if record.policy != expected_policy:
            raise ValueError(
                f"record policy {record.policy!r} disagrees with train_seed contract"
            )
        if record.steps <= 0 or record.failure is record.success:
            raise ValueError("legacy record outcome/step contract is invalid")
        if key in result:
            duplicates.append(key)
        else:
            result[key] = record
    return result, duplicates


def _validate_suite_seed_sets(
    suite_seeds: Mapping[str, Sequence[int]], expected_count: int
) -> dict[str, tuple[int, ...]]:
    if set(suite_seeds) != set(REQUIRED_SUITES):
        raise ValueError("legacy suite seed mapping must contain standard/challenge only")
    result: dict[str, tuple[int, ...]] = {}
    for suite in REQUIRED_SUITES:
        values = tuple(int(value) for value in suite_seeds[suite])
        if len(values) != expected_count or len(set(values)) != len(values):
            raise ValueError(f"suite {suite!r} must contain {expected_count} unique seeds")
        if any(value < 0 for value in values):
            raise ValueError("evaluation seeds must be non-negative")
        result[suite] = values
    if set(result["standard"]) & set(result["challenge"]):
        raise ValueError("standard and challenge seed sets must be disjoint")
    return result


def _three_unique_train_seeds(values: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or len(set(result)) != 3 or any(value < 0 for value in result):
        raise ValueError("formal legacy regression requires three unique train seeds")
    return result  # type: ignore[return-value]


def _checkpoint_contract(value: Mapping[str, Any], seeds: Sequence[int]) -> bool:
    if set(value) != {str(seed) for seed in seeds}:
        return False
    for seed in seeds:
        item = value.get(str(seed))
        if not isinstance(item, Mapping):
            return False
        if (
            int(item.get("train_seed", -1)) != seed
            or item.get("model_variant") != "state_vision_future"
            or item.get("strict_reload_passed") is not True
            or item.get("embedded_legacy_matches_source") is not True
            or not _is_sha256(item.get("tree_sha256"))
        ):
            return False
    return True


def _m1_observation_contract(record: LegacyRegressionEpisode) -> bool:
    required_presented = {
        "images.fixed",
        "image_frame_indices.fixed",
        "past_executed_actions",
        "proprioception",
        "task.id",
        "task.text",
    }
    required_consumed = {
        "image_frame_indices.fixed",
        "images.fixed",
        "proprioception",
        "task.id",
    }
    return bool(
        len(record.presented_observation_paths) == 1
        and required_presented <= set(record.presented_observation_paths[0])
        and len(record.consumed_observation_paths) == 1
        and required_consumed <= set(record.consumed_observation_paths[0])
        and not _forbidden_paths(
            (*record.presented_observation_paths[0], *record.consumed_observation_paths[0])
        )
    )


def _legacy_observation_contract(record: LegacyRegressionEpisode) -> bool:
    return bool(
        record.presented_observation_paths == (("proprioception",),)
        and record.consumed_observation_paths == (("proprioception",),)
    )


def _visual_pattern_valid(
    indices: Sequence[int], *, steps: int, stride: int, required: bool
) -> bool:
    if not required:
        return len(indices) == 0
    if steps <= 0 or len(indices) != steps:
        return False
    expected = tuple(index // stride for index in range(steps))
    return tuple(indices) == expected


def _leaf_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        if not value and prefix:
            paths.append(prefix)
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(value[key], child))
    elif prefix:
        paths.append(prefix)
    return tuple(paths)


def _forbidden_paths(paths: Sequence[str]) -> tuple[str, ...]:
    result = []
    for path in paths:
        segments = tuple(str(segment).lower() for segment in str(path).split("."))
        if any(
            segment in _FORBIDDEN_PATH_PARTS
            or segment.startswith("future_")
            or segment.startswith("next_observation")
            for segment in segments
        ):
            result.append(str(path))
    return tuple(sorted(set(result)))


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _integer_sequence_sha256(values: Sequence[int]) -> str:
    serialized = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


__all__ = [
    "AuditedLegacyDirectPolicy",
    "EXPECTED_LEGACY_ACTION_SOURCE",
    "EXPECTED_M1_ACTION_SOURCE",
    "FORMAT_VERSION",
    "FORMAL_EPISODES_PER_SUITE",
    "LEGACY_POLICY",
    "LegacyRegressionEpisode",
    "LegacyRegressionObserver",
    "M1_POLICY",
    "MAXIMUM_SUCCESS_REGRESSION",
    "REQUIRED_SUITES",
    "checkpoint_tree_sha256",
    "legacy_regression_report",
    "module_state_sha256",
    "rotating_train_seed",
]
