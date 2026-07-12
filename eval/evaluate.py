"""Paired evaluation utilities for decentralized FE-PC-WAM.

This module deliberately evaluates *episode records* rather than importing a
particular policy implementation.  A rollout driver may therefore change
without changing the comparison contract: every communication mode must be
run on the same episode/input records and seeds, and must emit the diagnostics
defined below.

The ``oracle_upper_bound`` mode is useful for measuring headroom, but it sees
privileged teammate-plan information and is never a deployable policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEPLOYABLE_COMMUNICATION_MODES = (
    "no_comm",
    "always_reply",
    "selective_vpi",
    "periodic",
    "random",
)
COMMUNICATION_MODES = (
    *DEPLOYABLE_COMMUNICATION_MODES,
    "oracle_upper_bound",
)
ORACLE_MODE = "oracle_upper_bound"
NON_DEPLOYABLE_MODES = frozenset({ORACLE_MODE})
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

_PAIRED_EPISODE_METRICS = (
    "success_rate",
    "safety_rate",
    "return_mean",
    "collision_count_mean",
    "force_violation_rate_mean",
    "max_force_mean",
    "request_rate",
    "reply_rate",
    "actual_request_reply_bits_per_episode",
    "communication_delay_mean",
    "expected_latency_cost_mean",
    "incurred_expected_latency_cost_mean",
    "vpi_mean",
    "code_surprise_mean",
    "residual_surprise_mean",
    "plan_surprise_mean",
    "G_before_mean",
    "G_after_mean",
    "G_improvement_mean",
    "replan_rate",
    "action_change_l2_mean",
)
_UNSPECIFIED_SCENARIO = "__unspecified__"

_MODE_ALIASES = {
    "none": "no_comm",
    "no_communication": "no_comm",
    "always": "always_reply",
    "always_comm": "always_reply",
    "selective": "selective_vpi",
    "selective_comm": "selective_vpi",
    "vpi": "selective_vpi",
    "periodic_reply": "periodic",
    "random_reply": "random",
    "oracle": ORACLE_MODE,
    "oracle_reply": ORACLE_MODE,
}


class EvaluationContractError(ValueError):
    """Raised when modes were not evaluated on identical episode inputs."""


def canonical_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    value = _MODE_ALIASES.get(value, value)
    if value not in COMMUNICATION_MODES:
        raise EvaluationContractError(
            f"unknown communication mode {mode!r}; expected one of {COMMUNICATION_MODES}"
        )
    return value


def compare_communication_modes(
    records_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_modes: Sequence[str] = COMMUNICATION_MODES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate paired inputs and aggregate all  communication modes.

    Each episode record must contain ``seed``.  ``episode_id`` (or one of its
    accepted aliases) is used with the seed as the pairing key; if no episode
    id is present the seed itself must be unique.  Supplying an ``input_digest``
    is strongly recommended and, if present in any mode for a pair, is required
    to match in every mode for that pair.
    """

    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise EvaluationContractError("bootstrap_samples must be a positive integer")
    if bootstrap_samples <= 0:
        raise EvaluationContractError("bootstrap_samples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise EvaluationContractError("bootstrap_seed must be an integer")

    canonical: dict[str, list[Mapping[str, Any]]] = {}
    for raw_mode, records in records_by_mode.items():
        mode = canonical_mode(raw_mode)
        if mode in canonical:
            raise EvaluationContractError(
                f"duplicate records for canonical mode {mode!r}"
            )
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise EvaluationContractError(f"records for {mode} must be a sequence")
        canonical[mode] = list(records)

    required = tuple(canonical_mode(mode) for mode in required_modes)
    missing = [mode for mode in required if mode not in canonical]
    if missing:
        raise EvaluationContractError(
            f"missing required communication modes: {missing}"
        )
    if not required:
        raise EvaluationContractError("required_modes must not be empty")

    reference_mode = "no_comm" if "no_comm" in required else required[0]
    indexed = {mode: _index_episode_records(mode, canonical[mode]) for mode in required}
    reference_keys = set(indexed[reference_mode])
    if not reference_keys:
        raise EvaluationContractError(f"{reference_mode} has no episode records")

    for mode in required:
        mode_keys = set(indexed[mode])
        if mode_keys != reference_keys:
            missing_keys = sorted(reference_keys - mode_keys, key=repr)
            extra_keys = sorted(mode_keys - reference_keys, key=repr)
            raise EvaluationContractError(
                f"{mode} does not use the same paired episodes as {reference_mode}; "
                f"missing={missing_keys}, extra={extra_keys}"
            )

    digest_verified = True
    scenario_field_verified = True
    for key in sorted(reference_keys, key=repr):
        digests = {
            mode: _input_digest(indexed[mode][key])
            for mode in required
        }
        provided = {mode: value for mode, value in digests.items() if value is not None}
        if provided and len(provided) != len(required):
            absent = [mode for mode, value in digests.items() if value is None]
            raise EvaluationContractError(
                f"paired episode {key!r} has input_digest in only some modes; missing={absent}"
            )
        if provided and len(set(provided.values())) != 1:
            raise EvaluationContractError(
                f"paired episode {key!r} input digests differ by mode: {provided}"
            )
        if not provided:
            digest_verified = False

        scenarios = {
            mode: _scenario(indexed[mode][key])
            for mode in required
        }
        provided_scenarios = {
            mode: value for mode, value in scenarios.items() if value is not None
        }
        if provided_scenarios and len(provided_scenarios) != len(required):
            absent = [mode for mode, value in scenarios.items() if value is None]
            raise EvaluationContractError(
                f"paired episode {key!r} has scenario in only some modes; missing={absent}"
            )
        if provided_scenarios and len(set(provided_scenarios.values())) != 1:
            raise EvaluationContractError(
                f"paired episode {key!r} scenarios differ by mode: {provided_scenarios}"
            )
        if not provided_scenarios:
            scenario_field_verified = False

    summaries = {
        mode: aggregate_episode_records(canonical[mode], mode=mode)
        for mode in required
    }
    reference = summaries[reference_mode]
    for mode, summary in summaries.items():
        summary["vs_no_comm"] = _paired_summary_delta(summary, reference)

    paired_comparisons: dict[str, dict[str, Any]] = {}
    comparison_baselines = tuple(
        mode for mode in ("no_comm", "always_reply") if mode in indexed
    )
    for mode in required:
        mode_comparisons: dict[str, Any] = {}
        for baseline in comparison_baselines:
            mode_comparisons[f"vs_{baseline}"] = _paired_episode_comparison(
                indexed[mode],
                indexed[baseline],
                mode=mode,
                baseline=baseline,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        summaries[mode]["paired_deltas"] = mode_comparisons
        if "always_reply" in summaries:
            summaries[mode]["vs_always_reply"] = _paired_summary_delta(
                summaries[mode], summaries["always_reply"]
            )
        paired_comparisons[mode] = mode_comparisons

    scenario_breakdown = _scenario_breakdown(
        indexed,
        required=required,
        reference_keys=reference_keys,
    )
    selective_acceptance = _selective_vpi_acceptance(
        summaries,
        paired_comparisons,
    )
    truncated_records = sum(
        int(bool(record.get("truncated", False)))
        for mode in required
        for record in canonical[mode]
    )
    if truncated_records:
        selective_acceptance = {
            **selective_acceptance,
            "status": "not_applicable_truncated",
            "passed": None,
            "reason": (
                "At least one episode was truncated by the evaluation driver; "
                "wiring smoke results are not a formal performance acceptance."
            ),
        }
    deployable_order = [
        mode for mode in required if mode in DEPLOYABLE_COMMUNICATION_MODES
    ]

    return {
        "contract": "fe_pc_wam_paired_communication_evaluation",
        "paired_inputs_verified": True,
        "input_digest_verified": digest_verified,
        "scenario_field_verified": scenario_field_verified,
        "reference_mode": reference_mode,
        "episode_count": len(reference_keys),
        "truncated_episode_record_count": truncated_records,
        "modes": summaries,
        "mode_order": list(required),
        "deployable_modes": {
            mode: summaries[mode]
            for mode in deployable_order
        },
        "deployable_mode_order": deployable_order,
        "paired_comparisons": paired_comparisons,
        "paired_delta_unit": "episode",
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "method": "paired_episode_percentile",
        },
        "latency_semantics": {
            "actual_communication_delay_mean": (
                "Observed simulator delay; synchronous rollout records should report 0."
            ),
            "expected_latency_cost_mean": (
                "Expected delay cost used by the VPI decision, not measured network latency."
            ),
            "incurred_expected_latency_cost_mean": (
                "Expected delay cost charged only after a request was actually issued."
            ),
            "real_network_latency_validated": False,
        },
        "scenario_breakdown": scenario_breakdown,
        "selective_vpi_acceptance": selective_acceptance,
        "oracle_notice": (
            "oracle_upper_bound is NON-DEPLOYABLE: it may use the true teammate plan "
            "and is reported only as an explicitly requested upper-bound diagnostic; "
            "it is excluded from the deployable-mode summary and CLI default."
        ),
    }


def aggregate_episode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    """Aggregate success/safety and request-reply diagnostics.

    Step-level values can be stored directly as vectors or inside a ``steps``
    list.  Scalar episode fields are also accepted.  Missing optional metrics
    are emitted as JSON ``null`` instead of being silently interpreted as zero.
    """

    mode = canonical_mode(mode)
    if not records:
        raise EvaluationContractError(f"{mode} has no episode records")

    success: list[float] = []
    safety: list[float] = []
    returns: list[float] = []
    decision_count = 0
    request_count = 0.0
    reply_count = 0.0
    request_observations = 0
    reply_observations = 0
    bits: list[float] = []
    delays: list[float] = []
    expected_latency_cost: list[float] = []
    incurred_expected_latency_cost: list[float] = []
    vpi: list[float] = []
    code_surprise: list[float] = []
    residual_surprise: list[float] = []
    plan_surprise: list[float] = []
    g_before: list[float] = []
    g_after: list[float] = []
    replanned: list[float] = []
    action_change: list[float] = []
    g_improvement: list[float] = []
    collision_count: list[float] = []
    collision_steps: list[float] = []
    collision_contact_instances: list[float] = []
    force_violation_rate: list[float] = []
    max_force: list[float] = []
    failure_reason_counts: Counter[str] = Counter()
    compact_sums: Counter[str] = Counter()
    compact_counts: Counter[str] = Counter()

    def collect_decision_metric(
        record: Mapping[str, Any],
        label: str,
        aliases: Sequence[str],
        target: list[float],
    ) -> None:
        stat = _decision_aggregate_stat(record, aliases)
        if stat is None:
            target.extend(_metric_values(record, aliases))
            return
        total, count, _ = stat
        compact_sums[label] += total
        compact_counts[label] += count

    for record in records:
        success_value = _episode_flag(
            record, ("success", "task_success"), require_all=False
        )
        if success_value is not None:
            success.append(success_value)
        safety_value = _episode_safety(record)
        if safety_value is not None:
            safety.append(safety_value)
        return_value = _episode_scalar(record, ("return", "episode_return", "reward_sum"))
        if return_value is not None:
            returns.append(return_value)

        request_aliases = (
            "request",
            "requested",
            "request_sent",
            "request_trigger",
            "trigger",
        )
        reply_aliases = (
            "reply",
            "replied",
            "reply_received",
            "response_received",
        )
        request_stat = _decision_aggregate_stat(record, request_aliases)
        reply_stat = _decision_aggregate_stat(record, reply_aliases)
        requests = [] if request_stat is not None else _metric_values(record, request_aliases)
        replies = [] if reply_stat is not None else _metric_values(record, reply_aliases)
        steps = _decision_count(record, requests, replies)
        decision_count += steps

        if request_stat is not None:
            request_count += request_stat[0]
            request_observations += request_stat[1]
        elif requests:
            request_count += float(sum(requests))
            request_observations += len(requests)
        else:
            count = _episode_scalar(record, ("request_count", "requests"))
            if count is not None:
                request_count += count
            elif mode == "no_comm":
                request_count += 0.0
        if reply_stat is not None:
            reply_count += reply_stat[0]
            reply_observations += reply_stat[1]
        elif replies:
            reply_count += float(sum(replies))
            reply_observations += len(replies)
        else:
            count = _episode_scalar(record, ("reply_count", "replies"))
            if count is not None:
                reply_count += count
            elif mode == "no_comm":
                reply_count += 0.0

        bits_aliases = (
            "modeled_protocol_round_trip_bits",
            "actual_request_reply_bits",
            "actual_round_trip_bits",
            "communication_bits",
            "actual_bits",
        )
        bits_stat = _decision_aggregate_stat(record, bits_aliases)
        if bits_stat is None:
            bits.extend(_actual_round_trip_bits(record, requests, replies))
        else:
            compact_sums["bits"] += bits_stat[0]
            compact_counts["bits"] += bits_stat[1]
        collect_decision_metric(
            record,
            "delay",
            (
                "actual_communication_delay",
                "actual_round_trip_delay",
                "communication_delay",
                "round_trip_delay",
                "delay_steps",
                "delay",
            ),
            delays,
        )
        collect_decision_metric(
            record,
            "expected_latency_cost",
            (
                "expected_latency_cost",
                "expected_delay_cost",
                "vpi_expected_latency_cost",
            ),
            expected_latency_cost,
        )
        collect_decision_metric(
            record,
            "incurred_expected_latency_cost",
            ("incurred_expected_latency_cost",),
            incurred_expected_latency_cost,
        )
        collect_decision_metric(
            record,
            "vpi",
            ("VPI", "vpi", "value_of_information"),
            vpi,
        )
        collect_decision_metric(
            record, "code_surprise", ("code_surprise",), code_surprise
        )
        collect_decision_metric(
            record,
            "residual_surprise",
            ("residual_surprise",),
            residual_surprise,
        )
        collect_decision_metric(
            record, "plan_surprise", ("plan_surprise",), plan_surprise
        )
        before_start = len(g_before)
        after_start = len(g_after)
        collect_decision_metric(
            record, "G_before", ("G_before", "g_before", "G_no", "g_no"), g_before
        )
        collect_decision_metric(
            record, "G_after", ("G_after", "g_after", "G_reply", "g_reply"), g_after
        )
        collect_decision_metric(
            record, "G_improvement", ("G_improvement",), g_improvement
        )
        if _decision_aggregate_stat(record, ("G_improvement",)) is None:
            record_before = g_before[before_start:]
            record_after = g_after[after_start:]
            g_improvement.extend(
                record_before[index] - record_after[index]
                for index in range(min(len(record_before), len(record_after)))
            )
        collect_decision_metric(
            record, "replanned", ("replanned", "replan", "plan_changed"), replanned
        )
        collect_decision_metric(
            record,
            "action_change",
            ("action_change_l2", "action_change_magnitude", "action_delta_l2"),
            action_change,
        )
        collision_count_value = _episode_scalar(record, ("collision_count",))
        if collision_count_value is not None:
            collision_count.append(collision_count_value)
        force_violation_value = _episode_scalar(record, ("force_violation_rate",))
        if force_violation_value is not None:
            force_violation_rate.append(force_violation_value)
        max_force_value = _episode_scalar(record, ("max_force",))
        if max_force_value is not None:
            max_force.append(max_force_value)
        collision_step_value = _episode_scalar(record, ("collision_step_count",))
        if collision_step_value is not None:
            collision_steps.append(collision_step_value)
        contact_instances_value = _episode_scalar(
            record, ("collision_contact_instances",)
        )
        if contact_instances_value is not None:
            collision_contact_instances.append(contact_instances_value)
        reason = record.get("failure_reason")
        if reason is not None:
            failure_reason_counts[str(reason)] += 1

    # When only episode-level rates are present, use them rather than claiming
    # the event stream was observed.  Otherwise rates use the true decision
    # denominator, including zero-communication decisions.
    request_rate = _safe_ratio(request_count, decision_count)
    reply_rate = _safe_ratio(reply_count, decision_count)
    if request_observations == 0 and request_count == 0 and mode != "no_comm":
        request_rate = _mean_episode_alias(records, ("request_rate",))
    if reply_observations == 0 and reply_count == 0 and mode != "no_comm":
        reply_rate = _mean_episode_alias(records, ("reply_rate",))

    bits_total = float(sum(bits) + compact_sums["bits"])

    return {
        "mode": mode,
        "deployable": mode not in NON_DEPLOYABLE_MODES,
        "non_deployable_reason": (
            "Privileged true teammate plan; diagnostic upper bound only."
            if mode in NON_DEPLOYABLE_MODES
            else None
        ),
        "episodes": len(records),
        "decision_count": decision_count,
        "success_rate": _mean_or_none(success),
        "safety_rate": _mean_or_none(safety),
        "return_mean": _mean_or_none(returns),
        "collision_count_mean": _mean_or_none(collision_count),
        "collision_step_count_mean": _mean_or_none(collision_steps),
        "collision_contact_instances_mean": _mean_or_none(
            collision_contact_instances
        ),
        "collision_count_semantics": "rising-edge collision events per episode",
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "force_violation_rate_mean": _mean_or_none(force_violation_rate),
        "max_force_mean": _mean_or_none(max_force),
        "request_rate": request_rate,
        "reply_rate": reply_rate,
        "actual_request_reply_bits_total": bits_total,
        "actual_request_reply_bits_per_episode": float(bits_total / len(records)),
        "actual_request_reply_bits_per_decision": _safe_ratio(bits_total, decision_count),
        "modeled_protocol_bits_total": bits_total,
        "modeled_protocol_bits_per_episode": float(bits_total / len(records)),
        "modeled_protocol_bits_per_decision": _safe_ratio(bits_total, decision_count),
        "communication_bits_semantics": (
            "Modeled request/reply protocol budget; no serializer or wire-byte "
            "measurement. actual_request_reply_bits_* are compatibility aliases."
        ),
        "communication_delay_mean": _combined_mean(
            delays, compact_sums["delay"], compact_counts["delay"]
        ),
        "actual_communication_delay_mean": _combined_mean(
            delays, compact_sums["delay"], compact_counts["delay"]
        ),
        "expected_latency_cost_mean": _combined_mean(
            expected_latency_cost,
            compact_sums["expected_latency_cost"],
            compact_counts["expected_latency_cost"],
        ),
        "incurred_expected_latency_cost_mean": _combined_mean(
            incurred_expected_latency_cost,
            compact_sums["incurred_expected_latency_cost"],
            compact_counts["incurred_expected_latency_cost"],
        ),
        "vpi_mean": _combined_mean(vpi, compact_sums["vpi"], compact_counts["vpi"]),
        "code_surprise_mean": _combined_mean(
            code_surprise,
            compact_sums["code_surprise"],
            compact_counts["code_surprise"],
        ),
        "residual_surprise_mean": _combined_mean(
            residual_surprise,
            compact_sums["residual_surprise"],
            compact_counts["residual_surprise"],
        ),
        "plan_surprise_mean": _combined_mean(
            plan_surprise,
            compact_sums["plan_surprise"],
            compact_counts["plan_surprise"],
        ),
        "G_before_mean": _combined_mean(
            g_before, compact_sums["G_before"], compact_counts["G_before"]
        ),
        "G_after_mean": _combined_mean(
            g_after, compact_sums["G_after"], compact_counts["G_after"]
        ),
        "G_improvement_mean": _combined_mean(
            g_improvement,
            compact_sums["G_improvement"],
            compact_counts["G_improvement"],
        ),
        "replan_rate": _combined_mean(
            replanned, compact_sums["replanned"], compact_counts["replanned"]
        ),
        "action_change_l2_mean": _combined_mean(
            action_change,
            compact_sums["action_change"],
            compact_counts["action_change"],
        ),
        "diagnostic_sample_counts": {
            "bits": len(bits) + compact_counts["bits"],
            "delay": len(delays) + compact_counts["delay"],
            "expected_latency_cost": len(expected_latency_cost)
            + compact_counts["expected_latency_cost"],
            "incurred_expected_latency_cost": len(
                incurred_expected_latency_cost
            )
            + compact_counts["incurred_expected_latency_cost"],
            "vpi": len(vpi) + compact_counts["vpi"],
            "code_surprise": len(code_surprise) + compact_counts["code_surprise"],
            "residual_surprise": len(residual_surprise)
            + compact_counts["residual_surprise"],
            "plan_surprise": len(plan_surprise) + compact_counts["plan_surprise"],
            "G_before": len(g_before) + compact_counts["G_before"],
            "G_after": len(g_after) + compact_counts["G_after"],
            "replanned": len(replanned) + compact_counts["replanned"],
            "action_change": len(action_change) + compact_counts["action_change"],
        },
    }


def load_evaluation_records(path: str | Path) -> dict[str, list[Mapping[str, Any]]]:
    """Load a JSON comparison manifest.

    Accepted layouts are ``{mode: [records...]}``, ``{"modes": ...}``, or a
    flat list in which each record contains a ``mode`` field.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "modes" in payload:
        payload = payload["modes"]
    if isinstance(payload, Mapping):
        result: dict[str, list[Mapping[str, Any]]] = {}
        for mode, records in payload.items():
            # Some rollout writers place metadata alongside the mode mapping.
            # It is manifest context, not an additional communication mode.
            if str(mode) == "metadata":
                continue
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                raise EvaluationContractError(
                    f"records for {mode} must be a sequence"
                )
            result[str(mode)] = list(records)
        return result
    if isinstance(payload, list):
        result: dict[str, list[Mapping[str, Any]]] = {}
        for index, record in enumerate(payload):
            if not isinstance(record, Mapping) or "mode" not in record:
                raise EvaluationContractError(
                    f"flat record {index} must be a mapping with a mode field"
                )
            result.setdefault(str(record["mode"]), []).append(record)
        return result
    raise EvaluationContractError("evaluation JSON must contain a mapping or list")


def _index_episode_records(
    mode: str, records: Sequence[Mapping[str, Any]]
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    indexed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise EvaluationContractError(f"{mode} record {index} is not a mapping")
        if "seed" not in record:
            raise EvaluationContractError(f"{mode} record {index} is missing seed")
        seed = _hashable(record["seed"])
        episode_id = _first_direct(
            record,
            ("episode_id", "episode_record_id", "record_id", "input_id", "source_episode"),
        )
        key = (seed,) if episode_id is None else (seed, _hashable(episode_id))
        if key in indexed:
            raise EvaluationContractError(
                f"{mode} has duplicate paired episode key {key!r}; add a unique episode_id"
            )
        indexed[key] = record
    return indexed


def _input_digest(record: Mapping[str, Any]) -> str | None:
    value = _first_direct(
        record,
        ("input_digest", "episode_input_digest", "input_hash", "record_digest"),
    )
    return None if value is None else str(value)


def _scenario(record: Mapping[str, Any]) -> str | None:
    value = _first_direct(
        record,
        ("scenario", "scenario_name", "scenario_id", "scenario_type"),
    )
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (Mapping, Sequence)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _scenario_breakdown(
    indexed: Mapping[str, Mapping[tuple[Any, ...], Mapping[str, Any]]],
    *,
    required: Sequence[str],
    reference_keys: set[tuple[Any, ...]],
) -> dict[str, Any]:
    reference_mode = "no_comm" if "no_comm" in required else required[0]
    grouped_keys: dict[str, list[tuple[Any, ...]]] = {}
    for key in sorted(reference_keys, key=repr):
        label = _scenario(indexed[reference_mode][key]) or _UNSPECIFIED_SCENARIO
        grouped_keys.setdefault(label, []).append(key)

    result: dict[str, Any] = {}
    for label in sorted(grouped_keys):
        keys = grouped_keys[label]
        mode_summaries = {
            mode: aggregate_episode_records(
                [indexed[mode][key] for key in keys],
                mode=mode,
            )
            for mode in required
        }
        if "no_comm" in mode_summaries:
            for summary in mode_summaries.values():
                summary["vs_no_comm"] = _paired_summary_delta(
                    summary, mode_summaries["no_comm"]
                )
        if "always_reply" in mode_summaries:
            for summary in mode_summaries.values():
                summary["vs_always_reply"] = _paired_summary_delta(
                    summary, mode_summaries["always_reply"]
                )
        result[label] = {
            "episode_count": len(keys),
            "descriptive_only": True,
            "modes": mode_summaries,
        }
    return result


def _paired_episode_comparison(
    records: Mapping[tuple[Any, ...], Mapping[str, Any]],
    baseline_records: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    mode: str,
    baseline: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    keys = sorted(records, key=repr)
    metrics: dict[str, Any] = {}
    for metric in _PAIRED_EPISODE_METRICS:
        deltas: list[float] = []
        for key in keys:
            value = _episode_paired_metric(records[key], mode=mode, metric=metric)
            baseline_value = _episode_paired_metric(
                baseline_records[key], mode=baseline, metric=metric
            )
            if value is not None and baseline_value is not None:
                deltas.append(float(value) - float(baseline_value))
        metrics[metric] = _bootstrap_paired_delta(
            deltas,
            total_episode_count=len(keys),
            bootstrap_samples=bootstrap_samples,
            seed=_stable_bootstrap_seed(
                bootstrap_seed,
                mode=mode,
                baseline=baseline,
                metric=metric,
            ),
        )
    return {
        "mode": mode,
        "baseline": baseline,
        "episode_count": len(keys),
        "metrics": metrics,
    }


def _bootstrap_paired_delta(
    deltas: Sequence[float],
    *,
    total_episode_count: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    paired_count = len(deltas)
    if paired_count == 0:
        return {
            "mean_delta": None,
            "ci95": None,
            "paired_episode_count": 0,
            "dropped_episode_count": total_episode_count,
            "ci_excludes_zero": None,
        }

    array = np.asarray(deltas, dtype=np.float64)
    mean_delta = float(np.mean(array))
    if paired_count == 1:
        low = high = mean_delta
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0,
            paired_count,
            size=(bootstrap_samples, paired_count),
        )
        sample_means = np.mean(array[indices], axis=1)
        low, high = (
            float(value)
            for value in np.quantile(sample_means, (0.025, 0.975))
        )
    return {
        "mean_delta": mean_delta,
        "ci95": [low, high],
        "paired_episode_count": paired_count,
        "dropped_episode_count": total_episode_count - paired_count,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def _stable_bootstrap_seed(
    seed: int,
    *,
    mode: str,
    baseline: str,
    metric: str,
) -> int:
    material = f"{seed}\0{mode}\0{baseline}\0{metric}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _episode_paired_metric(
    record: Mapping[str, Any],
    *,
    mode: str,
    metric: str,
) -> float | None:
    def aggregate_mean(aliases: Sequence[str]) -> float | None:
        stat = _decision_aggregate_stat(record, aliases)
        return None if stat is None else stat[2]

    if metric == "success_rate":
        return _episode_flag(record, ("success", "task_success"), require_all=False)
    if metric == "safety_rate":
        return _episode_safety(record)
    if metric == "return_mean":
        return _episode_scalar(record, ("return", "episode_return", "reward_sum"))
    if metric == "collision_count_mean":
        return _episode_scalar(record, ("collision_count",))
    if metric == "force_violation_rate_mean":
        return _mean_metric(record, ("force_violation_rate",))
    if metric == "max_force_mean":
        values = _metric_values(record, ("max_force",))
        return None if not values else float(max(values))
    if metric == "request_rate":
        compact = aggregate_mean(
            ("request", "requested", "request_sent", "request_trigger", "trigger")
        )
        if compact is not None:
            return compact
        return _episode_communication_rate(record, mode=mode, event="request")
    if metric == "reply_rate":
        compact = aggregate_mean(
            ("reply", "replied", "reply_received", "response_received")
        )
        if compact is not None:
            return compact
        return _episode_communication_rate(record, mode=mode, event="reply")
    if metric == "actual_request_reply_bits_per_episode":
        compact = _decision_aggregate_stat(
            record,
            (
                "modeled_protocol_round_trip_bits",
                "actual_request_reply_bits",
                "actual_round_trip_bits",
                "communication_bits",
                "actual_bits",
            ),
        )
        if compact is not None:
            return compact[0]
        requests = _metric_values(
            record,
            ("request", "requested", "request_sent", "request_trigger", "trigger"),
        )
        replies = _metric_values(
            record,
            ("reply", "replied", "reply_received", "response_received"),
        )
        values = _actual_round_trip_bits(record, requests, replies)
        if values:
            return float(sum(values))
        return 0.0 if mode == "no_comm" else None
    if metric == "communication_delay_mean":
        compact = aggregate_mean(
            (
                "actual_communication_delay",
                "actual_round_trip_delay",
                "communication_delay",
                "round_trip_delay",
                "delay_steps",
                "delay",
            )
        )
        if compact is not None:
            return compact
        return _mean_metric(
            record,
            (
                "actual_communication_delay",
                "actual_round_trip_delay",
                "communication_delay",
                "round_trip_delay",
                "delay_steps",
                "delay",
            ),
        )
    if metric == "expected_latency_cost_mean":
        compact = aggregate_mean(
            (
                "expected_latency_cost",
                "expected_delay_cost",
                "vpi_expected_latency_cost",
            )
        )
        if compact is not None:
            return compact
        return _mean_metric(
            record,
            (
                "expected_latency_cost",
                "expected_delay_cost",
                "vpi_expected_latency_cost",
            ),
        )
    if metric == "incurred_expected_latency_cost_mean":
        compact = aggregate_mean(("incurred_expected_latency_cost",))
        if compact is not None:
            return compact
        return _mean_metric(record, ("incurred_expected_latency_cost",))
    if metric == "vpi_mean":
        compact = aggregate_mean(("VPI", "vpi", "value_of_information"))
        if compact is not None:
            return compact
        return _mean_metric(record, ("VPI", "vpi", "value_of_information"))
    if metric == "code_surprise_mean":
        compact = aggregate_mean(("code_surprise",))
        if compact is not None:
            return compact
        return _mean_metric(record, ("code_surprise",))
    if metric == "residual_surprise_mean":
        compact = aggregate_mean(("residual_surprise",))
        if compact is not None:
            return compact
        return _mean_metric(record, ("residual_surprise",))
    if metric == "plan_surprise_mean":
        compact = aggregate_mean(("plan_surprise",))
        if compact is not None:
            return compact
        return _mean_metric(record, ("plan_surprise",))
    if metric == "G_before_mean":
        compact = aggregate_mean(("G_before", "g_before", "G_no", "g_no"))
        if compact is not None:
            return compact
        return _mean_metric(record, ("G_before", "g_before", "G_no", "g_no"))
    if metric == "G_after_mean":
        compact = aggregate_mean(("G_after", "g_after", "G_reply", "g_reply"))
        if compact is not None:
            return compact
        return _mean_metric(record, ("G_after", "g_after", "G_reply", "g_reply"))
    if metric == "G_improvement_mean":
        compact = aggregate_mean(("G_improvement",))
        if compact is not None:
            return compact
        before = _metric_values(record, ("G_before", "g_before", "G_no", "g_no"))
        after = _metric_values(record, ("G_after", "g_after", "G_reply", "g_reply"))
        paired_count = min(len(before), len(after))
        if paired_count == 0:
            return None
        return float(
            np.mean(
                [before[index] - after[index] for index in range(paired_count)]
            )
        )
    if metric == "replan_rate":
        compact = aggregate_mean(("replanned", "replan", "plan_changed"))
        if compact is not None:
            return compact
        return _mean_metric(record, ("replanned", "replan", "plan_changed"))
    if metric == "action_change_l2_mean":
        compact = aggregate_mean(
            ("action_change_l2", "action_change_magnitude", "action_delta_l2")
        )
        if compact is not None:
            return compact
        return _mean_metric(
            record,
            ("action_change_l2", "action_change_magnitude", "action_delta_l2"),
        )
    raise AssertionError(f"unsupported paired metric {metric!r}")


def _episode_communication_rate(
    record: Mapping[str, Any],
    *,
    mode: str,
    event: str,
) -> float | None:
    if event == "request":
        aliases = ("request", "requested", "request_sent", "request_trigger", "trigger")
        count_aliases = ("request_count", "requests")
        rate_aliases = ("request_rate",)
    elif event == "reply":
        aliases = ("reply", "replied", "reply_received", "response_received")
        count_aliases = ("reply_count", "replies")
        rate_aliases = ("reply_rate",)
    else:
        raise AssertionError(f"unsupported communication event {event!r}")

    values = _metric_values(record, aliases)
    if values:
        return _safe_ratio(sum(values), _decision_count(record, values, values))
    count = _episode_scalar(record, count_aliases)
    if count is not None:
        return _safe_ratio(count, _decision_count(record, (), ()))
    rate = _mean_metric(record, rate_aliases)
    if rate is not None:
        return rate
    return 0.0 if mode == "no_comm" else None


def _mean_metric(record: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    values = _metric_values(record, aliases)
    return None if not values else float(np.mean(values))


def _episode_safety(record: Mapping[str, Any]) -> float | None:
    safe = _episode_flag(
        record,
        ("safe", "safety", "safety_success"),
        require_all=True,
    )
    if safe is not None:
        return safe
    failures = _metric_values(
        record,
        ("safety_failure", "safety_failed", "unsafe"),
    )
    if not failures:
        return None
    return float(not any(value > 0.5 for value in failures))


def _selective_vpi_acceptance(
    summaries: Mapping[str, Mapping[str, Any]],
    paired_comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = ("no_comm", "always_reply", "selective_vpi")
    missing = [mode for mode in required if mode not in summaries]
    thresholds = {
        "max_success_distance_from_always_reply": 0.05,
        "max_safety_distance_from_always_reply": 0.05,
        "minimum_bits_reduction_vs_always_reply": 0.50,
        "minimum_success_delta_vs_no_comm": 0.0,
    }
    claim_policy = (
        "Point-estimate gates are not evidence of statistical superiority. "
        "Claim an advantage only when the paired 95% bootstrap CI excludes zero "
        "in the favorable direction."
    )
    if missing:
        return {
            "status": "unavailable",
            "passed": None,
            "missing_modes": missing,
            "thresholds": thresholds,
            "checks": {},
            "statistical_claim_policy": claim_policy,
        }

    no_comm = summaries["no_comm"]
    always = summaries["always_reply"]
    selective = summaries["selective_vpi"]
    success_distance = _absolute_difference(
        selective.get("success_rate"), always.get("success_rate")
    )
    safety_distance = _absolute_difference(
        selective.get("safety_rate"), always.get("safety_rate")
    )
    success_delta_no_comm = _difference(
        selective.get("success_rate"), no_comm.get("success_rate")
    )
    selective_bits = (
        selective.get("modeled_protocol_bits_per_episode")
        if selective.get("diagnostic_sample_counts", {}).get("bits", 0) > 0
        else None
    )
    always_bits = (
        always.get("modeled_protocol_bits_per_episode")
        if always.get("diagnostic_sample_counts", {}).get("bits", 0) > 0
        else None
    )
    bits_reduction = None
    if selective_bits is not None and always_bits is not None and float(always_bits) > 0.0:
        bits_reduction = 1.0 - float(selective_bits) / float(always_bits)

    checks = {
        "success_within_5pp_of_always_reply": {
            "observed_absolute_distance": success_distance,
            "threshold": thresholds["max_success_distance_from_always_reply"],
            "passed": _at_most(
                success_distance,
                thresholds["max_success_distance_from_always_reply"],
            ),
        },
        "safety_within_5pp_of_always_reply": {
            "observed_absolute_distance": safety_distance,
            "threshold": thresholds["max_safety_distance_from_always_reply"],
            "passed": _at_most(
                safety_distance,
                thresholds["max_safety_distance_from_always_reply"],
            ),
        },
        "bits_reduced_at_least_50pct_vs_always_reply": {
            "observed_reduction_fraction": bits_reduction,
            "threshold": thresholds["minimum_bits_reduction_vs_always_reply"],
            "passed": _at_least(
                bits_reduction,
                thresholds["minimum_bits_reduction_vs_always_reply"],
            ),
        },
        "success_not_below_no_comm": {
            "observed_delta": success_delta_no_comm,
            "threshold": thresholds["minimum_success_delta_vs_no_comm"],
            "passed": _at_least(
                success_delta_no_comm,
                thresholds["minimum_success_delta_vs_no_comm"],
            ),
        },
    }
    check_results = [item["passed"] for item in checks.values()]
    passed = None if any(value is None for value in check_results) else all(check_results)

    success_stats = (
        paired_comparisons.get("selective_vpi", {})
        .get("vs_no_comm", {})
        .get("metrics", {})
        .get("success_rate", {})
    )
    success_ci = success_stats.get("ci95")
    ci_supports_advantage = (
        None if success_ci is None else bool(float(success_ci[0]) > 0.0)
    )
    return {
        "status": "insufficient_data" if passed is None else ("passed" if passed else "failed"),
        "passed": passed,
        "missing_modes": [],
        "thresholds": thresholds,
        "checks": checks,
        "decision_basis": "predefined paired-run point-estimate gates",
        "statistical_claim_policy": claim_policy,
        "success_advantage_over_no_comm": {
            "mean_delta": success_stats.get("mean_delta"),
            "ci95": success_ci,
            "supported_by_paired_ci": ci_supports_advantage,
        },
    }


def _difference(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _absolute_difference(value: Any, baseline: Any) -> float | None:
    difference = _difference(value, baseline)
    return None if difference is None else abs(difference)


def _at_most(value: float | None, threshold: float) -> bool | None:
    return None if value is None else bool(value <= threshold + 1e-12)


def _at_least(value: float | None, threshold: float) -> bool | None:
    return None if value is None else bool(value + 1e-12 >= threshold)


def _actual_round_trip_bits(
    record: Mapping[str, Any],
    requests: Sequence[float],
    replies: Sequence[float],
) -> list[float]:
    direct = _metric_values(
        record,
        (
            "modeled_protocol_round_trip_bits",
            "actual_request_reply_bits",
            "actual_round_trip_bits",
            "communication_bits",
            "actual_bits",
        ),
    )
    if direct:
        return direct

    request_bits = _metric_values(record, ("actual_request_bits", "request_bits"))
    reply_bits = _metric_values(record, ("actual_reply_bits", "reply_bits"))
    if not request_bits and not reply_bits:
        return []
    length = max(len(request_bits), len(reply_bits), len(requests), len(replies), 1)
    request_bits = _broadcast_values(request_bits, length, "request_bits")
    reply_bits = _broadcast_values(reply_bits, length, "reply_bits")
    request_mask = _broadcast_values(requests, length, "request") if requests else [1.0] * length
    reply_mask = _broadcast_values(replies, length, "reply") if replies else request_mask
    return [
        request_bits[index] * request_mask[index]
        + reply_bits[index] * reply_mask[index]
        for index in range(length)
    ]


def _decision_count(
    record: Mapping[str, Any], requests: Sequence[float], replies: Sequence[float]
) -> int:
    explicit = _episode_scalar(record, ("decision_count", "num_decisions", "episode_steps"))
    if explicit is not None and explicit >= 0 and float(explicit).is_integer():
        return int(explicit)
    step_records = record.get("steps")
    if isinstance(step_records, Sequence) and not isinstance(step_records, (str, bytes)):
        return len(step_records)
    return max(len(requests), len(replies), 1)


def _metric_values(record: Mapping[str, Any], aliases: Sequence[str]) -> list[float]:
    containers: list[Mapping[str, Any]] = [record]
    for key in ("metrics", "communication", "diagnostics"):
        value = record.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        value = _first_direct(container, aliases)
        if value is not None:
            return _numeric_values(value)

    step_records = record.get("steps")
    if isinstance(step_records, Sequence) and not isinstance(step_records, (str, bytes)):
        values: list[float] = []
        for step in step_records:
            if isinstance(step, Mapping):
                value = _first_direct(step, aliases)
                if value is not None:
                    values.extend(_numeric_values(value))
        return values
    return []


def _decision_aggregate_stat(
    record: Mapping[str, Any], aliases: Sequence[str]
) -> tuple[float, int, float] | None:
    """Read one bounded-size rollout decision aggregate, when present."""

    aggregates = record.get("decision_aggregates")
    if not isinstance(aggregates, Mapping):
        return None
    raw = _first_direct(aggregates, aliases)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise EvaluationContractError("decision aggregate must be a mapping")
    try:
        total = float(raw["sum"])
        count_value = raw["count"]
        count = int(count_value)
        mean = float(raw["mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationContractError(
            "decision aggregate requires numeric sum/count/mean"
        ) from exc
    if count <= 0 or count != float(count_value):
        raise EvaluationContractError("decision aggregate count must be a positive integer")
    if not all(np.isfinite(value) for value in (total, mean)):
        raise EvaluationContractError("decision aggregate contains non-finite values")
    if not np.isclose(total, mean * count, rtol=1e-6, atol=1e-8):
        raise EvaluationContractError("decision aggregate sum/count/mean are inconsistent")
    return total, count, mean


def _combined_mean(values: Sequence[float], aggregate_sum: float, aggregate_count: int) -> float | None:
    count = len(values) + aggregate_count
    return None if count <= 0 else float((sum(values) + aggregate_sum) / count)


def _episode_flag(
    record: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    require_all: bool,
) -> float | None:
    values = _metric_values(record, aliases)
    if not values:
        return None
    flags = [value > 0.5 for value in values]
    return float(all(flags) if require_all else any(flags))


def _episode_scalar(record: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    values = _metric_values(record, aliases)
    if not values:
        return None
    return float(values[-1]) if len(values) == 1 else float(sum(values))


def _mean_episode_alias(
    records: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> float | None:
    values: list[float] = []
    for record in records:
        found = _metric_values(record, aliases)
        if found:
            values.append(float(np.mean(found)))
    return _mean_or_none(values)


def _first_direct(mapping: Mapping[str, Any], aliases: Sequence[str]) -> Any | None:
    for key in aliases:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise EvaluationContractError(f"metric value must be numeric, got {type(value).__name__}")
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(f"metric value is not numeric: {value!r}") from exc
    if not np.all(np.isfinite(array)):
        raise EvaluationContractError("metric values must be finite")
    return [float(item) for item in array]


def _broadcast_values(values: Sequence[float], length: int, label: str) -> list[float]:
    if not values:
        return [0.0] * length
    if len(values) == length:
        return list(values)
    if len(values) == 1:
        return [float(values[0])] * length
    raise EvaluationContractError(
        f"{label} has {len(values)} values and cannot broadcast to {length} decisions"
    )


def _hashable(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _safe_ratio(numerator: float, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _paired_summary_delta(
    summary: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in (
        "success_rate",
        "safety_rate",
        "return_mean",
        "collision_count_mean",
        "force_violation_rate_mean",
        "max_force_mean",
        "request_rate",
        "reply_rate",
        "actual_request_reply_bits_per_episode",
        "modeled_protocol_bits_per_episode",
        "communication_delay_mean",
        "actual_communication_delay_mean",
        "expected_latency_cost_mean",
        "incurred_expected_latency_cost_mean",
        "vpi_mean",
        "G_after_mean",
        "G_improvement_mean",
        "replan_rate",
        "action_change_l2_mean",
    ):
        value, baseline = summary.get(key), reference.get(key)
        result[f"{key}_delta"] = (
            None if value is None or baseline is None else float(value) - float(baseline)
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired FE-PC-WAM  communication-mode episode records"
    )
    parser.add_argument("--records", required=True, help="JSON record manifest")
    parser.add_argument("--output", required=True, help="output summary JSON")
    parser.add_argument(
        "--modes",
        default=",".join(DEPLOYABLE_COMMUNICATION_MODES),
        help="comma-separated required modes (default: five deployable  modes)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"paired bootstrap replicates (default: {DEFAULT_BOOTSTRAP_SAMPLES})",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"fixed paired bootstrap seed (default: {DEFAULT_BOOTSTRAP_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    report = compare_communication_modes(
        load_evaluation_records(args.records),
        required_modes=modes,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
