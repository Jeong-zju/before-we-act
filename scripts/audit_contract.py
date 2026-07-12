"""Strict data/checkpoint/input-firewall audit.

The command exits non-zero on legacy checkpoints, missing empirical plan-code
support, schema drift, deployable teammate/world truth, or sampled candidate
codes outside the measured support.  A JSON report is written on both success
and failure so CI and experiment launchers can retain the evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.schema import (  # noqa: E402
    DEPLOYABLE_POLICY,
    LEGACY_CONTACT_SEMANTICS,
    LEGACY_FORCE_SEMANTICS,
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
    TRANSITION_SEMANTICS,
    spec_from_hdf5,
)
from models.plan_tokenizer import PlanCodeSupport  # noqa: E402
from train.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    IncompatibleCheckpoint,
    CONTRACT_TAG,
    load_checkpoint,
    require_plan_code_support,
)


FORBIDDEN_DEPLOYABLE_TERMS = (
    "teammate",
    "other_agent",
    "global_state",
    "world_pose",
    "pose_world",
    "privileged",
    "ground_truth",
)
EXPECTED_SLOT_ROLES = (
    "self",
    "object-belief",
    "teammate-belief",
    "task-context",
)
EXPECTED_DEPLOYABLE_INPUT_KEYS = frozenset(
    {
        "ego_id",
        "local_history",
        "history_mask",
        "object_observation_history",
        "object_valid_history",
        "object_confidence_history",
        "object_age_history",
    }
)


class ContractAuditError(ValueError):
    """Raised when an artifact violates the  information contract."""


def audit_hdf5_file(path: str | Path) -> dict[str, Any]:
    """Validate one  HDF5 episode and its deployable-input firewall."""

    source = Path(path)
    if not source.is_file():
        raise ContractAuditError(f"HDF5 episode does not exist: {source}")
    try:
        file = h5py.File(source, "r")
    except OSError as exc:
        raise ContractAuditError(f"cannot open HDF5 episode {source}: {exc}") from exc

    with file:
        schema = _attr_text(file.attrs, "schema_version")
        if schema != SCHEMA_VERSION:
            raise ContractAuditError(
                f"{source}: legacy/incompatible schema_version={schema!r}; expected {SCHEMA_VERSION!r}"
            )
        if _attr_text(file.attrs, "transition_semantics") != TRANSITION_SEMANTICS:
            raise ContractAuditError(f"{source}: transition semantics are not  T+1/T")
        if _attr_text(file.attrs, "deployable_input_policy") != DEPLOYABLE_POLICY:
            raise ContractAuditError(f"{source}: deployable_input_policy does not match ")

        spec = spec_from_hdf5(file)
        num_agents = _positive_int_attr(file, "num_agents")
        num_observations = _positive_int_attr(file, "num_observations")
        num_transitions = _positive_int_attr(file, "num_transitions", allow_zero=True)
        if num_observations != num_transitions + 1:
            raise ContractAuditError(
                f"{source}: expected observations=transitions+1, got "
                f"{num_observations} and {num_transitions}"
            )

        for required_path in (
            "schema/local_observation",
            "schema/rgb",
            "observations",
            "transitions/actions",
            "privileged/observations",
            "privileged/transitions",
            "raw_sensors",
        ):
            if required_path not in file:
                raise ContractAuditError(f"{source}: missing required group /{required_path}")

        local_schema = file["schema/local_observation"]
        explicit_teammate = bool(
            local_schema.attrs.get("explicit_teammate_state_allowed", True)
        )
        if explicit_teammate:
            raise ContractAuditError(
                f"{source}: schema permits explicit teammate state in deployable input"
            )
        field_order = _json_attr(local_schema.attrs, "field_order_json")
        model_field_order = _json_attr(local_schema.attrs, "model_field_order_json")
        if field_order != list(spec.field_shapes()):
            raise ContractAuditError(f"{source}: local field order differs from  spec")
        if model_field_order != spec.model_field_names():
            raise ContractAuditError(f"{source}: model field order differs from  spec")
        if int(local_schema.attrs.get("flat_dim", -1)) != spec.flat_dim:
            raise ContractAuditError(f"{source}: flat_dim metadata is inconsistent")
        if int(local_schema.attrs.get("model_observation_dim", -1)) != spec.model_observation_dim:
            raise ContractAuditError(
                f"{source}: model_observation_dim metadata is inconsistent"
            )

        privileged = file["privileged"]
        if bool(privileged.attrs.get("policy_input_allowed", True)):
            raise ContractAuditError(
                f"{source}: /privileged is not explicitly forbidden as policy input"
            )

        expected_agent_names = {f"agent_{agent_id}" for agent_id in range(num_agents)}
        observed_agent_names = set(file["observations"].keys())
        action_agent_names = set(file["transitions/actions"].keys())
        if observed_agent_names != expected_agent_names:
            raise ContractAuditError(
                f"{source}: observation agents mismatch: {sorted(observed_agent_names)}"
            )
        if action_agent_names != expected_agent_names:
            raise ContractAuditError(
                f"{source}: action agents mismatch: {sorted(action_agent_names)}"
            )

        expected_fields = set(spec.field_shapes())
        contact_semantics = str(
            file.attrs.get("local_contact_semantics", LEGACY_CONTACT_SEMANTICS)
        )
        force_semantics = str(
            file.attrs.get("local_force_semantics", LEGACY_FORCE_SEMANTICS)
        )
        force_units = file.attrs.get("local_force_units")
        force_units = None if force_units is None else str(force_units)
        sensor_provenance = file.attrs.get("local_sensor_provenance")
        sensor_provenance = (
            None if sensor_provenance is None else str(sensor_provenance)
        )
        force_scale = file.attrs.get("local_force_scale_newtons")
        force_scale = None if force_scale is None else float(force_scale)
        strict_sensors = (
            contact_semantics == STRICT_LOCAL_CONTACT_SEMANTICS
            and force_semantics == STRICT_LOCAL_FORCE_SEMANTICS
        )
        if strict_sensors:
            if force_units != LOCAL_FORCE_UNITS:
                raise ContractAuditError(
                    f"{source}: strict local force has invalid units {force_units!r}"
                )
            if sensor_provenance != STRICT_LOCAL_SENSOR_PROVENANCE:
                raise ContractAuditError(
                    f"{source}: strict local sensors lack explicit provenance"
                )
            if force_scale is None or not np.isfinite(force_scale) or force_scale <= 0:
                raise ContractAuditError(
                    f"{source}: strict local force requires a finite positive scale"
                )
        action_dim: int | None = None
        deployable_dataset_count = 0
        for agent_name in sorted(expected_agent_names):
            deployable_path = f"observations/{agent_name}/deployable"
            if deployable_path not in file:
                raise ContractAuditError(f"{source}: missing /{deployable_path}")
            deployable = file[deployable_path]
            actual_fields = set(_dataset_paths(deployable))
            if actual_fields != expected_fields:
                missing = sorted(expected_fields - actual_fields)
                extra = sorted(actual_fields - expected_fields)
                raise ContractAuditError(
                    f"{source}: {agent_name} deployable fields mismatch; "
                    f"missing={missing}, extra={extra}"
                )
            for field_name, tail_shape in spec.field_shapes().items():
                lowered = field_name.lower()
                forbidden = [term for term in FORBIDDEN_DEPLOYABLE_TERMS if term in lowered]
                if forbidden:
                    raise ContractAuditError(
                        f"{source}: forbidden deployable field {field_name!r}: {forbidden}"
                    )
                dataset = deployable[field_name]
                expected_shape = (num_observations, *tail_shape)
                if dataset.shape != expected_shape:
                    raise ContractAuditError(
                        f"{source}: /{deployable_path}/{field_name} has shape "
                        f"{dataset.shape}, expected {expected_shape}"
                    )
                _require_finite_dataset(source, dataset)
                deployable_dataset_count += 1

            valid = np.asarray(deployable["estimates/object/valid"][:])
            confidence = np.asarray(deployable["estimates/object/confidence"][:])
            age = np.asarray(deployable["estimates/object/age"][:])
            contact = np.asarray(deployable["local/contact"][:])
            grasp = np.asarray(deployable["local/grasp"][:])
            local_force = np.asarray(deployable["local/force"][:])
            if not np.all(np.isin(valid, (0.0, 1.0))):
                raise ContractAuditError(f"{source}: object valid flag is not binary")
            if not np.all(np.isin(contact, (0.0, 1.0))):
                raise ContractAuditError(f"{source}: local contact flag is not binary")
            if not np.all(np.isin(grasp, (0.0, 1.0))):
                raise ContractAuditError(f"{source}: local grasp flag is not binary")
            if np.any((confidence < 0) | (confidence > 1)):
                raise ContractAuditError(f"{source}: object confidence lies outside [0,1]")
            if np.any(age < 0):
                raise ContractAuditError(f"{source}: object estimate age is negative")
            if strict_sensors and np.any((local_force < 0) | (local_force > 1)):
                raise ContractAuditError(f"{source}: strict local force lies outside [0,1]")

            action = file[f"transitions/actions/{agent_name}"]
            if (
                action.ndim != 2
                or action.shape[0] != num_transitions
                or action.shape[1] != 4
            ):
                raise ContractAuditError(
                    f"{source}: {agent_name} action shape {action.shape} violates [T,4]"
                )
            if action_dim is None:
                action_dim = int(action.shape[1])
            elif int(action.shape[1]) != action_dim:
                raise ContractAuditError(f"{source}: action dimension changes by agent")
            _require_finite_dataset(source, action)
            action_values = np.asarray(action[:])
            if np.any(
                np.linalg.norm(action_values[:, :2], axis=-1)
                > np.sqrt(2.0) + 1e-6
            ):
                raise ContractAuditError(
                    f"{source}: {agent_name} normalized planar action norm exceeds sqrt(2)"
                )
            if np.any(
                (action_values[:, 2:] < -1.0) | (action_values[:, 2:] > 1.0)
            ):
                raise ContractAuditError(
                    f"{source}: {agent_name} wz/grip action lies outside [-1,1]"
                )

        return {
            "path": str(source.resolve()),
            "schema_version": schema,
            "firewall_passed": True,
            "explicit_teammate_state_allowed": False,
            "num_agents": num_agents,
            "num_observations": num_observations,
            "num_transitions": num_transitions,
            "action_dim": action_dim,
            "deployable_fields": list(spec.field_shapes()),
            "deployable_dataset_count": deployable_dataset_count,
            "privileged_policy_input_allowed": False,
            "local_contact_semantics": contact_semantics,
            "strict_local_contact_semantics": contact_semantics
            == STRICT_LOCAL_CONTACT_SEMANTICS,
            "local_force_semantics": force_semantics,
            "strict_local_force_semantics": force_semantics
            == STRICT_LOCAL_FORCE_SEMANTICS,
            "local_force_units": force_units,
            "local_force_scale_newtons": force_scale,
            "local_sensor_provenance": sensor_provenance,
        }


def audit_plan_code_support(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate empirical support and recompute hard-code usage statistics."""

    required = {
        "format_version",
        "codebook_size",
        "min_count",
        "counts",
        "probabilities",
        "active_codes",
        "residual_mean",
        "residual_std",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ContractAuditError(
            f"plan_code_support is incomplete; missing={missing}. Hard-coded codes are forbidden."
        )
    raw_counts = torch.as_tensor(state["counts"])
    if raw_counts.ndim != 1 or not torch.isfinite(raw_counts.float()).all():
        raise ContractAuditError("plan support counts must be a finite rank-one tensor")
    if raw_counts.dtype.is_floating_point and not torch.equal(raw_counts, raw_counts.round()):
        raise ContractAuditError("plan support counts must be integers")
    if (raw_counts < 0).any():
        raise ContractAuditError("plan support counts cannot be negative")

    try:
        support = PlanCodeSupport.from_dict(state)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractAuditError(f"invalid empirical plan_code_support: {exc}") from exc
    if int(support.counts.sum().item()) <= 0:
        raise ContractAuditError("plan support contains no encoded action segments")

    stored_active = _integer_codes(state["active_codes"], label="active_codes")
    derived_active = [int(value) for value in support.active_codes.tolist()]
    if stored_active != derived_active:
        raise ContractAuditError(
            f"stored active_codes={stored_active} do not match counts/min_count={derived_active}"
        )

    total = float(support.counts.sum().item())
    empirical_probabilities = support.counts.float() / total
    if not torch.allclose(
        support.probabilities, empirical_probabilities, atol=1e-6, rtol=1e-5
    ):
        raise ContractAuditError(
            "plan support probabilities are not the empirical normalized code counts"
        )
    used_mask = support.counts > 0
    probabilities = empirical_probabilities[used_mask]
    entropy = float(-(probabilities * probabilities.log()).sum().item())
    used_codes = int(used_mask.sum().item())
    active_codes = len(derived_active)
    return {
        "format_version": int(state["format_version"]),
        "codebook_size": support.codebook_size,
        "min_count": support.min_count,
        "encoded_segments": int(total),
        "used_codes": used_codes,
        "dead_codes": support.codebook_size - used_codes,
        "active_codes": derived_active,
        "active_code_count": active_codes,
        "usage_ratio": used_codes / support.codebook_size,
        "active_ratio": active_codes / support.codebook_size,
        "entropy": entropy,
        "perplexity": math.exp(entropy),
        "empirical_probabilities": empirical_probabilities.tolist(),
        "residual_dim": support.residual_dim,
    }


def audit_checkpoint_file(path: str | Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Reject legacy checkpoints and validate their inherited plan support."""

    source = Path(path)
    try:
        checkpoint = load_checkpoint(source, map_location="cpu")
        support_state = require_plan_code_support(checkpoint)
    except (FileNotFoundError, IncompatibleCheckpoint) as exc:
        raise ContractAuditError(str(exc)) from exc

    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("schema_version") != SCHEMA_VERSION:
        raise ContractAuditError(
            f"{source}: checkpoint dataset metadata is not tagged {SCHEMA_VERSION}"
        )
    input_keys = dataset.get("deployable_input_keys")
    if not isinstance(input_keys, Sequence) or isinstance(input_keys, (str, bytes)):
        raise ContractAuditError(
            f"{source}: checkpoint is missing deployable_input_keys firewall metadata"
        )
    if set(str(value) for value in input_keys) != EXPECTED_DEPLOYABLE_INPUT_KEYS:
        raise ContractAuditError(
            f"{source}: checkpoint deployable_input_keys differ from the  firewall"
        )
    feature_names = dataset.get("input_feature_names")
    if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)):
        raise ContractAuditError(
            f"{source}: checkpoint is missing input_feature_names firewall metadata"
        )
    forbidden_features = [
        str(name)
        for name in feature_names
        if any(term in str(name).lower() for term in FORBIDDEN_DEPLOYABLE_TERMS)
    ]
    if forbidden_features:
        raise ContractAuditError(
            f"{source}: checkpoint input features contain forbidden state: {forbidden_features}"
        )
    support_report = audit_plan_code_support(support_state)
    stage = str(checkpoint["stage"])
    extra = checkpoint.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}

    if stage == "plan":
        if checkpoint.get("model_class") != "ActionOnlyPlanTokenizer":
            raise ContractAuditError(
                f"{source}: plan stage must use ActionOnlyPlanTokenizer"
            )
        if extra.get("encoder_input") != "ego_future_action_only":
            raise ContractAuditError(
                f"{source}: plan encoder is not explicitly action-only"
            )
        if extra.get("hardcoded_plan_codes_allowed") is not False:
            raise ContractAuditError(
                f"{source}: checkpoint does not explicitly forbid hard-coded plan IDs"
            )
        _validate_checkpoint_usage_metrics(source, checkpoint.get("metrics"), support_report)
    elif stage == "belief":
        if checkpoint.get("model_class") != "LocalBeliefSlotEncoder":
            raise ContractAuditError(f"{source}: belief stage has the wrong model class")
        if tuple(extra.get("slot_role_order", ())) != EXPECTED_SLOT_ROLES:
            raise ContractAuditError(f"{source}: belief slot roles are not the fixed  roles")
        if extra.get("privileged_values_are_forward_inputs") is not False:
            raise ContractAuditError(
                f"{source}: privileged values are not explicitly target-only"
            )
    elif stage == "wam":
        if checkpoint.get("model_class") != "EgoLocalWAM":
            raise ContractAuditError(f"{source}: WAM stage has the wrong model class")
        if extra.get("teammate_private_state_input") is not False:
            raise ContractAuditError(
                f"{source}: WAM does not explicitly exclude teammate-private state"
            )
    elif stage == "wam_robust":
        if checkpoint.get("model_class") != "EgoLocalWAM":
            raise ContractAuditError(f"{source}: robust WAM has the wrong model class")
        if extra.get("true_teammate_plan_used_as_input_for_non_oracle_rows") is not False:
            raise ContractAuditError(
                f"{source}: robust WAM may leak true teammate plans into non-oracle rows"
            )
    elif stage == "intention":
        if checkpoint.get("model_class") != "LocalIntentionPosterior":
            raise ContractAuditError(f"{source}: intention stage has the wrong model class")
        if extra.get("teammate_plan_is_target_only") is not True:
            raise ContractAuditError(
                f"{source}: intention checkpoint does not mark teammate plan target-only"
            )

    report = {
        "path": str(source.resolve()),
        "checkpoint_format_version": int(checkpoint["checkpoint_format_version"]),
        "contract_tag": checkpoint["contract_tag"],
        "schema_version": checkpoint["schema_version"],
        "stage": stage,
        "model_class": checkpoint["model_class"],
        "local_contact_semantics": str(
            dataset.get("local_contact_semantics", LEGACY_CONTACT_SEMANTICS)
        ),
        "strict_local_contact_semantics": str(
            dataset.get("local_contact_semantics", LEGACY_CONTACT_SEMANTICS)
        )
        == STRICT_LOCAL_CONTACT_SEMANTICS,
        "local_force_semantics": str(
            dataset.get("local_force_semantics", LEGACY_FORCE_SEMANTICS)
        ),
        "strict_local_force_semantics": str(
            dataset.get("local_force_semantics", LEGACY_FORCE_SEMANTICS)
        )
        == STRICT_LOCAL_FORCE_SEMANTICS,
        "local_force_units": dataset.get("local_force_units"),
        "local_force_scale_newtons": dataset.get("local_force_scale_newtons"),
        "local_sensor_provenance": dataset.get("local_sensor_provenance"),
        "plan_code_support": support_report,
    }
    return report, support_state


def audit_candidate_codes(
    candidate_codes: Any,
    support_state: Mapping[str, Any],
    *,
    source: str = "candidate_codes",
) -> dict[str, Any]:
    """Fail when any sampled/selected candidate is outside empirical support."""

    support_report = audit_plan_code_support(support_state)
    codes = _integer_codes(candidate_codes, label=source, require_sorted=False)
    if not codes:
        raise ContractAuditError(f"{source}: candidate code list is empty")
    active = set(support_report["active_codes"])
    outside = sorted(set(codes) - active)
    if outside:
        raise ContractAuditError(
            f"{source}: candidate codes {outside} are outside empirical active support "
            f"{sorted(active)}"
        )
    counts: dict[str, int] = {}
    for code in codes:
        counts[str(code)] = counts.get(str(code), 0) + 1
    return {
        "source": source,
        "candidate_count": len(codes),
        "unique_candidate_codes": sorted(set(codes)),
        "counts": counts,
        "all_candidates_within_empirical_support": True,
    }


def run_contract_audit(
    hdf5_paths: Sequence[str | Path],
    checkpoint_paths: Sequence[str | Path],
    *,
    candidate_codes: Any | None = None,
    candidate_source: str = "candidate_codes",
    require_candidate_codes: bool = False,
) -> dict[str, Any]:
    """Run the complete audit and return a JSON-serializable report."""

    if not hdf5_paths:
        raise ContractAuditError("at least one  HDF5 episode is required")
    if not checkpoint_paths:
        raise ContractAuditError("at least one checkpoint is required")
    data_reports = [audit_hdf5_file(path) for path in hdf5_paths]
    checkpoint_results = [audit_checkpoint_file(path) for path in checkpoint_paths]
    checkpoint_reports = [result[0] for result in checkpoint_results]
    supports = [result[1] for result in checkpoint_results]
    _require_identical_support(supports, checkpoint_paths)
    semantic_keys = (
        "local_contact_semantics",
        "local_force_semantics",
        "local_force_units",
        "local_force_scale_newtons",
        "local_sensor_provenance",
    )
    expected_semantics = {
        key: data_reports[0].get(key) for key in semantic_keys
    }
    for report in (*data_reports[1:], *checkpoint_reports):
        for key, expected in expected_semantics.items():
            actual = report.get(key)
            if key == "local_force_scale_newtons" and expected is not None:
                if actual is None or not np.isclose(float(actual), float(expected)):
                    raise ContractAuditError(
                        f"artifact local sensor contract differs for {key}: "
                        f"{expected!r} vs {actual!r}"
                    )
            elif actual != expected:
                raise ContractAuditError(
                    f"artifact local sensor contract differs for {key}: "
                    f"{expected!r} vs {actual!r}"
                )

    if candidate_codes is None:
        if require_candidate_codes:
            raise ContractAuditError(
                "candidate-code evidence is required; pass --candidate-codes"
            )
        candidate_report: dict[str, Any] = {
            "status": "not_provided",
            "all_candidates_within_empirical_support": None,
        }
    else:
        candidate_report = audit_candidate_codes(
            candidate_codes, supports[0], source=candidate_source
        )

    return {
        "passed": True,
        "contract": CONTRACT_TAG,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "input_firewall": "deployable observations contain ego-local fields only",
        "episodes": data_reports,
        "checkpoints": checkpoint_reports,
        "candidate_codes": candidate_report,
    }


def load_candidate_codes(paths: Sequence[str | Path]) -> list[int]:
    codes: list[int] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ContractAuditError(f"candidate-code file does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".jsonl":
            payload = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif suffix == ".npy":
            payload = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            key = "candidate_codes" if "candidate_codes" in archive else "codes"
            if key not in archive:
                raise ContractAuditError(
                    f"{path}: npz must contain candidate_codes or codes"
                )
            payload = archive[key]
        elif suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise ContractAuditError(
                f"unsupported candidate-code file extension for {path}"
            )
        codes.extend(_extract_candidate_codes(payload, source=str(path)))
    return codes


def _extract_candidate_codes(payload: Any, *, source: str) -> list[int]:
    if isinstance(payload, Mapping):
        for key in ("candidate_codes", "plan_codes", "code_indices", "codes"):
            if key in payload:
                return _integer_codes(payload[key], label=f"{source}:{key}", require_sorted=False)
        # JSONL frequently stores one decision per row.  Recursively collect
        # matching fields rather than accepting arbitrary numeric metadata.
        collected: list[int] = []
        for value in payload.values():
            if isinstance(value, Mapping):
                try:
                    collected.extend(_extract_candidate_codes(value, source=source))
                except ContractAuditError:
                    pass
        if collected:
            return collected
        raise ContractAuditError(f"{source}: no candidate-code field found")
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if payload and all(isinstance(value, Mapping) for value in payload):
            collected: list[int] = []
            for index, value in enumerate(payload):
                collected.extend(_extract_candidate_codes(value, source=f"{source}[{index}]"))
            return collected
    return _integer_codes(payload, label=source, require_sorted=False)


def _validate_checkpoint_usage_metrics(
    source: Path, metrics: Any, support_report: Mapping[str, Any]
) -> None:
    if not isinstance(metrics, Mapping):
        raise ContractAuditError(f"{source}: plan checkpoint has no metrics mapping")
    for key in ("used_codes", "usage_ratio", "entropy", "perplexity"):
        if key not in metrics:
            raise ContractAuditError(f"{source}: plan metrics missing {key!r}")
        try:
            value = float(metrics[key])
        except (TypeError, ValueError) as exc:
            raise ContractAuditError(f"{source}: plan metric {key} is not numeric") from exc
        expected = float(support_report[key])
        if not math.isclose(value, expected, rel_tol=1e-5, abs_tol=1e-6):
            raise ContractAuditError(
                f"{source}: plan metric {key}={value} disagrees with empirical support {expected}"
            )


def _require_identical_support(
    supports: Sequence[Mapping[str, Any]], paths: Sequence[str | Path]
) -> None:
    if not supports:
        return
    reference = audit_plan_code_support(supports[0])
    reference_counts = torch.as_tensor(supports[0]["counts"], dtype=torch.long)
    reference_probabilities = torch.as_tensor(
        supports[0]["probabilities"], dtype=torch.float32
    )
    reference_mean = torch.as_tensor(supports[0]["residual_mean"], dtype=torch.float32)
    reference_std = torch.as_tensor(supports[0]["residual_std"], dtype=torch.float32)
    for index, support in enumerate(supports[1:], start=1):
        report = audit_plan_code_support(support)
        counts = torch.as_tensor(support["counts"], dtype=torch.long)
        probabilities = torch.as_tensor(support["probabilities"], dtype=torch.float32)
        residual_mean = torch.as_tensor(support["residual_mean"], dtype=torch.float32)
        residual_std = torch.as_tensor(support["residual_std"], dtype=torch.float32)
        if (
            report["codebook_size"] != reference["codebook_size"]
            or report["min_count"] != reference["min_count"]
            or report["active_codes"] != reference["active_codes"]
            or not torch.equal(counts, reference_counts)
            or probabilities.shape != reference_probabilities.shape
            or not torch.allclose(probabilities, reference_probabilities)
            or residual_mean.shape != reference_mean.shape
            or not torch.allclose(residual_mean, reference_mean)
            or residual_std.shape != reference_std.shape
            or not torch.allclose(residual_std, reference_std)
        ):
            raise ContractAuditError(
                f"checkpoint {paths[index]} does not inherit the same empirical plan support"
            )


def _integer_codes(
    values: Any, *, label: str, require_sorted: bool = True
) -> list[int]:
    try:
        tensor = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ContractAuditError(f"{label} must be an integer array") from exc
    if tensor.numel() == 0:
        return []
    if not torch.isfinite(tensor.float()).all():
        raise ContractAuditError(f"{label} contains non-finite values")
    if tensor.dtype.is_floating_point and not torch.equal(tensor, tensor.round()):
        raise ContractAuditError(f"{label} contains non-integer codes")
    codes = [int(value) for value in tensor.reshape(-1).tolist()]
    if any(value < 0 for value in codes):
        raise ContractAuditError(f"{label} contains negative codes")
    if require_sorted and codes != sorted(set(codes)):
        raise ContractAuditError(f"{label} must be sorted and unique")
    return codes


def _dataset_paths(group: h5py.Group) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if isinstance(value, h5py.Dataset):
            paths.append(name)

    group.visititems(visitor)
    return paths


def _require_finite_dataset(source: Path, dataset: h5py.Dataset) -> None:
    if dataset.dtype.kind not in {"b", "i", "u", "f"}:
        raise ContractAuditError(
            f"{source}: deployable numeric dataset {dataset.name} has dtype {dataset.dtype}"
        )
    if dataset.dtype.kind == "f" and not np.all(np.isfinite(dataset[:])):
        raise ContractAuditError(f"{source}: dataset {dataset.name} contains non-finite values")


def _attr_text(attrs: h5py.AttributeManager, name: str) -> str:
    if name not in attrs:
        raise ContractAuditError(f"missing required HDF5 attribute {name!r}")
    value = attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_attr(attrs: h5py.AttributeManager, name: str) -> Any:
    try:
        return json.loads(_attr_text(attrs, name))
    except json.JSONDecodeError as exc:
        raise ContractAuditError(f"HDF5 attribute {name!r} is not valid JSON") from exc


def _positive_int_attr(
    file: h5py.File, name: str, *, allow_zero: bool = False
) -> int:
    if name not in file.attrs:
        raise ContractAuditError(f"missing required HDF5 attribute {name!r}")
    try:
        value = int(file.attrs[name])
    except (TypeError, ValueError) as exc:
        raise ContractAuditError(f"HDF5 attribute {name!r} is not an integer") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ContractAuditError(f"HDF5 attribute {name!r} must be >= {minimum}")
    return value


def _expand_hdf5_inputs(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("episode_*.hdf5")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            paths.extend(Path(match) for match in sorted(glob.glob(value)))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ContractAuditError("no HDF5 episodes matched --data")
    return unique


def _write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FE-PC-WAM  artifacts")
    parser.add_argument(
        "--data", nargs="+", required=True, help=" HDF5 files, directories, or globs"
    )
    parser.add_argument(
        "--checkpoint", nargs="+", required=True, help=" checkpoint paths"
    )
    parser.add_argument(
        "--candidate-codes",
        nargs="*",
        default=[],
        help="JSON/JSONL/NPY/NPZ/PT files containing actually generated candidate codes",
    )
    parser.add_argument(
        "--require-candidate-codes",
        action="store_true",
        help="fail if no candidate-code evidence is supplied",
    )
    parser.add_argument(
        "--output", default="outputs/contract_audit.json", help="JSON audit report"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        data_paths = _expand_hdf5_inputs(args.data)
        candidates = (
            load_candidate_codes(args.candidate_codes) if args.candidate_codes else None
        )
        report = run_contract_audit(
            data_paths,
            args.checkpoint,
            candidate_codes=candidates,
            candidate_source=",".join(args.candidate_codes) or "candidate_codes",
            require_candidate_codes=args.require_candidate_codes,
        )
    except Exception as exc:
        report = {
            "passed": False,
            "contract": CONTRACT_TAG,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from exc
    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
