"""Checkpoints for the decentralized FE-PC-WAM pipeline.

The legacy project checkpoints predate the deployable-input firewall.  Loading
one of them as a component would be a silent information-contract bug, so
all readers in this module require an explicit contract tag and schema version.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data.schema import SCHEMA_VERSION


CONTRACT_TAG = "fe_pc_wam/decentralized_local_belief"
CHECKPOINT_FORMAT_VERSION = 2
STAGES = ("plan", "belief", "wam", "intention", "wam_robust")


class IncompatibleCheckpoint(ValueError):
    """Raised when a legacy or differently-scoped checkpoint is supplied."""


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dataclass or mapping")


def upstream_reference(path: str | Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(path).resolve()
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "stage": str(checkpoint["stage"]),
        "checkpoint_format_version": int(checkpoint["checkpoint_format_version"]),
        "contract_tag": str(checkpoint["contract_tag"]),
        "schema_version": str(checkpoint["schema_version"]),
    }


def make_checkpoint(
    *,
    stage: str,
    model_class: str,
    model_config: Any,
    model_state_dict: Mapping[str, torch.Tensor],
    training_config: Any,
    dataset_metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    normalization: Mapping[str, Any] | None = None,
    plan_code_support: Mapping[str, Any] | None = None,
    upstream: Mapping[str, Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    checkpoint: dict[str, Any] = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "contract_tag": CONTRACT_TAG,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": str(model_class),
        "model_config": config_to_dict(model_config),
        "model_state_dict": dict(model_state_dict),
        "training_config": config_to_dict(training_config),
        "dataset": dict(dataset_metadata),
        "metrics": dict(metrics),
        "normalization": dict(normalization or {}),
        "plan_code_support": dict(plan_code_support) if plan_code_support is not None else None,
        "upstream": {name: dict(value) for name, value in (upstream or {}).items()},
    }
    if extra:
        checkpoint["extra"] = dict(extra)
    return checkpoint


def save_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> Path:
    """Validate and atomically write a checkpoint."""

    validate_checkpoint(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    expected_stage: str | Sequence[str] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise IncompatibleCheckpoint(f"{source} does not contain a checkpoint mapping")
    result = dict(checkpoint)
    validate_checkpoint(result, source=source, expected_stage=expected_stage)
    return result


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source: str | Path = "checkpoint",
    expected_stage: str | Sequence[str] | None = None,
) -> None:
    label = str(source)
    contract = checkpoint.get("contract_tag")
    if contract != CONTRACT_TAG:
        raise IncompatibleCheckpoint(
            f"{label} is not a deployable checkpoint: expected "
            f"contract_tag={CONTRACT_TAG!r}, got {contract!r}. "
            "Incompatible checkpoints must be retrained with the current pipeline."
        )
    version = checkpoint.get("checkpoint_format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise IncompatibleCheckpoint(
            f"{label} checkpoint_format_version={version!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    schema = checkpoint.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise IncompatibleCheckpoint(
            f"{label} schema_version={schema!r}; expected {SCHEMA_VERSION!r}"
        )
    stage = checkpoint.get("stage")
    if stage not in STAGES:
        raise IncompatibleCheckpoint(f"{label} has unknown stage {stage!r}")
    if expected_stage is not None:
        allowed = {expected_stage} if isinstance(expected_stage, str) else set(expected_stage)
        if stage not in allowed:
            raise IncompatibleCheckpoint(
                f"{label} is stage {stage!r}; expected one of {sorted(allowed)!r}"
            )
    for required in ("model_class", "model_config", "model_state_dict", "training_config", "dataset"):
        if required not in checkpoint:
            raise IncompatibleCheckpoint(f"{label} is missing required field {required!r}")


def require_plan_code_support(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    support = checkpoint.get("plan_code_support")
    if not isinstance(support, Mapping):
        raise IncompatibleCheckpoint(
            "checkpoint has no empirical plan_code_support; hard-coded plan IDs are not allowed"
        )
    return support
