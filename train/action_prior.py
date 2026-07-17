"""Train, evaluate, save, and load the frozen action-prior baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor
import yaml

from models.wam import (
    ActionPrior,
    ActionPriorConfig,
    RWMARWorldModel,
    WorldModelSequenceInputs,
)

CHECKPOINT_FORMAT_VERSION = "wam.action_prior/1"
ProgressCallback = Callable[[Mapping[str, float | int]], None]


@dataclass(frozen=True)
class ActionPriorTrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0
    max_steps: int = -1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid weight decay or gradient clip")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")


def train_action_prior(
    prior: ActionPrior,
    world_model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    config: ActionPriorTrainConfig,
    progress: ProgressCallback | None = None,
) -> tuple[list[float], int]:
    _freeze_world_model(world_model.to(device))
    prior.to(device).train()
    optimizer = torch.optim.AdamW(
        prior.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[float] = []
    completed_steps = 0
    for epoch in range(config.epochs):
        for raw_batch in loader:
            batch = _prepare_batch(raw_batch, device)
            with torch.no_grad():
                _, _, features = world_model.encode_planning_history(_history(batch))
            losses = prior.nll(features, batch["actions"])
            weights = batch["weights"]
            eligible = weights.sum()
            loss = (
                (losses * weights).sum() / eligible
                if float(eligible.detach()) > 0.0
                else prior(features).mean.sum() * 0.0
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite action-prior loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), config.gradient_clip_norm)
            optimizer.step()
            completed_steps += 1
            value = float(loss.detach().cpu())
            history.append(value)
            if progress is not None:
                progress(
                    {
                        "epoch": epoch + 1,
                        "epochs": config.epochs,
                        "step": completed_steps,
                        "loss": value,
                    }
                )
            if config.max_steps > 0 and completed_steps >= config.max_steps:
                return history, completed_steps
    return history, completed_steps


@torch.inference_mode()
def evaluate_action_prior(
    prior: ActionPrior,
    world_model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
) -> dict[str, float | int | None]:
    _freeze_world_model(world_model.to(device))
    prior.to(device).eval()
    samples = 0
    eligible_samples = 0
    squared_error = 0.0
    selected_squared_error = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = _prepare_batch(raw_batch, device)
        _, _, features = world_model.encode_planning_history(_history(batch))
        actions = prior.deterministic_action(features)
        error = (actions - batch["actions"]).square().mean(dim=-1)
        selected = batch["weights"] > 0.0
        samples += int(actions.shape[0])
        eligible_samples += int(selected.sum().cpu())
        squared_error += float(error.sum().cpu())
        if bool(selected.any()):
            selected_squared_error += float(error[selected].sum().cpu())
        if progress is not None:
            progress({"batch": batch_index + 1, "samples": samples})
        if max_batches > 0 and batch_index + 1 >= max_batches:
            break
    if samples == 0:
        raise RuntimeError("cannot evaluate action prior on an empty loader")
    return {
        "samples": samples,
        "eligible_samples": eligible_samples,
        "action_rmse": float(np.sqrt(squared_error / samples)),
        "selected_action_rmse": (
            float(np.sqrt(selected_squared_error / eligible_samples))
            if eligible_samples
            else None
        ),
    }


def world_model_checkpoint_fingerprint(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    schema = root / "schema.json"
    members = sorted((root / "members").glob("member_*.safetensors"))
    if not schema.is_file() or not members:
        raise FileNotFoundError("world-model ensemble checkpoint is incomplete")
    return {
        "schema_sha256": _sha256(schema),
        "member_sha256": {path.name: _sha256(path) for path in members},
    }


def world_model_member_fingerprint(
    directory: str | Path, member_index: int = 0
) -> dict[str, Any]:
    """Fingerprint one deployable world-model ensemble member without reading its siblings."""

    root = Path(directory)
    schema = root / "schema.json"
    member = root / "members" / f"member_{int(member_index):02d}.safetensors"
    if not schema.is_file() or not member.is_file():
        raise FileNotFoundError("world-model ensemble member checkpoint is incomplete")
    return {
        "member_index": int(member_index),
        "schema_sha256": _sha256(schema),
        "member_sha256": _sha256(member),
    }


def save_action_prior_checkpoint(
    directory: str | Path,
    prior: ActionPrior,
    *,
    world_model_checkpoint: str | Path,
    experiment_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
    normalization_sha256: str,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in prior.state_dict().items()
        },
        target / "action_prior.safetensors",
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "wam_action_prior",
        },
    )
    config = dict(experiment_config)
    config["action_prior_config"] = asdict(prior.config)
    (target / "config.yaml").write_text(
        yaml.safe_dump(_plain(config), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        target / "schema.json",
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "schema_version": schema_version,
            "runtime_inputs": ["states", "past_actions", "valid_mask"],
            "forbidden_runtime_inputs": [
                "privileged_state",
                "braking_agent",
                "braking_time",
            ],
            "normalization_sha256": normalization_sha256,
            "world_model_member_fingerprint": world_model_member_fingerprint(
                world_model_checkpoint, 0
            ),
        },
    )
    _write_json(target / "dataset_manifest.json", dataset_manifest)
    _write_json(target / "metrics.json", metrics)
    _write_json(target / "provenance.json", provenance)
    return target


def load_action_prior_checkpoint(
    directory: str | Path,
    *,
    world_model_checkpoint: str | Path,
    device: str | torch.device = "cpu",
    expected_schema_version: str | None = None,
    expected_normalization_sha256: str | None = None,
) -> tuple[ActionPrior, dict[str, Any]]:
    source = Path(directory)
    required = (
        "action_prior.safetensors",
        "config.yaml",
        "schema.json",
        "dataset_manifest.json",
        "metrics.json",
        "provenance.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"action-prior checkpoint is missing {missing}")
    schema = _read_json(source / "schema.json")
    if schema.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported action-prior checkpoint")
    if (
        expected_schema_version is not None
        and schema.get("schema_version") != expected_schema_version
    ):
        raise ValueError("action-prior data schema mismatch")
    if (
        expected_normalization_sha256 is not None
        and schema.get("normalization_sha256") != expected_normalization_sha256
    ):
        raise ValueError("action-prior normalization hash mismatch")
    _validate_world_model_binding(schema, world_model_checkpoint)
    payload = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "action_prior_config" not in payload:
        raise ValueError("checkpoint config has no action_prior_config")
    prior = ActionPrior(ActionPriorConfig(**dict(payload["action_prior_config"])))
    incompatible = prior.load_state_dict(
        load_file(source / "action_prior.safetensors", device=str(device)), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict action-prior load failed: {incompatible}")
    return prior.to(device).eval(), {
        "experiment_config": dict(payload),
        "schema": schema,
        "dataset_manifest": _read_json(source / "dataset_manifest.json"),
        "metrics": _read_json(source / "metrics.json"),
        "provenance": _read_json(source / "provenance.json"),
    }


def _freeze_world_model(model: RWMARWorldModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _validate_world_model_binding(
    schema: Mapping[str, Any], world_model_checkpoint: str | Path
) -> None:
    current = world_model_member_fingerprint(world_model_checkpoint, 0)
    expected = schema.get("world_model_member_fingerprint")
    if expected is None:
        expected = schema.get("phase2_member_fingerprint")
    if expected is not None:
        if expected != current:
            raise ValueError("world-model ensemble member 0 fingerprint does not match action prior")
        return
    legacy = schema.get("phase2_fingerprint")
    if not isinstance(legacy, Mapping):
        raise ValueError("action prior has no world-model ensemble member binding")
    member_hashes = legacy.get("member_sha256")
    if (
        legacy.get("schema_sha256") != current["schema_sha256"]
        or not isinstance(member_hashes, Mapping)
        or member_hashes.get("member_00.safetensors") != current["member_sha256"]
    ):
        raise ValueError("world-model ensemble member 0 fingerprint does not match action prior")


def _prepare_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    required = (
        "states",
        "past_actions",
        "valid_mask",
        "candidate_actions",
        "action_prior_weights",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"action-prior batch is missing {missing}")
    return {
        "states": batch["states"].to(device, non_blocking=True),
        "past_actions": batch["past_actions"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "actions": batch["candidate_actions"][:, 0].to(device, non_blocking=True),
        "weights": batch["action_prior_weights"].reshape(-1).to(
            device, non_blocking=True
        ),
    }


def _history(batch: Mapping[str, Tensor]) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["valid_mask"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True), encoding="utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "ActionPriorTrainConfig",
    "CHECKPOINT_FORMAT_VERSION",
    "evaluate_action_prior",
    "load_action_prior_checkpoint",
    "world_model_checkpoint_fingerprint",
    "world_model_member_fingerprint",
    "save_action_prior_checkpoint",
    "train_action_prior",
]
