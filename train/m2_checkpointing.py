"""Strict, self-describing Phase M2 checkpoint directories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch

from models.wam_multimodal import BlockCausalWAM, BlockCausalWAMConfig


M2_CHECKPOINT_FORMAT = "wam.robofactory.m2.checkpoint/5"
M2_LEGACY_CHECKPOINT_FORMAT = "wam.robofactory.m2.checkpoint/4"
M2_V3_CHECKPOINT_FORMAT = "wam.robofactory.m2.checkpoint/3"


def save_m2_checkpoint(
    directory: str | Path,
    *,
    model: BlockCausalWAM,
    task_runtime: Sequence[Mapping[str, Any]],
    vision_identity: Mapping[str, Any],
    action_generation: Mapping[str, Any],
    action_objective: Mapping[str, Any],
    training: Mapping[str, Any],
    metrics: Mapping[str, Any],
    allow_replace: bool = False,
) -> dict[str, Any]:
    target = Path(directory).expanduser().resolve()
    if target.exists() and any(target.iterdir()) and not allow_replace:
        raise FileExistsError(f"refusing to overwrite non-empty M2 checkpoint {target}")
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.safetensors"
    _save_safetensors_atomic(
        model_path,
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
    )
    runtime = [dict(value) for value in task_runtime]
    _validate_task_runtime(
        runtime,
        max_action_dim=model.config.max_action_dim,
        max_action_horizon=model.config.action_horizon,
        max_cameras=model.config.max_cameras,
        max_agents=model.config.max_agents,
        require_action_horizon=True,
        require_camera_identity=True,
    )
    generation = _validate_action_generation(action_generation)
    objective = _validate_action_objective(
        action_objective, require_current=True
    )
    _assert_json_finite(runtime)
    _write_json_atomic(target / "task_runtime.json", runtime)
    _write_json_atomic(target / "metrics.json", dict(metrics))
    files = {
        "model.safetensors": _sha256(model_path),
        "task_runtime.json": _sha256(target / "task_runtime.json"),
        "metrics.json": _sha256(target / "metrics.json"),
    }
    schema = {
        "format_version": M2_CHECKPOINT_FORMAT,
        "model_config": model.config.to_dict(),
        "trainable_parameters": model.trainable_parameters,
        "task_vocabulary": [str(value["task_id"]) for value in runtime],
        "vision_identity": dict(vision_identity),
        "action_space": "per_task_zscore_canonical_unit_action",
        "action_generation": generation,
        "action_objective": objective,
        "training": dict(training),
        "files": files,
    }
    _assert_json_finite(schema)
    _write_json_atomic(target / "schema.json", schema)
    return {
        "checkpoint": str(target),
        "format_version": M2_CHECKPOINT_FORMAT,
        "tree_sha256": m2_checkpoint_tree_sha256(target),
        "files": {**files, "schema.json": _sha256(target / "schema.json")},
    }


def load_m2_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[BlockCausalWAM, list[dict[str, Any]], dict[str, Any]]:
    root = Path(directory).expanduser().resolve(strict=True)
    schema = _read_json(root / "schema.json")
    checkpoint_format = schema.get("format_version")
    if checkpoint_format not in {
        M2_CHECKPOINT_FORMAT,
        M2_LEGACY_CHECKPOINT_FORMAT,
        M2_V3_CHECKPOINT_FORMAT,
    }:
        raise ValueError("checkpoint is not a Phase M2 artifact")
    files = schema.get("files")
    if not isinstance(files, dict) or set(files) != {
        "model.safetensors",
        "task_runtime.json",
        "metrics.json",
    }:
        raise ValueError("M2 checkpoint file manifest is incomplete")
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"M2 checkpoint file identity mismatch: {name}")
    config = BlockCausalWAMConfig.from_dict(_mapping(schema, "model_config"))
    if schema.get("action_space") != "per_task_zscore_canonical_unit_action":
        raise ValueError("M2 checkpoint action space is unsupported")
    _validate_action_generation(_mapping(schema, "action_generation"))
    _validate_action_objective(
        _mapping(schema, "action_objective"),
        require_current=checkpoint_format != M2_V3_CHECKPOINT_FORMAT,
    )
    model = BlockCausalWAM(config)
    state = load_file(root / "model.safetensors", device=str(device))
    if checkpoint_format == M2_CHECKPOINT_FORMAT:
        incompatible = model.load_state_dict(state, strict=True)
        allowed_missing: set[str] = set()
    else:
        nn_missing = {
            "camera_embedding.weight",
            "camera_agent_embedding.weight",
        }
        with torch.no_grad():
            model.camera_embedding.weight.zero_()
            model.camera_agent_embedding.weight.zero_()
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = nn_missing
    if (
        set(incompatible.missing_keys) != allowed_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(f"strict M2 state load failed: {incompatible}")
    model = model.to(device).eval()
    runtime_raw = _read_json(root / "task_runtime.json")
    if not isinstance(runtime_raw, list) or len(runtime_raw) != config.num_tasks:
        raise ValueError("M2 task runtime count differs from the model")
    runtime = [dict(_mapping(value)) for value in runtime_raw]
    _validate_task_runtime(
        runtime,
        max_action_dim=config.max_action_dim,
        max_action_horizon=config.action_horizon,
        max_cameras=config.max_cameras,
        max_agents=config.max_agents,
        require_action_horizon=checkpoint_format != M2_V3_CHECKPOINT_FORMAT,
        require_camera_identity=checkpoint_format == M2_CHECKPOINT_FORMAT,
    )
    if checkpoint_format == M2_V3_CHECKPOINT_FORMAT:
        for value in runtime:
            value.setdefault("action_horizon", config.action_horizon)
    vocabulary = [str(value["task_id"]) for value in runtime]
    if vocabulary != list(schema.get("task_vocabulary", [])):
        raise ValueError("M2 task vocabulary drifted between checkpoint files")
    if (
        checkpoint_format == M2_CHECKPOINT_FORMAT
        and int(schema.get("trainable_parameters", -1)) != model.trainable_parameters
    ):
        raise ValueError("M2 checkpoint parameter count disagrees with architecture")
    return model, runtime, schema


def m2_checkpoint_tree_sha256(directory: str | Path) -> str:
    root = Path(directory).expanduser().resolve(strict=True)
    files = sorted(path for path in root.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _save_safetensors_atomic(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        save_file(dict(state), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _assert_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint JSON {path}") from exc


def _mapping(value: Any, key: str | None = None) -> Mapping[str, Any]:
    selected = value if key is None else value.get(key)
    if not isinstance(selected, Mapping):
        raise ValueError(f"checkpoint field {key or '<root>'} must be an object")
    return selected


def _validate_action_generation(value: Mapping[str, Any]) -> dict[str, Any]:
    generation = dict(value)
    required = {
        "solver_steps",
        "solver",
        "normalized_action_clip",
        "execution_steps",
        "warm_start",
    }
    if set(generation) != required:
        raise ValueError("M2 action generation contract is incomplete")
    solver_steps = int(generation["solver_steps"])
    execution_steps = int(generation["execution_steps"])
    normalized_clip = float(generation["normalized_action_clip"])
    solver = str(generation["solver"])
    if (
        solver_steps <= 0
        or execution_steps <= 0
        or not torch.isfinite(torch.tensor(normalized_clip))
        or normalized_clip <= 0.0
        or solver not in {"euler", "heun"}
        or not isinstance(generation["warm_start"], bool)
    ):
        raise ValueError("M2 action generation contract is invalid")
    return {
        "solver_steps": solver_steps,
        "solver": solver,
        "normalized_action_clip": normalized_clip,
        "execution_steps": execution_steps,
        "warm_start": bool(generation["warm_start"]),
    }


def _validate_task_runtime(
    runtime: Sequence[Mapping[str, Any]],
    *,
    max_action_dim: int,
    max_action_horizon: int,
    max_cameras: int,
    max_agents: int,
    require_action_horizon: bool,
    require_camera_identity: bool,
) -> None:
    for value in runtime:
        task_id = str(value.get("task_id", ""))
        action_dim = int(value.get("action_dim", -1))
        if not task_id or not 0 < action_dim <= max_action_dim:
            raise ValueError("M2 task runtime action identity is invalid")
        action_horizon = value.get("action_horizon")
        if require_action_horizon and action_horizon is None:
            raise ValueError(
                f"M2 task {task_id!r} runtime lacks action_horizon"
            )
        if action_horizon is not None and not (
            0 < int(action_horizon) <= int(max_action_horizon)
        ):
            raise ValueError(
                f"M2 task {task_id!r} action horizon is invalid"
            )
        cameras = value.get("camera_order")
        slots = value.get("camera_slot_indices")
        agents = value.get("camera_agent_indices")
        if require_camera_identity and (
            not isinstance(cameras, list)
            or not cameras
            or not isinstance(slots, list)
            or not isinstance(agents, list)
            or len(slots) != len(cameras)
            or len(agents) != len(cameras)
            or len(set(map(str, cameras))) != len(cameras)
            or len(set(map(int, slots))) != len(slots)
            or any(not 0 <= int(slot) < max_cameras for slot in slots)
            or any(not 0 <= int(agent) <= max_agents for agent in agents)
        ):
            raise ValueError(
                f"M2 task {task_id!r} camera identity contract is invalid"
            )
        mean = value.get("action_mean")
        std = value.get("action_std")
        if (
            not isinstance(mean, list)
            or not isinstance(std, list)
            or len(mean) != action_dim
            or len(std) != action_dim
        ):
            raise ValueError(
                f"M2 task {task_id!r} action normalization shape is invalid"
            )
        mean_tensor = torch.tensor(mean, dtype=torch.float64)
        std_tensor = torch.tensor(std, dtype=torch.float64)
        if (
            not bool(torch.isfinite(mean_tensor).all())
            or not bool(torch.isfinite(std_tensor).all())
            or not bool(std_tensor.gt(0.0).all())
        ):
            raise ValueError(
                f"M2 task {task_id!r} action normalization values are invalid"
            )


def _validate_action_objective(
    value: Mapping[str, Any],
    *,
    require_current: bool,
) -> dict[str, Any]:
    objective = dict(value)
    legacy_keys = {"tail_windows", "executed_prefix_weight"}
    current_keys = legacy_keys | {
        "visual_prefix_windows",
        "task_horizons",
        "loss_reduction",
    }
    expected = current_keys if require_current else legacy_keys
    if set(objective) != expected:
        raise ValueError("M2 action objective contract is incomplete")
    prefix_weight = float(objective["executed_prefix_weight"])
    if (
        objective["tail_windows"] != "repeat_last_with_validity_masks"
        or not bool(torch.isfinite(torch.tensor(prefix_weight)))
        or prefix_weight < 1.0
    ):
        raise ValueError("M2 action objective contract is invalid")
    result = {
        "tail_windows": "repeat_last_with_validity_masks",
        "executed_prefix_weight": prefix_weight,
    }
    if require_current:
        if (
            objective["visual_prefix_windows"]
            != "left_zero_pad_with_validity_mask"
            or objective["task_horizons"]
            != "max_tensor_with_task_validity_masks"
            or objective["loss_reduction"]
            != "per_sample_valid_element_mean"
        ):
            raise ValueError("M2 current action objective contract is invalid")
        result.update(
            visual_prefix_windows="left_zero_pad_with_validity_mask",
            task_horizons="max_tensor_with_task_validity_masks",
            loss_reduction="per_sample_valid_element_mean",
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_json_finite(value: Any) -> None:
    json.dumps(value, allow_nan=False)


__all__ = [
    "M2_CHECKPOINT_FORMAT",
    "M2_LEGACY_CHECKPOINT_FORMAT",
    "M2_V3_CHECKPOINT_FORMAT",
    "load_m2_checkpoint",
    "m2_checkpoint_tree_sha256",
    "save_m2_checkpoint",
]
