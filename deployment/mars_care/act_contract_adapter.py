"""ACT-side adapter for the shared MARS-Control action contract.

This module is intentionally a thin compatibility layer.  It does not
implement ACT and it does not alter CARE's candidate/scorer/selector math.  A
caller uses it at four boundaries:

1. canonicalize raw HDF5 actions before accumulating ACT normalization;
2. canonicalize each target chunk before padding and loss computation;
3. canonicalize decoded model output immediately before environment execution;
4. embed and verify an action-contract/statistics receipt in checkpoints.

The external ``before-we-act`` checkout has local, uncommitted changes, so the
adapter is kept here and can be imported through ``PYTHONPATH`` or copied next
to the ACT launcher.  No source file in that checkout is modified by this
repository.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HIGH,
    ACTION_LOW,
    action_contract_hash,
    audit_action_array,
    canonicalize_action,
    canonicalize_normalized_action,
    checkpoint_action_contract,
    contract_metadata,
    normalization_stats_hash,
    validate_action_space_bounds,
    validate_action_stats,
    validate_checkpoint_action_contract,
)


FORMAT_VERSION = "before-we-act.mars-act-contract-adapter/1"
TASK_LAYOUT: dict[str, int] = {
    "place_cube_in_cup": 2,
    "strike_cube_hard": 2,
    "three_robots_place_shoes": 3,
    "four_robots_stack_cube": 4,
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _trajectory_names(handle: h5py.File) -> list[str]:
    return sorted(
        (name for name in handle if name.startswith("traj_")),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )


def _task_paths(root: Path, task: str) -> list[Path]:
    directory = root / task / "motionplanning"
    merged = directory / f"{task}.h5"
    if merged.is_file():
        return [merged]
    return sorted(directory.glob(f"{task}.shard*.h5"))


class ACTActionMomentAccumulator:
    """Streaming population moments over canonicalized per-arm actions.

    The raw corpus remains untouched.  Statistics are accumulated in float64
    from the canonical float32 projection, which is the same representation
    that ACT receives in ``__getitem__``.  Keeping audit counters here makes it
    impossible to accidentally report a clean corpus after clipping values.
    """

    def __init__(self, *, std_floor: float = 1e-4) -> None:
        if not np.isfinite(std_floor) or std_floor <= 0:
            raise ValueError("std_floor must be a positive finite value")
        self.std_floor = float(std_floor)
        self._sum = np.zeros(ACTION_DIM, dtype=np.float64)
        self._sq_sum = np.zeros(ACTION_DIM, dtype=np.float64)
        self._rows = 0
        self._raw_values = 0
        self._out_of_bounds_values = 0
        self._max_abs_change = 0.0
        self._by_dimension = np.zeros(ACTION_DIM, dtype=np.int64)
        self._sources = 0

    def update(self, actions: Any, *, source: str | None = None) -> dict[str, Any]:
        canonical, report = audit_action_array(actions, source=source)
        rows = canonical.reshape(-1, ACTION_DIM).astype(np.float64, copy=False)
        self._sum += rows.sum(axis=0)
        self._sq_sum += np.square(rows).sum(axis=0)
        self._rows += int(rows.shape[0])
        self._raw_values += int(report["raw_values"])
        self._out_of_bounds_values += int(report["out_of_bounds_values"])
        self._max_abs_change = max(
            self._max_abs_change, float(report["max_abs_change"])
        )
        self._by_dimension += np.asarray(
            report["out_of_bounds_by_dimension"], dtype=np.int64
        )
        self._sources += 1
        return report

    def finalize(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._rows <= 0:
            raise ValueError("cannot finalize empty action moments")
        mean = self._sum / self._rows
        variance = np.maximum(self._sq_sum / self._rows - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), self.std_floor)
        stats: dict[str, Any] = {
            "a_mean": mean.astype(np.float32).tolist(),
            "a_std": std.astype(np.float32).tolist(),
            "action_encoding": ACTION_ENCODING,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "action_contract": contract_metadata(),
        }
        stats["normalization_sha256"] = normalization_stats_hash(stats)
        audit = {
            "contract_version": ACTION_CONTRACT_VERSION,
            "contract_sha256": action_contract_hash(),
            "raw_rows": self._rows,
            "raw_values": self._raw_values,
            "out_of_bounds_values": self._out_of_bounds_values,
            "out_of_bounds_fraction": (
                self._out_of_bounds_values / self._raw_values
                if self._raw_values
                else 0.0
            ),
            "max_abs_change": self._max_abs_change,
            "out_of_bounds_by_dimension": self._by_dimension.tolist(),
            "sources": self._sources,
        }
        return stats, audit


def canonicalize_act_target(
    actions: Any, *, horizon: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize an ACT chunk, repeat-pad it, and return its valid mask."""

    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("ACT horizon must be positive")
    canonical = canonicalize_action(actions)
    if canonical.ndim != 2:
        raise ValueError(
            f"ACT target must be [time,{ACTION_DIM}], got {canonical.shape}"
        )
    if canonical.shape[0] < 1:
        raise ValueError("ACT target cannot be empty")
    valid = min(int(canonical.shape[0]), horizon)
    result = np.empty((horizon, ACTION_DIM), dtype=np.float32)
    result[:valid] = canonical[:valid]
    result[valid:] = canonical[valid - 1]
    mask = np.zeros(horizon, dtype=np.float32)
    mask[:valid] = 1.0
    return result, mask


def canonicalize_act_target_torch(
    actions: Any, *, horizon: int = 100
) -> tuple[Any, Any]:
    """Torch target variant preserving device/dtype for ACT's data loader."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("torch is required for the torch ACT adapter") from error
    if not torch.is_tensor(actions):
        raise ValueError("ACT torch target must be a tensor")
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"ACT torch target must be [time,{ACTION_DIM}], got {tuple(actions.shape)}"
        )
    if actions.shape[0] < 1 or int(horizon) < 1:
        raise ValueError("ACT torch target and horizon must be non-empty")
    values = actions if torch.is_floating_point(actions) else actions.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("ACT torch target must be finite")
    low = torch.as_tensor(ACTION_LOW, device=values.device, dtype=values.dtype)
    high = torch.as_tensor(ACTION_HIGH, device=values.device, dtype=values.dtype)
    values = torch.clamp(values, min=low, max=high)
    valid = min(int(values.shape[0]), int(horizon))
    result = torch.empty((int(horizon), ACTION_DIM), device=values.device, dtype=values.dtype)
    result[:valid] = values[:valid]
    result[valid:] = values[valid - 1]
    mask = torch.zeros(int(horizon), device=values.device, dtype=values.dtype)
    mask[:valid] = 1
    return result, mask


def canonicalize_runtime_action(action: Any, action_space: Any) -> np.ndarray:
    """Validate live bounds and return the exact physical command."""

    validate_action_space_bounds(action_space)
    result = canonicalize_action(action)
    if result.shape != (ACTION_DIM,):
        raise ValueError(f"runtime action must be width {ACTION_DIM}, got {result.shape}")
    return result


def decode_and_canonicalize_runtime_chunk(
    normalized_chunk: Any, stats: Mapping[str, Any]
) -> np.ndarray:
    """Decode an ACT output chunk, clip in physical space, and return float32."""

    mean, std = validate_action_stats(stats)
    normalized = np.asarray(normalized_chunk)
    if normalized.ndim != 2 or normalized.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"decoded ACT chunk must be [time,{ACTION_DIM}], got {normalized.shape}"
        )
    return canonicalize_action(normalized.astype(np.float32) * std + mean)


def act_checkpoint_contract(
    stats: Mapping[str, Any], *, corpus_receipt_sha256: str | None = None
) -> dict[str, Any]:
    """Create checkpoint metadata binding ACT to canonical action moments."""

    validate_action_stats(stats)
    annotations: dict[str, Any] = {
        "adapter_format": FORMAT_VERSION,
        "normalization_sha256": normalization_stats_hash(stats),
    }
    if corpus_receipt_sha256 is not None:
        if len(str(corpus_receipt_sha256)) != 64:
            raise ValueError("corpus receipt hash must be a SHA256 hex digest")
        annotations["corpus_receipt_sha256"] = str(corpus_receipt_sha256)
    return checkpoint_action_contract(**annotations)


def validate_act_checkpoint_contract(
    checkpoint: Mapping[str, Any], *, require_corpus_receipt: bool = False
) -> dict[str, Any]:
    """Validate action metadata and the stats hash embedded in an ACT payload."""

    metadata = validate_checkpoint_action_contract(checkpoint)
    stats = checkpoint.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("ACT checkpoint is missing stats mapping")
    validate_action_stats(stats)
    expected_stats_hash = normalization_stats_hash(stats)
    annotations = metadata.get("annotations", {})
    if not isinstance(annotations, Mapping):
        raise ValueError("ACT action-contract annotations are malformed")
    if annotations.get("normalization_sha256") != expected_stats_hash:
        raise ValueError("ACT checkpoint action statistics hash differs")
    if require_corpus_receipt and not annotations.get("corpus_receipt_sha256"):
        raise ValueError("ACT checkpoint is missing corpus receipt hash")
    return metadata


def _iter_task_actions(root: Path, task: str) -> Iterable[tuple[Path, str, int, np.ndarray]]:
    arms = TASK_LAYOUT[task]
    paths = _task_paths(root, task)
    if not paths:
        raise FileNotFoundError(root / task / "motionplanning")
    for path in paths:
        with h5py.File(path, "r", swmr=True) as handle:
            for trajectory in _trajectory_names(handle):
                group = handle[trajectory]
                lengths = [int(group[f"actions/panda-{arm}"].shape[0]) for arm in range(arms)]
                if not lengths or min(lengths) < 1:
                    raise ValueError(f"empty action stream: {path}:{trajectory}")
                length = min(lengths)
                for arm in range(arms):
                    values = np.asarray(
                        group[f"actions/panda-{arm}"][:length], dtype=np.float32
                    )
                    if values.ndim != 2 or values.shape != (length, ACTION_DIM):
                        raise ValueError(
                            f"action shape drift: {path}:{trajectory}:arm{arm}: {values.shape}"
                        )
                    yield path, trajectory, arm, values


def compute_act_action_normalization(
    raw_root: str | Path,
    *,
    std_floor: float = 1e-4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute ACT-compatible all-data moments and an immutable audit report."""

    root = Path(raw_root).resolve(strict=True)
    accumulator = ACTActionMomentAccumulator(std_floor=std_floor)
    episode_counts: dict[str, int] = {}
    local_steps: dict[str, int] = {}
    shard_counts: dict[str, int] = {}
    for task, arms in TASK_LAYOUT.items():
        paths = _task_paths(root, task)
        if not paths:
            raise FileNotFoundError(root / task / "motionplanning")
        shard_counts[task] = len(paths)
        episodes: set[tuple[Path, str]] = set()
        steps = 0
        for path, trajectory, arm, actions in _iter_task_actions(root, task):
            episodes.add((path, trajectory))
            steps += int(actions.shape[0])
            accumulator.update(actions, source=f"{path}:{trajectory}:panda-{arm}")
        episode_counts[task] = len(episodes)
        local_steps[task] = steps
        if episode_counts[task] == 0:
            raise ValueError(f"task has no trajectories: {task}")
    stats, action_audit = accumulator.finalize()
    stats.update(
        {
            "format_version": "before-we-act.mars-act.normalization/1",
            "tasks": TASK_LAYOUT,
            "episodes": int(sum(episode_counts.values())),
            "local_steps": int(sum(local_steps.values())),
            "training_policy": "all_data_no_split",
        }
    )
    receipt = {
        "format_version": FORMAT_VERSION,
        "status": "PASSED",
        "contract": contract_metadata(),
        "contract_sha256": action_contract_hash(),
        "tasks": TASK_LAYOUT,
        "episodes_by_task": episode_counts,
        "shards_by_task": shard_counts,
        "local_steps_by_task": local_steps,
        "episodes": int(sum(episode_counts.values())),
        "local_steps": int(sum(local_steps.values())),
        "training_policy": "all_data_no_split",
        "action_audit": action_audit,
        "normalization_sha256": normalization_stats_hash(stats),
    }
    return stats, receipt


def write_corpus_receipt(
    raw_root: str | Path,
    output: str | Path,
    *,
    expected_episodes_per_task: int = 150,
    expected_shards_per_task: int = 10,
    expected_local_steps: int | None = None,
) -> dict[str, Any]:
    """Audit all four task streams and atomically write a JSON receipt."""

    stats, receipt = compute_act_action_normalization(raw_root)
    if any(
        int(receipt["episodes_by_task"][task]) != int(expected_episodes_per_task)
        for task in TASK_LAYOUT
    ):
        raise ValueError(
            "MARS ACT corpus episode count differs from the requested contract"
        )
    if any(
        int(receipt["shards_by_task"][task]) != int(expected_shards_per_task)
        for task in TASK_LAYOUT
    ):
        raise ValueError("MARS ACT corpus shard count differs from the requested contract")
    if expected_local_steps is not None and int(receipt["local_steps"]) != int(expected_local_steps):
        raise ValueError(
            f"MARS ACT corpus local-step count differs: "
            f"{receipt['local_steps']} != {expected_local_steps}"
        )
    receipt = dict(receipt)
    receipt["normalization"] = stats
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(Path(output), receipt)
    return receipt


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes-per-task", type=int, default=150)
    parser.add_argument("--expected-shards-per-task", type=int, default=10)
    parser.add_argument("--expected-local-steps", type=int)
    args = parser.parse_args()
    receipt = write_corpus_receipt(
        args.data_root,
        args.output,
        expected_episodes_per_task=args.expected_episodes_per_task,
        expected_shards_per_task=args.expected_shards_per_task,
        expected_local_steps=args.expected_local_steps,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "episodes": receipt["episodes"],
                "local_steps": receipt["local_steps"],
                "out_of_bounds_values": receipt["action_audit"]["out_of_bounds_values"],
                "normalization_sha256": receipt["normalization_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "ACTActionMomentAccumulator",
    "FORMAT_VERSION",
    "TASK_LAYOUT",
    "act_checkpoint_contract",
    "canonicalize_act_target",
    "canonicalize_act_target_torch",
    "canonicalize_runtime_action",
    "compute_act_action_normalization",
    "decode_and_canonicalize_runtime_chunk",
    "validate_act_checkpoint_contract",
    "write_corpus_receipt",
]
