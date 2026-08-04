"""Shared, reproducible five-task corpus contract.

The paths in this corpus are intentionally heterogeneous: most tasks are packed
in HDF5 shards while LPD uses one successful rollout per seed.  Never use a
parent directory name as the task label, since that would turn every LPD seed
into a separate "task".  This module is the single source of truth used by all
four method trainers.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path


CANONICAL_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "three_robots_stack_cube",
    "long_pipeline_delivery",
    "take_photo",
)


def task_from_path(path: str) -> str:
    """Return the benchmark task without leaking an ID into the policy."""
    normal = Path(path).as_posix().lower()
    aliases = {
        "lift_barrier": "lift_barrier",
        "camera_alignment": "camera_alignment",
        "three_robots_stack_cube": "three_robots_stack_cube",
        "long_pipeline_delivery": "long_pipeline_delivery",
        "take_photo": "take_photo",
    }
    for marker, task in aliases.items():
        if marker in normal:
            return task
    raise ValueError(f"Unrecognised five-task corpus path: {path}")


def relabel_trajectories(trajectories):
    """Attach a canonical *sampling-only* task label to trajectory records."""
    return [(path, key, n, present, task_from_path(path))
            for path, key, n, _old_task in trajectories]


def hierarchical_item_weights(kept_trajectories, items):
    """Equalize task, episode, local arm, then time within each local arm.

    The sampling label never reaches model inputs.  A task gets probability
    1/T; its valid demonstrations share that mass; each present local arm in a
    demo shares it; and its time indices share it.  This prevents 800-step LPD
    streams from dominating short tasks merely because they contain more frames.
    """
    episodes_per_task = Counter(task for *_prefix, task in kept_trajectories)
    stream_weights = {}
    for path, key, n, present, task in kept_trajectories:
        weight = 1.0 / (episodes_per_task[task] * len(present) * n)
        for arm in present:
            stream_weights[(path, key, arm)] = weight
    return [stream_weights[(path, key, arm)] for path, key, _t, arm, _task in items]
