"""Strictly decentralized DuoBench adapter for CARE's DINO B0-H policy.

The adapter is intentionally isolated from :mod:`deployment.duo_care`: B0-H
predicts absolute local action chunks and is the frozen RoboFactory reference
backbone, while ``duo_care`` is the later branch-selection stack.
"""

from .data import (
    ACTION_HORIZON,
    EFFECTIVE_BATCH,
    HISTORY_STEPS,
    TASKS,
    DuoBalancedDistributedBatchSampler,
    DuoTemporalDataset,
    DuoTemporalEpisode,
    DuoTemporalRequest,
    load_duo_episodes,
)

__all__ = [
    "ACTION_HORIZON",
    "EFFECTIVE_BATCH",
    "HISTORY_STEPS",
    "TASKS",
    "DuoBalancedDistributedBatchSampler",
    "DuoTemporalDataset",
    "DuoTemporalEpisode",
    "DuoTemporalRequest",
    "load_duo_episodes",
]
