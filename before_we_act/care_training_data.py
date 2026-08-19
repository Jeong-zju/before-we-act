"""Prepared-data contract for the owner-authorized A6 CARE diagnostic."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from before_we_act.care_belief import CARE_HORIZONS


ORDINARY_WEIGHTS = np.asarray(
    (
        0.3409090909090909,
        0.3409090909090909,
        0.0,
        0.09090909090909091,
        0.06818181818181818,
        0.06818181818181818,
        0.03409090909090909,
        0.056818181818181816,
    ),
    dtype=np.float64,
)
SPLIT_NAMES = ("train", "validation", "calibration", "test")
SPLIT_IDS = {name: index for index, name in enumerate(SPLIT_NAMES)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def ordinary_utility(outcome: Mapping[str, Any]) -> float:
    vector = np.asarray(outcome["bounded_utility_vector"], dtype=np.float64)
    if vector.shape != (8,):
        raise ValueError("CARE outcome vector must contain eight entries")
    return float(np.dot(ORDINARY_WEIGHTS, vector))


def branch_by_key(
    family: Mapping[str, Any], candidate: int, regime: str, repeat: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in family["branches"]
        if int(row["candidate_id"]) == int(candidate)
        and str(row["regime"]) == regime
        and int(row["repeat_id"]) == int(repeat)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one CARE branch candidate={candidate} regime={regime} repeat={repeat}"
        )
    return rows[0]


def family_targets(
    family: Mapping[str, Any], quality: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build [horizon,candidate,repeat,(direct,response,total)] labels."""

    targets = np.zeros((len(CARE_HORIZONS), 6, 2, 3), dtype=np.float32)
    hard_safety = np.zeros((len(CARE_HORIZONS), 6, 2), dtype=np.float32)
    usable = np.zeros((len(CARE_HORIZONS),), dtype=bool)
    for horizon_index, horizon in enumerate(CARE_HORIZONS):
        quality_row = quality["horizons"][str(horizon)]
        if str(quality_row["label"]) != "USE":
            continue
        usable[horizon_index] = True
        for repeat in (0, 1):
            reference_reactive = branch_by_key(
                family, 0, "reactive", repeat
            )["outcomes"][str(horizon)]
            reference_replay = branch_by_key(
                family, 0, "replay", repeat
            )["outcomes"][str(horizon)]
            u0_reactive = ordinary_utility(reference_reactive)
            u0_replay = ordinary_utility(reference_replay)
            for candidate in range(6):
                reactive = branch_by_key(
                    family, candidate, "reactive", repeat
                )["outcomes"][str(horizon)]
                replay = branch_by_key(
                    family, candidate, "replay", repeat
                )["outcomes"][str(horizon)]
                direct = ordinary_utility(replay) - u0_replay
                total = ordinary_utility(reactive) - u0_reactive
                targets[horizon_index, candidate, repeat] = (
                    direct,
                    total - direct,
                    total,
                )
                hard_safety[horizon_index, candidate, repeat] = float(
                    bool(reactive["hard_safety_violation"])
                )
    return targets, hard_safety, usable


@dataclass(frozen=True)
class PreparedCAREData:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    candidate_chunks: torch.Tensor
    targets: torch.Tensor
    hard_safety: torch.Tensor
    usable: torch.Tensor
    split_id: torch.Tensor
    task_id: torch.Tensor
    snapshot_ids: tuple[str, ...]
    tasks: tuple[str, ...]
    manifest: Mapping[str, Any]


def load_prepared_care(path: str | Path) -> PreparedCAREData:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("format_version") != "before-we-act.a6r1-care-prepared-data/1":
        raise ValueError("wrong prepared CARE data format")
    return PreparedCAREData(
        memory=value["memory"].float(),
        memory_mask=value["memory_mask"].bool(),
        candidate_chunks=value["candidate_chunks"].float(),
        targets=value["targets"].float(),
        hard_safety=value["hard_safety"].float(),
        usable=value["usable"].bool(),
        split_id=value["split_id"].long(),
        task_id=value["task_id"].long(),
        snapshot_ids=tuple(value["snapshot_ids"]),
        tasks=tuple(value["tasks"]),
        manifest=value["manifest"],
    )


class CARETrainingDataset(Dataset):
    """Family/horizon/repeat rows without sibling-branch leakage."""

    def __init__(
        self,
        prepared: PreparedCAREData,
        split: str,
        *,
        primary_horizon_only: bool = False,
        primary_horizon: int = 16,
    ) -> None:
        if split not in SPLIT_IDS:
            raise ValueError(f"unknown CARE split: {split}")
        rows: list[tuple[int, int, int]] = []
        allowed_horizons = (
            (CARE_HORIZONS.index(primary_horizon),)
            if primary_horizon_only
            else tuple(range(len(CARE_HORIZONS)))
        )
        for family_index in range(len(prepared.snapshot_ids)):
            if int(prepared.split_id[family_index]) != SPLIT_IDS[split]:
                continue
            for horizon_index in allowed_horizons:
                if not bool(prepared.usable[family_index, horizon_index]):
                    continue
                for repeat in (0, 1):
                    rows.append((family_index, horizon_index, repeat))
        if not rows:
            raise ValueError(f"CARE split has no usable rows: {split}")
        self.prepared = prepared
        self.split = split
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        family, horizon, repeat = self.rows[index]
        return {
            "memory": self.prepared.memory[family],
            "memory_mask": self.prepared.memory_mask[family],
            "candidate_chunks": self.prepared.candidate_chunks[family],
            "target": self.prepared.targets[family, horizon, :, repeat],
            "hard_safety": self.prepared.hard_safety[family, horizon, :, repeat],
            "horizon_index": torch.tensor(horizon, dtype=torch.long),
            "family_index": torch.tensor(family, dtype=torch.long),
            "task_id": self.prepared.task_id[family],
        }


def family_indices(prepared: PreparedCAREData, split: str) -> list[int]:
    split_id = SPLIT_IDS[split]
    return [
        index
        for index, value in enumerate(prepared.split_id.tolist())
        if int(value) == split_id
    ]


__all__ = [
    "CARETrainingDataset",
    "ORDINARY_WEIGHTS",
    "PreparedCAREData",
    "SPLIT_IDS",
    "SPLIT_NAMES",
    "atomic_json",
    "branch_by_key",
    "canonical_sha256",
    "family_indices",
    "family_targets",
    "load_prepared_care",
    "ordinary_utility",
    "sha256_file",
]
