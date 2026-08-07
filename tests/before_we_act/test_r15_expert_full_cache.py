from __future__ import annotations

import json

import h5py
import numpy as np

from before_we_act.data.full_episode_windows import FULL_EPISODE_PROTOCOL
from before_we_act.data.raw_team_windows import TASKS
from scripts.before_we_act.prepare_r15_expert_full_cache import (
    TASK,
    compose_index,
    terminal_steps,
)


def test_terminal_steps_removes_post_success_tail(tmp_path):
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("terminated", data=[False, False, True, True])
        group.create_dataset("truncated", data=[False, False, False, False])
        group.create_dataset("success", data=[False, False, True, True])
        assert terminal_steps(group) == 3


def test_compose_index_appends_expert_without_mutating_base(tmp_path):
    base_path = tmp_path / "base.json"
    receipt_path = tmp_path / "receipt.json"
    base = {
        "schema_version": 1,
        "round": "R12-R4",
        "protocol_variant": FULL_EPISODE_PROTOCOL,
        "episodes": [
            {
                "task": task,
                "split": "train",
                "seed": 3000 + index,
                "episode_index": index,
                "steps": 10,
                "path": f"/{task}.hdf5",
                "hdf5_sha256": "a" * 64,
            }
            for index, task in enumerate(TASKS)
        ],
        "step_counts": {
            split: {task: (10 if split == "train" else 0) for task in TASKS}
            for split in ("train", "validation")
        },
    }
    base_path.write_text(json.dumps(base))
    receipt_path.write_text("{}")
    expert = {
        "task": TASK,
        "split": "train",
        "seed": 5000,
        "episode_index": 99,
        "steps": 401,
        "path": "/expert.hdf5",
        "hdf5_sha256": "b" * 64,
    }
    combined = compose_index(
        base,
        [expert],
        base_index_path=base_path,
        receipt_path=receipt_path,
    )
    assert len(base["episodes"]) == len(TASKS)
    assert len(combined["episodes"]) == len(TASKS) + 1
    assert combined["step_counts"]["train"][TASK] == 411
    assert combined["extension"]["expert_episodes"] == 1


def test_terminal_steps_rejects_unsuccessful_terminal(tmp_path):
    path = tmp_path / "raw_failed.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("terminated", data=[False, True])
        group.create_dataset("truncated", data=[False, False])
        group.create_dataset("success", data=[False, False])
        try:
            terminal_steps(group)
        except ValueError as error:
            assert "without a success" in str(error)
        else:
            raise AssertionError("unsuccessful terminal was accepted")
