from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from before_we_act.train_r11_candidate import (
    WorkerHeartbeat,
    _resume_contract,
    atomic_torch_save,
    checkpoint_alias,
    process_start_time_ticks,
    training_contract,
)


def test_training_contract_accepts_both_frozen_config_layouts():
    a = {
        "data": {"micro_batch": 2, "accumulation": 24},
        "model_config": {},
        "optimization": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "dtype": "bfloat16",
        },
    }
    d = {
        "training": {
            "micro_batch_size": 2,
            "gradient_accumulation": 24,
            "learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "optimizer": "AdamW8bit",
            "precision": "bfloat16",
        },
        "model_config": {},
    }
    assert training_contract(a)["effective_batch"] == 48
    assert training_contract(d)["optimizer"] == "AdamW8bit"
    a["data"]["accumulation"] = 23
    with pytest.raises(ValueError, match="effective batch"):
        training_contract(a)


def test_milestone_alias_preserves_old_checkpoint_when_latest_changes(tmp_path):
    latest = tmp_path / "checkpoint_latest.pt"
    milestone = tmp_path / "checkpoint_000001.pt"
    atomic_torch_save({"update": 1}, latest)
    checkpoint_alias(latest, milestone)
    assert os.stat(latest).st_ino == os.stat(milestone).st_ino
    atomic_torch_save({"update": 2}, latest)
    assert torch.load(milestone, weights_only=False)["update"] == 1
    assert torch.load(latest, weights_only=False)["update"] == 2


class _CursorValidator:
    def validate_resume_receipt(self, receipt):
        assert receipt == {"completed_update": 17}
        return 17


def test_resume_contract_fails_closed_on_optimizer_layout():
    identity = {
        "candidate": "D",
        "model": "R11LaWAMSubgoalFlow",
        "base_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "source_receipt_sha256": "c" * 64,
        "dataset_receipt_sha256": "d" * 64,
        "baseline_checkpoint_sha256": "e" * 64,
        "micro_batch_size": 2,
        "gradient_accumulation": 24,
    }
    saved = {
        "provenance": dict(identity),
        "sample_cursor": {"completed_update": 17},
        "update": 17,
    }
    assert _resume_contract(saved, identity=identity, sampler=_CursorValidator()) == 17
    saved["provenance"]["micro_batch_size"] = 1
    with pytest.raises(ValueError, match="micro_batch_size"):
        _resume_contract(saved, identity=identity, sampler=_CursorValidator())


def test_worker_heartbeat_binds_pid_start_time(tmp_path):
    path = tmp_path / "heartbeat.json"
    with WorkerHeartbeat(path, candidate="A", stage="f1", interval=0.01) as beat:
        beat.update(update=3, micro_step=4)
    payload = json.loads(path.read_text())
    assert payload["pid"] == os.getpid()
    assert payload["pid_start_time_ticks"] == process_start_time_ticks()
    assert payload["worker_identity_alive"] is True
    assert payload["update"] == 3
    assert payload["micro_step"] == 4
