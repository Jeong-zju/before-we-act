from __future__ import annotations

import json
from pathlib import Path
import pytest

from deployment.mars_dp.verify_frozen_config import CONFIG, verify


def test_frozen_manifest_matches_v3_source() -> None:
    receipt = verify()
    assert receipt["status"] == "complete"
    assert len(receipt["checkpoint_sha256"]) == 64


def test_manifest_pins_all_huggingface_sources() -> None:
    config = json.loads(CONFIG.read_text())
    sources = config["data"]["sources"]
    assert set(sources) == set(config["data"]["tasks"])
    assert all(len(row["revision"]) == 40 for row in sources.values())
    assert all(row["formal_shards"] == 10 for row in sources.values())


def test_activity_sampler_is_balanced_and_deterministic() -> None:
    pytest.importorskip("h5py")
    from deployment.mars_dp.dataset import TaskBalancedBatchSampler

    rows = [list(range(task * 10, task * 10 + 10)) for task in range(4)]
    activity = [float(index + 1) for index in range(40)]
    first = list(TaskBalancedBatchSampler(rows, 16, 3, 20260827, activity, 0.75))
    second = list(TaskBalancedBatchSampler(rows, 16, 3, 20260827, activity, 0.75))
    assert first == second
    for batch in first:
        assert len(batch) == 16
        assert [sum(task * 10 <= index < task * 10 + 10 for index in batch) for task in range(4)] == [4, 4, 4, 4]
