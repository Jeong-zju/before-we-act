from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "vendor" / "stereo-core"
CORE_SOURCE = CORE_ROOT / "stereo_core"
RESULT = ROOT / "docs" / "reports" / "20260804_S10_CORE_USER_DATA_FROZEN100.json"


@pytest.fixture(scope="module")
def trainer_module():
    # The repository test environment intentionally does not install the
    # reproduction server's torchvision build.  These two stubs let the tests
    # exercise the manifest and sampler contracts without importing vision.
    model_stub = ModuleType("no_wrist_pair_model")
    model_stub.NoWristPAIRRoute = object
    train_act_stub = ModuleType("train_act")
    train_act_stub.seed_everything = lambda _seed: None
    previous = {
        name: sys.modules.get(name)
        for name in ("no_wrist_pair_model", "train_act")
    }
    sys.modules["no_wrist_pair_model"] = model_stub
    sys.modules["train_act"] = train_act_stub
    sys.path.insert(0, str(CORE_SOURCE))
    try:
        yield importlib.import_module("train_no_wrist_pair")
    finally:
        sys.path.remove(str(CORE_SOURCE))
        sys.modules.pop("train_no_wrist_pair", None)
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_verified_server_sources_are_preserved_byte_for_byte() -> None:
    expected = {
        "no_wrist_pair_model.py": "056fae41f2da17767c3b6af54fc0373324fec4972fc8a7ffa0fae07a95ae8673",
        "train_no_wrist_pair.py": "ba9d07fa5c3a69ca2deb344b43dcd6788ef4f0a5c15cb77086e54aef33a99b20",
        "evaluate_no_wrist_pair.py": "be474a410bb40bd116942997592e279942a2f8f200347ee4b5c48fdc418519b6",
    }
    actual = {
        name: hashlib.sha256((CORE_SOURCE / name).read_bytes()).hexdigest()
        for name in expected
    }

    assert actual == expected


def test_user_data_loader_requires_all_five_tasks(
    trainer_module, tmp_path: Path
) -> None:
    manifests = []
    for index, task in enumerate(trainer_module.CANONICAL_TASKS):
        task_root = tmp_path / task
        task_root.mkdir()
        episode = task_root / "episode.h5"
        episode.touch()
        manifest = task_root / "training_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "task": {"id": task},
                    "action": {"dimension": 8 * (index % 4 + 1)},
                    "episodes": [
                        {
                            "split": "train",
                            "hdf5_path": episode.name,
                            "steps": 10 + index,
                            "seed": 100 + index,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)

    episodes = trainer_module.load_episodes(manifests)

    assert [episode["task"] for episode in episodes] == list(
        trainer_module.CANONICAL_TASKS
    )
    assert episodes[-1]["arms"] == (0,)
    with pytest.raises(ValueError, match="expected all five tasks"):
        trainer_module.load_episodes(manifests[:-1])


def test_exact_five_task_sampler_is_balanced_and_resumable(trainer_module) -> None:
    episodes = [
        {"task": task, "arms": (0, 1), "length": 17}
        for task in trainer_module.CANONICAL_TASKS
    ]
    complete = list(
        trainer_module.ExactFiveTaskBatchSampler(
            episodes, updates=4, seed=20260803
        )
    )
    resumed = list(
        trainer_module.ExactFiveTaskBatchSampler(
            episodes, updates=4, seed=20260803, start_update=2
        )
    )

    assert resumed == complete[2:]
    for batch in complete:
        assert len(batch) == 40
        counts = Counter(episodes[episode_index]["task"] for episode_index, _, _ in batch)
        assert counts == Counter({task: 8 for task in trainer_module.CANONICAL_TASKS})


def test_recorded_result_is_bound_to_formal_user_data_protocol() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["method"] == "core"
    assert result["protocol"]["optimizer_updates"] == 120_000
    assert result["protocol"]["local_action_chunks"] == 4_800_000
    assert result["successes"] == {
        "lift_barrier": 100,
        "camera_alignment": 60,
        "three_robots_stack_cube": 0,
        "long_pipeline_delivery": 100,
        "take_photo": 97,
    }
    assert result["macro_success_rate"] == pytest.approx(0.714)
    assert len(result["training_manifest_sha256"]) == 5


def test_portable_launcher_keeps_the_verified_training_contract() -> None:
    launcher = (ROOT / "scripts" / "train_s10_core_user_data.sh").read_text(
        encoding="utf-8"
    )

    assert "--batch-size 40" in launcher
    assert "S10_CORE_UPDATES:-120000" in launcher
    assert "--save-every 1000" in launcher
    assert "checkpoint_latest.pt" in launcher
    assert "/workspace/no_wrist_stereo_core" not in launcher
