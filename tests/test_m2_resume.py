from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts import train_robofactory_m2
from train.m2_resume import (
    load_latest_m2_resume_checkpoint,
    save_m2_resume_checkpoint,
)
from train.robofactory_multitask_dataset import (
    CoverageTemperatureDistributedSampler,
)


ROOT = Path(__file__).resolve().parents[1]


def _trained_linear() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_resume_snapshot_restores_model_optimizer_progress_and_rng(tmp_path) -> None:
    torch.manual_seed(13)
    np.random.seed(13)
    random.seed(13)
    model, optimizer = _trained_linear()
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    identity = {"config_sha256": "abc", "world_size": 1}
    progress = {
        "stage_index": 0,
        "stage_step": 7,
        "global_step": 7,
        "epoch": 0,
        "samples_consumed_in_epoch": 14,
        "history": [],
        "current_stage_last": {"total": 1.25},
        "elapsed_seconds": 2.0,
    }
    save_m2_resume_checkpoint(
        tmp_path / "resume",
        model=model,
        optimizer=optimizer,
        identity=identity,
        progress=progress,
        coverage_seen=torch.tensor([True, False, True]),
        keep_last=2,
    )
    expected_torch = torch.rand(4)
    expected_numpy = np.random.rand(4)
    expected_python = [random.random() for _ in range(4)]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)
    restored = load_latest_m2_resume_checkpoint(
        tmp_path / "resume",
        model=model,
        optimizer=optimizer,
        expected_identity=identity,
        device="cpu",
    )

    assert restored is not None
    assert restored["global_step"] == 7
    assert restored["coverage_seen"].tolist() == [True, False, True]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name])
    torch.testing.assert_close(torch.rand(4), expected_torch)
    np.testing.assert_array_equal(np.random.rand(4), expected_numpy)
    assert [random.random() for _ in range(4)] == expected_python


def test_resume_snapshot_retention_and_identity_are_fail_closed(tmp_path) -> None:
    model, optimizer = _trained_linear()
    root = tmp_path / "resume"
    identity = {"dataset": "one"}
    for step in range(3):
        save_m2_resume_checkpoint(
            root,
            model=model,
            optimizer=optimizer,
            identity=identity,
            progress={"global_step": step},
            coverage_seen=torch.ones(2, dtype=torch.bool),
            keep_last=2,
        )
    generations = [
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")
    ]
    assert len(generations) == 2
    with pytest.raises(ValueError, match="identity differs"):
        load_latest_m2_resume_checkpoint(
            root,
            model=model,
            optimizer=optimizer,
            expected_identity={"dataset": "two"},
            device="cpu",
        )


def test_sampler_can_resume_at_a_batch_aligned_local_offset() -> None:
    class Dataset:
        datasets = (range(5), range(7))
        _offsets = (0, 5, 12)

        def __len__(self) -> int:
            return 12

    dataset = Dataset()
    sampler = CoverageTemperatureDistributedSampler(
        dataset,  # type: ignore[arg-type]
        samples_per_epoch=12,
        coverage_epochs=1,
        seed=4,
    )
    complete = list(sampler)
    sampler.set_epoch(0, start_offset=4)
    assert len(sampler) == 8
    assert list(sampler) == complete[4:]
    with pytest.raises(ValueError, match="start_offset"):
        sampler.set_epoch(0, start_offset=13)


def test_native_640x480_training_uses_bounded_host_parallelism() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/wam_multimodal/m2_liftbarrier_longpipeline_joint.yaml"
        ).read_text(encoding="utf-8")
    )
    training = config["training"]
    checkpoint = config["checkpoint"]
    assert config["data"]["image_height"] == 480
    assert config["data"]["image_width"] == 640
    assert training["num_workers"] == 8
    assert training["prefetch_factor"] == 1
    assert checkpoint["interval_steps"] == 100
    assert checkpoint["keep_last"] == 2


def test_native_rgb_loader_memory_guard_rejects_previous_parallelism(
    monkeypatch,
) -> None:
    dataset = SimpleNamespace(
        image_shape_hwc=(480, 640, 3),
        visual_history=4,
        future_horizons=(1, 4, 8, 16, 32),
        camera_order=("global", "agent_0", "agent_1", "agent_2", "agent_3"),
    )
    monkeypatch.setattr(
        train_robofactory_m2,
        "_available_memory_bytes",
        lambda: 64 * 1024**3,
    )
    safe = train_robofactory_m2._loader_memory_plan(
        dataset,  # type: ignore[arg-type]
        training={
            "batch_size": 16,
            "num_workers": 8,
            "prefetch_factor": 1,
            "loader_queue_max_available_fraction": 0.25,
        },
        smoke=False,
    )
    assert safe["resident_batches_upper_bound"] == 10
    with pytest.raises(MemoryError, match="native-RGB loader queues"):
        train_robofactory_m2._loader_memory_plan(
            dataset,  # type: ignore[arg-type]
            training={
                "batch_size": 16,
                "num_workers": 16,
                "prefetch_factor": 4,
                "loader_queue_max_available_fraction": 0.25,
            },
            smoke=False,
        )
