from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION, schema_profile
from models import (
    ActionPrior,
    ActionPriorConfig,
    LinearWorldModel,
    LinearWorldModelConfig,
    PolicyInputs,
    WorldModelInputs,
)
from scripts.collect_wam_proprio_dataset import main as collect_main
from scripts.train_wam_baselines import main as train_main
from train.trajectory_dataset import ProprioSequenceDataset, split_episode_paths


def test_wam_proprio_schema_has_no_images_or_privileged_inputs():
    schema = schema_profile("wam_proprio")
    names = {field.name for field in schema.fields}
    sources = {field.source for field in schema.fields}

    assert schema.version == PROPRIO_WAM_SCHEMA_VERSION
    assert not any(field.is_image for field in schema.fields)
    assert "observation.state" in names
    assert "commanded_action" in names
    assert "executed_action" in names
    assert "next_observation.state" in names
    assert not any("privileged_state" in source for source in sources)


def test_proprio_collection_exports_aligned_actions_and_metadata_without_images(
    tmp_path,
):
    output = tmp_path / "dataset"
    assert (
        collect_main(
            [
                "--out-dir",
                str(output),
                "--episodes",
                "1",
                "--max-steps",
                "3",
                "--no-progress",
            ]
        )
        == 0
    )

    path = output / "hdf5" / "episode_000000.hdf5"
    with h5py.File(path, "r") as file:
        assert file.attrs["schema_version"] == PROPRIO_WAM_SCHEMA_VERSION
        assert file.attrs["behavior_id"] == "scripted_oracle_v1"
        assert file["data/commanded_action"].shape == (3, 8)
        assert file["data/executed_action"].shape == (3, 8)
        assert not np.array_equal(
            file["data/commanded_action"][0], file["data/executed_action"][0]
        )
        all_paths: list[str] = []
        file.visit(all_paths.append)
        assert not any("image" in name for name in all_paths)
        assert not any("privileged" in name for name in all_paths)
        metadata = json.loads(file.attrs["episode_metadata_json"])
        assert json.loads(metadata["randomization_config"])["seed"] == 0


def test_sequence_loader_pads_without_crossing_episode_boundaries(tmp_path):
    paths = [
        _write_episode(tmp_path / "episode_000000.hdf5", seed=11, offset=0.0),
        _write_episode(tmp_path / "episode_000001.hdf5", seed=12, offset=100.0),
    ]
    dataset = ProprioSequenceDataset(
        paths=paths,
        history_horizon=3,
        forecast_horizon=3,
        hdf5_cache_size=1,
    )
    try:
        first = dataset[0]
        assert first["valid_mask"].tolist() == [False, False, True]
        assert first["past_action_mask"].tolist() == [False, False]
        assert first["states"][-1, 0].item() == 0.0
        assert first["target_states"][0, 0].item() == 1.0

        second = dataset[1]
        assert second["valid_mask"].tolist() == [False, True, True]
        assert second["past_action_mask"].tolist() == [False, True]
        assert second["past_actions"][-1, 0].item() == 0.0

        end_of_first_episode = dataset[3]
        assert end_of_first_episode["forecast_mask"].tolist() == [True, False, False]
        assert end_of_first_episode["target_states"][0, 0].item() == 4.0
        assert end_of_first_episode["target_states"][1:].count_nonzero().item() == 0

        start_of_second_episode = dataset[4]
        assert start_of_second_episode["states"][-1, 0].item() == 100.0
        assert start_of_second_episode["states"][:-1].count_nonzero().item() == 0
    finally:
        dataset.close()


def test_episode_split_keeps_duplicate_seeds_together(tmp_path):
    paths = [
        _write_episode(tmp_path / f"episode_{index:06d}.hdf5", seed=seed, offset=index)
        for index, seed in enumerate((4, 4, 5, 6, 7))
    ]
    split = split_episode_paths(paths, seed=3)
    seed_to_partition: dict[int, str] = {}
    for partition, partition_paths in split.items():
        for path in partition_paths:
            with h5py.File(path, "r") as file:
                episode_seed = int(file.attrs["seed"])
            assert seed_to_partition.setdefault(episode_seed, partition) == partition
    assert set().union(*map(set, split.values())) == set(paths)


def test_phase0_models_and_training_entrypoint_smoke(tmp_path):
    linear = LinearWorldModel(LinearWorldModelConfig(state_dim=22, action_dim=8))
    world_output = linear(
        WorldModelInputs(state=torch.zeros(2, 22), action=torch.zeros(2, 8))
    )
    assert world_output.next_state.shape == (2, 22)

    prior = ActionPrior(
        ActionPriorConfig(state_dim=22, action_dim=8, hidden_dim=16, hidden_layers=1)
    )
    prior_output = prior(PolicyInputs(state=torch.zeros(2, 22)))
    assert prior_output.action.shape == (2, 8)

    data_dir = tmp_path / "episodes"
    data_dir.mkdir()
    for index in range(3):
        _write_episode(
            data_dir / f"episode_{index:06d}.hdf5",
            seed=index,
            offset=float(index),
        )
    output_dir = tmp_path / "baselines"
    assert (
        train_main(
            [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--models",
                "linear",
                "mlp",
                "action_prior",
                "--history-horizon",
                "3",
                "--forecast-horizon",
                "2",
                "--hidden-dim",
                "16",
                "--hidden-layers",
                "1",
                "--batch-size",
                "4",
                "--epochs",
                "1",
                "--max-steps",
                "1",
                "--device",
                "cpu",
                "--no-progress",
            ]
        )
        == 0
    )
    metrics = json.loads((output_dir / "baseline_metrics.json").read_text())
    assert set(metrics) == {"linear", "mlp", "action_prior"}
    assert metrics["linear"]["splits"]["train"]["samples"] == 4
    for name in metrics:
        assert (output_dir / f"{name}.safetensors").exists()


def _write_episode(path: Path, *, seed: int, offset: float, steps: int = 4) -> Path:
    state = np.repeat(
        (offset + np.arange(steps, dtype=np.float32))[:, None], 22, axis=1
    )
    next_state = state + 1.0
    commanded = np.repeat(
        np.arange(steps, dtype=np.float32)[:, None], 8, axis=1
    ) / max(steps - 1, 1)
    executed = commanded * 0.5
    with h5py.File(path, "w") as file:
        file.attrs.update(
            {
                "schema_profile": "wam_proprio",
                "schema_version": PROPRIO_WAM_SCHEMA_VERSION,
                "num_steps": steps,
                "seed": seed,
                "episode_index": int(path.stem.rsplit("_", 1)[-1]),
            }
        )
        data = file.create_group("data")
        data.create_dataset("observation/state", data=state)
        data.create_dataset("next_observation/state", data=next_state)
        data.create_dataset("commanded_action", data=commanded)
        data.create_dataset("executed_action", data=executed)
        data.create_dataset("reward", data=np.linspace(0.0, 1.0, steps, dtype=np.float32))
        data.create_dataset("done", data=np.arange(steps) == steps - 1)
        data.create_dataset("success", data=np.arange(steps) == steps - 1)
        data.create_dataset("failure", data=np.zeros(steps, dtype=np.bool_))
        data.create_dataset("response_progress", data=np.zeros(steps, dtype=np.float32))
        data.create_dataset("coordination_error", data=np.zeros(steps, dtype=np.float32))
    return path
