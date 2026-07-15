from __future__ import annotations

import json
from collections import Counter
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import pytest
from rich.console import Console
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
from policies.collection import (
    PHASE0_BEHAVIOR_WEIGHTS,
    CooperativeStopCollectionPolicy,
)
from scripts.audit_phase0_gate_a import main as audit_main
from scripts.collect_modular_dataset import CollectionProgressObserver
from scripts.collect_wam_proprio_dataset import main as collect_main
from scripts.train_phase0_baselines import (
    TrainingProgress,
    _PhaseDisplay,
    _loss_point_chart,
    main as train_main,
)
from train.phase0_baselines import (
    binary_classification_metrics,
    calibrate_binary_threshold,
)
from train.trajectory_dataset import (
    InMemoryOneStepDataset,
    ProprioSequenceDataset,
    split_episode_paths,
)


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


def test_progress_displays_fill_terminal_width_without_wrapping(capsys):
    collection = CollectionProgressObserver(10000)
    collection._update(
        step=125,
        successes=30,
        failures=7,
        phase="failed: response timeout",
    )
    training = TrainingProgress(enabled=True, total_stages=2)
    first_phase = training.add_phase("dataset statistics", 20)
    first_phase.finish("normalization ready")
    second_phase = training.add_phase("train action_prior", 1234)
    second_phase.advance({"epoch": 1, "epochs": 10, "loss": 0.12345})
    second_phase.advance({"epoch": 1, "epochs": 10, "loss": 0.10000})
    second_phase.finish("final loss 0.10000")

    assert first_phase.progress is not second_phase.progress
    assert first_phase.progress.tasks[0].finished
    assert second_phase.progress.tasks[0].finished
    assert len(training._phases) == 2
    assert second_phase.loss_history == [0.12345, 0.1]
    for width in (40, 60, 80, 100, 140):
        collection_output = StringIO()
        collection_console = Console(
            file=collection_output, width=width, color_system=None
        )
        collection_console.print(
            collection._progress.make_tasks_table(collection._progress.tasks)
        )
        collection_lines = collection_output.getvalue().splitlines()
        assert len(collection_lines) == 1
        assert len(collection_lines[0]) == width

        for phase, expected_lines in ((first_phase, 1), (second_phase, 7)):
            output = StringIO()
            console = Console(file=output, width=width, color_system=None)
            console.print(_PhaseDisplay(phase))
            lines = output.getvalue().splitlines()
            assert len(lines) == expected_lines
            assert all(len(line) == width for line in lines)
    assert any("\u2801" <= character <= "\u28ff" for character in output.getvalue())
    assert "completed stage 1/2" not in capsys.readouterr().err


def test_training_completion_is_printed_when_progress_is_disabled(capsys):
    training = TrainingProgress(enabled=False, total_stages=1)
    phase = training.add_phase("train linear", 2)
    phase.finish("final loss 0.12500")

    assert (
        "✓ completed stage 1/1 | train linear | final loss 0.12500"
        in capsys.readouterr().err
    )


def test_loss_chart_keeps_full_step_range_without_outlier_flattening():
    early_outlier = np.asarray([1000.0])
    recent_decline = np.linspace(1.0, 0.5, 1000)
    chart = _loss_point_chart(
        np.concatenate((early_outlier, recent_decline)),
        width=80,
        height=5,
    )
    output = StringIO()
    Console(file=output, width=80, color_system=None).print(chart)
    lines = output.getvalue().splitlines()

    assert len(lines) == 6
    assert all(len(line) == 80 for line in lines)
    assert "1e+03" not in output.getvalue()
    assert "steps 1→1001" in output.getvalue()
    active_rows = sum(
        any("\u2801" <= character <= "\u28ff" for character in line)
        for line in lines[:5]
    )
    assert active_rows >= 4


def test_binary_metrics_and_validation_threshold_calibration():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    logits = np.log(probabilities / (1.0 - probabilities))

    calibration = calibrate_binary_threshold(labels, logits)
    metrics = binary_classification_metrics(
        labels,
        logits,
        threshold=float(calibration["threshold"]),
    )

    assert calibration["status"] == "calibrated"
    assert calibration["threshold"] == pytest.approx(0.35)
    assert calibration["objective_value"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5.0 / 6.0)
    assert metrics["baselines"]["majority_accuracy"] == pytest.approx(0.5)
    assert metrics["threshold_metrics"]["recall"] == pytest.approx(1.0)

    unavailable = calibrate_binary_threshold(np.zeros(4), np.zeros(4))
    assert unavailable["status"] == "unavailable_single_class"
    assert unavailable["threshold"] == 0.5


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
        assert file.attrs["behavior_id"] in dict(PHASE0_BEHAVIOR_WEIGHTS)
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
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["behavior_profile"] == "phase0_mixed_v1"


def test_phase0_mixture_schedule_matches_section_11_2_exactly():
    policy = CooperativeStopCollectionPolicy(
        object(), profile="phase0_mixed_v1", mixture_seed=17
    )
    behaviors = [
        policy.configure_episode(episode_index=index, episode_seed=1000 + index)
        for index in range(100)
    ]
    counts = Counter(item.behavior_id for item in behaviors)

    assert counts == Counter(dict(PHASE0_BEHAVIOR_WEIGHTS))
    for behavior in behaviors:
        json.dumps(behavior.perturbation_config)


def test_gate_a_audit_accepts_complete_mixed_proprio_contract(tmp_path):
    data_dir = tmp_path / "mixed" / "hdf5"
    data_dir.mkdir(parents=True)
    schedule = [
        behavior_id
        for behavior_id, weight in PHASE0_BEHAVIOR_WEIGHTS
        for _ in range(weight)
    ]
    for index, behavior_id in enumerate(schedule):
        _write_episode(
            data_dir / f"episode_{index:06d}.hdf5",
            seed=index,
            offset=float(index),
            steps=2,
            behavior_id=behavior_id,
            success=index % 2 == 0,
        )
    report = tmp_path / "gate_a.json"

    assert (
        audit_main(
            [
                "--data-dir",
                str(data_dir),
                "--report",
                str(report),
                "--expected-episodes",
                "100",
                "--mixture-tolerance",
                "0",
                "--no-progress",
            ]
        )
        == 0
    )
    payload = json.loads(report.read_text())
    assert payload["passed"]
    assert all(check["passed"] for check in payload["checks"].values())


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


def test_in_memory_one_step_loader_matches_sequence_targets(tmp_path):
    paths = [
        _write_episode(tmp_path / "episode_000000.hdf5", seed=11, offset=0.0),
        _write_episode(tmp_path / "episode_000001.hdf5", seed=12, offset=100.0),
    ]
    sequence = ProprioSequenceDataset(
        paths=paths,
        history_horizon=3,
        forecast_horizon=3,
        hdf5_cache_size=1,
    )
    progress: list[dict[str, int]] = []
    cached = InMemoryOneStepDataset(paths=paths, progress=progress.append)
    try:
        assert len(cached) == len(sequence) == 8
        assert cached.num_episodes == 2
        assert cached.nbytes == len(cached) * (22 + 8 + 22 + 1 + 1 + 1 + 1) * 4
        assert progress[-1] == {"episode": 2, "episodes": 2, "samples": 8}
        for index in range(len(sequence)):
            sequence_item = sequence[index]
            cached_item = cached[index]
            torch.testing.assert_close(
                cached_item["states"][-1], sequence_item["states"][-1]
            )
            torch.testing.assert_close(
                cached_item["candidate_actions"][0],
                sequence_item["candidate_actions"][0],
            )
            torch.testing.assert_close(
                cached_item["target_states"][0],
                sequence_item["target_states"][0],
            )
            torch.testing.assert_close(
                cached_item["rewards"][0], sequence_item["rewards"][0]
            )
            torch.testing.assert_close(
                cached_item["dones"][0], sequence_item["dones"][0]
            )
            torch.testing.assert_close(
                cached_item["successes"][0], sequence_item["successes"][0]
            )
            torch.testing.assert_close(
                cached_item["failures"][0], sequence_item["failures"][0]
            )
    finally:
        sequence.close()
        cached.close()


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
    assert world_output.success_logit.shape == (2, 1)
    assert world_output.failure_logit.shape == (2, 1)

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
    assert set(metrics["linear"]["splits"]["test"]["classification"]) == {
        "done",
        "success",
        "failure",
    }
    assert metrics["linear"]["outcome_calibration"]["source_split"] == "validation"
    assert (output_dir / "outcome_label_stats.json").exists()
    assert (output_dir / "linear_outcome_calibration.json").exists()
    assert (output_dir / "mlp_outcome_calibration.json").exists()
    for name in metrics:
        assert (output_dir / f"{name}.safetensors").exists()


def _write_episode(
    path: Path,
    *,
    seed: int,
    offset: float,
    steps: int = 4,
    behavior_id: str = "scripted_oracle_v1",
    success: bool = True,
) -> Path:
    state = np.repeat(
        (offset + np.arange(steps, dtype=np.float32))[:, None], 22, axis=1
    )
    next_state = state + 1.0
    commanded = np.repeat(np.arange(steps, dtype=np.float32)[:, None], 8, axis=1) / max(
        steps - 1, 1
    )
    executed = commanded * 0.5
    with h5py.File(path, "w") as file:
        file.attrs.update(
            {
                "schema_profile": "wam_proprio",
                "schema_version": PROPRIO_WAM_SCHEMA_VERSION,
                "num_steps": steps,
                "seed": seed,
                "episode_index": int(path.stem.rsplit("_", 1)[-1]),
                "behavior_id": behavior_id,
            }
        )
        data = file.create_group("data")
        data.create_dataset("observation/state", data=state)
        data.create_dataset("next_observation/state", data=next_state)
        data.create_dataset("commanded_action", data=commanded)
        data.create_dataset("executed_action", data=executed)
        data.create_dataset(
            "reward", data=np.linspace(0.0, 1.0, steps, dtype=np.float32)
        )
        data.create_dataset("done", data=np.arange(steps) == steps - 1)
        terminal = np.arange(steps) == steps - 1
        data.create_dataset("success", data=terminal & success)
        data.create_dataset("failure", data=terminal & (not success))
        data.create_dataset("response_progress", data=np.zeros(steps, dtype=np.float32))
        data.create_dataset(
            "coordination_error", data=np.zeros(steps, dtype=np.float32)
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for name, value in (
            ("behavior_id", behavior_id),
            ("perturbation_config", "{}"),
            ("environment_config", "{}"),
            ("randomization_config", json.dumps({"seed": seed})),
        ):
            data.create_dataset(
                name,
                data=np.asarray([value] * steps, dtype=object),
                dtype=string_dtype,
            )
    return path
