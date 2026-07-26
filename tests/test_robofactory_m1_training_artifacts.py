from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from models.wam.action_codec import AffineActionCodec, AffineActionCodecConfig
from models.wam.normalizer import NormalizationStats
from train.generic_m1_trajectory_dataset import (
    GENERIC_M1_DATASET_PROTOCOL,
    GenericM1ManifestIndex,
    GenericM1WindowDataset,
)
from train.m1_data_protocol import (
    build_m1_window_dataset,
    load_m1_data_manifest,
    m1_data_capabilities,
)
from train.robofactory_m1_training_artifacts import (
    TRAINING_MANIFEST_VERSION,
    prepare_robofactory_m1_training_artifacts,
)


def test_generic_m1_loader_consumes_training_manifest_without_cue_fields(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    artifacts = _prepare(
        root,
        transition_selection="through-first-done-inclusive",
    )

    manifest = load_m1_data_manifest(artifacts.manifest_path)
    assert isinstance(manifest, GenericM1ManifestIndex)
    assert manifest.state_dim == 2
    assert manifest.action_dim == 1
    assert manifest.camera_order == ("global",)
    assert manifest.task_order == ("lift_barrier",)
    assert manifest.normalization_verified
    capabilities = m1_data_capabilities(manifest)
    assert capabilities.dataset_protocol == GENERIC_M1_DATASET_PROTOCOL
    assert not capabilities.causal_pairs
    assert not capabilities.event_probe_labels
    assert not capabilities.decision_window_sampling

    dataset = build_m1_window_dataset(
        manifest,
        split="train",
        state_history=4,
        action_chunk=1,
        visual_history=1,
        future_horizons=(1,),
        hdf5_cache_size=1,
    )
    assert isinstance(dataset, GenericM1WindowDataset)
    try:
        # Four train episodes, each selected through its first done at row 1.
        assert len(dataset) == 8
        sample = dataset[0]
        assert frozenset(sample) == GenericM1WindowDataset.SAMPLE_KEYS
        assert sample["states"].shape == (4, 2)
        assert sample["state_valid_mask"].tolist() == [False, False, False, True]
        assert sample["past_actions"].shape == (3, 1)
        assert sample["past_action_valid_mask"].tolist() == [False, False, False]
        assert sample["images"].shape == (1, 1, 3, 2, 3)
        assert sample["image_valid_mask"].tolist() == [[True]]
        assert sample["action_targets"].shape == (1, 1)
        assert sample["future_states"].shape == (1, 2)
        assert sample["future_images"].shape == (1, 1, 3, 2, 3)
        assert sample["future_image_novelty_mask"].tolist() == [[True]]
        assert sample["future_horizons"].tolist() == [1]
        assert sample["task_index"].item() == 0

        second = dataset[1]
        assert second["past_action_valid_mask"].tolist() == [False, False, True]
        assert second["past_actions"][-1].item() == pytest.approx(
            sample["action_targets"][0].item()
        )
        assert dataset.sample_lineage(0).decision_t == 0
        assert dataset.sample_lineage(1).decision_t == 1
        assert dataset.decision_window_indices == ()
        assert dataset.observationally_ambiguous_window_indices == ()
        assert dataset.sampling_weights().sum().item() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="no decision windows"):
            dataset.sampling_weights(decision_window_boost=2.0)

        summary = dataset.window_summary()
        assert summary["dataset_protocol"] == GENERIC_M1_DATASET_PROTOCOL
        assert summary["causal_pairs"] == "not_supported"
        assert summary["transition_selection"] == "through-first-done-inclusive"
        checkpoint = dataset.checkpoint_lineage()
        assert checkpoint["manifest_sha256"] == artifacts.manifest_sha256
        assert checkpoint["normalization_verified"] is True
    finally:
        dataset.close()


def test_generic_m1_tail_windows_repeat_pad_without_supervising_padding(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    artifacts = _prepare(
        root,
        transition_selection="through-first-done-inclusive",
    )
    dataset = GenericM1WindowDataset(
        artifacts.manifest_path,
        split="train",
        state_history=4,
        action_chunk=2,
        visual_history=1,
        future_horizons=(1, 2),
        allow_incomplete_horizon=True,
    )
    try:
        # Four train episodes have two selected transitions each.  The final
        # decision remains a training context instead of being dropped merely
        # because fewer than two future transitions remain.
        assert len(dataset) == 8
        tail = dataset[1]
        assert frozenset(tail) == GenericM1WindowDataset.AVAILABLE_SAMPLE_KEYS
        assert dataset.sample_lineage(1).decision_t == 1
        assert tail["action_target_valid_mask"].tolist() == [True, False]
        assert tail["future_state_valid_mask"].tolist() == [True, False]
        assert tail["future_visual_valid_mask"].tolist() == [True, False]
        torch.testing.assert_close(
            tail["action_targets"][0], tail["action_targets"][1]
        )
        torch.testing.assert_close(
            tail["future_states"][0], tail["future_states"][1]
        )
    finally:
        dataset.close()


def test_generic_m1_prefix_windows_right_align_short_visual_history(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    artifacts = _prepare(
        root,
        transition_selection="through-first-done-inclusive",
    )
    dataset = GenericM1WindowDataset(
        artifacts.manifest_path,
        split="train",
        state_history=4,
        action_chunk=2,
        visual_history=2,
        future_horizons=(1, 2),
        allow_incomplete_horizon=True,
        allow_incomplete_visual_history=True,
    )
    try:
        assert len(dataset) == 8
        reset = dataset[0]
        assert dataset.sample_lineage(0).decision_t == 0
        assert reset["images"].shape == (2, 1, 3, 2, 3)
        assert reset["image_valid_mask"].tolist() == [
            [False],
            [True],
        ]
        assert bool(reset["images"][0].eq(0).all())
        second = dataset[1]
        assert second["image_valid_mask"].tolist() == [
            [True],
            [True],
        ]
    finally:
        dataset.close()


def test_generic_m1_ram_preload_matches_hdf5_and_is_worker_serializable(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    artifacts = _prepare(
        root,
        transition_selection="through-first-done-inclusive",
    )
    manifest = GenericM1ManifestIndex.from_path(artifacts.manifest_path)
    dataset = GenericM1WindowDataset(
        manifest,
        split="train",
        state_history=4,
        action_chunk=1,
        visual_history=1,
        future_horizons=(1,),
    )
    try:
        baseline = {name: value.clone() for name, value in dataset[0].items()}
        callbacks: list[tuple[int, int, int]] = []
        estimate = dataset.estimate_ram_preload_bytes()
        report = dataset.preload_to_ram(
            shared_memory=True,
            progress_callback=lambda current, total, loaded: callbacks.append(
                (current, total, loaded)
            ),
        )
        assert report["enabled"] is True
        assert report["shared_memory"] is True
        assert report["episodes"] == 4
        assert report["bytes"] == estimate
        assert callbacks[0] == (0, 4, 0)
        assert callbacks[-1] == (4, 4, estimate)
        cached = dataset[0]
        assert cached.keys() == baseline.keys()
        for name, expected in baseline.items():
            torch.testing.assert_close(cached[name], expected)

        reloaded = pickle.loads(pickle.dumps(dataset))
        try:
            for name, expected in baseline.items():
                torch.testing.assert_close(reloaded[0][name], expected)
        finally:
            reloaded.close()
            reloaded.clear_ram_preload()

    finally:
        dataset.close()
        dataset.clear_ram_preload()


def test_training_artifacts_are_seed_disjoint_train_only_and_deterministic(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    artifacts = _prepare(root, transition_selection="through-first-done-inclusive")
    manifest = artifacts.manifest

    assert manifest["format_version"] == TRAINING_MANIFEST_VERSION
    assert manifest["dataset_protocol"] == "generic_multimodal_trajectory"
    assert {
        name: values["episodes"]
        for name, values in manifest["split_counts"].items()
    } == {"train": 4, "validation": 1, "test": 1}
    assert {
        name: values["selected_transitions"]
        for name, values in manifest["split_counts"].items()
    } == {"train": 8, "validation": 2, "test": 2}
    assert manifest["transition_selection"] == {
        "mode": "through-first-done-inclusive",
        "terminal_field": "data/done",
        "includes_first_terminal_transition": True,
        "recorded_transitions": 18,
        "selected_transitions": 12,
        "post_first_done_transitions": 6,
        "excluded_post_first_done_transitions": 6,
    }
    assert manifest["normalization"]["source_split"] == "train"
    assert manifest["normalization"]["transition_count"] == 8
    assert manifest["normalization"]["sample_unit"] == "selected_raw_transition"
    assert manifest["normalization"]["action_domain"] == (
        "raw_pd_joint_pos_commanded"
    )
    assert manifest["normalization"]["reward"] == {
        "available": False,
        "placeholder": True,
        "usage": "unused_model_compatibility_only",
        "mean": [0.0],
        "std": [1.0],
    }

    partitions = {
        name: {
            int(entry["seed"])
            for entry in manifest["episodes"]
            if entry["split"] == name
        }
        for name in ("train", "validation", "test")
    }
    assert not partitions["train"] & partitions["validation"]
    assert not partitions["train"] & partitions["test"]
    assert not partitions["validation"] & partitions["test"]
    assert sum(len(values) for values in partitions.values()) == 6

    for entry in manifest["episodes"]:
        relative = Path(entry["hdf5_path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert entry["hdf5_sha256"] == _sha256(root / relative)
        assert entry["recorded_steps"] == 3
        assert entry["steps"] == 2

    stats = NormalizationStats.load(artifacts.normalization_path)
    train_entries = [
        entry for entry in manifest["episodes"] if entry["split"] == "train"
    ]
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    for entry in train_entries:
        with h5py.File(root / entry["hdf5_path"], "r") as file:
            stop = int(entry["steps"])
            current = file["data/observation/state"][:stop]
            states.append(current)
            actions.append(file["data/action/commanded"][:stop])
            deltas.append(file["data/next_observation/state"][:stop] - current)
    expected_state = np.concatenate(states)
    expected_action = np.concatenate(actions)
    expected_delta = np.concatenate(deltas)
    np.testing.assert_allclose(stats.state_mean, expected_state.mean(axis=0))
    np.testing.assert_allclose(
        stats.state_std,
        np.maximum(expected_state.std(axis=0), 1e-3),
    )
    np.testing.assert_allclose(stats.action_mean, expected_action.mean(axis=0))
    np.testing.assert_allclose(
        stats.action_std,
        np.maximum(expected_action.std(axis=0), 1e-3),
    )
    np.testing.assert_allclose(stats.delta_mean, expected_delta.mean(axis=0))
    np.testing.assert_allclose(
        stats.delta_std,
        np.maximum(expected_delta.std(axis=0), 1e-3),
    )
    assert stats.state_std[1] == pytest.approx(1e-3)
    assert stats.delta_std[1] == pytest.approx(1e-3)
    assert artifacts.normalization_semantic_sha256 == stats.sha256()
    assert artifacts.normalization_file_sha256 == _sha256(
        artifacts.normalization_path
    )
    assert artifacts.manifest_sha256 == _sha256(artifacts.manifest_path)
    assert (
        artifacts.manifest_path.with_suffix(".json.sha256").read_text(
            encoding="utf-8"
        )
        == f"{artifacts.manifest_sha256}  training_manifest.json\n"
    )

    first_assignment = manifest["split_protocol"]["assignment_sha256"]
    first_semantic_hash = artifacts.normalization_semantic_sha256
    repeated = _prepare(
        root,
        transition_selection="through-first-done-inclusive",
        overwrite=True,
    )
    assert repeated.manifest["split_protocol"]["assignment_sha256"] == (
        first_assignment
    )
    assert repeated.normalization_semantic_sha256 == first_semantic_hash
    assert repeated.manifest_sha256 == artifacts.manifest_sha256


def test_training_artifacts_apply_one_codec_to_stats_and_windows(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    codec_config = AffineActionCodecConfig(
        codec_id="test.lift-barrier-action/1",
        low=(-10.0,),
        high=(100.0,),
        raw_domain="raw_pd_joint_pos_commanded",
    )
    artifacts = _prepare(
        root,
        transition_selection="through-first-done-inclusive",
        action_codec=codec_config,
    )
    action = artifacts.manifest["action"]
    assert action["storage_domain"] == codec_config.raw_domain
    assert action["domain"] == codec_config.encoded_domain
    assert action["codec"]["applied"] is True
    assert action["codec"]["semantic_sha256"] == codec_config.sha256()
    assert artifacts.manifest["normalization"]["action_domain"] == (
        codec_config.encoded_domain
    )
    assert artifacts.manifest["normalization"]["sample_unit"] == (
        "selected_transition_after_action_codec"
    )

    index = GenericM1ManifestIndex.from_path(artifacts.manifest_path)
    assert index.action_codec is not None
    dataset = GenericM1WindowDataset(
        index,
        split="train",
        state_history=4,
        action_chunk=1,
        visual_history=1,
        future_horizons=(1,),
    )
    try:
        lineage = dataset.sample_lineage(0)
        with h5py.File(lineage.path, "r") as file:
            raw = np.asarray(
                file["data/action/commanded"][lineage.decision_t : lineage.decision_t + 1],
                dtype=np.float32,
            )
        expected = AffineActionCodec(codec_config).encode(raw)
        np.testing.assert_allclose(dataset[0]["action_targets"].numpy(), expected)
    finally:
        dataset.close()


def test_transition_selection_and_manifest_input_order_are_explicit(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    through = _prepare(root, transition_selection="through-first-done-inclusive")

    conversion = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    conversion["episodes"].reverse()
    reversed_path = root / "manifest_reversed.json"
    reversed_path.write_text(
        json.dumps(conversion, indent=2, sort_keys=True), encoding="utf-8"
    )
    reversed_artifacts = _prepare(
        root,
        transition_selection="through-first-done-inclusive",
        conversion_manifest_path="manifest_reversed.json",
        training_manifest_path="training_manifest_reversed.json",
        normalization_path="normalization_reversed.npz",
    )
    assert reversed_artifacts.manifest["split_protocol"]["assignment_sha256"] == (
        through.manifest["split_protocol"]["assignment_sha256"]
    )
    assert reversed_artifacts.normalization_semantic_sha256 == (
        through.normalization_semantic_sha256
    )

    all_recorded = _prepare(
        root,
        transition_selection="all-recorded",
        training_manifest_path="training_manifest_all.json",
        normalization_path="normalization_all.npz",
    )
    assert all_recorded.manifest["totals"]["selected_transitions"] == 18
    assert all_recorded.manifest["normalization"]["transition_count"] == 12
    assert all_recorded.manifest["transition_selection"][
        "excluded_post_first_done_transitions"
    ] == 0
    assert all_recorded.normalization_semantic_sha256 != (
        through.normalization_semantic_sha256
    )


def test_training_artifacts_fail_closed_on_duplicate_episode_seed(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", episode_count=6)
    conversion_path = root / "manifest.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    duplicate_seed = int(conversion["episodes"][0]["seed"])
    conversion["episodes"][1]["seed"] = duplicate_seed
    conversion_path.write_text(
        json.dumps(conversion, indent=2, sort_keys=True), encoding="utf-8"
    )
    with h5py.File(root / "hdf5/episode_000001.hdf5", "r+") as file:
        file.attrs["seed"] = duplicate_seed
        file["data/seed"][:] = duplicate_seed

    with pytest.raises(ValueError, match="seed values must be globally unique"):
        _prepare(root, transition_selection="through-first-done-inclusive")


def _prepare(
    root: Path,
    *,
    transition_selection: str,
    conversion_manifest_path: str = "manifest.json",
    training_manifest_path: str = "training_manifest.json",
    normalization_path: str = "normalization.npz",
    action_codec=None,
    overwrite: bool = False,
):
    return prepare_robofactory_m1_training_artifacts(
        root,
        transition_selection=transition_selection,
        conversion_manifest_path=conversion_manifest_path,
        training_manifest_path=training_manifest_path,
        normalization_path=normalization_path,
        split_seed=7,
        expected_episodes=6,
        expected_state_dim=2,
        expected_action_dim=1,
        expected_task_id="lift_barrier",
        expected_cameras=("global",),
        expected_fps=20.0,
        action_codec=action_codec,
        overwrite=overwrite,
    )


def _write_dataset(root: Path, *, episode_count: int) -> Path:
    hdf5_dir = root / "hdf5"
    hdf5_dir.mkdir(parents=True)
    episodes: list[dict[str, object]] = []
    for episode_index in range(episode_count):
        seed = 100 + episode_index
        path = hdf5_dir / f"episode_{episode_index:06d}.hdf5"
        _write_episode(path, episode_index=episode_index, seed=seed)
        episodes.append(
            {
                "episode_index": episode_index,
                "source_episode_id": 900 + episode_index,
                "seed": seed,
                "steps": 3,
                "success": True,
                "terminated": True,
                "truncated": False,
            }
        )
    manifest = {
        "format_version": "robofactory.conversion_manifest/2.0",
        "schema_profile": "robofactory_m1",
        "schema_version": "wam.robofactory.multimodal/1.0",
        "fps": 20.0,
        "task_id": "lift_barrier",
        "task": "Lift the barrier together",
        "source": {
            "hdf5_sha256": "a" * 64,
            "metadata_json_sha256": "b" * 64,
            "metadata": {"source_type": "motionplanning"},
        },
        "layout": {
            "state_size": 2,
            "action_size": 1,
            "exported_cameras": [{"target_name": "global"}],
        },
        "data_semantics": {
            "action": {
                "commanded_field": "action.commanded",
                "executed_action_source": "command_echo",
                "independent_actuator_feedback_available": False,
            },
            "timing": {"image_hz": 20.0},
        },
        "field_mapping": {
            "centralized_state": [
                {
                    "source": "synthetic/state",
                    "target": "observation.state",
                    "slice": [0, 2],
                }
            ],
            "centralized_action": [
                {
                    "source": "synthetic/action",
                    "target": "action.commanded",
                    "slice": [0, 1],
                }
            ],
        },
        "episodes": episodes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def _write_episode(path: Path, *, episode_index: int, seed: int) -> None:
    steps = 3
    full_state = np.stack(
        (
            episode_index * 100.0 + np.arange(steps + 1, dtype=np.float32),
            np.full(steps + 1, 5.0, dtype=np.float32),
        ),
        axis=-1,
    )
    action = (
        episode_index * 10.0 + np.arange(steps, dtype=np.float32)
    ).reshape(steps, 1)
    done = np.asarray([False, True, True], dtype=np.bool_)
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as file:
        file.attrs["schema_profile"] = "robofactory_m1"
        file.attrs["schema_version"] = "wam.robofactory.multimodal/1.0"
        file.attrs["format_version"] = "wam.trajectory.hdf5/1"
        file.attrs["episode_index"] = episode_index
        file.attrs["seed"] = seed
        file.attrs["num_steps"] = steps
        file.attrs["task_id"] = "lift_barrier"
        file.attrs["task"] = "Lift the barrier together"
        file.attrs["fps"] = 20.0
        file.attrs["camera_order_json"] = '["global"]'
        file.create_dataset("data/observation/state", data=full_state[:-1])
        file.create_dataset("data/next_observation/state", data=full_state[1:])
        file.create_dataset("data/action/commanded", data=action)
        file.create_dataset("data/action/executed", data=action)
        file.create_dataset(
            "data/timestamp", data=np.arange(steps, dtype=np.float64) / 20.0
        )
        file.create_dataset("data/frame_index", data=np.arange(steps, dtype=np.int64))
        file.create_dataset(
            "data/episode_index",
            data=np.full(steps, episode_index, dtype=np.int64),
        )
        file.create_dataset("data/seed", data=np.full(steps, seed, dtype=np.int64))
        file.create_dataset(
            "data/task/id",
            data=np.asarray(["lift_barrier"] * steps, dtype=object),
            dtype=strings,
        )
        file.create_dataset(
            "data/task/text",
            data=np.asarray(["Lift the barrier together"] * steps, dtype=object),
            dtype=strings,
        )
        file.create_dataset("data/terminated", data=done)
        file.create_dataset("data/truncated", data=np.zeros(steps, dtype=np.bool_))
        file.create_dataset("data/done", data=done)
        file.create_dataset("data/success", data=done)
        image = np.full((steps, 2, 3, 3), episode_index + 1, dtype=np.uint8)
        file.create_dataset("data/observation/images/global", data=image)
        file.create_dataset("data/next_observation/images/global", data=image + 1)
        file.create_dataset(
            "data/observation/image_frame_index/global",
            data=np.arange(steps, dtype=np.int64),
        )
        file.create_dataset(
            "data/next_observation/image_frame_index/global",
            data=np.arange(1, steps + 1, dtype=np.int64),
        )
        file.create_dataset(
            "data/observation/image_timestamp/global",
            data=np.arange(steps, dtype=np.float64) / 20.0,
        )
        file.create_dataset(
            "data/next_observation/image_timestamp/global",
            data=np.arange(1, steps + 1, dtype=np.float64) / 20.0,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
