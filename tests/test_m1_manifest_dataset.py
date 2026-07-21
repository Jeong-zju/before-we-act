from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from train import m1_manifest_dataset as manifest_dataset_module
from train.m1_manifest_dataset import (
    CANONICAL_CAMERA_ORDER,
    M1CausalPairDataset,
    M1ManifestIndex,
    M1StateCausalPairDataset,
    M1WindowDataset,
    load_m1_manifest,
)


TASKS = ("visual_target_select", "visual_event_stop")
SPLITS = ("train", "validation", "test")


def test_manifest_audits_pairs_splits_hdf5_and_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)
    manifest = load_m1_manifest(manifest_path)

    assert manifest.camera_order == CANONICAL_CAMERA_ORDER
    assert manifest.task_order == TASKS
    assert dict(manifest.task_to_index) == {
        "visual_target_select": 0,
        "visual_event_stop": 1,
    }
    assert manifest.hdf5_sha256_verified
    assert manifest.hdf5_contract_verified

    summary = manifest.split_summary("train")
    assert summary["episodes"] == 4
    assert summary["transitions"] == 56
    assert summary["physical_seed_pairs"] == 2
    assert summary["cue_counts"] == {"0": 2, "1": 2}
    assert summary["task_to_index"] == dict(manifest.task_to_index)
    assert summary["task_counts"]["visual_target_select"] == {
        "episodes": 2,
        "transitions": 28,
        "physical_seed_pairs": 1,
    }
    assert len(summary["episode_lineage_sha256"]) == 64
    assert manifest.split_summary("val") == manifest.split_summary("validation")
    assert manifest.split_summary_sha256("train") == manifest.split_summary_sha256(
        "train"
    )

    lineage = manifest.checkpoint_lineage("train")
    assert lineage["manifest_sha256"] == _sha256(manifest_path)
    assert lineage["split_summary_sha256"] == manifest.split_summary_sha256("train")
    assert lineage["hdf5_sha256_verified"] is True
    assert lineage["hdf5_contract_verified"] is True


def test_m1_windows_are_complete_causal_compact_and_leakage_free(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)
    manifest = M1ManifestIndex.from_path(manifest_path)
    dataset = M1WindowDataset(
        manifest,
        split="train",
        state_history=6,
        action_chunk=8,
        visual_history=2,
        future_horizons=(1, 2, 4, 8),
        hdf5_cache_size=1,
    )
    try:
        # Four 14-step episodes, decisions t=2..6: two distinct past RGB
        # captures and a complete H=8 future are both mandatory.
        assert len(dataset) == 20
        assert dataset.sample_lineage(0).decision_t == 2
        assert dataset.sample_lineage(4).decision_t == 6
        assert dataset.sample_lineage(5).decision_t == 2

        sample = dataset[0]
        assert frozenset(sample) == M1WindowDataset.SAMPLE_KEYS
        assert not any(
            forbidden in key.lower()
            for key in sample
            for forbidden in ("event", "cue", "seed", "privileged", "lineage", "probe")
        )
        assert sample["states"].shape == (6, 22)
        assert sample["state_valid_mask"].tolist() == [
            False,
            False,
            False,
            True,
            True,
            True,
        ]
        assert sample["past_actions"].shape == (5, 8)
        assert sample["past_action_valid_mask"].tolist() == [
            False,
            False,
            False,
            True,
            True,
        ]
        assert sample["past_actions"][-2:, 0].tolist() == pytest.approx([10.0, 11.0])
        assert sample["images"].shape == (2, 1, 3, 6, 8)
        assert sample["images"].dtype == torch.uint8
        # t=2 has sample-held frame IDs [0,0,1]; only rows 1 and 2 survive.
        assert sample["images"][:, 0, 0, 0, 0].tolist() == [0, 10]
        assert sample["task_index"].item() == 0
        assert sample["action_targets"].shape == (8, 8)
        assert sample["action_targets"][:, 0].tolist() == pytest.approx(
            [102.0 + row for row in range(8)]
        )
        assert sample["future_states"].shape == (8, 22)
        assert sample["future_states"][:, 0].tolist() == pytest.approx(
            [3.0 + row for row in range(8)]
        )
        assert sample["future_images"].shape == (4, 1, 3, 6, 8)
        assert sample["future_horizons"].tolist() == [1, 2, 4, 8]
        assert sample["future_images"][:, 0, 0, 0, 0].tolist() == [10, 20, 30, 50]
        assert sample["future_image_novelty_mask"][:, 0].tolist() == [
            False,
            True,
            True,
            True,
        ]

        labels = dataset.probe_labels(0)
        assert set(labels) == {"h8_center_xy", "h8_event_active"}
        assert labels["h8_center_xy"].tolist() == pytest.approx([11.0, 11.25])
        assert labels["h8_event_active"].item() is True
        assert dataset.sample_lineage(0).path.name == "episode_000000.hdf5"

        checkpoint = dataset.checkpoint_lineage()
        assert checkpoint["window_summary"]["sample_keys"] == sorted(sample)
        assert checkpoint["window_summary"]["windows"] == 20
        assert len(checkpoint["window_summary_sha256"]) == 64
    finally:
        dataset.close()


def test_camera_subset_order_task_balancing_and_decision_sampler(
    tmp_path: Path,
) -> None:
    manifest = load_m1_manifest(_write_manifest_fixture(tmp_path))
    dataset = M1WindowDataset(
        manifest,
        split="train",
        state_history=4,
        cameras=CANONICAL_CAMERA_ORDER,
        decision_window_radius=0,
        hdf5_cache_size=0,
    )
    try:
        sample = dataset[0]
        assert sample["images"].shape == (2, 3, 3, 6, 8)
        assert sample["future_images"].shape == (4, 3, 3, 6, 8)
        assert sample["images"][0, :, 0, 0, 0].tolist() == [0, 1, 2]
        # Event onset step 4 maps to transition decision row 3.
        assert len(dataset.decision_window_indices) == 4
        assert {
            dataset.sample_lineage(index).decision_t
            for index in dataset.decision_window_indices
        } == {3}

        weights = dataset.sampling_weights(decision_window_boost=3.0)
        assert weights.dtype == torch.float64
        assert weights.sum().item() == pytest.approx(1.0)
        task_mass = {task: 0.0 for task in TASKS}
        for index, weight in enumerate(weights.tolist()):
            # Resolve via the public task index without relying on cue/seed labels.
            task = TASKS[dataset[index]["task_index"].item()]
            task_mass[task] += weight
        assert task_mass == pytest.approx(
            {"visual_target_select": 0.5, "visual_event_stop": 0.5}
        )
        decision_weight = weights[dataset.decision_window_indices[0]].item()
        ordinary_index = next(
            index
            for index in range(len(dataset))
            if index not in set(dataset.decision_window_indices)
            and dataset[index]["task_index"].item()
            == dataset[dataset.decision_window_indices[0]]["task_index"].item()
        )
        assert decision_weight > weights[ordinary_index].item()

        first = list(dataset.make_weighted_sampler(num_samples=12, seed=7))
        second = list(dataset.make_weighted_sampler(num_samples=12, seed=7))
        assert first == second
    finally:
        dataset.close()

    with pytest.raises(ValueError, match="canonical manifest order"):
        M1WindowDataset(
            manifest,
            split="train",
            cameras=("robot_1_camera", "fixed"),
        )


def test_m1_window_projection_is_exact_and_skips_unrequested_rgb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_m1_manifest(_write_manifest_fixture(tmp_path))
    dataset = M1WindowDataset(
        manifest,
        split="train",
        state_history=6,
        hdf5_cache_size=1,
    )
    try:
        full = dataset[0]
        rgb_reads: list[str] = []
        original = manifest_dataset_module._read_rgb_rows

        def tracked_rgb_rows(*args: object, **kwargs: object) -> np.ndarray:
            rgb_reads.append(str(kwargs["prefix"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            manifest_dataset_module,
            "_read_rgb_rows",
            tracked_rgb_rows,
        )

        state_keys = frozenset(
            {
                "states",
                "state_valid_mask",
                "past_actions",
                "task_index",
                "action_targets",
                "future_horizons",
            }
        )
        state_sample = dataset.project(state_keys)[0]
        assert frozenset(state_sample) == state_keys
        assert rgb_reads == []
        for name in state_keys:
            assert torch.equal(state_sample[name], full[name]), name

        vision_keys = frozenset({"images", "task_index", "action_targets"})
        vision_sample = dataset.project(vision_keys)[0]
        assert rgb_reads == ["data/observation/images"]
        for name in vision_keys:
            assert torch.equal(vision_sample[name], full[name]), name

        future_keys = vision_keys.union({"future_images", "future_image_novelty_mask"})
        future_sample = dataset.project(future_keys)[0]
        assert rgb_reads[-2:] == [
            "data/observation/images",
            "data/next_observation/images",
        ]
        for name in future_keys:
            assert torch.equal(future_sample[name], full[name]), name

        with pytest.raises(ValueError, match="cannot be empty"):
            dataset.project(())
        with pytest.raises(ValueError, match="unknown M1 projected"):
            dataset.project(("privileged_cue",))
    finally:
        dataset.close()


def test_m1_window_projection_supports_persistent_hdf5_workers(
    tmp_path: Path,
) -> None:
    manifest = load_m1_manifest(_write_manifest_fixture(tmp_path))
    dataset = M1WindowDataset(
        manifest,
        split="train",
        state_history=6,
        hdf5_cache_size=1,
    )
    loader = DataLoader(
        dataset.project(("task_index", "action_targets")),
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    try:
        first = next(iter(loader))
        second = next(iter(loader))
        assert set(first) == set(second) == {"task_index", "action_targets"}
        for name in first:
            assert torch.equal(first[name], second[name]), name
    finally:
        del loader
        dataset.close()


def test_causal_pair_dataset_finds_observable_anchors_without_cue_inputs(
    tmp_path: Path,
) -> None:
    manifest = load_m1_manifest(
        _write_manifest_fixture(tmp_path, causal_pair_signals=True)
    )
    dataset = M1CausalPairDataset(
        manifest,
        split="train",
        state_history=6,
        hdf5_cache_size=1,
    )
    try:
        assert len(dataset) == 2
        samples = [dataset[index] for index in range(len(dataset))]
        by_task = {TASKS[sample["task_index"][0].item()]: sample for sample in samples}
        target = by_task["visual_target_select"]
        event = by_task["visual_event_stop"]

        assert frozenset(target) == M1CausalPairDataset.SAMPLE_KEYS
        assert target["states"].shape == (2, 6, 22)
        assert target["state_valid_mask"].shape == (2, 6)
        assert target["past_actions"].shape == (2, 5, 8)
        assert target["images"].shape == (2, 2, 1, 3, 6, 8)
        assert target["image_valid_mask"][:, :, 0].tolist() == [
            [True, False],
            [True, False],
        ]
        assert torch.count_nonzero(target["images"][:, 1]).item() == 0
        assert event["image_valid_mask"][:, :, 0].tolist() == [
            [True, True],
            [True, True],
        ]
        assert torch.equal(target["states"][0], target["states"][1])
        assert torch.equal(target["past_actions"][0], target["past_actions"][1])
        assert not torch.equal(target["images"][0], target["images"][1])
        assert not torch.equal(
            target["action_targets"][0, :, 0],
            target["action_targets"][1, :, 0],
        )
        assert target["task_index"].tolist() == [0, 0]
        assert all(
            isinstance(sample_id, str) and len(sample_id) == 64
            for sample_id in target["audit_sample_ids"]
        )
        assert not any(
            key in target
            for key in ("pair_id", "physical_seed", "cue_id", "decision_t")
        )

        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        assert batch["states"].shape == (2, 2, 6, 22)
        assert batch["images"].shape == (2, 2, 2, 1, 3, 6, 8)
        assert batch["image_valid_mask"].shape == (2, 2, 2, 1)

        summary = dataset.pair_summary()
        assert summary["pairs"] == 2
        assert summary["branch_samples"] == 4
        assert summary["single_effective_frame_pairs"] == 1
        assert summary["two_effective_frame_pairs"] == 1
        assert summary["visual_history_alignment"] == "deployable_prefix_right_padding"
        assert summary["anchor_t_by_task"] == {
            "visual_target_select": {"0": 1},
            "visual_event_stop": {"4": 1},
        }
        assert len(summary["audit_sample_ids_sha256"]) == 64
        lineage = dataset.checkpoint_lineage()
        assert lineage["causal_pair_summary"] == summary
        assert len(lineage["causal_pair_summary_sha256"]) == 64
    finally:
        dataset.close()


def test_state_causal_pairs_isolate_lateral_feedback_without_labels(
    tmp_path: Path,
) -> None:
    manifest = load_m1_manifest(
        _write_manifest_fixture(tmp_path, state_causal_pair_signals=True)
    )
    dataset = M1StateCausalPairDataset(
        manifest,
        split="train",
        hdf5_cache_size=1,
    )
    try:
        # Both cue episodes of the target task qualify.  The event task has an
        # H2 look-ahead difference but no current lateral feedback delta, so it
        # is rejected without consulting its task or event labels.
        assert len(dataset) == 2
        samples = [dataset[index] for index in range(len(dataset))]
        assert all(
            TASKS[sample["task_index"][0].item()] == "visual_target_select"
            for sample in samples
        )
        sample = samples[0]
        assert frozenset(sample) == M1StateCausalPairDataset.SAMPLE_KEYS
        assert sample["states"].shape == (2, 32, 22)
        assert sample["state_valid_mask"].shape == (2, 32)
        assert sample["state_valid_mask"].sum(dim=1).tolist() == [4, 4]
        assert not sample["state_valid_mask"][:, :-4].any()
        assert sample["state_valid_mask"][:, -4:].all()
        assert torch.count_nonzero(sample["states"][:, :-4]).item() == 0
        assert not torch.equal(sample["states"][0], sample["states"][1])

        assert sample["past_actions"].shape == (2, 31, 8)
        assert sample["past_action_valid_mask"].sum(dim=1).tolist() == [3, 3]
        assert not sample["past_action_valid_mask"][:, :-3].any()
        assert sample["past_action_valid_mask"][:, -3:].all()
        assert torch.count_nonzero(sample["past_actions"][:, :-3]).item() == 0
        assert torch.equal(sample["past_actions"][0], sample["past_actions"][1])

        assert sample["images"].shape == (2, 2, 1, 3, 6, 8)
        assert sample["image_valid_mask"].all()
        assert torch.equal(sample["images"][0], sample["images"][1])
        execute_delta = sample["action_targets"][1, 0] - sample["action_targets"][0, 0]
        assert execute_delta[[0, 4]].tolist() == pytest.approx([-1.0, -1.0])
        assert torch.count_nonzero(execute_delta[[1, 2, 3, 5, 6, 7]]).item() == 0
        robot_x_delta = (
            sample["states"][1, -1, [0, 11]].mean()
            - sample["states"][0, -1, [0, 11]].mean()
        )
        assert execute_delta[[0, 4]].mean() * robot_x_delta < 0
        assert all(
            isinstance(sample_id, str) and len(sample_id) == 64
            for sample_id in sample["audit_sample_ids"]
        )
        assert not any(
            key in sample
            for key in (
                "pair_id",
                "episode_index",
                "decision_t",
                "physical_seed",
                "cue_id",
                "event_active",
            )
        )

        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        assert batch["states"].shape == (2, 2, 32, 22)
        assert batch["past_actions"].shape == (2, 2, 31, 8)
        assert batch["images"].shape == (2, 2, 2, 1, 3, 6, 8)

        summary = dataset.pair_summary()
        assert summary["pairs"] == 2
        assert summary["state_valid_steps"] == 4
        assert summary["past_action_valid_steps"] == 3
        assert summary["pairs_by_task"] == {
            "visual_target_select": 2,
            "visual_event_stop": 0,
        }
        assert summary["anchor_t_by_task"] == {
            "visual_target_select": {"4": 2},
            "visual_event_stop": {},
        }
        assert summary["selector_counts"] == {
            "episodes_considered": 4,
            "candidate_steps_considered": 12,
            "visual_history_incomplete": 0,
            "visual_frame_id_history_mismatch": 8,
            "past_action_history_mismatch": 0,
            "current_state_equal": 0,
            "execute_step0_delta_below_threshold": 2,
            "execute_step0_non_lateral_delta": 0,
            "execute_step0_lateral_mismatch": 0,
            "state_feedback_sign_mismatch": 0,
            "visual_tensor_history_mismatch": 0,
            "eligible_pairs": 2,
            "later_eligible_pairs_excluded": 0,
            "episodes_without_pair": 2,
            "selected_pairs": 2,
        }
        assert len(dataset.pair_summary_sha256()) == 64
        lineage = dataset.checkpoint_lineage()
        assert lineage["state_causal_pair_summary"] == summary
        assert len(lineage["state_causal_pair_summary_sha256"]) == 64
    finally:
        dataset.close()


def test_sampling_zeroes_identical_observation_conflicts_and_hashes_count(
    tmp_path: Path,
) -> None:
    manifest = load_m1_manifest(
        _write_manifest_fixture(tmp_path, causal_pair_signals=True)
    )
    dataset = M1WindowDataset(
        manifest,
        split="train",
        state_history=6,
        action_chunk=8,
        visual_history=2,
        hdf5_cache_size=0,
    )
    try:
        ambiguous = dataset.observationally_ambiguous_window_indices
        assert len(ambiguous) == 4
        assert {dataset.sample_lineage(index).decision_t for index in ambiguous} == {
            2,
            3,
        }
        assert {TASKS[dataset[index]["task_index"].item()] for index in ambiguous} == {
            "visual_event_stop"
        }

        weights = dataset.sampling_weights(decision_window_boost=4.0)
        assert torch.count_nonzero(weights[list(ambiguous)]).item() == 0
        task_mass = {task: 0.0 for task in TASKS}
        for index, weight in enumerate(weights.tolist()):
            task = TASKS[dataset[index]["task_index"].item()]
            task_mass[task] += weight
        assert task_mass == pytest.approx(
            {"visual_target_select": 0.5, "visual_event_stop": 0.5}
        )

        summary = dataset.window_summary()
        assert summary["observationally_ambiguous_windows"] == 4
        assert summary["sampling_eligible_windows"] == len(dataset) - 4
        assert summary["observationally_ambiguous_windows_by_task"] == {
            "visual_target_select": 0,
            "visual_event_stop": 4,
        }
        assert len(summary["observationally_ambiguous_sample_ids_sha256"]) == 64
        checkpoint = dataset.checkpoint_lineage()
        assert checkpoint["window_summary"] == summary
        assert checkpoint["window_summary_sha256"] == dataset.window_summary_sha256()
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("sha", "SHA256"),
        ("pair_split", "cue pair.*split"),
        ("template_overlap", "template_id overlaps"),
        ("attr", "root attr seed"),
        ("metadata", "randomization_config.physical_seed"),
        ("unsafe_path", "unsafe hdf5_path"),
    ),
)
def test_manifest_loader_rejects_lineage_split_and_hdf5_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["episodes"][0]
    first_path = manifest_path.parent / first["hdf5_path"]

    if mutation == "sha":
        with first_path.open("ab") as stream:
            stream.write(b"tamper")
    elif mutation == "pair_split":
        manifest["episodes"][1]["split"] = "validation"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "template_overlap":
        for episode in manifest["episodes"]:
            if episode["split"] == "validation":
                episode["template_id"] = "train_template"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "attr":
        with h5py.File(first_path, "r+") as file:
            file.attrs["seed"] = int(first["seed"]) + 1
        first["hdf5_sha256"] = _sha256(first_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "metadata":
        with h5py.File(first_path, "r+") as file:
            metadata = json.loads(str(file.attrs["episode_metadata_json"]))
            randomization = json.loads(metadata["randomization_config"])
            randomization["physical_seed"] += 1
            metadata["randomization_config"] = json.dumps(randomization)
            file.attrs["episode_metadata_json"] = json.dumps(metadata)
        first["hdf5_sha256"] = _sha256(first_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "unsafe_path":
        first["hdf5_path"] = "../escape.hdf5"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:  # pragma: no cover - parameter exhaustiveness.
        raise AssertionError(mutation)

    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        load_m1_manifest(manifest_path)


def test_manifest_only_mode_still_preserves_explicit_verification_status(
    tmp_path: Path,
) -> None:
    manifest = load_m1_manifest(
        _write_manifest_fixture(tmp_path),
        verify_hdf5_sha256=False,
        verify_hdf5_contract=False,
    )
    lineage = manifest.checkpoint_lineage("test")
    assert lineage["hdf5_sha256_verified"] is False
    assert lineage["hdf5_contract_verified"] is False

    # Dataset construction still validates selected HDF5 contracts for safe I/O.
    dataset = M1WindowDataset(manifest, split="test", hdf5_cache_size=0)
    try:
        assert dataset[0]["future_states"].shape == (8, 22)
    finally:
        dataset.close()


def _write_manifest_fixture(
    root: Path,
    *,
    steps: int = 14,
    causal_pair_signals: bool = False,
    state_causal_pair_signals: bool = False,
) -> Path:
    dataset_root = root / "canonical"
    hdf5_root = dataset_root / "hdf5"
    hdf5_root.mkdir(parents=True)
    episodes: list[dict[str, object]] = []
    episode_index = 0
    for split_index, split in enumerate(SPLITS):
        template_id = f"{split}_template"
        for task_index, task_id in enumerate(TASKS):
            physical_seed = 10_000 + split_index * 100 + task_index
            scene_id = f"{task_id}:{template_id}:scene"
            object_id = f"{task_id}:{template_id}:objects"
            task_text = f"perform {task_id} from fixed RGB"
            for cue_id in (0, 1):
                seed = physical_seed * 2 + cue_id
                path = hdf5_root / f"episode_{episode_index:06d}.hdf5"
                _write_hdf5_episode(
                    path,
                    episode_index=episode_index,
                    seed=seed,
                    physical_seed=physical_seed,
                    cue_id=cue_id,
                    split=split,
                    template_id=template_id,
                    scene_id=scene_id,
                    object_id=object_id,
                    task_id=task_id,
                    task_text=task_text,
                    task_index=task_index,
                    steps=steps,
                    causal_pair_signals=causal_pair_signals,
                    state_causal_pair_signals=state_causal_pair_signals,
                )
                episodes.append(
                    {
                        "behavior_id": "fixture_oracle",
                        "cue_id": cue_id,
                        "episode_index": episode_index,
                        "hdf5_path": f"hdf5/{path.name}",
                        "hdf5_sha256": _sha256(path),
                        "object_combination_id": object_id,
                        "physical_seed": physical_seed,
                        "scene_id": scene_id,
                        "seed": seed,
                        "split": split,
                        "steps": steps,
                        "task_id": task_id,
                        "task_text": task_text,
                        "template_id": template_id,
                    }
                )
                episode_index += 1

    split_counts: dict[str, object] = {}
    for split in SPLITS:
        selected = [episode for episode in episodes if episode["split"] == split]
        split_counts[split] = {
            "cue_counts": {"0": 2, "1": 2},
            "episodes": 4,
            "physical_seeds": 2,
            "tasks": sorted(TASKS),
            "template_ids": [f"{split}_template"],
        }
        assert len(selected) == 4

    manifest = {
        "camera_order": list(CANONICAL_CAMERA_ORDER),
        "control_hz": 20.0,
        "cue_variants": [0, 1],
        "episodes": episodes,
        "formal_protocol": True,
        "format_version": "wam.multimodal.m0.dataset/2",
        "image_hz": 10.0,
        "phase": "M0",
        "raw_unannotated": True,
        "resolution": [6, 8],
        "schema_profile": "wam_multimodal",
        "schema_version": "wam.multimodal/1.1",
        "split_counts": split_counts,
        "tasks": list(TASKS),
    }
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return manifest_path


def _write_hdf5_episode(
    path: Path,
    *,
    episode_index: int,
    seed: int,
    physical_seed: int,
    cue_id: int,
    split: str,
    template_id: str,
    scene_id: str,
    object_id: str,
    task_id: str,
    task_text: str,
    task_index: int,
    steps: int,
    causal_pair_signals: bool,
    state_causal_pair_signals: bool,
) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    rows = np.arange(steps, dtype=np.float32)
    frame_ids = np.arange(steps, dtype=np.int64) // 2
    next_frame_ids = (np.arange(steps, dtype=np.int64) + 1) // 2
    states = np.zeros((steps, 22), dtype=np.float32)
    next_states = np.zeros((steps, 22), dtype=np.float32)
    states[:, 0] = rows
    states[:, 1] = rows + 0.25
    states[:, 11] = rows + 2.0
    states[:, 12] = rows + 2.25
    next_states[:, 0] = rows + 1.0
    next_states[:, 1] = rows + 1.25
    next_states[:, 11] = rows + 3.0
    next_states[:, 12] = rows + 3.25
    states[:, 2] = float(task_index)
    next_states[:, 2] = float(task_index)
    commanded = np.repeat((rows + 100.0)[:, None], 8, axis=1).astype(np.float32)
    executed = np.repeat((rows + 10.0)[:, None], 8, axis=1).astype(np.float32)
    if causal_pair_signals and cue_id == 1:
        action_start = 0 if task_id == "visual_target_select" else 4
        commanded[action_start:, 0] += 7.0
    if state_causal_pair_signals:
        executed[:5] = 42.0
        commanded[5] = commanded[4]
        if task_id == "visual_target_select":
            commanded[5, [0, 4]] = commanded[4, [0, 4]] - 1.0

    randomization = {
        "cue_id": cue_id,
        "cue_variant": cue_id,
        "episode_seed": seed,
        "object_combination_id": object_id,
        "physical_seed": physical_seed,
        "scene_id": scene_id,
        "split": split,
        "template_id": template_id,
    }
    environment = {
        "camera_order": list(CANONICAL_CAMERA_ORDER),
        "image_height": 6,
        "image_width": 8,
        "randomization_template_id": template_id,
        "raw_unannotated": True,
        "task_id": task_id,
    }
    metadata = {
        "behavior_id": "fixture_oracle",
        "environment_config": json.dumps(environment),
        "randomization_config": json.dumps(randomization),
        "schema_version": "wam.multimodal/1.1",
        "seed": seed,
        "task_id": task_id,
    }
    with h5py.File(path, "w") as file:
        file.attrs.update(
            {
                "behavior_id": "fixture_oracle",
                "camera_order_json": json.dumps(
                    list(CANONICAL_CAMERA_ORDER), separators=(",", ":")
                ),
                "episode_index": episode_index,
                "episode_metadata_json": json.dumps(metadata),
                "format_version": "wam.trajectory.hdf5/1",
                "fps": 20.0,
                "num_steps": steps,
                "schema_profile": "wam_multimodal",
                "schema_version": "wam.multimodal/1.1",
                "seed": seed,
                "task": task_text,
                "transition_semantics": "observation[t], action[t], observation[t+1]",
            }
        )
        data = file.create_group("data")
        data.create_dataset("timestamp", data=np.arange(steps) / 20.0)
        data.create_dataset("frame_index", data=np.arange(steps, dtype=np.int64))
        data.create_dataset(
            "episode_index", data=np.full(steps, episode_index, dtype=np.int64)
        )
        data.create_dataset("seed", data=np.full(steps, seed, dtype=np.int64))
        task = data.create_group("task")
        task.create_dataset(
            "id", data=np.asarray([task_id] * steps, dtype=string_dtype)
        )
        task.create_dataset(
            "text", data=np.asarray([task_text] * steps, dtype=string_dtype)
        )
        observation = data.create_group("observation")
        next_observation = data.create_group("next_observation")
        observation.create_dataset("state", data=states)
        next_observation.create_dataset("state", data=next_states)
        observation_images = observation.create_group("images")
        next_images = next_observation.create_group("images")
        observation_indices = observation.create_group("image_frame_index")
        next_indices = next_observation.create_group("image_frame_index")
        for camera_index, camera in enumerate(CANONICAL_CAMERA_ORDER):
            current_rgb = np.empty((steps, 6, 8, 3), dtype=np.uint8)
            future_rgb = np.empty((steps, 6, 8, 3), dtype=np.uint8)
            for row in range(steps):
                current_rgb[row].fill(int(frame_ids[row]) * 10 + camera_index)
                future_rgb[row].fill(int(next_frame_ids[row]) * 10 + camera_index)
                visual_start = 0 if task_id == "visual_target_select" else 4
                if causal_pair_signals and cue_id == 1 and row >= visual_start:
                    current_rgb[row, 0, 0] = np.asarray(
                        (251, 17 + camera_index, 3), dtype=np.uint8
                    )
            observation_images.create_dataset(camera, data=current_rgb)
            next_images.create_dataset(camera, data=future_rgb)
            observation_indices.create_dataset(camera, data=frame_ids)
            next_indices.create_dataset(camera, data=next_frame_ids)
        action = data.create_group("action")
        action.create_dataset("commanded", data=commanded)
        action.create_dataset("executed", data=executed)
        event = data.create_group("event")
        event.create_dataset(
            "visual_signal_active",
            data=np.arange(steps, dtype=np.int64) >= 3,
        )
        event.create_dataset(
            "visual_signal_onset_step", data=np.full(steps, 4, dtype=np.int64)
        )
        event.create_dataset(
            "rendered_cue_variant", data=np.full(steps, cue_id, dtype=np.int64)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
