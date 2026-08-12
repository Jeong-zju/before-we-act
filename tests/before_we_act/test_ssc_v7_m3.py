from __future__ import annotations

import gzip
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.before_we_act import run_ssc_v7_m3 as m3


def test_stage_revision_is_m3_r3() -> None:
    assert m3.STAGE_ID == "SSC-V7-M3-R3"
    assert m3.CONDITIONS == (
        "E0",
        "HC",
        "label_shuffled",
        "time_phase_only",
        "B",
        "B_hat",
    )


def _label(frame: int) -> dict[str, object]:
    return {
        "stage_id": "approach" if frame < 20 else "controlled",
        "within_stage_progress": frame / 40.0,
        "remaining_goal_mask": {"goal": frame < 40},
        "factorized_predicates": {"controlled": frame >= 20},
        "causal_automaton_state": {
            "completed_goal_mask": {},
            "completed_handoff_mask": [],
            "custody_transfer_count": 0,
            "previous_stage_id": "approach",
            "last_confirmed_custodian": {},
        },
        "per_agent_contribution": [
            {
                "agent_slot": 0,
                "active": frame >= 20,
                "contact_objects": [],
                "grasp_objects": [],
                "objects": [],
                "roles": [],
            },
            {
                "agent_slot": 1,
                "active": False,
                "contact_objects": [],
                "grasp_objects": [],
                "objects": [],
                "roles": [],
            },
        ],
        "agent_object_role_slots": [
            {"agent_slot": 0, "object": "barrier", "roles": ["support"]},
            {"agent_slot": 1, "object": "barrier", "roles": ["none"]},
        ],
        "grasp_contact_custody_state": {
            "barrier": {
                "contact_agents": [0] if frame >= 20 else [],
                "grasp_agents": [0] if frame >= 20 else [],
                "controller_agents": [0] if frame >= 20 else [],
                "current_custodian": 0 if frame >= 20 else None,
                "last_confirmed_custodian": 0 if frame >= 20 else None,
                "shared_control": False,
            }
        },
        "collision_drop_contention_risk": {
            "contested_objects": [],
            "dropped_objects": [],
            "robot_collision": False,
            "robot_proximity_risk": False,
        },
        "label_validity_mask": {
            "stage_id": True,
            "within_stage_progress": True,
            "remaining_goal_mask": True,
        },
        "ambiguity_code": 0,
    }


def _episode(root: Path) -> tuple[Path, Path]:
    hdf5_path = root / "episode.hdf5"
    rng = np.random.default_rng(7)
    with h5py.File(hdf5_path, "w") as stream:
        stream.attrs["agent_count"] = 2
        image_group = stream.require_group("data/observation/images")
        global_rgb = rng.integers(0, 256, size=(41, 60, 80, 3), dtype=np.uint8)
        image_group.create_dataset("global", data=global_rgb)
        image_group.create_dataset("agent_0", data=global_rgb)
        image_group.create_dataset("agent_1", data=np.flip(global_rgb, axis=2))
        agent_group = stream.require_group("data/observation/agents")
        agent_group.create_dataset("panda_0", data=rng.normal(size=(41, 9)))
        agent_group.create_dataset("panda_1", data=rng.normal(size=(41, 9)))
        stream.require_group("data/action").create_dataset(
            "commanded", data=rng.normal(size=(40, 16)).astype(np.float32)
        )
    sidecar_path = root / "episode.oracle.jsonl.gz"
    with gzip.open(sidecar_path, "wt", encoding="utf-8") as stream:
        for frame in range(41):
            stream.write(
                json.dumps({"oracle_label": _label(frame)}, sort_keys=True) + "\n"
            )
    return hdf5_path, sidecar_path


def test_compact_rgb_is_exact_area_mean() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:8, :8] = 5
    image[:4, :4] = 7
    compact = m3.compact_rgb(image)
    assert compact.shape == (60, 80, 3)
    assert compact.dtype == np.uint8
    assert compact[0, 0, 0] == 6
    assert np.count_nonzero(compact[1:]) == 0


def test_social_features_are_deterministic_and_factorized() -> None:
    first = m3.social_features(_label(25), own_slot=0)
    second = m3.social_features(_label(25), own_slot=0)
    assert np.array_equal(first, second)
    assert first.shape == (192,)
    assert np.any(first[m3.SOURCE_SLICES["P"]])
    assert np.any(first[m3.SOURCE_SLICES["T"]])
    assert np.any(first[m3.SOURCE_SLICES["B"]])


def test_loader_never_opens_sealed_test_path(tmp_path: Path) -> None:
    hdf5_path, sidecar_path = _episode(tmp_path)
    manifest = {
        "episodes": [
            {
                "split": "train",
                "task": "lift_barrier",
                "hdf5_path": str(hdf5_path),
                "sidecar_path": str(sidecar_path),
                "hdf5_sha256": "a" * 64,
            },
            {
                "split": "read_only_test",
                "task": "lift_barrier",
                "hdf5_path": str(tmp_path / "must-not-open.hdf5"),
                "sidecar_path": str(tmp_path / "must-not-open.jsonl.gz"),
                "hdf5_sha256": "b" * 64,
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data, audit = m3.load_probe_data(manifest_path, {"train"})
    assert len(data) > 0
    assert data.legal.shape[1] == data.e0.shape[1]
    assert data.social.shape[1] == 192
    assert data.target.shape[1:] == (100, 8)
    assert audit["test_paths_opened"] == 0


def test_all_action_conditions_have_identical_parameter_count() -> None:
    counts = {
        sum(parameter.numel() for parameter in m3.build_action_model(1896, 256, 3).parameters())
        for _condition in m3.CONDITIONS
    }
    assert len(counts) == 1


def test_nested_social_masks_are_episode_level_and_leak_free() -> None:
    episode_ids = []
    tasks = []
    for task in m3.TASKS:
        for index in range(36):
            episode_ids.extend([f"{task}-{index}"] * 2)
            tasks.extend([task] * 2)
    count = len(episode_ids)
    data = m3.ProbeData(
        legal=np.zeros((count, 1), dtype=np.float32),
        e0=np.zeros((count, 1), dtype=np.float32),
        social=np.zeros((count, 192), dtype=np.float32),
        time=np.zeros((count, 192), dtype=np.float32),
        target=np.zeros((count, 100, 8), dtype=np.float32),
        target_mask=np.ones((count, 100), dtype=np.float32),
        tasks=np.asarray(tasks),
        episode_ids=np.asarray(episode_ids),
        frame_indices=np.zeros(count, dtype=np.int32),
        agent_slots=np.zeros(count, dtype=np.int16),
    )
    all_held: set[str] = set()
    for fold in range(3):
        held, inner_fit, inner_validation = m3.nested_social_masks(data, fold)
        assert np.all(held.astype(int) + inner_fit.astype(int) + inner_validation.astype(int) == 1)
        for episode_id in set(episode_ids):
            rows = data.episode_ids == episode_id
            assert held[rows].all() or inner_fit[rows].all() or inner_validation[rows].all()
        held_ids = set(data.episode_ids[held].tolist())
        assert not (held_ids & set(data.episode_ids[inner_validation].tolist()))
        assert all(
            len(set(data.episode_ids[held & (data.tasks == task)].tolist())) == 12
            for task in m3.TASKS
        )
        all_held.update(held_ids)
    assert all_held == set(episode_ids)


def test_holm_adjustment_is_monotone_in_rank() -> None:
    adjusted = m3.holm({"P": 0.01, "T": 0.03, "B": 0.20})
    assert adjusted["P"] == 0.03
    assert adjusted["T"] == 0.06
    assert adjusted["B"] == 0.20
