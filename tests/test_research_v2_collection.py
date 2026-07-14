from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import collect_research_v2_dataset as collector


def test_disk_preflight_preserves_reserve_and_accounts_for_worker_temporaries(
    tmp_path, monkeypatch
):
    gib = 1024**3
    monkeypatch.setattr(
        collector.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * gib, used=90 * gib, free=10 * gib),
    )
    report = collector._check_disk_space(
        tmp_path,
        remaining_episodes=4,
        workers=2,
        min_free_gib=9.0,
        estimated_episode_mib=100.0,
    )
    assert report["estimated_required_gib"] == pytest.approx(0.5859375)
    with pytest.raises(OSError, match="insufficient disk space"):
        collector._check_disk_space(
            tmp_path,
            remaining_episodes=4,
            workers=2,
            min_free_gib=9.5,
            estimated_episode_mib=100.0,
        )


def test_resume_removes_only_research_v2_temporary_episode_files(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    stale = split / "episode_000003.hdf5.v2tmp"
    unrelated = split / "keep.tmp"
    stale.write_bytes(b"partial")
    unrelated.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="--resume"):
        collector._handle_stale_temporaries(tmp_path, resume=False)
    assert collector._handle_stale_temporaries(tmp_path, resume=True) == 1
    assert not stale.exists()
    assert unrelated.read_bytes() == b"keep"


def test_parallel_resume_accepts_missing_indices_for_recollection(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    (split / "episode_000000.hdf5").touch()
    (split / "episode_000002.hdf5").touch()
    indexed = collector._indexed_episode_paths(split)
    assert sorted(indexed) == [0, 2]
    assert [index for index in range(4) if index not in indexed] == [1, 3]


def test_formal_and_pilot_recipes_are_deterministic_but_use_distinct_mode_sets():
    for seed in range(20_000, 20_100):
        assert collector._episode_recipe(seed) == collector._episode_recipe(seed)
        pilot = collector._episode_recipe(seed, pilot=True)
        assert pilot["mode"] in collector.RESEARCH_V2_PILOT_MODES
        assert 0.0 <= float(pilot["object_dropout_prob"]) <= 1.0
    assert {collector._episode_recipe(seed)["mode"] for seed in range(20_000, 20_100)} >= {
        "exploratory",
        "near_miss",
    }
    scenarios = {
        collector._episode_recipe(seed)["scenario"] for seed in range(20_000, 20_400)
    }
    assert "private_gates" in scenarios
    assert scenarios >= {"nominal", "occlusion", "hard_comm"}


def test_formal_private_event_quality_gate_rejects_missing_coverage():
    report = {
        "episodes": 100,
        "private_event_quality": {
            "private_gate_episodes": 0,
            "event_type_observation_counts": {
                "decisive_private": 0,
                "locally_inferable": 0,
                "redundant": 0,
            },
            "informed_agent_observation_counts": {"0": 0, "1": 0},
            "maneuver_observation_counts": {"left": 0, "hold": 100, "right": 0},
            "active_observations": 0,
            "cued_agent_observations": 0,
        },
    }
    with pytest.raises(RuntimeError, match="private-event quality gate failed"):
        collector._validate_formal_split_quality("train", report, smoke=False)
    collector._validate_formal_split_quality("train", report, smoke=True)


def test_formal_private_event_quality_gate_accepts_all_required_classes():
    report = {
        "episodes": 100,
        "private_event_quality": {
            "private_gate_episodes": 12,
            "event_type_observation_counts": {
                "decisive_private": 30,
                "locally_inferable": 30,
                "redundant": 30,
            },
            "informed_agent_observation_counts": {"0": 45, "1": 45},
            "maneuver_observation_counts": {"left": 30, "hold": 30, "right": 30},
            "active_observations": 20,
            "cued_agent_observations": 20,
        },
    }
    collector._validate_formal_split_quality("validation", report, smoke=False)
