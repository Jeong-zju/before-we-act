from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from before_we_act.mars_temporal_data import ARMS, ENV_DIR, MARS_TASKS
from scripts.before_we_act.prepare_mars_care_optimization_smoke import (
    ACT_VALIDATION_SEED_RANGES,
    CARE_VALIDATION_SEED_RANGE,
    FAMILIES_PER_TASK,
    branch_completeness_report,
    build_smoke_manifest,
    inclusive_range,
    load_formal_family_seeds,
    load_sidecar_episodes,
)


def _write_source(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for task_index, task in enumerate(MARS_TASKS):
        directory = root / ENV_DIR[task] / "motionplanning"
        directory.mkdir(parents=True)
        for shard in range(10):
            h5_path = directory / f"{task}.shard{shard:02d}.h5"
            h5_path.touch()
            rows = []
            for episode_id in range(15):
                # Include the old formal seeds 0..29; all other rows remain
                # eligible.  Validation seeds are much larger but still part
                # of the explicit exclusion contract.
                seed = task_index * 100_000 + shard * 15 + episode_id
                rows.append(
                    {
                        "episode_id": episode_id,
                        "episode_seed": seed,
                        "elapsed_steps": 200 + episode_id,
                        "reset_kwargs": {"seed": seed},
                        "success": 1,
                    }
                )
            sidecar = h5_path.with_suffix(".json")
            sidecar.write_text(json.dumps({"episodes": rows}), encoding="utf-8")
    return root


def _write_formal_manifest(tmp_path: Path) -> Path:
    rows = []
    for task_index, task in enumerate(MARS_TASKS):
        rows.extend(
            {"task": task, "episode_seed": task_index * 100_000 + offset}
            for offset in range(30)
        )
    path = tmp_path / "old_formal.json"
    path.write_text(
        json.dumps(
            {
                "format_version": "before-we-act.care-mars-family-manifest/1",
                "families": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_act_validation20_seed_ranges_are_the_archived_per_task_contract() -> None:
    assert ACT_VALIDATION_SEED_RANGES == {
        "place_cube_in_cup": (20260820, 20260839),
        "strike_cube_hard": (20261820, 20261839),
        "three_robots_place_shoes": (20262820, 20262839),
        "four_robots_stack_cube": (20263820, 20263839),
    }
    assert CARE_VALIDATION_SEED_RANGE == (20260827, 20260846)
    assert all(
        len(inclusive_range(bounds)) == 20
        for bounds in ACT_VALIDATION_SEED_RANGES.values()
    )


def test_smoke_manifest_is_disjoint_stratified_balanced_and_reproducible(
    tmp_path: Path,
) -> None:
    raw_root = _write_source(tmp_path)
    formal_path = _write_formal_manifest(tmp_path)
    episodes, sidecars = load_sidecar_episodes(raw_root)
    formal_seeds, provenance = load_formal_family_seeds(formal_path)

    first = build_smoke_manifest(
        episodes,
        formal_seeds=formal_seeds,
        formal_provenance=provenance,
        raw_root=raw_root,
        sidecars=sidecars,
    )
    second = build_smoke_manifest(
        episodes,
        formal_seeds=formal_seeds,
        formal_provenance=provenance,
        raw_root=raw_root,
        sidecars=sidecars,
    )
    assert len(episodes) == 600
    assert len(sidecars) == 40
    assert first["family_count"] == len(MARS_TASKS) * FAMILIES_PER_TASK == 16
    # Timestamps are the sole intentionally non-deterministic field.
    first.pop("created_at_utc")
    second.pop("created_at_utc")
    assert first == second

    care_validation = inclusive_range(CARE_VALIDATION_SEED_RANGE)
    for task in MARS_TASKS:
        rows = [row for row in first["families"] if row["task"] == task]
        selected = {int(row["episode_seed"]) for row in rows}
        assert len(rows) == len(selected) == 4
        assert not selected & formal_seeds[task]
        assert not selected & care_validation
        assert not selected & inclusive_range(ACT_VALIDATION_SEED_RANGES[task])
        assert Counter(row["sampling_stratum"] for row in rows) == Counter(
            {"critical": 2, "uniform": 2}
        )
        focal = Counter(int(row["focal_agent"]) for row in rows)
        counts = [focal[arm] for arm in range(ARMS[task])]
        assert max(counts) - min(counts) <= 1
        assert all(row["sampling_protocol"] == "fixed_stratified_smoke_v1" for row in rows)
        assert first["seed_exclusions"]["counts_by_task"][task][
            "excluded_by_reason"
        ]["old_formal_care_family_episode_seed"] == 30


def _family(path: Path, *, observed: int, requested: int = 16) -> Path:
    payload = {
        "snapshot_id": path.stem,
        "branches": [
            {
                "candidate_id": 0,
                "regime": "reactive",
                "repeat_id": 0,
                "outcomes": {
                    str(requested): {
                        "requested_steps": requested,
                        "observed_steps": observed,
                    }
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_branch_completeness_requires_observed_steps_to_cover_request(
    tmp_path: Path,
) -> None:
    complete = branch_completeness_report(
        [_family(tmp_path / "complete.json", observed=16)], expected_families=1
    )
    assert complete["status"] == "PASSED"
    assert complete["outcome_count"] == 1
    assert complete["issue_count"] == 0

    short = branch_completeness_report(
        [_family(tmp_path / "short.json", observed=15)], expected_families=1
    )
    assert short["status"] == "FAILED"
    assert short["issues"] == [
        {
            "snapshot_id": "short",
            "branch_index": 0,
            "horizon": "16",
            "requested_steps": 16,
            "observed_steps": 15,
            "reason": "observed_steps_shorter_than_requested",
        }
    ]


def test_sidecar_loader_rejects_non_600_contract(tmp_path: Path) -> None:
    raw_root = _write_source(tmp_path)
    sidecar = next((raw_root / ENV_DIR[MARS_TASKS[0]] / "motionplanning").glob("*.json"))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["episodes"].pop()
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="150 unique seeds"):
        load_sidecar_episodes(raw_root)
