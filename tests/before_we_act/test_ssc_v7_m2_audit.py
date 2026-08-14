from __future__ import annotations

import sys
from types import ModuleType

import pytest

try:
    import h5py as _h5py  # noqa: F401
except ModuleNotFoundError:
    # These unit tests exercise pure selection helpers and do not need HDF5.
    # Keep the import shim local so it cannot contaminate later test modules.
    sys.modules["h5py"] = ModuleType("h5py")
    try:
        from scripts.before_we_act.audit_ssc_v7_m2 import (
            causal_review_indices,
            select_manual_events,
        )
    finally:
        sys.modules.pop("h5py", None)
else:
    from scripts.before_we_act.audit_ssc_v7_m2 import (
        causal_review_indices,
        select_manual_events,
    )


def test_causal_review_window_uses_only_past_and_current_frames() -> None:
    assert causal_review_indices(center=10, total_frames=20) == [6, 7, 8, 9, 10]


def test_causal_review_window_left_pads_episode_start_without_future() -> None:
    assert causal_review_indices(center=2, total_frames=20) == [0, 0, 0, 1, 2]
    assert causal_review_indices(center=0, total_frames=20) == [0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("center", "total", "width"),
    ((-1, 10, 5), (10, 10, 5), (0, 0, 5), (0, 10, 0)),
)
def test_causal_review_window_rejects_invalid_bounds(
    center: int, total: int, width: int
) -> None:
    with pytest.raises(ValueError):
        causal_review_indices(center=center, total_frames=total, width=width)


def test_manual_selection_appends_only_missing_episode_coverage() -> None:
    collection = {
        "episodes": [
            {"task": "lift_barrier", "hdf5_sha256": "episode-a"},
            {"task": "lift_barrier", "hdf5_sha256": "episode-b"},
            {"task": "camera_alignment", "hdf5_sha256": "episode-c"},
        ]
    }
    candidates = {
        task: []
        for task in (
            "lift_barrier",
            "camera_alignment",
            "long_pipeline_delivery",
            "take_photo",
            "pass_shoe",
            "place_food",
        )
    }
    candidates["lift_barrier"] = [
        {"rank": "00", "episode_sha256": "episode-a", "frame_index": 1},
        {"rank": "01", "episode_sha256": "episode-a", "frame_index": 2},
        {"rank": "02", "episode_sha256": "episode-b", "frame_index": 3},
    ]
    candidates["camera_alignment"] = [
        {"rank": "00", "episode_sha256": "episode-c", "frame_index": 4},
        {"rank": "01", "episode_sha256": "episode-c", "frame_index": 5},
    ]
    gate = {"manual_audit": {"transition_windows_per_task_min": 2}}

    selected, missing = select_manual_events(collection, candidates, gate)

    assert missing == 0
    assert [(item["episode_sha256"], item["frame_index"]) for item in selected] == [
        ("episode-a", 1),
        ("episode-a", 2),
        ("episode-c", 4),
        ("episode-c", 5),
        ("episode-b", 3),
    ]
    assert "coverage_supplement" not in selected[0]
    assert selected[-1]["coverage_supplement"] is True
