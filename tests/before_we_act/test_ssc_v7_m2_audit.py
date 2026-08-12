from __future__ import annotations

import sys
from types import ModuleType

import pytest

sys.modules.setdefault("h5py", ModuleType("h5py"))

from scripts.before_we_act.audit_ssc_v7_m2 import causal_review_indices


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
