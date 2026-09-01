from __future__ import annotations

import numpy as np


def sample_indices(start: int, end: int) -> list[int]:
    """Small deterministic audit sample including both ends of an episode."""
    if end - start < 2: return []
    return sorted(set((start, start + 1, max(start, end - 2))))
