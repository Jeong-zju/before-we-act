"""Lightweight data helpers shared by R10 training and contract tests."""
from __future__ import annotations

import numpy as np


def take_hdf5_rows(dataset, indices: list[int]) -> np.ndarray:
    """Read a short ordered window without h5py's strict fancy-index rule."""
    return np.stack(
        [np.asarray(dataset[index], dtype=np.float32) for index in indices]
    )
