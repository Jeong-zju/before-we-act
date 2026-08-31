"""BiCoord-native CARE deployment adapter.

Only this package knows about BiCoord's raw HDF5/JPEG layout.  The policy
modules in :mod:`before_we_act` remain the upstream CARE implementation; this
adapter supplies a seven-dimensional local-arm codec and benchmark metadata.
"""

from .config import *  # noqa: F401,F403
from .data import (  # noqa: F401
    BiCoordBalancedDistributedBatchSampler,
    BiCoordEpisode,
    BiCoordTemporalDataset,
    BiCoordTemporalRequest,
    BiCoordVisualCache,
    BalancedDistributedBatchSampler,
    compute_normalization,
    discover_bicoord_episodes,
    episode_manifest,
    load_bicoord_episodes,
    load_normalization_receipt,
    write_normalization_receipt,
)
from .hdf5_data import (  # noqa: F401
    BiCoordHDF5Reader,
    CAMERA_NAMES,
    StageSegment,
    discover_all_episode_files,
    discover_episode_files,
    validate_hdf5_schema,
)

__all__ = [name for name in globals() if not name.startswith("_")]
