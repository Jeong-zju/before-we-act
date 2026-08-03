from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train.s4_future_feature_cache import (
    CACHE_FORMAT,
    HORIZONS,
    S4ProjectedFutureFeatureCache,
    file_sha256,
)


def test_projected_future_cache_indexes_and_normalizes_exactly(tmp_path: Path) -> None:
    manifests = []
    manifest_rows = []
    episodes = []
    offset = 0
    for task_index in range(5):
        manifest = tmp_path / f"task_{task_index}.json"
        manifest.write_text(json.dumps({"task": task_index}) + "\n")
        manifests.append(manifest)
        manifest_rows.append(
            {"path": str(manifest), "sha256": file_sha256(manifest)}
        )
        for episode_index in range(150):
            steps = 100 if task_index == 0 and episode_index == 0 else 1
            episodes.append(
                {
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "offset": offset,
                    "steps": steps,
                }
            )
            offset += steps
    root = tmp_path / "cache"
    root.mkdir()
    features_path = root / "features.npy"
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(offset, 5, 4, 256),
    )
    features[:] = 0.0
    for row in (0, 24, 49, 99):
        features[row] = float(row + 1)
    features.flush()
    del features
    features_sha256 = file_sha256(features_path)
    metadata = {
        "format_version": CACHE_FORMAT,
        "features_sha256": features_sha256,
        "features_size_bytes": features_path.stat().st_size,
        "pca_artifact_sha256": "a" * 64,
        "vision_weights_sha256": "b" * 64,
        "shape": [offset, 5, 4, 256],
        "manifests": manifest_rows,
        "episodes": episodes,
    }
    (root / "metadata.json").write_text(json.dumps(metadata) + "\n")
    cache = S4ProjectedFutureFeatureCache(
        root,
        manifests=manifests,
        expected_features_sha256=features_sha256,
        expected_pca_sha256="a" * 64,
        expected_vision_weights_sha256="b" * 64,
    )
    grouped = {
        "task_index": torch.tensor([0]),
        "episode_index": torch.tensor([0]),
        "decision_t": torch.tensor([0]),
        "future_horizons": torch.tensor([HORIZONS]),
        "future_agent_visual_valid_mask": torch.ones(1, 4, 4, dtype=torch.bool),
        "future_shared_visual_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    artifact = {
        "visual_delta_mean": torch.zeros(4, 256),
        "visual_delta_std": torch.ones(4, 256),
        "shared_visual_delta_mean": torch.zeros(4, 256),
        "shared_visual_delta_std": torch.ones(4, 256),
    }
    local, shared = cache.normalized_targets(
        grouped,
        artifact,
        current_local=torch.zeros(1, 4, 4, 256),
        current_shared=torch.zeros(1, 4, 256),
        device=torch.device("cpu"),
    )
    expected = torch.tensor([1.0, 25.0, 50.0, 100.0])
    torch.testing.assert_close(local[0, 0, :, 0, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(shared[0, :, 0, 0], expected, rtol=0, atol=0)
