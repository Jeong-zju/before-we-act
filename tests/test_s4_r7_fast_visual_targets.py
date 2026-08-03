from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from train.s2_future_prediction import (
    encode_current_visual_context,
    encode_future_visual_targets,
    encode_local_visual_targets,
    encode_shared_visual_targets,
)


class _BatchIndependentVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_height=2, image_width=2)
        self.patch_size = 1
        self.calls = 0

    def forward(self, images: Tensor) -> SimpleNamespace:
        self.calls += 1
        pixels = images.float().mean(dim=1).reshape(images.shape[0], 4, 1)
        features = torch.arange(1024, dtype=torch.float32).reshape(1, 1, -1)
        return SimpleNamespace(spatial_tokens=pixels + features / 1024.0)

    def forward_spatial_grid(
        self, images: Tensor, *, grid_height: int, grid_width: int
    ) -> SimpleNamespace:
        assert (grid_height, grid_width) == (2, 2)
        return self(images)


def _artifact() -> dict[str, Tensor]:
    components = torch.zeros(1024, 256)
    components[:256] = torch.eye(256)
    return {
        "pca_mean": torch.zeros(1024),
        "pca_components": components,
        "pca_projected_std": torch.ones(256),
        "visual_delta_mean": torch.zeros(4, 256),
        "visual_delta_std": torch.ones(4, 256),
        "shared_visual_delta_mean": torch.zeros(4, 256),
        "shared_visual_delta_std": torch.ones(4, 256),
    }


def _grouped() -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        "valid_agent_mask": torch.tensor([[True, True]]),
        "shared_observation_valid_mask": torch.tensor([True]),
        "future_agent_visual_valid_mask": torch.tensor(
            [[[True, True, True, True], [True, False, True, True]]]
        ),
        "future_shared_visual_valid_mask": torch.tensor(
            [[True, True, False, True]]
        ),
        "agent_observations": torch.rand(1, 2, 3, 2, 2, generator=generator),
        "shared_observation": torch.rand(1, 3, 2, 2, generator=generator),
        "future_agent_observations": torch.rand(
            1, 2, 4, 3, 2, 2, generator=generator
        ),
        "future_shared_observations": torch.rand(
            1, 4, 3, 2, 2, generator=generator
        ),
    }


def test_fast_combined_targets_match_separate_fp32_target_construction() -> None:
    device = torch.device("cpu")
    grouped = _grouped()
    artifact = _artifact()
    vision = _BatchIndependentVision()

    _, current_local, current_shared = encode_current_visual_context(
        vision,
        grouped,
        artifact,
        device=device,
        grid_height=2,
        grid_width=2,
    )
    calls_after_current = vision.calls
    fast_local, fast_shared = encode_future_visual_targets(
        vision,
        grouped,
        artifact,
        current_local=current_local,
        current_shared=current_shared,
        device=device,
        grid_height=2,
        grid_width=2,
    )
    fast_target_calls = vision.calls - calls_after_current

    _, reference_local = encode_local_visual_targets(
        vision,
        grouped,
        artifact,
        device=device,
        grid_height=2,
        grid_width=2,
    )
    _, reference_shared, _ = encode_shared_visual_targets(
        vision,
        grouped,
        artifact,
        device=device,
        grid_height=2,
        grid_width=2,
    )

    torch.testing.assert_close(fast_local, reference_local, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fast_shared, reference_shared, rtol=0.0, atol=0.0)
    assert calls_after_current == 1
    assert fast_target_calls == 1
