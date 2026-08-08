from pathlib import Path

import pytest
import torch
from torch import nn

from before_we_act.action_generator.r13n_baseline import (
    R13NActionGenerator,
    load_r13n_config,
)
from before_we_act.benchmark import ACTIVE_TASKS as BENCHMARK_TASKS
from before_we_act.r13n import (
    SPLIT_EPISODES,
    TASKS,
    TASK_SPECS,
    camera_sensor_key,
    clamp_action_to_space,
)


def test_r13n_contract_has_exact_no_stack_portfolio():
    assert TASKS == (
        "lift_barrier",
        "camera_alignment",
        "long_pipeline_delivery",
        "take_photo",
        "pass_shoe",
        "place_food",
    )
    assert tuple(BENCHMARK_TASKS) == TASKS
    assert not any("stack" in task for task in TASKS)
    assert set(TASK_SPECS) == set(TASKS)
    assert SPLIT_EPISODES == {"train": 120, "validation": 15, "test": 15}


def test_place_food_is_deliberately_global_only():
    assert TASK_SPECS["place_food"]["camera_order"] == ("global",)
    assert TASK_SPECS["place_food"]["agents"] == 2


def test_r13n_dataset_views_map_to_real_robofactory_sensor_keys():
    assert camera_sensor_key("global") == "head_camera_global"
    assert camera_sensor_key("agent_0") == "head_camera_agent0"
    assert camera_sensor_key("agent_3") == "head_camera_agent3"
    with pytest.raises(ValueError, match="unknown R13N camera view"):
        camera_sensor_key("agent_4")


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(96, 32)

    def sample(self, tokens, token_mask):
        del token_mask
        return self.head(tokens.mean(dim=1))[:, None].expand(-1, 100, -1)

    def training_loss(self, tokens, token_mask, actions, mask):
        prediction = self.sample(tokens, token_mask)
        loss = (prediction - actions).abs().masked_select(mask).mean()
        return {"loss": loss, "l1": loss}


def test_r13n_model_consumes_six_task_masks_and_is_candidate_native():
    config = load_r13n_config(
        Path(__file__).parents[2] / "configs/before_we_act/r13n/b6.yaml"
    )
    model = R13NActionGenerator(config)
    model.core = _Core()
    batch = {
        "visual": torch.randn(2, 3, 16, 15),
        "view_mask": torch.tensor(
            [[[1, 1, 1, 0, 0]] * 3, [[1, 0, 0, 0, 0]] * 3],
            dtype=torch.float32,
        ),
        "qpos": torch.randn(2, 3, 4, 9),
        "actions": torch.randn(2, 3, 4, 8),
        "agent_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool),
    }
    spatial = torch.randn(2, 5, 48, 768)
    spatial_mask = batch["view_mask"][:, -1].bool()
    tasks = torch.tensor([TASKS.index("pass_shoe"), TASKS.index("place_food")])
    proposals = model.sample(
        batch,
        spatial_tokens=spatial,
        spatial_view_mask=spatial_mask,
        task_index=tasks,
    )
    assert proposals.actions.shape == (2, 1, 4, 100, 8)
    assert torch.equal(proposals.actions[:, :, 2:], torch.zeros_like(proposals.actions[:, :, 2:]))
    assert proposals.diagnostics["candidate_native"] is True


class _LargeCore(nn.Module):
    def sample(self, tokens, token_mask):
        del token_mask
        values = torch.full((len(tokens), 100, 32), 50.0, device=tokens.device)
        values[:, :, 1] = 100.0
        return values


def test_r13n_normalized_guard_preserves_valid_six_task_range():
    config = load_r13n_config(
        Path(__file__).parents[2] / "configs/before_we_act/r13n/b6.yaml"
    )
    model = R13NActionGenerator(config)
    model.core = _LargeCore()
    batch = {
        "visual": torch.randn(1, 3, 16, 15),
        "view_mask": torch.ones(1, 3, 5),
        "qpos": torch.randn(1, 3, 4, 9),
        "actions": torch.randn(1, 3, 4, 8),
        "agent_mask": torch.tensor([[1, 0, 0, 0]], dtype=torch.bool),
    }
    proposals = model.sample(
        batch,
        spatial_tokens=torch.randn(1, 5, 48, 768),
        spatial_view_mask=torch.ones(1, 5, dtype=torch.bool),
        task_index=torch.tensor([0]),
    )
    assert config.action["normalized_clip"] == 96.0
    assert torch.all(proposals.actions[0, 0, 0, :, 0] == 50.0)
    assert torch.all(proposals.actions[0, 0, 0, :, 1] == 96.0)
    assert torch.count_nonzero(proposals.actions[0, 0, 1:]) == 0


class _Box:
    def __init__(self, low, high):
        self.low = low
        self.high = high


class _DictSpace:
    def __init__(self):
        self.spaces = {
            "panda-0": _Box(
                torch.tensor([-1.0, -2.0]).numpy(),
                torch.tensor([1.0, 2.0]).numpy(),
            )
        }


def test_r13n_applies_physical_bounds_after_denormalization():
    bounded, clipped, total = clamp_action_to_space(
        _DictSpace(), {"panda-0": torch.tensor([1.5, -1.5]).numpy()}
    )
    assert bounded["panda-0"].tolist() == pytest.approx([1.0, -1.5])
    assert (clipped, total) == (1, 2)
