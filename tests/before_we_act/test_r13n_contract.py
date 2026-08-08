from pathlib import Path

import torch
from torch import nn

from before_we_act.action_generator.r13n_baseline import (
    R13NActionGenerator,
    load_r13n_config,
)
from before_we_act.benchmark import ACTIVE_TASKS as BENCHMARK_TASKS
from before_we_act.r13n import SPLIT_EPISODES, TASKS, TASK_SPECS


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
