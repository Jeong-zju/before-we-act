from __future__ import annotations

from pathlib import Path

import pytest
import torch

from before_we_act.action_generator.evolution import (
    TaskConditionedActionGenerator,
    deduplicate_exact_spatial_views,
    load_r12_evolution_config,
)
from before_we_act.contracts import TeamBeliefState


def test_exact_view_dedup_keeps_first_active_copy_and_preserves_near_equal_view():
    tokens = torch.zeros(1, 5, 3, 2)
    tokens[:, 2] = 1.0
    tokens[:, 3] = 1.0
    tokens[:, 4] = 1.0
    tokens[:, 4, 0, 0] += torch.finfo(tokens.dtype).eps
    mask = torch.tensor([[True, True, True, False, True]])

    deduplicated = deduplicate_exact_spatial_views(tokens, mask)

    assert deduplicated.tolist() == [[True, False, True, False, True]]


@pytest.mark.parametrize(
    ("tokens", "mask", "message"),
    [
        (torch.zeros(1, 5, 3), torch.ones(1, 5, dtype=torch.bool), "must be"),
        (torch.zeros(1, 5, 3, 2), torch.ones(1, 4, dtype=torch.bool), "shape"),
        (torch.zeros(1, 5, 3, 2), torch.zeros(1, 5, dtype=torch.bool), "active"),
    ],
)
def test_exact_view_dedup_rejects_invalid_contract(tokens, mask, message):
    with pytest.raises(ValueError, match=message):
        deduplicate_exact_spatial_views(tokens, mask)


def test_model_condition_sends_deduplicated_mask_to_role_query_bridge():
    root = Path(__file__).resolve().parents[2]
    config = load_r12_evolution_config(
        root / "configs/before_we_act/r12_action/e1_p2.yaml"
    )
    model = TaskConditionedActionGenerator(config)
    belief = TeamBeliefState(
        tokens=torch.randn(1, 16, 96),
        agent_tokens=torch.randn(1, 4, 96),
        consensus_token=torch.randn(1, 96),
        uncertainty=torch.zeros(1, 1),
        agent_mask=torch.tensor([[True, True, True, False]]),
    )
    spatial = torch.randn(1, 5, 48, 768)
    spatial[:, 1:4] = spatial[:, :1]
    captured: dict[str, torch.Tensor] = {}
    original = model.bridge.forward

    def capture(*args, **kwargs):
        captured["mask"] = args[2].detach().clone()
        return original(*args, **kwargs)

    model.bridge.forward = capture
    tokens, token_mask = model.condition(
        belief,
        spatial,
        torch.tensor([[True, True, True, True, False]]),
        torch.tensor([2]),
    )

    assert captured["mask"].tolist() == [[True, False, False, False, False]]
    assert tuple(tokens.shape) == (1, 37, 96)
    assert token_mask[:, -16:].sum().item() == 12


def test_view_dedup_branch_keeps_isolated_promotion_and_launcher_identity():
    root = Path(__file__).resolve().parents[2]
    names = [
        "launch_r15_expert_finetune_tmux.sh",
        "run_r15_expert_finetune_candidate.sh",
        "launch_r15_temporal_screens_tmux.sh",
        "launch_r15_formal_stack_tmux.sh",
    ]
    content = {
        name: (root / "scripts/before_we_act" / name).read_text() for name in names
    }
    for text in content.values():
        assert "role-query-view-dedup" in text
    assert "role_query_view_dedup_act_temporal_ensemble" in content[
        "launch_r15_temporal_screens_tmux.sh"
    ]
    assert "role_query_view_dedup_act_temporal_ensemble" in content[
        "launch_r15_formal_stack_tmux.sh"
    ]
    handoff = (
        root
        / "scripts/before_we_act/handoff_r15_role_query_view_dedup_promotion.sh"
    ).read_text()
    assert handoff.index("while tmux has-session -t \"$PREDECESSOR\"") < handoff.index(
        "launch_r15_expert_finetune_tmux.sh"
    )
    assert handoff.index("UPSTREAM_PRODUCER=bwa-r15s-expert-e22") < handoff.index(
        "launch_r15_expert_finetune_tmux.sh"
    )
    assert handoff.index("DISCOVERY_ACCEPTANCE=") < handoff.index(
        "VALIDATION_ACCEPTANCE="
    ) < handoff.index("FORMAL_ACCEPTANCE=")
    assert "bwa-r15s-dedup-e54" in handoff
    assert "bwa-r15s-dedup-e55" in handoff
    assert "bwa-r15s-dedup-e56" in handoff
