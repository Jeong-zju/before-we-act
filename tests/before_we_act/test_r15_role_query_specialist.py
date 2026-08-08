from __future__ import annotations

from pathlib import Path

import pytest
import torch

from before_we_act.action_generator.evolution import (
    bind_role_conditioned_spatial_queries,
)
from before_we_act.action_generator.spatial_bridge import SpatialQueryBridge
from before_we_act.contracts import TeamBeliefState


def test_role_queries_are_grouped_by_action_slot_and_mask_inactive_agents():
    agent_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    slot_delta = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0], [7.0, 7.0, 7.0]],
        ]
    )

    conditioned, mask = bind_role_conditioned_spatial_queries(
        slot_delta, agent_mask, 8
    )

    for batch in range(2):
        for agent in range(4):
            expected = slot_delta[batch, agent].expand(2, -1)
            assert torch.equal(
                conditioned[batch, 2 * agent : 2 + 2 * agent], expected
            )
            assert mask[batch, 2 * agent : 2 + 2 * agent].tolist() == [
                bool(agent_mask[batch, agent])
            ] * 2


@pytest.mark.parametrize("query_count", [3, 6, 14])
def test_role_query_grouping_rejects_non_exact_action_slot_partition(query_count):
    with pytest.raises(ValueError, match="grouping contract"):
        bind_role_conditioned_spatial_queries(
            torch.zeros(1, 4, 3),
            torch.ones(1, 4, dtype=torch.bool),
            query_count,
        )


def test_role_bias_enters_spatial_cross_attention_and_masks_unused_slot():
    torch.manual_seed(7)
    bridge = SpatialQueryBridge(spatial_dim=8)
    belief = TeamBeliefState(
        tokens=torch.randn(1, 16, 96),
        agent_tokens=torch.randn(1, 4, 96),
        consensus_token=torch.randn(1, 96),
        uncertainty=torch.zeros(1, 1),
        agent_mask=torch.tensor([[True, True, True, False]]),
    )
    spatial = torch.randn(1, 5, 48, 8)
    view_mask = torch.tensor([[True, True, True, True, False]])
    baseline, baseline_mask = bridge(belief, spatial, view_mask)
    slot_delta = torch.randn(1, 4, 96) * belief.agent_mask[:, :, None]
    query_bias, query_mask = bind_role_conditioned_spatial_queries(
        slot_delta, belief.agent_mask, bridge.query_count
    )
    conditioned, conditioned_mask = bridge(
        belief,
        spatial,
        view_mask,
        query_bias=query_bias,
        query_mask=query_mask,
    )

    assert not torch.equal(conditioned[:, -16:], baseline[:, -16:])
    assert baseline_mask[:, -16:].all()
    assert conditioned_mask[:, -16:].sum().item() == 12
    assert not conditioned_mask[0, -4:].any()


def test_role_query_launchers_keep_branch_session_and_phase_identity():
    root = Path(__file__).resolve().parents[2]
    expert = (
        root / "scripts/before_we_act/launch_r15_expert_finetune_tmux.sh"
    ).read_text()
    runner = (
        root / "scripts/before_we_act/run_r15_expert_finetune_candidate.sh"
    ).read_text()
    temporal = (
        root / "scripts/before_we_act/launch_r15_temporal_screens_tmux.sh"
    ).read_text()
    formal = (
        root / "scripts/before_we_act/launch_r15_formal_stack_tmux.sh"
    ).read_text()
    handoff = (
        root / "scripts/before_we_act/handoff_r15_role_query_promotion.sh"
    ).read_text()

    assert "role-query-specialist" in expert
    assert "role-query-specialist" in runner
    assert "role-query-specialist" in temporal
    assert "role-query-specialist" in formal
    assert "--phase-manifest" in expert
    assert "--session" in temporal
    assert "--session" in formal
    assert "role_query_act_temporal_ensemble" in temporal
    assert "role_query_act_temporal_ensemble" in formal
    assert handoff.index("DISCOVERY_ACCEPTANCE=") < handoff.index(
        "VALIDATION_ACCEPTANCE="
    ) < handoff.index("FORMAL_ACCEPTANCE=")
    assert "bwa-r15s-role-e51" in handoff
    assert "bwa-r15s-role-e52" in handoff
    assert "bwa-r15s-role-e53" in handoff
