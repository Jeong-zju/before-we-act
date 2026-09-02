from __future__ import annotations

import pytest
import torch

from before_we_act.mars_care_learned_proposal import (
    FIXED_LIBRARY_LABEL_USAGE,
    MARSCARELearnedProposalConfig,
    MARSCARELearnedProposalHead,
    MARSCAREProposalBootstrapLossConfig,
    mars_care_proposal_bootstrap_loss,
    normalize_fixed_library_candidates,
)


def _inputs(
    *, batch: int = 3, tokens: int = 7, d_model: int = 32, candidates: int = 4
) -> tuple[
    MARSCARELearnedProposalHead,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(17)
    config = MARSCARELearnedProposalConfig(
        d_model=d_model,
        heads=4,
        hidden_width=64,
        candidates=candidates,
        dropout=0.0,
    )
    model = MARSCARELearnedProposalHead(config)
    reference = torch.randn(batch, 100, 8)
    memory = torch.randn(batch, tokens, d_model)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    mask[:, -1] = False
    return model, reference, memory, mask


def test_candidate_zero_is_bit_exact_reference_and_contract_is_100x8() -> None:
    model, reference, memory, mask = _inputs()
    output = model(reference, memory, mask)

    assert output.candidates_normalized.shape == (3, 4, 100, 8)
    assert output.residuals_normalized.shape == (3, 4, 100, 8)
    # ``equal`` catches any arithmetic/cast drift, not just close values.
    assert torch.equal(output.candidates_normalized[:, 0], reference)
    assert torch.equal(output.reference_normalized, reference)
    assert torch.equal(
        output.residuals_normalized[:, 0], torch.zeros_like(reference)
    )


def test_alternative_residuals_are_bounded_in_normalized_space() -> None:
    limits = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    config = MARSCARELearnedProposalConfig(
        d_model=32,
        heads=4,
        hidden_width=64,
        candidates=5,
        dropout=0.0,
        normalized_residual_limit=limits,
    )
    model = MARSCARELearnedProposalHead(config)
    reference = torch.zeros(2, 100, 8)
    memory = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    # Force a very large pre-tanh output to exercise the bound itself.
    with torch.no_grad():
        model.residual_output.bias.fill_(100.0)
    output = model(reference, memory, mask)
    alternatives = output.residuals_normalized[:, 1:]
    limit = torch.tensor(limits).view(1, 1, 1, 8)
    assert torch.all(alternatives.abs() <= limit + 1e-6)
    assert torch.all(alternatives.abs() > limit * 0.99)


def test_slots_share_reader_fusion_and_output_parameters() -> None:
    model, _reference, _memory, _mask = _inputs(candidates=6)
    # One projection is shared by all K-1 slots; there are no per-slot heads.
    assert model.residual_output.weight.shape == (100 * 8, 32)
    assert model.slot_embedding.shape == (1, 5, 32)
    assert not hasattr(model, "residual_outputs")
    assert not hasattr(model, "slot_heads")


def test_each_batch_row_reads_only_its_own_local_memory() -> None:
    model, reference, memory, mask = _inputs(batch=2, candidates=4)
    model.eval()
    with torch.no_grad():
        model.residual_output.weight.normal_(std=0.02)
        baseline = model(reference, memory, mask).candidates_normalized
        changed_memory = memory.clone()
        changed_memory[1].add_(10.0 * torch.randn_like(changed_memory[1]))
        changed = model(reference, changed_memory, mask).candidates_normalized
    # Batching is only scheduling: modifying arm/row 1 cannot affect arm/row 0.
    assert torch.equal(baseline[0], changed[0])
    assert not torch.equal(baseline[1, 1:], changed[1, 1:])
    assert torch.equal(changed[:, 0], reference)


def test_bootstrap_loss_uses_fixed_library_only_as_candidate_initialization() -> None:
    model, reference, memory, mask = _inputs(candidates=4)
    output = model(reference, memory, mask)
    # A prepared fixed library has the same normalized reference in slot 0;
    # alternatives are treated as initialization targets, not branch outcomes.
    fixed = output.candidates_normalized.detach().clone()
    fixed[:, 1, :, 0] += 0.15
    fixed[:, 2, :, 1] -= 0.10
    fixed[:, 3, :, 2] += 0.05
    loss, pieces = mars_care_proposal_bootstrap_loss(output, fixed)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert set(pieces) == {"bootstrap_coverage", "diversity", "total"}
    assert FIXED_LIBRARY_LABEL_USAGE == "bootstrap_only"
    loss.backward()
    assert model.residual_output.weight.grad is not None
    # The bootstrap target is detached internally, so no gradient can leak
    # into a caller-owned fixed-library tensor.
    fixed.requires_grad_(True)
    output = model(reference, memory, mask)
    loss, _ = mars_care_proposal_bootstrap_loss(output, fixed)
    loss.backward()
    assert fixed.grad is None


def test_diversity_term_penalizes_collapsed_slots() -> None:
    model, reference, memory, mask = _inputs(candidates=4)
    collapsed_output = model(reference, memory, mask)
    collapsed = collapsed_output.candidates_normalized.detach().clone()
    distinct_residuals = collapsed_output.residuals_normalized.detach().clone()
    distinct_residuals[:, 1, :, 0] += 0.20
    distinct_residuals[:, 2, :, 1] -= 0.20
    distinct_residuals[:, 3, :, 2] += 0.20
    distinct_residuals.requires_grad_(True)
    distinct_candidates = torch.cat(
        (
            reference.unsqueeze(1),
            reference.unsqueeze(1) + distinct_residuals[:, 1:],
        ),
        dim=1,
    )
    distinct_output = type(collapsed_output)(
        candidates_normalized=distinct_candidates,
        residuals_normalized=distinct_residuals,
        slot_state=collapsed_output.slot_state,
        reference_normalized=reference,
    )
    distinct = distinct_candidates.detach()
    collapsed_loss, collapsed_parts = mars_care_proposal_bootstrap_loss(
        collapsed_output,
        collapsed,
        config=MARSCAREProposalBootstrapLossConfig(
            coverage_weight=0.0, diversity_weight=1.0, diversity_margin=0.1
        ),
    )
    distinct_loss, distinct_parts = mars_care_proposal_bootstrap_loss(
        distinct_output,
        distinct,
        config=MARSCAREProposalBootstrapLossConfig(
            coverage_weight=0.0, diversity_weight=1.0, diversity_margin=0.1
        ),
    )
    assert collapsed_parts["diversity"] > distinct_parts["diversity"]
    assert collapsed_loss > distinct_loss
    distinct_loss.backward()
    assert distinct_residuals.grad is not None


def test_fixed_library_normalization_requires_reference_statistics() -> None:
    physical = torch.zeros(2, 4, 100, 8)
    normalized = normalize_fixed_library_candidates(
        physical,
        action_mean=torch.arange(8),
        action_std=torch.ones(8),
    )
    assert normalized.shape == physical.shape
    expected = -torch.arange(8).float().expand(2, -1)
    assert torch.equal(normalized[:, 0, 0], expected)
    with pytest.raises(ValueError, match="standard deviation"):
        normalize_fixed_library_candidates(
            physical, torch.zeros(8), torch.tensor([1.0] * 7 + [0.0])
        )


def test_proposal_rejects_all_masked_local_memory_and_shape_drift() -> None:
    model, reference, memory, mask = _inputs()
    mask.zero_()
    with pytest.raises(ValueError, match="at least one"):
        model(reference, memory, mask)
    with pytest.raises(ValueError, match="100,8"):
        model(torch.zeros(3, 99, 8), memory, mask.bool().fill_(True))
