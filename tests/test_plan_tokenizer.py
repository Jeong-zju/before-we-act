import inspect
import io

import torch

from models.plan_tokenizer import (
    ActionOnlyPlanTokenizer,
    ActionOnlyPlanTokenizerConfig,
    PlanCodeSupport,
    PlanCodeSupportAccumulator,
    build_plan_code_support,
    compute_action_only_plan_losses,
)


def test_action_only_tokenizer_excludes_outcomes_and_has_balanced_usage_loss():
    cfg = ActionOnlyPlanTokenizerConfig(
        horizon=6,
        action_dim=3,
        latent_dim=12,
        hidden_dim=32,
        codebook_size=8,
        residual_dropout=0.0,
        auxiliary_traj_dim=2,
        auxiliary_traj_weight=0.5,
    )
    model = ActionOnlyPlanTokenizer(cfg)
    actions = torch.randn(5, 6, 3)
    trajectory = torch.randn(5, 6, 2)

    # There is structurally no trajectory/phase argument on the  encoder.
    assert tuple(inspect.signature(model.encode).parameters) == ("actions",)
    out = model(actions)
    assert out["recon_actions"].shape == (5, 6, 3)
    assert out["recon_auxiliary_trajectory"].shape == (5, 6, 2)
    assert out["soft_code_usage"].shape == (8,)
    assert torch.allclose(out["soft_code_usage"].sum(), torch.tensor(1.0), atol=1e-5)
    assert out["usage_balance_loss"].ndim == 0
    assert torch.isfinite(out["usage_balance_loss"])

    losses = compute_action_only_plan_losses(model, {"actions": actions, "trajectory": trajectory})
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.vq.embedding.weight.grad is not None
    assert torch.isfinite(model.vq.embedding.weight.grad).all()

    diagnostics = compute_action_only_plan_losses(
        model,
        {"actions": actions, "trajectory": trajectory},
        include_rate_distortion_metrics=True,
    )
    assert diagnostics["loss_code_only_action"].ndim == 0
    assert diagnostics["loss_mean_action_baseline"].ndim == 0
    assert torch.isfinite(diagnostics["loss_code_only_action"])
    assert torch.isfinite(diagnostics["loss_mean_action_baseline"])


def test_action_only_tokenizer_residual_dropout_prevents_continuous_bypass():
    cfg = ActionOnlyPlanTokenizerConfig(
        horizon=4,
        action_dim=2,
        latent_dim=8,
        hidden_dim=16,
        codebook_size=4,
        residual_dropout=1.0,
    )
    model = ActionOnlyPlanTokenizer(cfg).train()
    z_q = torch.randn(3, 8)

    decoded_a = model.decode(z_q, torch.randn(3, 8))["recon_actions"]
    decoded_b = model.decode(z_q, torch.randn(3, 8) * 100.0)["recon_actions"]
    assert torch.equal(decoded_a, decoded_b)


def test_plan_code_support_uses_empirical_codes_and_residual_statistics():
    codes = torch.tensor([0, 0, 2, 2, 2, 3])
    residual = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 20.0],
            [12.0, 22.0],
            [14.0, 24.0],
            [99.0, 99.0],
        ]
    )
    support = build_plan_code_support(codes, residual, codebook_size=5, min_count=2, std_floor=1e-4)

    assert torch.equal(support.counts, torch.tensor([2, 0, 3, 1, 0]))
    assert torch.equal(support.active_codes, torch.tensor([0, 2]))
    assert torch.allclose(support.residual_mean[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(support.residual_std[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(support.residual_mean[2], torch.tensor([12.0, 22.0]))

    sampled = support.sample(128, generator=torch.Generator().manual_seed(7))
    assert sampled["code_indices"].shape == (128,)
    assert sampled["residual"].shape == (128, 2)
    assert set(sampled["code_indices"].tolist()) <= {0, 2}
    diverse = support.sample(
        2,
        generator=torch.Generator().manual_seed(4),
        ensure_code_diversity=True,
    )
    assert set(diverse["code_indices"].tolist()) == {0, 2}

    # The support artifact survives ordinary checkpoint serialization.
    checkpoint = io.BytesIO()
    torch.save(support.to_dict(), checkpoint)
    checkpoint.seek(0)
    restored = PlanCodeSupport.from_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(restored.counts, support.counts)
    assert torch.equal(restored.active_codes, support.active_codes)
    assert torch.allclose(restored.residual_mean, support.residual_mean)


def test_plan_code_support_accumulator_is_streaming():
    accumulator = PlanCodeSupportAccumulator(codebook_size=4, residual_dim=2)
    accumulator.update(torch.tensor([1, 1]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    accumulator.update(torch.tensor([3]), torch.tensor([[5.0, 6.0]]))
    support = accumulator.build(min_count=1)

    assert torch.equal(support.active_codes, torch.tensor([1, 3]))
    assert torch.allclose(support.probabilities, torch.tensor([0.0, 2 / 3, 0.0, 1 / 3]))
