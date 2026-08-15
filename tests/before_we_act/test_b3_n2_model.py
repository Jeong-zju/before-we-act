from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from before_we_act.team_belief.n2_core import (  # noqa: E402
    B3N2Config,
    PredictiveTeamBeliefCore,
    TeacherBeliefInputs,
)


def tiny_config(**overrides) -> B3N2Config:
    values = {
        "n_belief_tokens": 4,
        "n_evidence_queries": 2,
        "event_capacity": 2,
        "temporal_layers": 1,
        "d_model": 8,
        "vision_dim": 6,
        "state_dim": 3,
        "action_dim": 2,
        "max_views": 3,
        "heads": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return B3N2Config(**values)


def runtime_inputs(config: B3N2Config, *, batch: int = 2, steps: int = 6):
    generator = torch.Generator().manual_seed(73)
    visual = torch.randn(
        batch,
        steps,
        2,
        3,
        config.vision_dim,
        generator=generator,
    )
    visual_mask = torch.ones(batch, steps, 2, 3, dtype=torch.bool)
    qpos = torch.randn(batch, steps, config.state_dim, generator=generator)
    action = torch.randn(batch, steps, config.action_dim, generator=generator)
    history_mask = torch.ones(batch, steps, dtype=torch.bool)
    action_mask = torch.ones(batch, steps, dtype=torch.bool)
    task = torch.randn(batch, config.d_model, generator=generator)
    reset = torch.zeros(batch, steps, dtype=torch.bool)
    reset[:, 0] = True
    return visual, visual_mask, qpos, action, history_mask, action_mask, task, reset


def run_core(model: PredictiveTeamBeliefCore, values, *, state=None):
    return model(*values, initial_state=state)


def test_capacity_is_explicit_and_future_contract_is_fail_closed() -> None:
    with pytest.raises(TypeError):
        B3N2Config()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="future step anchors"):
        tiny_config(future_offsets_steps=(4, 8, 16, 24))
    with pytest.raises(ValueError, match="two agents"):
        tiny_config(n_agent_anchors=3)
    with pytest.raises(ValueError, match="belief_unimix"):
        tiny_config(belief_unimix=0.0)


def test_discrete_belief_is_normalized_bounded_and_uncertainty_aware() -> None:
    torch.manual_seed(3)
    config = tiny_config(belief_factors=3, belief_classes=5, belief_unimix=0.01)
    model = PredictiveTeamBeliefCore(config, include_teacher=False).eval()
    hidden = torch.randn(2, config.n_belief_tokens, config.d_model)
    valid = torch.ones(2, dtype=torch.bool)
    low = model._distribution(hidden, torch.zeros(2), valid)
    high = model._distribution(hidden, torch.full((2,), 100.0), valid)
    _, low_sigma, low_log_probs, low_probs, low_entropy = low
    _, high_sigma, _, high_probs, high_entropy = high

    assert low_probs.shape == (2, 4, 3, 5)
    torch.testing.assert_close(low_probs.sum(-1), torch.ones_like(low_probs[..., 0]))
    torch.testing.assert_close(low_log_probs.exp(), low_probs)
    assert float(low_probs.min()) >= config.belief_unimix / config.belief_classes
    assert torch.all(high_entropy > low_entropy)
    assert torch.all(high_sigma > low_sigma)
    assert torch.max((high_probs - 0.2).abs()) < torch.max((low_probs - 0.2).abs())


def test_balanced_categorical_kl_is_bounded_and_splits_gradients() -> None:
    from before_we_act.team_belief.n2_losses import _balanced_categorical_kl

    def distribution(logits):
        base = logits.softmax(-1)
        probs = 0.99 * base + 0.01 / logits.shape[-1]
        return probs.log(), probs

    student_logits = torch.tensor([[[[40.0, -40.0, -40.0]]]], requires_grad=True)
    teacher_logits = torch.tensor([[[[-40.0, 40.0, -40.0]]]], requires_grad=True)
    student_log, student = distribution(student_logits)
    teacher_log, teacher = distribution(teacher_logits)
    total, dynamics, representation = _balanced_categorical_kl(
        student_log,
        student,
        teacher_log,
        teacher,
        free_nats=0.0,
        representation_scale=0.1,
    )
    theoretical_bound = torch.log(torch.tensor((3 * 0.99 + 0.01) / 0.01))
    assert dynamics <= theoretical_bound + 1e-5
    assert representation <= theoretical_bound + 1e-5
    assert torch.isfinite(total)

    dynamics.backward(retain_graph=True)
    assert student_logits.grad is not None and student_logits.grad.abs().sum() > 0
    assert teacher_logits.grad is None
    student_logits.grad = None
    representation.backward()
    assert student_logits.grad is None
    assert teacher_logits.grad is not None and teacher_logits.grad.abs().sum() > 0


def test_runtime_update_is_strictly_causal_and_reset_clears_old_episode() -> None:
    torch.manual_seed(5)
    model = PredictiveTeamBeliefCore(tiny_config(), include_teacher=False).eval()
    values = list(runtime_inputs(model.config, batch=1))
    baseline = run_core(model, values)

    future_changed = list(values)
    future_changed[0] = values[0].clone()
    future_changed[0][:, 3:] += 1000
    changed = run_core(model, future_changed)
    torch.testing.assert_close(
        baseline.mu_sequence[:, :3], changed.mu_sequence[:, :3], rtol=0, atol=0
    )

    reset_values = list(values)
    reset_values[-1] = values[-1].clone()
    reset_values[-1][:, 3] = True
    reset_baseline = run_core(model, reset_values)
    old_episode_changed = list(reset_values)
    old_episode_changed[0] = reset_values[0].clone()
    old_episode_changed[0][:, :3] -= 777
    after_reset = run_core(model, old_episode_changed)
    torch.testing.assert_close(
        reset_baseline.mu_sequence[:, 3:],
        after_reset.mu_sequence[:, 3:],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        reset_baseline.event_memory, after_reset.event_memory, rtol=0, atol=0
    )


def test_incremental_runtime_matches_one_pass_and_memory_is_bounded() -> None:
    torch.manual_seed(11)
    model = PredictiveTeamBeliefCore(tiny_config(), include_teacher=False).eval()
    values = runtime_inputs(model.config, batch=1)
    full = run_core(model, values)

    first = tuple(
        value[:, :3] if value.ndim >= 2 and value.shape[1] == 6 else value
        for value in values
    )
    second_values = []
    for index, value in enumerate(values):
        if value.ndim >= 2 and value.shape[1] == 6:
            sliced = value[:, 3:]
            if index == 7:
                sliced = torch.zeros_like(sliced)
            second_values.append(sliced)
        else:
            second_values.append(value)
    first_output = run_core(model, first)
    second = run_core(model, tuple(second_values), state=first_output.runtime_state)
    # Batched attention kernels may round differently when the same sequence is
    # presented as 6 steps or two 3-step chunks.  State carry must agree to
    # normal float32 numerical precision; bitwise identity is not a valid GPU
    # resume contract across different batch shapes.
    torch.testing.assert_close(full.mu, second.mu, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        full.event_memory, second.event_memory, rtol=1e-6, atol=1e-6
    )
    assert full.event_memory.shape[1] == model.config.event_capacity
    assert full.event_mask.sum() == model.config.event_capacity


def test_mixed_padding_backward_is_finite() -> None:
    torch.manual_seed(13)
    model = PredictiveTeamBeliefCore(tiny_config(), include_teacher=False).train()
    values = list(runtime_inputs(model.config, batch=2))
    values[1][1, :4] = False
    values[4][1, :4] = False
    values[5][1, :4] = False
    values[7][1] = False
    values[7][1, 4] = True
    output = run_core(model, values)
    loss = output.mu.square().mean() + output.future_latent_prediction.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def teacher_inputs(config: B3N2Config) -> TeacherBeliefInputs:
    generator = torch.Generator().manual_seed(91)
    current = torch.randn(1, 3, 2, config.vision_dim, generator=generator)
    future = torch.randn(1, 4, 3, 2, config.vision_dim, generator=generator)
    current_mask = torch.ones(1, 3, 2, dtype=torch.bool)
    future_mask = torch.ones(1, 4, 3, 2, dtype=torch.bool)
    anchor_mask = torch.tensor([[True, False, True, False]])
    agent_state = torch.randn(1, 2, config.state_dim, generator=generator)
    return TeacherBeliefInputs(
        current,
        current_mask,
        future,
        future_mask,
        anchor_mask,
        agent_state,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 1]]),
    )


def test_teacher_masks_missing_anchors_is_agent_order_invariant_and_strippable() -> None:
    torch.manual_seed(17)
    model = PredictiveTeamBeliefCore(tiny_config(), include_teacher=True).train()
    task = torch.randn(1, model.config.d_model)
    inputs = teacher_inputs(model.config)
    original = model.forward_teacher(inputs, task)
    assert torch.equal(
        original.future_latent_target[:, 1],
        torch.zeros_like(original.future_latent_target[:, 1]),
    )
    assert torch.equal(
        original.future_latent_target[:, 3],
        torch.zeros_like(original.future_latent_target[:, 3]),
    )

    permuted = TeacherBeliefInputs(
        inputs.current_visual_tokens,
        inputs.current_visual_mask,
        inputs.future_visual_tokens,
        inputs.future_visual_mask,
        inputs.future_anchor_mask,
        inputs.agent_state[:, [1, 0]],
        inputs.agent_mask[:, [1, 0]],
        inputs.relative_agent_role[:, [1, 0]],
    )
    reordered = model.forward_teacher(permuted, task)
    torch.testing.assert_close(original.mu, reordered.mu, rtol=1e-6, atol=1e-6)

    model.eval()
    evaluated = model.forward_teacher(inputs, task)
    torch.testing.assert_close(original.mu, evaluated.mu, rtol=0, atol=0)
    model.strip_teacher_()
    assert not any(key.startswith("teacher_branch.") for key in model.state_dict())
    with pytest.raises(RuntimeError, match="stripped"):
        model.forward_teacher(inputs, task)


def test_uncertainty_reduces_reliability() -> None:
    low = torch.full((2, 4, 8), 0.1)
    high = torch.full((2, 4, 8), 3.0)
    low_reliability = PredictiveTeamBeliefCore.reliability_from_sigma(low)
    high_reliability = PredictiveTeamBeliefCore.reliability_from_sigma(high)
    assert torch.all(low_reliability > high_reliability)


def test_auxiliary_losses_mask_tail_anchors_and_report_each_horizon() -> None:
    from before_we_act.b3_n2_model import B3N2PolicyOutput
    from before_we_act.team_belief.n2_losses import (
        B3N2LossWeights,
        compute_b3_n2_losses,
    )

    torch.manual_seed(19)
    core = PredictiveTeamBeliefCore(tiny_config(), include_teacher=True).train()
    values = runtime_inputs(core.config, batch=1)
    belief = run_core(core, values)
    teacher = core.forward_teacher(teacher_inputs(core.config), values[-2])
    # A masked tail anchor must not affect the aggregate even if its prediction
    # is deliberately pathological.
    belief.future_latent_prediction[:, 1] = 1e6
    belief.future_latent_prediction[:, 3] = -1e6
    prediction = torch.zeros(1, 3, core.config.action_dim)
    output = B3N2PolicyOutput(
        prediction=prediction,
        base_prediction=prediction.clone(),
        belief_residual=prediction.clone(),
        residual_gate=torch.ones(1, 3, 1),
        action_posterior_mu=None,
        action_posterior_logvar=None,
        belief=belief,
        teacher=teacher,
        current_visual_raw=torch.zeros(1, 2, core.config.vision_dim),
        dense_routes=torch.ones(1, 3, 4) / 4,
    )
    losses = compute_b3_n2_losses(
        output,
        torch.ones_like(prediction),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 4, core.config.state_dim),
        torch.ones(1, 4, dtype=torch.bool),
        torch.zeros(1, 16, core.config.action_dim),
        torch.ones(1, 16, dtype=torch.bool),
        B3N2LossWeights(1, 1, 1, 1, 1, 1, 1, 1, 1),
    )
    assert torch.isfinite(losses["total"])
    assert losses["future_0.4s"] == 0
    assert losses["future_1.6s"] == 0
    assert {"future_0.2s", "future_0.4s", "future_0.8s", "future_1.6s"} <= set(
        losses
    )
    losses["total"].backward()
    teacher_gradients = [
        parameter.grad
        for parameter in core.teacher_branch.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert teacher_gradients
    assert all(torch.isfinite(gradient).all() for gradient in teacher_gradients)


class _FakeProcessor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.25, 0.25, 0.25)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


class _FakeVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=768, num_register_tokens=0)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        spatial = torch.linspace(
            -1, 1, 1201, device=pixel_values.device, dtype=pixel_values.dtype
        ).view(1, 1201, 1)
        signal = pixel_values.mean((1, 2, 3)).view(batch, 1, 1)
        return types.SimpleNamespace(
            last_hidden_state=(spatial + signal).expand(batch, 1201, 768)
        )


@pytest.fixture
def fake_policy_dependencies(monkeypatch):
    module = types.ModuleType("transformers")
    module.AutoImageProcessor = _FakeProcessor
    module.AutoModel = _FakeVision
    monkeypatch.setitem(sys.modules, "transformers", module)
    torchvision = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")
    models.resnet18 = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("resnet18 is outside the frozen DINO policy")
    )
    torchvision.models = models
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)


def test_policy_zero_init_and_b_off_are_exact(fake_policy_dependencies) -> None:
    # Import only after dependency stubs are installed; train_act binds these
    # classes at module-import time.
    for name in tuple(sys.modules):
        if name == "before_we_act.b0h_model" or name == "before_we_act.b3_n2_model":
            sys.modules.pop(name)
        elif name.startswith("stereo_core"):
            sys.modules.pop(name)
    from before_we_act.b3_n2_model import B3N2Policy

    config = tiny_config(vision_dim=768, state_dim=9, action_dim=8)
    torch.manual_seed(23)
    model = B3N2Policy(
        config,
        state_dim=9,
        action_dim=8,
        horizon=3,
        d_model=8,
        enc_layers=1,
        dec_layers=1,
        roles=4,
        role_rank=4,
        history_layers=1,
        dino_model="offline/fake",
    ).eval()
    history_mask = torch.zeros(1, 16, dtype=torch.bool)
    history_mask[:, -1] = True
    action_mask = torch.zeros_like(history_mask)
    kwargs = {
        "global_rgb": torch.zeros(1, 3, 480, 640),
        "local_rgb": torch.ones(1, 3, 480, 640),
        "history_visual_raw": torch.zeros(1, 16, 2, 768),
        "history_qpos": torch.zeros(1, 16, 9),
        "history_action": torch.zeros(1, 16, 8),
        "history_mask": history_mask,
        "action_history_mask": action_mask,
        "task_bytes": torch.tensor([[65] + [256] * 63]),
        "task_text_mask": torch.tensor([[True] + [False] * 63]),
        "episode_reset": torch.tensor([True]),
    }
    with torch.no_grad():
        initial = model(**kwargs)
    assert torch.equal(initial.prediction, initial.base_prediction)

    with torch.no_grad():
        model.direct_belief_residual.output.weight.fill_(0.1)
        model.direct_belief_residual.output.bias.fill_(0.1)
        enabled = model(**kwargs, belief_enabled=True)
        disabled = model(**kwargs, belief_enabled=False)
    assert not torch.equal(enabled.prediction, enabled.base_prediction)
    assert torch.equal(disabled.prediction, disabled.base_prediction)
    assert torch.equal(disabled.belief_residual, torch.zeros_like(disabled.belief_residual))
    assert not any(
        key.startswith("belief_core.teacher_branch.")
        for key in model.deployment_state_dict()
    )
    deployment = B3N2Policy(
        config,
        state_dim=9,
        action_dim=8,
        horizon=3,
        d_model=8,
        enc_layers=1,
        dec_layers=1,
        roles=4,
        role_rank=4,
        history_layers=1,
        dino_model="offline/fake",
        include_teacher=False,
    ).eval()
    deployment.load_state_dict(model.deployment_state_dict(), strict=True)
    with torch.no_grad():
        deployed = deployment(**kwargs)
    torch.testing.assert_close(deployed.prediction, enabled.prediction, rtol=0, atol=0)
