from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from models.static_rgb_act import StaticRGBMoEACTConfig
from models.wam_multimodal import AgentFactorizedFlowWAM
from scripts.run_agent_factorized_flow_inference import _validated_generation
from scripts.serve_robofactory_m2_rollout import _validate_client
from train.agent_factorized_flow_training import (
    make_flow_matching_batch,
    uniform_masked_flow_mse,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> StaticRGBMoEACTConfig:
    return StaticRGBMoEACTConfig(
        horizon=6,
        vision_dim=16,
        d_model=32,
        encoder_layers=1,
        decoder_layers=2,
        heads=4,
        ffn_dim=64,
        latent_dim=8,
        experts=4,
        dropout=0.0,
        dense_ffn_dim=128,
    )


def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_flow_model_is_per_agent_shared_decoder_without_cvae_or_future():
    config = _config()
    model = AgentFactorizedFlowWAM(config)
    vision = torch.randn(3, 12, config.vision_dim)
    state = torch.randn(3, config.state_dim)
    noisy_actions = torch.randn(3, config.horizon, config.action_dim)
    flow_time = torch.rand(3)

    velocity, router_aux = model(vision, state, noisy_actions, flow_time)
    (velocity.square().mean() + 0.01 * router_aux).backward()

    assert velocity.shape == noisy_actions.shape
    assert torch.isfinite(velocity).all()
    assert model.action_projection.weight.grad is not None
    assert model.decoder.layers[0].moe.router.weight.grad is not None
    assert not hasattr(model, "posterior")
    assert not hasattr(model, "latent")
    assert not hasattr(model, "future")


def test_flow_matching_batch_matches_rectified_path_definition():
    target = torch.randn(4, 6, 8)
    sampled = make_flow_matching_batch(
        target,
        generator=torch.Generator().manual_seed(17),
    )
    interpolation = sampled.flow_time[:, None, None]

    torch.testing.assert_close(
        sampled.action_inputs,
        (1.0 - interpolation) * sampled.initial_actions
        + interpolation * target,
    )
    torch.testing.assert_close(
        sampled.target_velocity,
        target - sampled.initial_actions,
    )
    assert bool(sampled.flow_time.ge(0).all())
    assert bool(sampled.flow_time.lt(1).all())


def test_uniform_masked_flow_loss_averages_agents_equally():
    prediction = torch.zeros(2, 2, 2)
    target = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[3.0, 3.0], [100.0, 100.0]],
        ]
    )
    valid = torch.tensor([[True, True], [True, False]])

    loss = uniform_masked_flow_mse(prediction, target, valid)

    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_four_step_euler_is_the_deployed_atomic_solver():
    config = _config()
    model = AgentFactorizedFlowWAM(config).eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.velocity_head.bias.data.fill_(1.0)
    vision = torch.zeros(2, 3, config.vision_dim)
    state = torch.zeros(2, config.state_dim)
    initial = torch.zeros(2, config.horizon, config.action_dim)

    generated = model.generate_actions(
        vision,
        state,
        initial_actions=initial,
        solver_steps=4,
        solver="euler",
    )

    torch.testing.assert_close(generated, torch.ones_like(generated))


def test_f1_config_changes_only_the_action_generator_vertical_slice():
    s0 = _yaml(ROOT / "configs/wam_flow/s0_candidate.yaml")
    f1 = _yaml(ROOT / "configs/wam_flow/s1_r1_f1_flow_cold.yaml")

    assert f1["round"]["action_generator"] == "rectified_flow_cold"
    assert f1["data"] == s0["data"]
    assert f1["vision"] == s0["vision"]
    assert f1["model"] == s0["model"]
    for key in (
        "seed",
        "batch_size",
        "num_workers",
        "updates",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "router_aux_weight",
        "save_interval",
    ):
        assert f1["training"][key] == s0["training"][key]
    assert f1["training"]["active_agent_loss_weighting"] is False
    assert f1["training"]["objective"] == "rectified_flow_velocity_mse"
    assert f1["generation"] == {
        "source_distribution": "standard_normal",
        "solver_steps": 4,
        "solver": "euler",
        "normalized_action_clip": 10.0,
    }
    assert f1["inference"]["chunk_aggregation"] == (
        s0["inference"]["chunk_aggregation"]
    )
    assert f1["inference"]["temporal_ensemble_decay"] == (
        s0["inference"]["temporal_ensemble_decay"]
    )


def test_generation_contract_rejects_warm_or_nondefault_solver():
    expected = {
        "source_distribution": "standard_normal",
        "solver_steps": 4,
        "solver": "euler",
        "normalized_action_clip": 10.0,
    }

    assert _validated_generation(expected, expected) == expected
    with pytest.raises(ValueError, match="S1-R1 F1"):
        _validated_generation(
            {**expected, "source_distribution": "previous_chunk"},
            expected,
        )


def test_rollout_server_accepts_only_the_frozen_f1_checkpoint_source_pair():
    contract = {"task_id": "lift_barrier", "future_path": False}
    client = {
        "checkpoint_format": (
            "wam.robofactory.agent_factorized_flow.checkpoint/1"
        ),
        "task_vocabulary": ["lift_barrier", "long_pipeline_delivery"],
        "future_path": False,
        "policy": {
            "action_source": "agent_factorized_rectified_flow_cold",
        },
    }

    assert _validate_client(client, contract=contract) == client
    with pytest.raises(RuntimeError, match="supported direct action source"):
        _validate_client(
            {
                **client,
                "policy": {"action_source": "static_rgb_dino_act_moe"},
            },
            contract=contract,
        )


def test_f1_runtime_routes_training_and_inference_to_agent_flow():
    env = (
        ROOT / "experiments/wam_flow/s1_r1/candidate.env"
    ).read_text(encoding="utf-8")
    card = _yaml(ROOT / "experiments/wam_flow/s1_r1/candidate_card.yaml")
    runner = (ROOT / "scripts/run_lpd_single_5090.sh").read_text(
        encoding="utf-8"
    )
    gate = (ROOT / "scripts/run_lpd_fixed_seed_gate.sh").read_text(
        encoding="utf-8"
    )
    retry = (ROOT / "scripts/retry_s1_r1_f1_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "S1_R1_CANDIDATE_ID=F1" in env
    assert "LPD_POLICY_KIND=agent_flow" in env
    assert "train_agent_factorized_flow_wam.py" in runner
    assert "run_agent_factorized_flow_inference.py" in gate
    assert 's1/r1-f1-flow-cold' in retry
    assert 'LPD_POLICY_KIND=agent_flow' in retry
    assert '"${FE_ROOT}/scripts/run_lpd_single_5090.sh" gate' in retry
    assert 'test -f "${CHECKPOINT}"' in retry
    assert card["runtime"]["gpu_index"] == 1
    assert card["runtime"]["future_path"] is False
    assert card["runtime"]["warm_start"] is False
