from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from rich.console import Console

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION
from eval.rwm_ar_open_loop import evaluate_open_loop
from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    WorldModelRolloutInputs,
    WorldModelSequenceInputs,
)
from scripts.train_phase1_rwm_ar import main as train_phase1_rwm_ar_main
from train.rwm_ar_checkpointing import load_wam_checkpoint, save_wam_checkpoint
from train.rwm_ar_losses import RWMLossWeights, compute_rwm_loss
from train.progress import PhaseDisplay, TrainingProgress
from train.trajectory_dataset import (
    InMemoryProprioSequenceDataset,
    ProprioSequenceDataset,
)


def test_rwm_ar_shapes_outer_autoregression_and_no_privileged_inputs():
    stats = _stats()
    config = RWMARConfig(
        history_horizon=4,
        train_forecast_horizon=3,
        encoder_hidden_dim=16,
        gru_hidden_dim=12,
        gru_layers=2,
    )
    model = RWMARWorldModel(config, stats)
    history = WorldModelSequenceInputs(
        states=torch.zeros(2, 4, 22),
        past_actions=torch.zeros(2, 3, 8),
        valid_mask=torch.tensor([[False, False, True, True], [True, True, True, True]]),
    )
    actions = torch.zeros(2, 3, 8)
    predictions = model.predict(history, actions)

    assert predictions.next_state_mean.shape == (2, 3, 22)
    assert predictions.state_delta_log_std.shape == (2, 3, 22)
    assert predictions.gripper_closed_logit.shape == (2, 3, 2)
    assert predictions.reward.shape == (2, 3, 1)
    assert torch.isfinite(predictions.next_state_mean).all()
    assert set(WorldModelSequenceInputs.__dataclass_fields__) == {
        "states",
        "past_actions",
        "valid_mask",
    }

    public = model(
        WorldModelRolloutInputs(
            history=history, candidate_actions=actions, num_particles=3
        )
    )
    assert public.state_distribution["mean"].shape == (3, 2, 3, 22)
    assert public.rewards.shape == (3, 2, 3, 1)
    assert public.uncertainty["aleatoric_std"].min() > 0.0


def test_amp_safe_missing_previous_action_with_constant_action_dimension():
    base = _stats()
    stats = NormalizationStats(
        state_mean=base.state_mean,
        state_std=base.state_std,
        action_mean=np.ones(8, dtype=np.float32),
        action_std=np.full(8, 1e-6, dtype=np.float32),
        delta_mean=base.delta_mean,
        delta_std=base.delta_std,
        reward_mean=base.reward_mean,
        reward_std=base.reward_std,
    )
    model = RWMARWorldModel(
        RWMARConfig(
            history_horizon=4,
            encoder_hidden_dim=16,
            gru_hidden_dim=12,
            gru_layers=1,
        ),
        stats,
    ).half()
    history = WorldModelSequenceInputs(
        states=torch.zeros(2, 4, 22, dtype=torch.float16),
        past_actions=torch.ones(2, 3, 8, dtype=torch.float16),
        valid_mask=torch.tensor([[False, False, True, True], [True, True, True, True]]),
    )
    predictions = model.predict(history, torch.ones(2, 4, 8, dtype=torch.float16))

    for name in predictions.__dataclass_fields__:
        assert torch.isfinite(getattr(predictions, name)).all(), name


def test_phase1_progress_matches_phase0_adaptive_braille_style():
    progress = TrainingProgress(enabled=True, total_stages=1)
    phase = progress.add_phase("train RWM-AR H=4", 3)
    phase.advance({"epoch": 1, "epochs": 1, "loss": 10.0})
    phase.advance({"epoch": 1, "epochs": 1, "loss": 5.0})
    for width in (40, 80, 120):
        output = StringIO()
        Console(file=output, width=width, color_system=None).print(PhaseDisplay(phase))
        lines = output.getvalue().splitlines()
        assert len(lines) == 7
        assert all(len(line) == width for line in lines)
        assert "steps 1→2" in output.getvalue()
        assert any("⠀" < character <= "⣿" for character in output.getvalue())
    phase.finish("done")


def test_multistep_loss_is_finite_and_backpropagates():
    model = RWMARWorldModel(
        RWMARConfig(
            history_horizon=3,
            train_forecast_horizon=2,
            encoder_hidden_dim=16,
            gru_hidden_dim=12,
            gru_layers=1,
        ),
        _stats(),
    )
    batch = _tensor_batch(batch_size=4, history=3, forecast=2)
    predictions = model.predict(
        WorldModelSequenceInputs(
            batch["states"], batch["past_actions"], batch["valid_mask"]
        ),
        batch["candidate_actions"],
    )
    loss, components = compute_rwm_loss(
        predictions,
        batch,
        delta_std=model.delta_std,
        yaw_indices=model.config.yaw_indices,
        closed_indices=model.config.gripper_closed_indices,
        positive_weights={
            name: torch.tensor(1.0) for name in ("done", "success", "failure")
        },
        horizon_decay=0.95,
        weights=RWMLossWeights(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_state_mean_objective_cannot_be_bypassed_by_variance_inflation():
    model = RWMARWorldModel(
        RWMARConfig(
            history_horizon=3,
            train_forecast_horizon=2,
            encoder_hidden_dim=16,
            gru_hidden_dim=12,
            gru_layers=1,
        ),
        _stats(),
    )
    batch = _tensor_batch(batch_size=4, history=3, forecast=2)
    predictions = model.predict(
        WorldModelSequenceInputs(
            batch["states"], batch["past_actions"], batch["valid_mask"]
        ),
        batch["candidate_actions"],
    )
    weights = RWMLossWeights(
        state_mean=1.0,
        state_nll=0.0,
        gripper_closed=0.0,
        reward=0.0,
        done=0.0,
        terminal=0.0,
        auxiliary=0.0,
    )

    def objective(log_std: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adjusted = replace(
            predictions,
            normalized_delta_log_std=torch.full_like(
                predictions.normalized_delta_log_std, log_std
            ),
        )
        return compute_rwm_loss(
            adjusted,
            batch,
            delta_std=model.delta_std,
            yaw_indices=model.config.yaw_indices,
            closed_indices=model.config.gripper_closed_indices,
            positive_weights={
                name: torch.tensor(1.0) for name in ("done", "success", "failure")
            },
            horizon_decay=0.95,
            weights=weights,
        )

    low_variance_loss, low_components = objective(-8.0)
    high_variance_loss, high_components = objective(2.0)
    torch.testing.assert_close(low_variance_loss, high_variance_loss)
    torch.testing.assert_close(
        low_variance_loss, low_components["state_mean_mse"]
    )
    torch.testing.assert_close(
        low_components["state_mean_mse"], high_components["state_mean_mse"]
    )


def test_in_memory_phase1_windows_match_disk_loader(tmp_path):
    paths = [
        _write_episode(tmp_path / f"episode_{index:06d}.hdf5", index, steps=5)
        for index in range(2)
    ]
    disk = ProprioSequenceDataset(
        paths=paths,
        history_horizon=3,
        forecast_horizon=4,
        allow_legacy_wam=False,
    )
    memory = InMemoryProprioSequenceDataset(
        paths=paths,
        history_horizon=3,
        forecast_horizon=4,
        allow_legacy_wam=False,
    )
    try:
        assert len(disk) == len(memory) == 10
        assert memory.nbytes > 0
        for index in (0, 3, 4, 5, 9):
            expected = disk[index]
            actual = memory[index]
            assert set(actual) == set(expected)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name])
    finally:
        disk.close()
        memory.close()


def test_checkpoint_strict_reload_is_elementwise_identical(tmp_path):
    stats = _stats()
    model = RWMARWorldModel(
        RWMARConfig(
            history_horizon=3,
            encoder_hidden_dim=16,
            gru_hidden_dim=12,
            gru_layers=1,
        ),
        stats,
    ).eval()
    save_wam_checkpoint(
        tmp_path,
        model,
        stats,
        experiment_config={"version": "test"},
        dataset_manifest={"partitions": {}},
        metrics={},
        provenance={},
        schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    loaded, metadata = load_wam_checkpoint(
        tmp_path, expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION
    )
    batch = _tensor_batch(batch_size=2, history=3, forecast=2)
    history = WorldModelSequenceInputs(
        batch["states"], batch["past_actions"], batch["valid_mask"]
    )
    expected = model.predict(history, batch["candidate_actions"])
    actual = loaded.predict(history, batch["candidate_actions"])

    for name in expected.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), rtol=0, atol=0
        )
    assert metadata["schema"]["normalization_sha256"] == stats.sha256()
    assert (tmp_path / "ema_model.safetensors").exists()


def test_open_loop_reports_all_required_horizons():
    stats = _stats()
    model = RWMARWorldModel(
        RWMARConfig(
            history_horizon=3,
            encoder_hidden_dim=16,
            gru_hidden_dim=12,
            gru_layers=1,
        ),
        stats,
    )
    batch = _tensor_batch(batch_size=4, history=3, forecast=40)
    metrics, example = evaluate_open_loop(
        model,
        [batch],
        stats,
        device=torch.device("cpu"),
        horizons=(1, 5, 10, 20, 40),
        calibrate_thresholds=True,
    )

    assert set(metrics["models"]) == {"rwm_ar", "constant_velocity"}
    assert set(metrics["models"]["rwm_ar"]["exact_horizon"]) == {
        "1",
        "5",
        "10",
        "20",
        "40",
    }
    one_step = metrics["models"]["rwm_ar"]["exact_horizon"]["1"]
    assert "continuous_state_nrmse" in one_step
    assert "gripper_closed_rmse" in one_step
    assert example["actual"].shape == (41, 22)


def test_phase1_training_entrypoint_smoke(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index in range(6):
        _write_episode(data_dir / f"episode_{index:06d}.hdf5", index, steps=4)
    checkpoint = tmp_path / "checkpoint"
    config = {
        "version": "wam.cooperative_stop/test",
        "data": {
            "directory": str(data_dir),
            "state_dim": 22,
            "action_dim": 8,
            "history_horizon": 3,
            "planning_horizon": 3,
            "split_seed": 7,
        },
        "state_features": {
            "yaw_indices": [2, 13],
            "gripper_closed_indices": [7, 18],
            "predict_delta": True,
        },
        "model": {
            "ensemble_size": 1,
            "encoder_hidden_dim": 16,
            "gru_hidden_dim": 12,
            "gru_layers": 1,
            "dropout": 0.0,
            "min_log_std": -8.0,
            "max_log_std": 2.0,
        },
        "training": {
            "mode": "autoregressive",
            "privileged_inputs": False,
            "forecast_curriculum": [2],
            "curriculum_epochs": [1],
            "batch_size": 4,
            "num_workers": 0,
            "seed": 7,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "overfit_refine_epochs": 1,
            "overfit_refine_learning_rate": 0.0005,
            "gradient_clip_norm": 10.0,
            "horizon_decay": 0.95,
            "normalization_std_floor": 0.001,
            "use_amp": False,
            "preload_data": True,
            "loss_weights": {
                "state_mean": 1.0,
                "state_nll": 0.1,
                "gripper_closed": 1.0,
                "reward": 0.1,
                "done": 0.1,
                "terminal": 0.1,
                "auxiliary": 0.05,
            },
        },
        "evaluation": {"validation_max_batches": 1},
        "checkpoint": {"directory": str(checkpoint)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert (
        train_phase1_rwm_ar_main(
            [
                "--config",
                str(config_path),
                "--data-dir",
                str(data_dir),
                "--checkpoint-dir",
                str(checkpoint),
                "--overfit-samples",
                "4",
                "--device",
                "cpu",
                "--max-steps-per-stage",
                "1",
                "--no-progress",
            ]
        )
        == 0
    )
    loaded, metadata = load_wam_checkpoint(checkpoint)
    assert isinstance(loaded, RWMARWorldModel)
    assert metadata["metrics"]["checkpoint_reload"]["passed"]
    assert metadata["metrics"]["curriculum"][0]["optimizer_steps"] == 1
    assert metadata["metrics"]["curriculum"][1]["stage"] == "H=1 overfit refine"
    assert metadata["metrics"]["curriculum"][1]["learning_rate"] == 0.0005


def _stats() -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(22, dtype=np.float32),
        state_std=np.ones(22, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        delta_mean=np.zeros(22, dtype=np.float32),
        delta_std=np.ones(22, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )


def _tensor_batch(
    *, batch_size: int, history: int, forecast: int
) -> dict[str, torch.Tensor]:
    states = torch.randn(batch_size, history, 22) * 0.1
    states[..., [7, 18]] = 1.0
    actions = torch.zeros(batch_size, forecast, 8)
    targets = states[:, -1:, :].repeat(1, forecast, 1)
    targets[..., 1] += torch.arange(1, forecast + 1) * 0.01
    targets[..., [7, 18]] = 1.0
    return {
        "states": states,
        "past_actions": torch.zeros(batch_size, history - 1, 8),
        "valid_mask": torch.ones(batch_size, history, dtype=torch.bool),
        "candidate_actions": actions,
        "executed_actions": actions.clone(),
        "target_states": targets,
        "forecast_mask": torch.ones(batch_size, forecast, dtype=torch.bool),
        "rewards": torch.zeros(batch_size, forecast, 1),
        "dones": torch.zeros(batch_size, forecast, 1),
        "successes": torch.zeros(batch_size, forecast, 1),
        "failures": torch.zeros(batch_size, forecast, 1),
        "response_progress": torch.zeros(batch_size, forecast, 1),
        "coordination_error": torch.zeros(batch_size, forecast, 1),
    }


def _write_episode(path: Path, seed: int, *, steps: int) -> Path:
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 0.1, size=(steps, 22)).astype(np.float32)
    state[:, [7, 18]] = 1.0
    next_state = state.copy()
    next_state[:, [1, 12]] += 0.01
    commanded = rng.uniform(-0.2, 0.2, size=(steps, 8)).astype(np.float32)
    executed = commanded * 0.8
    with h5py.File(path, "w") as file:
        file.attrs.update(
            {
                "schema_profile": "wam_proprio",
                "schema_version": PROPRIO_WAM_SCHEMA_VERSION,
                "num_steps": steps,
                "seed": seed,
                "episode_index": seed,
            }
        )
        data = file.create_group("data")
        data.create_dataset("observation/state", data=state)
        data.create_dataset("next_observation/state", data=next_state)
        data.create_dataset("commanded_action", data=commanded)
        data.create_dataset("executed_action", data=executed)
        data.create_dataset("reward", data=np.zeros(steps, dtype=np.float32))
        terminal = np.arange(steps) == steps - 1
        data.create_dataset("done", data=terminal)
        data.create_dataset("success", data=terminal & (seed % 2 == 0))
        data.create_dataset("failure", data=terminal & (seed % 2 == 1))
        data.create_dataset(
            "response_progress", data=np.linspace(0.0, 1.0, steps, dtype=np.float32)
        )
        data.create_dataset(
            "coordination_error", data=np.zeros(steps, dtype=np.float32)
        )
        data.attrs["metadata"] = json.dumps({"seed": seed})
    return path
