from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from eval.uncertainty import (
    OODActionPerturbation,
    evaluate_rwm_u,
    fit_variance_calibration,
)
from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    RWMUEnsemble,
    RWMUEnsembleConfig,
    WorldModelRolloutInputs,
    WorldModelSequenceInputs,
)
from scripts.train_phase2_rwm_u import _training_source_hashes
from train.rwm_u_checkpointing import (
    load_rwm_u_checkpoint,
    load_teacher_forcing_ablation,
    save_rwm_u_checkpoint,
)
from train.rwm_u_trainer import make_episode_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def test_phase2_config_matches_rwm_u_contract():
    payload = yaml.safe_load(
        (ROOT / "configs/wam/phase2_rwm_u_v1.yaml").read_text(encoding="utf-8")
    )
    assert payload["phase"] == "phase2_rwm_u"
    assert payload["model"]["family"] == "rwm_u"
    assert payload["model"]["ensemble_size"] == 5
    assert payload["model"]["bootstrap"] is True
    assert payload["training"]["teacher_forcing_ablation"] is True
    assert payload["checkpoint"]["format_version"] == "wam.rwm_u/1"
    assert payload["evaluation"]["gate_c_horizon"] == 20


def test_phase2_training_provenance_fingerprints_source_files():
    hashes = _training_source_hashes(ROOT / "configs/wam/phase2_rwm_u_v1.yaml")

    assert "configs/wam/phase2_rwm_u_v1.yaml" in hashes
    assert "scripts/train_phase2_rwm_u.py" in hashes
    assert "models/wam/ensemble.py" in hashes
    assert "train/rwm_u_trainer.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_ensemble_particle_rollout_keeps_member_identity_and_exposes_risk():
    ensemble = _ensemble(3)
    history = _history(batch=2, horizon=3)
    actions = torch.zeros(2, 4, 8)
    output = ensemble(
        WorldModelRolloutInputs(
            history=history,
            candidate_actions=actions,
            num_particles=2,
        )
    )

    assert output.state_distribution["mean"].shape == (6, 2, 4, 22)
    assert output.rewards.shape == (6, 2, 4, 1)
    assert output.uncertainty["epistemic_std"].shape == (2, 4, 22)
    assert output.uncertainty["aleatoric_std"].min() > 0.0
    assert output.diagnostics["member_index"].tolist() == [0, 0, 1, 1, 2, 2]
    assert output.diagnostics["particle_index"].tolist() == [0, 1, 0, 1, 0, 1]
    assert output.diagnostics["risk_total"].shape == (2, 4)
    assert bool((output.diagnostics["risk_total"] >= 0.0).all())

    deterministic = ensemble(
        WorldModelRolloutInputs(history=history, candidate_actions=actions)
    )
    assert deterministic.state_distribution["mean"].shape == (3, 2, 4, 22)
    assert deterministic.diagnostics["member_index"].tolist() == [0, 1, 2]


def test_teacher_forcing_is_explicit_and_changes_only_recursive_feedback():
    torch.manual_seed(5)
    model = RWMARWorldModel(_member_config(), _stats())
    history = _history(batch=2, horizon=3)
    actions = torch.zeros(2, 3, 8)
    teacher_states = torch.full((2, 3, 22), 0.75)
    teacher_states[..., [7, 18]] = 1.0

    autoregressive = model.predict(history, actions)
    teacher_forced = model.predict_teacher_forced(history, actions, teacher_states)

    torch.testing.assert_close(
        autoregressive.next_state_mean[:, 0],
        teacher_forced.next_state_mean[:, 0],
    )
    assert not torch.allclose(
        autoregressive.next_state_mean[:, 1],
        teacher_forced.next_state_mean[:, 1],
    )


def test_episode_bootstrap_repeats_complete_episode_fragment_groups():
    dataset = _EpisodeIndexedDataset([0, 0, 1, 1, 1, 2])
    bootstrap = make_episode_bootstrap(dataset, seed=13)
    source = np.asarray(dataset._sample_episode)
    bootstrapped = source[list(bootstrap.sample_indices)]

    cursor = 0
    for episode in bootstrap.episode_draws:
        expected_count = int((source == episode).sum())
        assert np.all(bootstrapped[cursor : cursor + expected_count] == episode)
        cursor += expected_count
    assert cursor == len(bootstrap.sample_indices)
    assert len(bootstrap.episode_draws) == 3


def test_rwm_u_checkpoint_roundtrip_is_strict_and_loads_ablation(tmp_path: Path):
    ensemble = _ensemble(2)
    teacher = RWMARWorldModel(_member_config(), _stats())
    checkpoint = tmp_path / "rwm_u"
    save_rwm_u_checkpoint(
        checkpoint,
        ensemble,
        _stats(),
        teacher_forcing_model=teacher,
        experiment_config={"phase": "phase2_rwm_u"},
        dataset_manifest={"partitions": {}},
        bootstrap_manifest={"members": []},
        metrics={"parameter_diversity": {"passed": True}},
        provenance={"seed": 7},
        schema_version="wam.proprio/1.0",
    )
    loaded, metadata = load_rwm_u_checkpoint(
        checkpoint, expected_schema_version="wam.proprio/1.0"
    )
    loaded_teacher = load_teacher_forcing_ablation(checkpoint)
    history = _history(batch=2, horizon=3)
    actions = torch.zeros(2, 2, 8)
    first = ensemble.predict(history, actions)
    second = loaded.predict(history, actions)

    for name in first.__dataclass_fields__:
        torch.testing.assert_close(getattr(first, name), getattr(second, name))
    assert metadata["schema"]["particle_member_semantics"] == (
        "fixed_member_for_complete_trajectory"
    )
    assert isinstance(loaded_teacher, RWMARWorldModel)


def test_validation_calibration_and_ood_evaluation_are_finite():
    ensemble = _ensemble(3)
    teacher = RWMARWorldModel(_member_config(), _stats())
    batch = _batch(batch_size=6, history=3, forecast=2)
    loader = [batch]
    calibration = fit_variance_calibration(
        ensemble,
        loader,
        device=torch.device("cpu"),
        horizon=2,
    )
    metrics = evaluate_rwm_u(
        ensemble,
        loader,
        _stats(),
        device=torch.device("cpu"),
        calibration=calibration,
        horizons=(1, 2),
        teacher_forcing_model=teacher,
        event_horizon=1,
        event_progress_min=0.0,
        event_slowdown_min=0.01,
        event_asymmetry_min=0.01,
    )

    exact = metrics["exact_horizon"]["2"]
    assert np.isfinite(exact["ensemble_mean_continuous_nrmse"])
    assert np.isfinite(exact["gaussian_nll"])
    assert set(exact["interval_coverage"]) == {"50", "90", "95"}
    assert metrics["ood"]["exact_horizon"]["2"]["auroc"] is not None
    assert metrics["event_aligned"]["available"]
    assert metrics["risk"]["ood_mean"] >= 0.0


def test_ood_action_transform_is_bounded_and_increases_action_risk():
    ensemble = _ensemble(2)
    history = _history(batch=3, horizon=3)
    actions = torch.zeros(3, 2, 8)
    transform = OODActionPerturbation(scale=2.0, offset_std=5.0)
    shifted = transform.apply(actions, ensemble.action_mean, ensemble.action_std)
    id_predictions = ensemble.predict(history, actions)
    ood_predictions = ensemble.predict(history, shifted)
    id_risk = ensemble.risk_scores(id_predictions, actions)["total"]
    ood_risk = ensemble.risk_scores(ood_predictions, shifted)["total"]

    assert float(shifted.min()) >= -1.0
    assert float(shifted.max()) <= 1.0
    assert float(ood_risk.mean().detach()) > float(id_risk.mean().detach())


class _EpisodeIndexedDataset(Dataset):
    def __init__(self, sample_episode: list[int]) -> None:
        self._sample_episode = np.asarray(sample_episode, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._sample_episode.size)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index)


def _member_config() -> RWMARConfig:
    return RWMARConfig(
        history_horizon=3,
        train_forecast_horizon=2,
        planning_horizon=2,
        encoder_hidden_dim=16,
        gru_hidden_dim=12,
        gru_layers=1,
    )


def _stats() -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(22, dtype=np.float32),
        state_std=np.ones(22, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.full(8, 0.2, dtype=np.float32),
        delta_mean=np.zeros(22, dtype=np.float32),
        delta_std=np.ones(22, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )


def _ensemble(size: int) -> RWMUEnsemble:
    members = []
    for seed in range(size):
        torch.manual_seed(seed + 1)
        members.append(RWMARWorldModel(_member_config(), _stats()))
    return RWMUEnsemble(members, RWMUEnsembleConfig(size), _stats())


def _history(*, batch: int, horizon: int) -> WorldModelSequenceInputs:
    states = torch.zeros(batch, horizon, 22)
    states[..., 3] = 1.0
    states[..., 14] = 1.0
    states[..., [7, 18]] = 1.0
    return WorldModelSequenceInputs(
        states=states,
        past_actions=torch.zeros(batch, horizon - 1, 8),
        valid_mask=torch.ones(batch, horizon, dtype=torch.bool),
    )


def _batch(*, batch_size: int, history: int, forecast: int) -> dict[str, torch.Tensor]:
    sequence = _history(batch=batch_size, horizon=history)
    targets = sequence.states[:, -1:, :].repeat(1, forecast, 1)
    targets[:, 0, 3] = 0.5
    targets[..., [7, 18]] = 1.0
    return {
        "states": sequence.states,
        "past_actions": sequence.past_actions,
        "valid_mask": sequence.valid_mask,
        "candidate_actions": torch.zeros(batch_size, forecast, 8),
        "target_states": targets,
        "forecast_mask": torch.ones(batch_size, forecast, dtype=torch.bool),
        "response_progress": torch.ones(batch_size, forecast, 1),
    }
