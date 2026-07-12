"""Component evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from data.decentralized_dataset import DecentralizedTransitionDataset
from data.schema import SCHEMA_VERSION
from eval.evaluate_components import (
    ComponentEvaluationConfig,
    evaluate_components,
)
from models.decentralized import (
    EgoLocalWAM,
    EgoLocalWAMConfig,
    LocalIntentionConfig,
    LocalIntentionPosterior,
)
from models.plan_tokenizer import (
    ActionOnlyPlanTokenizer,
    ActionOnlyPlanTokenizerConfig,
    PlanCodeSupport,
)
from models.slot_encoder import LocalBeliefSlotEncoder, LocalBeliefSlotEncoderConfig
from tests.test_training import _write_synthetic_episode
from train.checkpoint import (
    file_sha256,
    make_checkpoint,
    save_checkpoint,
    upstream_reference,
)


def test_frozen_components_report_held_out_metrics_without_mutating_checkpoints(
    tmp_path,
):
    data_dir = tmp_path / "val"
    data_dir.mkdir()
    _write_synthetic_episode(data_dir / "episode_000000.hdf5", transitions=5)
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "splits": {
                    "val": {
                        "path": str(data_dir.resolve()),
                        "episodes": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoints = _write_synthetic_stack(tmp_path, data_dir)
    output = tmp_path / "component_metrics.json"
    hashes_before = {
        name: file_sha256(path)
        for name, path in checkpoints.items()
        if name != "wam"
    }

    report = evaluate_components(
        ComponentEvaluationConfig(
            data_dir=str(data_dir),
            plan_checkpoint=str(checkpoints["plan"]),
            belief_checkpoint=str(checkpoints["belief"]),
            wam_checkpoint=str(checkpoints["wam_robust"]),
            intention_checkpoint=str(checkpoints["intention"]),
            split="validation",
            batch_size=2,
            max_batches=1,
            device="cpu",
            output=str(output),
        )
    )

    assert report["evaluation"]["samples_evaluated"] == 2
    assert report["evaluation"]["batches_evaluated"] == 1
    assert report["evaluation"]["weights_unchanged"] is True
    assert report["dataset"]["held_out_from_checkpoint_training_data"] is True
    assert sum(report["plan"]["code_counts"]) == 2
    assert report["plan"]["encoded_segments"] == 2
    assert report["plan"]["perplexity"] >= 1.0
    assert report["plan"]["action_reconstruction_mse"] >= 0.0
    assert set(report["belief"]) >= {
        "loss",
        "loss_aux_self_state",
        "loss_aux_object_pose",
        "loss_aux_teammate_pose",
        "loss_aux_task_progress",
    }
    assert set(report["wam_robust"]) >= {
        "loss",
        "loss_slots",
        "loss_ego_actions",
        "loss_privileged_teammate_actions",
        "loss_contact",
        "loss_force",
        "loss_progress",
    }
    intention = report["intention"]
    assert intention["examples_evaluated"] == 2
    assert 0.0 <= intention["accuracy"] <= 1.0
    assert 0.0 <= intention["macro_f1"] <= 1.0
    assert 0.0 <= intention["brier_score"] <= 2.0
    assert 0.0 <= intention["ece"] <= 1.0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert hashes_before == {
        name: file_sha256(path)
        for name, path in checkpoints.items()
        if name != "wam"
    }

    with pytest.raises(ValueError, match="manifest has no 'test' split"):
        evaluate_components(
            ComponentEvaluationConfig(
                data_dir=str(data_dir),
                plan_checkpoint=str(checkpoints["plan"]),
                belief_checkpoint=str(checkpoints["belief"]),
                wam_checkpoint=str(checkpoints["wam_robust"]),
                intention_checkpoint=str(checkpoints["intention"]),
                split="test",
                batch_size=2,
                max_batches=1,
                device="cpu",
            )
        )


def _write_synthetic_stack(
    tmp_path: Path, evaluation_data_dir: Path
) -> dict[str, Path]:
    dataset = DecentralizedTransitionDataset(
        evaluation_data_dir, history=2, horizon=2
    )
    training_dataset = {
        "data_dir": str((tmp_path / "different_training_split").resolve()),
        "history": 2,
        "horizon": 2,
        "action_dim": 4,
        "local_contact_semantics": dataset.local_contact_semantics,
        "local_force_semantics": dataset.local_force_semantics,
        "local_force_units": dataset.local_force_units,
        "local_force_scale_newtons": dataset.local_force_scale_newtons,
        "local_sensor_provenance": dataset.local_sensor_provenance,
    }
    support = PlanCodeSupport(
        codebook_size=4,
        min_count=1,
        counts=torch.ones(4, dtype=torch.long),
        probabilities=torch.full((4,), 0.25),
        residual_mean=torch.zeros(4, 4),
        residual_std=torch.ones(4, 4),
    )
    paths = {
        name: tmp_path / f"{name}.pt"
        for name in ("plan", "belief", "wam", "intention", "wam_robust")
    }

    plan_cfg = ActionOnlyPlanTokenizerConfig(
        horizon=2,
        action_dim=4,
        latent_dim=4,
        hidden_dim=8,
        codebook_size=4,
        residual_dropout=0.0,
    )
    plan = ActionOnlyPlanTokenizer(plan_cfg)
    plan_checkpoint = make_checkpoint(
        stage="plan",
        model_class="ActionOnlyPlanTokenizer",
        model_config=plan_cfg,
        model_state_dict=plan.state_dict(),
        training_config={},
        dataset_metadata=training_dataset,
        metrics={},
        normalization={
            "action_mean": torch.zeros(4),
            "action_std": torch.ones(4),
            "action_count": 1,
        },
        plan_code_support=support.to_dict(),
    )
    save_checkpoint(paths["plan"], plan_checkpoint)

    belief_cfg = LocalBeliefSlotEncoderConfig(
        history=2,
        local_dim=dataset.local_history_dim,
        object_dim=3,
        slot_dim=8,
        hidden_dim=16,
        num_heads=2,
        num_history_layers=1,
        num_slot_layers=1,
        dropout=0.0,
        privileged_aux_dims={
            "self_state": 3,
            "object_pose": 3,
            "teammate_pose": 3,
            "task_progress": 1,
        },
        privileged_aux_roles={
            "self_state": "self",
            "object_pose": "object-belief",
            "teammate_pose": "teammate-belief",
            "task_progress": "task-context",
        },
    )
    belief = LocalBeliefSlotEncoder(belief_cfg)
    belief_checkpoint = make_checkpoint(
        stage="belief",
        model_class="LocalBeliefSlotEncoder",
        model_config=belief_cfg,
        model_state_dict=belief.state_dict(),
        training_config={},
        dataset_metadata=training_dataset,
        metrics={},
        plan_code_support=support.to_dict(),
        upstream={"plan": upstream_reference(paths["plan"], plan_checkpoint)},
    )
    save_checkpoint(paths["belief"], belief_checkpoint)

    wam_cfg = EgoLocalWAMConfig(
        horizon=2,
        slots_per_agent=4,
        slot_dim=8,
        plan_codebook_size=4,
        plan_latent_dim=4,
        action_dim_per_agent=4,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        dropout=0.0,
    )
    wam = EgoLocalWAM(wam_cfg)
    wam_checkpoint = make_checkpoint(
        stage="wam",
        model_class="EgoLocalWAM",
        model_config=wam_cfg,
        model_state_dict=wam.state_dict(),
        training_config={},
        dataset_metadata=training_dataset,
        metrics={},
        plan_code_support=support.to_dict(),
        upstream={
            "plan": upstream_reference(paths["plan"], plan_checkpoint),
            "belief": upstream_reference(paths["belief"], belief_checkpoint),
        },
    )
    save_checkpoint(paths["wam"], wam_checkpoint)

    intention_cfg = LocalIntentionConfig(
        slots_per_agent=4,
        slot_dim=8,
        plan_codebook_size=4,
        plan_latent_dim=4,
        message_metadata_dim=4,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        dropout=0.0,
    )
    intention = LocalIntentionPosterior(intention_cfg)
    intention_checkpoint = make_checkpoint(
        stage="intention",
        model_class="LocalIntentionPosterior",
        model_config=intention_cfg,
        model_state_dict=intention.state_dict(),
        training_config={},
        dataset_metadata=training_dataset,
        metrics={},
        plan_code_support=support.to_dict(),
        upstream={
            "plan": upstream_reference(paths["plan"], plan_checkpoint),
            "belief": upstream_reference(paths["belief"], belief_checkpoint),
            "wam": upstream_reference(paths["wam"], wam_checkpoint),
        },
    )
    save_checkpoint(paths["intention"], intention_checkpoint)

    robust_checkpoint = make_checkpoint(
        stage="wam_robust",
        model_class="EgoLocalWAM",
        model_config=wam_cfg,
        model_state_dict=wam.state_dict(),
        training_config={},
        dataset_metadata=training_dataset,
        metrics={},
        plan_code_support=support.to_dict(),
        upstream={
            "plan": upstream_reference(paths["plan"], plan_checkpoint),
            "belief": upstream_reference(paths["belief"], belief_checkpoint),
            "wam": upstream_reference(paths["wam"], wam_checkpoint),
            "intention": upstream_reference(paths["intention"], intention_checkpoint),
        },
    )
    save_checkpoint(paths["wam_robust"], robust_checkpoint)
    return paths
