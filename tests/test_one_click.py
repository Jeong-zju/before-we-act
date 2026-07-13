"""One-command workflow tests."""

from __future__ import annotations

import h5py
import json
import pytest
import sys

from data.schema import SCHEMA_VERSION
from scripts.collect_fe_pc_wam_dataset import (
    CollectionConfig,
    collect_dataset,
    episode_recipe,
)
from scripts.train_fe_pc_wam_pipeline import (
    build_parser,
    main as pipeline_main,
    run_pipeline,
)


def test_episode_recipe_is_seed_deterministic_and_uses_current_profiles(tmp_path):
    config = CollectionConfig(
        out_dir=str(tmp_path),
        train_episodes=1,
        val_episodes=0,
        test_episodes=0,
        pilot_episodes=0,
        profile="balanced",
    )
    assert episode_recipe(1234, config) == episode_recipe(1234, config)
    recipe = episode_recipe(1234, config)
    assert recipe["mode"] in {"scripted", "noisy", "recovery"}
    assert recipe["scenario"] in {
        "nominal",
        "narrow",
        "occlusion",
        "asymmetric_obstacle",
        "blocked_passage",
        "false_belief",
        "hard_comm",
    }
    assert 0.0 <= recipe["object_dropout_prob"] <= 1.0


def test_one_click_collection_and_training_pipeline_from_dataset_root(
    tmp_path, capsys, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    manifest = collect_dataset(
        CollectionConfig(
            out_dir=str(dataset_root),
            train_episodes=2,
            val_episodes=1,
            test_episodes=1,
            seed=31,
            episode_len=5,
            pilot_episodes=0,
        )
    )
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["legacy_data_compatible"] is False
    assert manifest["splits"]["train"]["episodes"] == 2
    assert manifest["splits"]["val"]["episodes"] == 1
    assert manifest["splits"]["test"]["episodes"] == 1

    with h5py.File(dataset_root / "train/episode_000000.hdf5", "r") as file:
        assert file.attrs["schema_version"] == SCHEMA_VERSION
        assert file["metadata"].attrs["split"] == "train"
        assert not bool(
            file["schema/local_observation"].attrs["explicit_teammate_state_allowed"]
        )

    # Existing files are never silently overwritten.
    with pytest.raises(FileExistsError, match="--resume"):
        collect_dataset(
            CollectionConfig(
                out_dir=str(dataset_root),
                train_episodes=2,
                val_episodes=1,
                test_episodes=1,
                seed=31,
                episode_len=5,
                pilot_episodes=0,
            )
        )

    checkpoint_root = tmp_path / "checkpoints"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--out-dir",
            str(checkpoint_root),
            "--history",
            "2",
            "--horizon",
            "2",
            "--num-workers",
            "0",
            "--min-active-codes",
            "1",
            "--min-usage-ratio",
            "0.01",
            "--smoke",
        ]
    )
    capsys.readouterr()
    trained = run_pipeline(args)
    trained_output = capsys.readouterr()
    assert trained_output.out == ""
    assert "\r" in trained_output.err
    assert "[pipeline] dataset=" in trained_output.err
    assert "[stage 1/5:plan] start" in trained_output.err
    assert "[stage 1/5:plan] normalization" in trained_output.err
    assert "[stage 1/5:plan] training" in trained_output.err
    assert "epoch=1/1" in trained_output.err
    assert "[stage 1/5:plan] codebook scan" in trained_output.err
    assert "[stage 5/5:wam_robust] completed" in trained_output.err
    assert trained["stage_order"] == [
        "plan",
        "belief",
        "wam",
        "intention",
        "wam_robust",
    ]
    assert trained["validation_data_dir"] == str((dataset_root / "val").resolve())
    assert trained["test_data_dir"] == str((dataset_root / "test").resolve())
    assert all(stage["status"] == "trained" for stage in trained["stages"].values())

    resume_args = parser.parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--out-dir",
            str(checkpoint_root),
            "--history",
            "2",
            "--horizon",
            "2",
            "--num-workers",
            "0",
            "--min-active-codes",
            "1",
            "--min-usage-ratio",
            "0.01",
            "--smoke",
            "--resume",
        ]
    )
    resumed = run_pipeline(resume_args)
    resumed_output = capsys.readouterr()
    assert resumed_output.err.count(" reused checkpoint=") == 5
    assert all(stage["status"] == "reused" for stage in resumed["stages"].values())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fe_pc_wam_pipeline.py",
            "--dataset-root",
            str(dataset_root),
            "--out-dir",
            str(checkpoint_root),
            "--history",
            "2",
            "--horizon",
            "2",
            "--num-workers",
            "0",
            "--min-active-codes",
            "1",
            "--min-usage-ratio",
            "0.01",
            "--smoke",
            "--resume",
            "--quiet",
        ],
    )
    pipeline_main()
    quiet_output = capsys.readouterr()
    assert quiet_output.err == ""
    quiet_manifest = json.loads(quiet_output.out)
    assert quiet_manifest["stage_order"] == list(trained["stage_order"])
    assert all(
        stage["status"] == "reused" for stage in quiet_manifest["stages"].values()
    )
