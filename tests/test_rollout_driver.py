"""Rollout driver tests."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.evaluate_fe_pc_wam_rollouts as rollout_driver
from eval.evaluate import aggregate_episode_records, compare_communication_modes
from policies.decentralized import LocalDecision, PairDecision
from scripts.evaluate_fe_pc_wam_rollouts import (
    EpisodeRecipe,
    DEPLOYABLE_MODES,
    REQUIRED_VALIDATION_FREEZE_CONDITIONS,
    _attach_failure_replay_video,
    _digest_json,
    _compact_episode_record,
    _episode_metrics,
    _load_frozen_runtime_configs,
    _load_resumable_record,
    _mode_policy_config,
    _runtime_config,
    _record_set_attestation,
    _validate_resume_snapshot,
    _video_artifact_path,
    build_parser,
    resolve_checkpoints,
    run_paired_evaluation,
    run_episode,
)
from train.checkpoint import file_sha256


def _args(tmp_path: Path):
    return build_parser().parse_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-steps",
            "2",
            "--num-candidates",
            "2",
            "--num-teammate-hypotheses",
            "1",
            "--residual-sigma-points",
            "1",
        ]
    )


def test_parser_defaults_to_failure_videos_and_requires_explicit_base_wam(tmp_path):
    args = _args(tmp_path)
    assert args.save_failure_videos is True
    assert args.use_base_wam is False
    disabled = build_parser().parse_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-save-failure-videos",
            "--use-base-wam",
        ]
    )
    assert disabled.save_failure_videos is False
    assert disabled.use_base_wam is True


def test_checkpoint_resolution_can_explicitly_select_base_wam(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    for name in ("plan", "belief", "wam", "intention"):
        (root / f"{name}.pt").touch()

    with pytest.raises(FileNotFoundError, match="--use-base-wam"):
        resolve_checkpoints(root)

    diagnostic = resolve_checkpoints(root, use_base_wam=True)
    assert diagnostic.uses_base_wam is True
    assert diagnostic.deployment_wam == (root / "wam.pt").resolve()
    assert set(diagnostic.runtime_paths()) == {"plan", "belief", "wam", "intention"}
    assert diagnostic.audit_paths().count((root / "wam.pt").resolve()) == 1

    (root / "wam_robust.pt").touch()
    formal = resolve_checkpoints(root)
    assert formal.uses_base_wam is False
    assert formal.deployment_wam_stage == "wam_robust"
    assert set(formal.runtime_paths()) == {
        "plan",
        "belief",
        "wam_robust",
        "intention",
    }


class _FakeRuntime:
    def __init__(self) -> None:
        self.seed = None
        self.calls = 0

    def reset(self, *, seed=None) -> None:
        self.seed = seed
        self.calls = 0

    def step(self, packets) -> PairDecision:
        assert len(packets) == 2
        for packet in packets:
            assert all(
                "teammate" not in name and "global" not in name
                for name in packet.as_mapping()
            )
        decisions = []
        for agent_id in (0, 1):
            diagnostics = {
                "request_sent": False,
                "reply_received": False,
                "actual_round_trip_bits": 0,
                "actual_delay_steps": 1.0,
                "VPI": 0.2,
                "G_before": 2.0,
                "G_after": 2.0,
                "replanned": False,
                "action_change_l2": 0.0,
                "candidate_codes": [1, 3],
            }
            decisions.append(
                LocalDecision(
                    agent_id=agent_id,
                    step=self.calls,
                    action=torch.tensor([0.0, 0.25, 0.0, 1.0]),
                    plan_code=1,
                    plan_residual=torch.zeros(2),
                    communicated=False,
                    diagnostics=diagnostics,
                )
            )
        self.calls += 1
        return PairDecision(
            joint_action=torch.cat((decisions[0].action, decisions[1].action)),
            agents=(decisions[0], decisions[1]),
            routed_messages=0,
        )


def test_mode_configuration_makes_always_literal_and_matches_baseline_rate(tmp_path):
    args = _args(tmp_path)
    always = _mode_policy_config("always_reply", args, matched_request_rate=None)
    assert always.communication_mode == "always_reply"
    assert always.cooldown_steps == 0
    assert always.plan_valid_steps == 0
    assert always.metadata_available_index == -1
    assert always.metadata_delay_index == -1

    periodic = _mode_policy_config("periodic", args, matched_request_rate=0.2)
    random = _mode_policy_config("random", args, matched_request_rate=0.2)
    assert periodic.periodic_interval == args.periodic_interval
    assert periodic.periodic_request_rate == 0.2
    assert periodic.cooldown_steps == 0
    assert random.random_request_probability == pytest.approx(0.2)
    disabled_periodic = _mode_policy_config(
        "periodic", args, matched_request_rate=0.0
    )
    assert disabled_periodic.periodic_enabled is False
    assert disabled_periodic.periodic_request_rate == 0.0


def test_test_split_cannot_tune_or_run_without_frozen_validation_config(tmp_path):
    args = _args(tmp_path)
    args.split = "test"
    with pytest.raises(ValueError, match="requires --frozen-config-from"):
        run_paired_evaluation(args)

    args.match_baselines_to_selective = True
    with pytest.raises(ValueError, match="test cannot tune baseline rates"):
        run_paired_evaluation(args)


def test_episode_metrics_treat_timeout_as_safe_but_force_failure_as_unsafe():
    safe = _episode_metrics(
        [{"failure": True, "failure_reason": "timeout", "force_violation": False}],
        [1.0],
        done=True,
        truncated=False,
    )
    unsafe = _episode_metrics(
        [
            {
                "failure": True,
                "failure_reason": "force_violation",
                "force_violation": True,
                "force_proxy": 2.0,
            }
        ],
        [-1.0],
        done=True,
        truncated=False,
    )
    assert safe["safe"] is True
    assert unsafe["safe"] is False
    assert unsafe["max_force"] == 2.0


def test_run_episode_emits_agent_decisions_and_mode_independent_pair_digest(tmp_path):
    args = _args(tmp_path)
    recipe = EpisodeRecipe(
        source_path=tmp_path / "episode_000000.hdf5",
        split="val",
        episode_id="val/episode_000000",
        episode_index=0,
        seed=41,
        scenario="nominal",
        object_dropout_prob=0.0,
        source_sha256="source-hash",
    )
    manifest = {
        "config": {
            "episode_len": 5,
            "randomize": False,
            "object_position_std": 0.0,
            "object_yaw_std": 0.0,
        }
    }
    runtime = _FakeRuntime()
    first = run_episode(
        runtime,
        recipe,
        manifest=manifest,
        mode="no_comm",
        checkpoint_hashes={"plan": "p", "belief": "b", "wam_robust": "w", "intention": "i"},
        evaluation_config_digest="no-comm-config",
        args=args,
    )
    second = run_episode(
        runtime,
        recipe,
        manifest=manifest,
        mode="selective_vpi",
        checkpoint_hashes={"plan": "p", "belief": "b", "wam_robust": "w", "intention": "i"},
        evaluation_config_digest="selective-config",
        args=args,
    )

    assert runtime.seed == recipe.seed + args.policy_seed
    assert first["environment_steps"] == 2
    assert first["decision_count"] == 4
    assert first["truncated"] is True
    assert first["candidate_codes"] == [1, 3]
    assert first["candidate_code_counts"] == {"1": 4, "3": 4}
    assert first["candidate_code_observations"] == 8
    assert first["expected_candidate_code_observations"] == 8
    assert all("candidate_codes" not in row for row in first["steps"])
    assert all(row["communication_delay"] == 0.0 for row in first["steps"])
    assert all(row["expected_delay_cost_steps"] == 1.0 for row in first["steps"])
    assert all(row["expected_latency_cost"] == 0.05 for row in first["steps"])
    assert all(row["incurred_expected_latency_cost"] == 0.0 for row in first["steps"])
    assert first["input_digest"] == second["input_digest"]
    assert first["evaluation_config_digest"] != second["evaluation_config_digest"]
    assert np.isfinite(first["return"])
    aggregate = aggregate_episode_records([first], mode="no_comm")
    assert aggregate["actual_communication_delay_mean"] == 0.0
    assert aggregate["expected_latency_cost_mean"] == 0.05
    assert aggregate["incurred_expected_latency_cost_mean"] == 0.0


def test_run_episode_streams_video_and_records_artifact(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.video_width = 320
    args.video_height = 240
    args.video_fps = 12
    recipe = EpisodeRecipe(
        source_path=tmp_path / "episode_000000.hdf5",
        split="val",
        episode_id="val/episode_000000",
        episode_index=0,
        seed=43,
        scenario="nominal",
        object_dropout_prob=0.0,
        source_sha256="source-hash",
    )
    manifest = {
        "config": {
            "episode_len": 5,
            "randomize": False,
            "object_position_std": 0.0,
            "object_yaw_std": 0.0,
        }
    }

    class FakeVideoRecorder:
        def __init__(self, path, **kwargs):
            self.path = Path(path)
            self.kwargs = kwargs
            self.frame_count = 0

        def capture(self, data):
            assert data is not None
            self.frame_count += 1

        def close(self, *, commit):
            assert commit is True
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"fake-mp4")
            return {
                "path": str(self.path),
                "sha256": file_sha256(self.path),
                "frame_count": self.frame_count,
                "fps": self.kwargs["fps"],
                "width": self.kwargs["width"],
                "height": self.kwargs["height"],
                "codec": "mp4v",
            }

    monkeypatch.setattr(rollout_driver, "_RolloutVideoRecorder", FakeVideoRecorder)
    video_path = tmp_path / "videos" / "no_comm" / "episode_000000.mp4"
    record = run_episode(
        _FakeRuntime(),
        recipe,
        manifest=manifest,
        mode="no_comm",
        checkpoint_hashes={
            "plan": "p",
            "belief": "b",
            "wam_robust": "w",
            "intention": "i",
        },
        evaluation_config_digest="video-config",
        args=args,
        video_path=video_path,
    )

    assert video_path.is_file()
    assert record["video"]["frame_count"] == 3  # reset frame + two steps
    assert record["video"]["fps"] == 12
    assert record["video"]["width"] == 320
    assert record["video"]["height"] == 240
    assert record["video"]["sha256"] == file_sha256(video_path)


def test_failure_replay_video_is_attached_only_after_determinism_check(tmp_path):
    video_path = tmp_path / "failure.mp4"
    video_path.write_bytes(b"failure-video")
    common = {
        "input_digest": "input",
        "evaluation_config_digest": "config",
        "success": False,
        "failure": True,
        "failure_reason": "timeout",
        "done": True,
        "truncated": False,
        "environment_steps": 12,
        "return": -3.5,
    }
    original = dict(common)
    replay = {
        **common,
        "video": {
            "path": str(video_path),
            "sha256": file_sha256(video_path),
            "frame_count": 13,
        },
    }
    _attach_failure_replay_video(original, replay, video_path=video_path)
    assert original["video"]["selection"] == "failure_replay"
    assert original["video"]["replay_verified"] is True

    changed = dict(replay)
    changed["success"] = True
    with pytest.raises(RuntimeError, match="not deterministic"):
        _attach_failure_replay_video(dict(common), changed, video_path=video_path)


def test_failure_video_path_includes_sanitized_reason(tmp_path):
    recipe = EpisodeRecipe(
        source_path=tmp_path / "episode_000007.hdf5",
        split="val",
        episode_id="val/episode_000007",
        episode_index=7,
        seed=57,
        scenario="private_gates",
        object_dropout_prob=0.0,
        source_sha256="source-hash",
    )

    path = _video_artifact_path(
        tmp_path,
        "selective_vpi",
        recipe,
        failure_reason="Private Event/Mismatch",
    )

    assert path == (
        tmp_path
        / "videos"
        / "selective_vpi"
        / "episode_000007__private_event_mismatch.mp4"
    )


def test_compact_manifest_records_preserve_full_paired_evaluation(tmp_path):
    args = _args(tmp_path)
    recipe = EpisodeRecipe(
        source_path=tmp_path / "episode_000000.hdf5",
        split="val",
        episode_id="val/episode_000000",
        episode_index=0,
        seed=51,
        scenario="nominal",
        object_dropout_prob=0.0,
        source_sha256="source-hash",
    )
    manifest = {
        "config": {
            "episode_len": 5,
            "randomize": False,
            "object_position_std": 0.0,
            "object_yaw_std": 0.0,
        }
    }
    full_by_mode = {}
    compact_by_mode = {}
    for mode in DEPLOYABLE_MODES:
        full = run_episode(
            _FakeRuntime(),
            recipe,
            manifest=manifest,
            mode=mode,
            checkpoint_hashes={
                "plan": "p",
                "belief": "b",
                "wam_robust": "w",
                "intention": "i",
            },
            evaluation_config_digest=f"config-{mode}",
            args=args,
        )
        path = tmp_path / "records" / mode / "episode_000000.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(full), encoding="utf-8")
        full_by_mode[mode] = [full]
        compact_by_mode[mode] = [
            _compact_episode_record(full, record_path=path, output_dir=tmp_path)
        ]

    assert compare_communication_modes(
        full_by_mode,
        required_modes=DEPLOYABLE_MODES,
        bootstrap_samples=10,
    ) == compare_communication_modes(
        compact_by_mode,
        required_modes=DEPLOYABLE_MODES,
        bootstrap_samples=10,
    )


def test_resume_rejects_changed_episode_or_frozen_snapshot(tmp_path):
    recipe = EpisodeRecipe(
        source_path=tmp_path / "episode.hdf5",
        split="val",
        episode_id="val/episode_000000",
        episode_index=0,
        seed=7,
        scenario="nominal",
        object_dropout_prob=0.1,
        source_sha256="source-a",
    )
    record_path = tmp_path / "record.json"
    input_recipe = {"source_sha256": recipe.source_sha256}
    record_path.write_text(
        json.dumps(
            {
                "mode": "no_comm",
                "evaluation_config_digest": "config",
                "episode_id": recipe.episode_id,
                "seed": recipe.seed,
                "scenario": recipe.scenario,
                "input_recipe": input_recipe,
                "input_digest": _digest_json(input_recipe),
            }
        ),
        encoding="utf-8",
    )
    assert (
        _load_resumable_record(
            record_path,
            mode="no_comm",
            config_digest="config",
            recipe=recipe,
        )
        is not None
    )
    changed_recipe = replace(recipe, source_sha256="source-b")
    assert (
        _load_resumable_record(
            record_path,
            mode="no_comm",
            config_digest="config",
            recipe=changed_recipe,
        )
        is None
    )

    snapshot_path = tmp_path / "snapshot.json"
    frozen_snapshot = {
        "source_tree_sha256": "tree-a",
        "dataset_manifest_sha256": "data",
        "checkpoints": {"plan": "hash"},
        "environment_sha256": "environment",
        "evaluation_arguments_sha256": "randomize-1",
        "selected_episode_set_sha256": "episodes-a",
        "frozen_validation_config": {"path": "frozen", "sha256": "freeze-a"},
    }
    snapshot_path.write_text(json.dumps(frozen_snapshot), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different frozen experiment snapshot"):
        _validate_resume_snapshot(
            snapshot_path,
            {**frozen_snapshot, "source_tree_sha256": "tree-b"},
        )
    with pytest.raises(RuntimeError, match="evaluation_arguments_sha256"):
        _validate_resume_snapshot(
            snapshot_path,
            {**frozen_snapshot, "evaluation_arguments_sha256": "randomize-0"},
        )

    with pytest.raises(RuntimeError, match="selected_episode_set_sha256"):
        _validate_resume_snapshot(
            snapshot_path,
            {**frozen_snapshot, "selected_episode_set_sha256": "episodes-b"},
        )
    with pytest.raises(RuntimeError, match="frozen_validation_config"):
        _validate_resume_snapshot(
            snapshot_path,
            {
                **frozen_snapshot,
                "frozen_validation_config": {
                    "path": "frozen",
                    "sha256": "freeze-b",
                },
            },
        )


def test_frozen_test_config_requires_complete_validation_protocol(tmp_path):
    args = _args(tmp_path)
    checkpoint_hashes = {
        "plan": "p",
        "belief": "b",
        "wam_robust": "w",
        "intention": "i",
    }
    effective = {}
    for mode in DEPLOYABLE_MODES:
        policy = _mode_policy_config(mode, args, matched_request_rate=None)
        effective[mode] = {
            "checkpoint_hashes": checkpoint_hashes,
            "source_tree_sha256": "tree",
            "dataset_manifest_sha256": "manifest",
            "environment_sha256": "environment",
            "resolved_device": "cuda",
            "runtime": asdict(_runtime_config(policy, args)),
        }
    records = {
        mode: [
            {
                "record_contract": "fe_pc_wam_closed_loop_episode_compact",
                "mode": mode,
                "seed": index,
                "episode_id": f"val/episode_{index:06d}",
                "input_digest": f"input-{index}",
                "evaluation_config_digest": _digest_json(effective[mode]),
                "full_record_sha256": f"{index:064x}"[-64:],
                "truncated": False,
            }
            for index in range(160)
        ]
        for mode in DEPLOYABLE_MODES
    }
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps({"modes": records}), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "passed": True,
                "candidate_code_coverage": {"complete": True},
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_payload = {
        "source_tree_sha256": "tree",
        "dataset_manifest_sha256": "manifest",
        "selected_episode_set_sha256": "episodes",
        "environment_sha256": "environment",
    }
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    path = tmp_path / "frozen_config.json"
    payload = {
        "frozen_config_contract": "fe_pc_wam_validation_freeze",
        "split": "val",
        "episode_count": 160,
        "expected_split_episode_count": 160,
        "validation_freeze_eligible": True,
        "validation_freeze_conditions": {
            name: True for name in REQUIRED_VALIDATION_FREEZE_CONDITIONS
        },
        "modes": list(DEPLOYABLE_MODES),
        "validation_record_counts": {mode: 160 for mode in DEPLOYABLE_MODES},
        "validation_record_set_sha256": _record_set_attestation(records),
        "paired_inputs_verified": True,
        "input_digest_verified": True,
        "baseline_budget_match": {"passed": True},
        "checkpoint_hashes": checkpoint_hashes,
        "effective_mode_configs": effective,
        "records_manifest": {
            "path": str(records_path),
            "sha256": file_sha256(records_path),
        },
        "artifact_audit": {
            "path": str(audit_path),
            "sha256": file_sha256(audit_path),
            "passed": True,
        },
        "experiment_snapshot": {
            "path": str(snapshot_path),
            "sha256": file_sha256(snapshot_path),
            **snapshot_payload,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    configs, source = _load_frozen_runtime_configs(
        path,
        modes=DEPLOYABLE_MODES,
        checkpoint_hashes=checkpoint_hashes,
        device="cuda",
        resolved_device="cuda",
    )
    assert set(configs) == set(DEPLOYABLE_MODES)
    assert all(config.device == "cuda" for config in configs.values())
    assert source["path"] == str(path.resolve())

    payload["validation_freeze_conditions"].pop("cuda_execution")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete/unknown freeze conditions"):
        _load_frozen_runtime_configs(
            path,
            modes=DEPLOYABLE_MODES,
            checkpoint_hashes=checkpoint_hashes,
            device="cuda",
            resolved_device="cuda",
        )

    payload["validation_freeze_conditions"]["cuda_execution"] = True
    records["random"] = []
    records_path.write_text(json.dumps({"modes": records}), encoding="utf-8")
    payload["records_manifest"]["sha256"] = file_sha256(records_path)
    payload["validation_record_set_sha256"] = _record_set_attestation(records)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="random must contain 160"):
        _load_frozen_runtime_configs(
            path,
            modes=DEPLOYABLE_MODES,
            checkpoint_hashes=checkpoint_hashes,
            device="cuda",
            resolved_device="cuda",
        )
