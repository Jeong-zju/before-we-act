from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.bicoord_care import supervisor as sup
from deployment.bicoord_care import asset_stage
from deployment.bicoord_care.evaluate_b0h import run as run_b0h_evaluation
from deployment.bicoord_care.paired_evaluate import run as run_paired_evaluation


def _settings(tmp_path: Path, *, revision: str = "a" * 40) -> sup.Settings:
    repo = tmp_path / "care"
    benchmark = tmp_path / "benchmark"
    dataset = tmp_path / "dataset"
    dino = tmp_path / "dino"
    for path in (repo, benchmark, dataset, dino):
        path.mkdir(parents=True)
    return sup.Settings(
        repo=repo,
        benchmark_repo=benchmark,
        dataset=dataset,
        run=tmp_path / "run",
        dino_model=dino,
        python="python",
        care_source_revision=revision,
        modules=dict(sup.DEFAULT_MODULES),
    )


def test_settings_binds_a_valid_care_source_revision(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.validate()
    frozen = settings.frozen_config()
    assert frozen["care_source_revision"] == "a" * 40
    assert frozen["care_oof"] == {
        "variant": "care",
        "public_seed": 20260904,
        "folds": [0, 1, 2],
        "training_seeds": [20261904, 20261905, 20261906],
        "deployment_candidate": False,
    }
    assert frozen["asset_contract"]["legacy_shovel"] == {
        "task": "sweep_block",
        "object": "082_smallshovel",
        "model_id": 3,
        "metadata": "model_data3.json",
        "pristine_metadata_sha256": sup.PRISTINE_SHOVEL_METADATA_SHA256,
        "legacy_fields": ["contact_pose", "trans_matrix"],
        "derived_fields": ["contact_points_pose"],
        "derived_contact_points_pose_sha256": (
            sup.SHOVEL_CONTACT_POINTS_POSE_SHA256
        ),
        "conversion": (
            "scale(contact_pose) @ trans_matrix -> scale(contact_points_pose)"
        ),
        "scale": [0.167, 0.167, 0.167],
        "collision_mesh": {
            "relative_path": "collision/base3.glb",
            "bytes": sup.SHOVEL_COLLISION_BYTES,
            "sha256": sup.SHOVEL_COLLISION_SHA256,
        },
        "visual_mesh": {
            "relative_path": "visual/base3.glb",
            "bytes": sup.SHOVEL_VISUAL_BYTES,
            "sha256": sup.SHOVEL_VISUAL_SHA256,
        },
        "model_variant_replaced": False,
    }
    with pytest.raises(ValueError, match="40-character"):
        _settings(tmp_path / "bad", revision="not-a-commit").validate()


def test_base_commands_explicitly_isolate_smoke_cache_and_branch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    # Avoid import discovery in this command-construction test.
    settings = sup.Settings(
        **{
            **settings.__dict__,
            "modules": {
                **settings.modules,
                "bcore_cache": "deployment.bicoord_care.cache_bcore",
                "branch_collect": "deployment.bicoord_care.branch_collection",
            },
        }
    )
    supervisor = sup.Supervisor(settings)
    cache_spec = sup.STAGES["bcore_smoke_cache"]
    cache_command = supervisor._base_command(cache_spec, tmp_path / "cache.json")
    assert "--smoke" in cache_command
    assert cache_command[cache_command.index("--stage-name") + 1] == "bcore_smoke_cache"

    branch_spec = sup.STAGES["branch_smoke"]
    # _sharded_action is exercised with a fake scheduler to inspect every
    # worker command without launching a simulator.
    commands: list[list[str]] = []

    class FakeScheduler:
        def run_wave(self, jobs):
            commands.extend(command for _name, command, _gpu, _log in jobs)

    supervisor.scheduler = FakeScheduler()  # type: ignore[assignment]
    # Aggregation is intentionally bypassed; this test only checks argv.
    supervisor._aggregate_worker_results = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    supervisor._sharded_action(branch_spec, tmp_path / "branch.json")
    assert len(commands) == 4
    for command in commands:
        assert command[command.index("--families-per-task") + 1] == "1"
        assert command[command.index("--branches-per-family") + 1] == "24"
        assert "--smoke" in command


def test_seed_discovery_is_gpu_task_queued() -> None:
    assert sup.STAGES["seed_discovery"].gpu_plan == "seed_task_queue4"
    assert sup.STAGES["seed_discovery_smoke"].gpu_plan == "seed_task_queue4"


def test_asset_contract_gates_dataset_audit_and_runtime_overlay(tmp_path: Path) -> None:
    assert sup.STAGES["asset_contract"].dependencies == ("dataset_download",)
    assert sup.STAGES["dataset_audit"].dependencies == ("asset_contract",)
    assert tuple(sup.STAGES).index("asset_contract") < tuple(sup.STAGES).index(
        "dataset_audit"
    )
    supervisor = sup.Supervisor(_settings(tmp_path))
    environment = supervisor.scheduler.environment((0,))
    expected_plate = (
        supervisor.s.run
        / "artifacts"
        / "asset_contract"
        / "overlay"
        / "003_plate"
        / "model_data0.json"
    )
    expected_shovel = (
        supervisor.s.run
        / "artifacts"
        / "asset_contract"
        / "overlay"
        / "082_smallshovel"
        / "model_data3.json"
    )
    assert environment["BICOORD_PLATE_ASSET_OVERLAY"] == str(expected_plate)
    assert environment["BICOORD_SHOVEL_ASSET_OVERLAY"] == str(expected_shovel)
    assert environment["BICOORD_REQUIRE_ASSET_OVERLAY"] == "1"


def test_preserves_mapping_except_ignores_only_overlay_field() -> None:
    pristine = {
        "scale": [0.025, 0.025, 0.025],
        "contact_points_pose": [],
        "center": [0.0, 0.0, 0.0],
    }
    effective = {
        "scale": [0.025, 0.025, 0.025],
        "contact_points_pose": [[1.0]],
        "center": [0.0, 0.0, 0.0],
    }
    assert sup._preserves_mapping_except(
        pristine, effective, ("contact_points_pose",)
    )
    effective["scale"] = [0.026, 0.025, 0.025]
    assert not sup._preserves_mapping_except(
        pristine, effective, ("contact_points_pose",)
    )


def test_asset_contract_paths_are_canonical_and_non_symbolic(tmp_path: Path) -> None:
    expected = (
        tmp_path
        / "run"
        / "artifacts"
        / "asset_contract"
        / "overlay"
        / "003_plate"
        / "model_data0.json"
    )
    expected.parent.mkdir(parents=True)
    expected.write_text("{}\n", encoding="utf-8")

    assert sup._canonical_stage_path(
        str(expected), expected, label="plate runtime overlay"
    ) == expected
    with pytest.raises(sup.InvalidArtifact, match="canonical run artifact"):
        sup._canonical_stage_path(
            str(expected.parent / ".." / "003_plate" / expected.name),
            expected,
            label="plate runtime overlay",
        )

    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = expected.parent / "linked.json"
    link.symlink_to(outside)
    with pytest.raises(sup.InvalidArtifact, match="canonical run artifact"):
        sup._canonical_stage_path(str(link), expected, label="plate runtime overlay")

    # A symlinked parent must be rejected even when the final spelling happens
    # to resolve to the expected file.
    moved = tmp_path / "real-overlay"
    moved.mkdir()
    real = moved / expected.name
    real.write_text("{}\n", encoding="utf-8")
    parent = expected.parent
    expected.unlink()
    link.unlink()
    parent.rmdir()
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(sup.InvalidArtifact, match="symbolic component"):
        sup._canonical_stage_path(str(expected), expected, label="plate runtime overlay")


def _minimal_asset_result(
    supervisor: sup.Supervisor, receipt: Path
) -> dict[str, object]:
    return {
        "schema": sup.RESULT_SCHEMA,
        "stage": "asset_contract",
        "status": "PASSED",
        "benchmark_adapter": "BiCoord",
        "config_sha256": supervisor.config_hash,
        "artifacts": [],
        "dataset_archive_sha256": asset_stage.BICOORD_OBJECTS_SHA256,
        "base_archive_sha256": asset_stage.ROBOTWIN_OBJECTS_SHA256,
        "contact_points_pose_count": 4,
        "shovel_contact_points_pose_count": 1,
        "shovel_contact_points_pose_sha256": (
            sup.SHOVEL_CONTACT_POINTS_POSE_SHA256
        ),
        "shovel_metadata_sha256": "b" * 64,
        "copied_fields": ["contact_points_pose"],
        "task_source_modified": False,
        "upstream_model_modified": False,
        "normalization_modified": False,
        "task_asset_references_checked": {
            "tasks": len(sup.TASKS),
            "actors": 21,
            "interactions": 95,
        },
        "task_asset_task_count": len(sup.TASKS),
        "task_asset_actor_reference_count": 21,
        "task_asset_interaction_reference_count": 95,
        "task_asset_dynamic_inventory_sha256": (
            sup.TASK_ASSET_DYNAMIC_INVENTORY_SHA256
        ),
        "task_asset_unresolved_inventory_sha256": (
            sup.TASK_ASSET_UNRESOLVED_INTERACTION_INVENTORY_SHA256
        ),
        "asset_contract": str(receipt),
        "asset_contract_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }


def test_asset_result_rejects_an_external_receipt(tmp_path: Path) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    outside = tmp_path / "outside-receipt.json"
    outside.write_text("{}\n", encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(_minimal_asset_result(supervisor, outside)), encoding="utf-8"
    )

    with pytest.raises(sup.InvalidArtifact, match="canonical run artifact"):
        supervisor._validate_result(sup.STAGES["asset_contract"], candidate)


def test_asset_result_requires_shovel_result_identity(tmp_path: Path) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    value = _minimal_asset_result(supervisor, receipt)
    value.pop("shovel_contact_points_pose_sha256")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        sup.InvalidArtifact, match="asset contract result shovel overlay/hash differs"
    ):
        supervisor._validate_result(sup.STAGES["asset_contract"], candidate)


def test_asset_result_rejects_an_external_overlay(tmp_path: Path) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    receipt = (
        supervisor.s.run
        / "artifacts"
        / "asset_contract"
        / "asset_contract.json"
    )
    receipt.parent.mkdir(parents=True)
    outside_overlay = tmp_path / "outside-overlay.json"
    outside_overlay.write_text("{}\n", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "schema": "before-we-act.bicoord.asset-contract/1",
                "status": "PASSED",
                "dataset_repo_id": sup.FORMAL_DATASET_REPO,
                "dataset_revision": sup.FORMAL_DATASET_REVISION,
                "benchmark_revision": sup.BICOORD_CODE_REVISION,
                "tasks": list(sup.TASKS),
                "supplemental_assets_installed": True,
                "benchmark_tracked_source_modified": False,
                "task_source_modified": False,
                "upstream_model_modified": False,
                "normalization_modified": False,
                "plate_overlay": {
                    "overlay_metadata": str(outside_overlay),
                    "target_metadata_sha256": hashlib.sha256(
                        outside_overlay.read_bytes()
                    ).hexdigest(),
                    "copied_fields": ["contact_points_pose"],
                    "small_scale_preserved": True,
                    "contact_points_pose_count": 4,
                    "task_source_modified": False,
                    "planner_modified": False,
                    "model_modified": False,
                    "normalization_modified": False,
                    "benchmark_asset_source_modified": False,
                    "mutation_scope": (
                        "run_artifact_and_actor_config_in_memory_only"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(_minimal_asset_result(supervisor, receipt)), encoding="utf-8"
    )

    with pytest.raises(sup.InvalidArtifact, match="canonical run artifact"):
        supervisor._validate_result(sup.STAGES["asset_contract"], candidate)


def _runtime_overlay_expectations(tmp_path: Path) -> dict[str, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    receipt = tmp_path / "asset_contract.json"
    receipt.write_text("{}\n", encoding="utf-8")
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    return {
        "place_plate_and_cup": {
            "overlay": str(tmp_path / "003_plate" / "model_data0.json"),
            "contact_points_pose_sha256": "a" * 64,
            "contact_points_pose_count": 4,
            "receipt": str(receipt),
            "receipt_sha256": receipt_hash,
        },
        "sweep_block": {
            "overlay": str(tmp_path / "082_smallshovel" / "model_data3.json"),
            "contact_points_pose_sha256": (
                sup.SHOVEL_CONTACT_POINTS_POSE_SHA256
            ),
            "contact_points_pose_count": 1,
            "receipt": str(receipt),
            "receipt_sha256": receipt_hash,
        },
    }


def _runtime_overlay_attempts(
    expectations: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    plate = expectations["place_plate_and_cup"]
    shovel = expectations["sweep_block"]
    return {
        "place_plate_and_cup": {
            "structural_error": False,
            "asset_overlay": {
                "task": "place_plate_and_cup",
                "applied": True,
                "overlay": plate["overlay"],
                "receipt": plate["receipt"],
                "receipt_sha256": plate["receipt_sha256"],
                "contact_points_pose_sha256": plate[
                    "contact_points_pose_sha256"
                ],
                "actors": {
                    name: {
                        "before_sha256": "0" * 64,
                        "after_sha256": plate["contact_points_pose_sha256"],
                        "contact_points_pose_count": 4,
                        "scale_preserved": True,
                        "changed_fields": ["contact_points_pose"],
                    }
                    for name in ("plate", "plate_2")
                },
                "copied_fields": ["contact_points_pose"],
                "task_source_modified": False,
            },
        },
        "sweep_block": {
            "structural_error": False,
            "asset_overlay": {
                "task": "sweep_block",
                "applied": True,
                "overlay": shovel["overlay"],
                "receipt": shovel["receipt"],
                "receipt_sha256": shovel["receipt_sha256"],
                "contact_points_pose_sha256": shovel[
                    "contact_points_pose_sha256"
                ],
                "actors": {
                    "shovel": {
                        "before_sha256": "0" * 64,
                        "after_sha256": shovel[
                            "contact_points_pose_sha256"
                        ],
                        "contact_points_pose_count": 1,
                        "scale_preserved": True,
                        "changed_fields": ["contact_points_pose"],
                    }
                },
                "copied_fields": ["contact_points_pose"],
                "derived_fields": ["contact_points_pose"],
                "source_fields": ["contact_pose", "trans_matrix"],
                "legacy_conversion": True,
                "task_source_modified": False,
            },
        },
    }


def test_seed_attempts_bind_both_runtime_overlays(tmp_path: Path) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    expectations = _runtime_overlay_expectations(tmp_path / "evidence")
    attempts = _runtime_overlay_attempts(expectations)
    for task, attempt in attempts.items():
        supervisor._validate_seed_asset_overlay(
            task,
            attempt,
            expectations,
            context=f"{task} attempt 1",
        )


@pytest.mark.parametrize(
    ("task", "field", "value", "message"),
    [
        (
            "place_plate_and_cup",
            "overlay",
            "/outside/plate.json",
            "overlay path differs",
        ),
        (
            "sweep_block",
            "contact_points_pose_sha256",
            "f" * 64,
            "converted contact hash differs",
        ),
    ],
)
def test_seed_attempt_rejects_runtime_overlay_drift(
    tmp_path: Path, task: str, field: str, value: object, message: str
) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    expectations = _runtime_overlay_expectations(tmp_path / "evidence")
    attempt = _runtime_overlay_attempts(expectations)[task]
    overlay = attempt["asset_overlay"]
    assert isinstance(overlay, dict)
    overlay[field] = value
    with pytest.raises(sup.InvalidArtifact, match=message):
        supervisor._validate_seed_asset_overlay(
            task,
            attempt,
            expectations,
            context=f"{task} attempt 1",
        )


def test_seed_attempt_retains_explicit_pre_overlay_structural_failure(
    tmp_path: Path,
) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    expectations = _runtime_overlay_expectations(tmp_path / "evidence")
    shovel = expectations["sweep_block"]
    attempt = {
        "valid": False,
        "plan_success": False,
        "expert_success": False,
        "structural_error": True,
        "error_type": "RuntimeError",
        "error_signature": "a" * 64,
        "asset_overlay": {
            "task": "sweep_block",
            "applied": False,
            "overlay": shovel["overlay"],
            "contact_points_pose_sha256": None,
            "reason": "environment_construction_failed_before_overlay_receipt",
        },
    }
    supervisor._validate_seed_asset_overlay(
        "sweep_block",
        attempt,
        expectations,
        context="sweep_block attempt 1",
    )


def test_interrupted_gpu_wave_is_terminal_not_retryable(tmp_path: Path) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)
    active = object()
    scheduler._spawn = lambda *_args, **_kwargs: active  # type: ignore[method-assign]

    def interrupted(_active):
        raise sup.Interrupted("service stop")

    scheduler._wait = interrupted  # type: ignore[method-assign]
    with pytest.raises(sup.Interrupted, match="service stop"):
        scheduler.run_wave(
            [("worker", ["python", "-V"], 0, tmp_path / "worker.log")]
        )


def test_interrupted_scheduler_cannot_spawn_another_worker(tmp_path: Path) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)
    scheduler.interrupt()
    with pytest.raises(sup.Interrupted, match="supervisor interrupted"):
        scheduler._spawn(
            "worker", ["python", "-V"], (0,), tmp_path / "worker.log"
        )
    assert not (tmp_path / "worker.log").exists()


class _FakeSchedulerProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode: int | None = None
        self.wait_calls: list[float | None] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_interrupt_race_before_popen_never_launches_child(
    tmp_path: Path, monkeypatch
) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)
    original_environment = scheduler.environment
    popen_calls: list[list[str]] = []

    def interrupting_environment(gpus):
        scheduler.interrupt()
        return original_environment(gpus)

    def fake_popen(command, **_kwargs):
        popen_calls.append(list(command))
        return _FakeSchedulerProcess(4101)

    monkeypatch.setattr(scheduler, "environment", interrupting_environment)
    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    with pytest.raises(sup.Interrupted, match="supervisor interrupted"):
        scheduler._spawn(
            "worker", ["python", "-V"], (0,), tmp_path / "worker.log"
        )
    assert popen_calls == []
    assert scheduler.active == {}
    assert scheduler.snapshot() == []
    assert not (tmp_path / "worker.log").exists()


def test_interrupt_race_inside_popen_terminates_reaps_and_unregisters_child(
    tmp_path: Path, monkeypatch
) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)
    process = _FakeSchedulerProcess(4102)
    kill_calls: list[tuple[int, int]] = []
    captured_stdout = None

    def fake_popen(_command, **kwargs):
        nonlocal captured_stdout
        captured_stdout = kwargs["stdout"]
        # This models SIGTERM arriving after the pre-Popen check but before
        # Popen has returned an object that the scheduler can register.
        scheduler.interrupt()
        return process

    def fake_killpg(pgid, signum):
        kill_calls.append((pgid, signum))
        process.returncode = -signum

    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sup.os, "killpg", fake_killpg)
    with pytest.raises(sup.Interrupted, match="supervisor interrupted"):
        scheduler._spawn(
            "worker", ["python", "-V"], (0,), tmp_path / "worker.log"
        )
    assert kill_calls == [(process.pid, sup.signal.SIGTERM)]
    assert process.wait_calls == [sup.CHILD_TERMINATION_GRACE_SECONDS]
    assert captured_stdout is not None and captured_stdout.closed
    assert scheduler.active == {}
    assert scheduler.snapshot() == []


def test_interrupt_race_after_popen_terminates_reaps_and_unregisters_child(
    tmp_path: Path, monkeypatch
) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)
    kill_calls: list[tuple[int, int]] = []

    class InterruptAfterPopenProcess(_FakeSchedulerProcess):
        def __init__(self, pid: int):
            self._pid = pid
            self._interrupt_on_pid_read = True
            self.returncode = None
            self.wait_calls = []

        @property
        def pid(self):
            # Popen has returned; the next scheduler operation reads the PID
            # to register it.  Deliver SIGTERM immediately before that write.
            if self._interrupt_on_pid_read:
                self._interrupt_on_pid_read = False
                scheduler.interrupt()
            return self._pid

    process = InterruptAfterPopenProcess(4105)

    def fake_killpg(pgid, signum):
        kill_calls.append((pgid, signum))
        process.returncode = -signum

    monkeypatch.setattr(sup.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(sup.os, "killpg", fake_killpg)
    with pytest.raises(sup.Interrupted, match="supervisor interrupted"):
        scheduler._spawn(
            "worker", ["python", "-V"], (0,), tmp_path / "worker.log"
        )
    assert kill_calls == [(4105, sup.signal.SIGTERM)]
    assert process.wait_calls == [sup.CHILD_TERMINATION_GRACE_SECONDS]
    assert scheduler.active == {}
    assert scheduler.snapshot() == []


def test_partial_wave_spawn_failure_reaps_every_started_owned_group(
    tmp_path: Path, monkeypatch
) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)
    first = _FakeSchedulerProcess(4103)
    popen_count = 0
    kill_calls: list[tuple[int, int]] = []

    def fake_popen(_command, **_kwargs):
        nonlocal popen_count
        popen_count += 1
        if popen_count == 1:
            return first
        raise OSError("second spawn failed")

    def fake_killpg(pgid, signum):
        kill_calls.append((pgid, signum))
        assert pgid == first.pid
        first.returncode = -signum

    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sup.os, "killpg", fake_killpg)
    with pytest.raises(OSError, match="second spawn failed"):
        scheduler.run_wave(
            [
                ("first", ["python", "-V"], 0, tmp_path / "first.log"),
                ("second", ["python", "-V"], 1, tmp_path / "second.log"),
            ]
        )
    assert kill_calls == [(first.pid, sup.signal.SIGTERM)]
    assert first.wait_calls == [sup.CHILD_TERMINATION_GRACE_SECONDS]
    assert scheduler.active == {}
    assert scheduler.snapshot() == []


def test_group_signal_requires_exact_active_registry_identity(
    tmp_path: Path, monkeypatch
) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)
    owned_process = _FakeSchedulerProcess(4104)
    stale_process = _FakeSchedulerProcess(4104)
    owned = sup.ActiveProcess(
        "owned", owned_process, (0,), tmp_path / "owned.log", 1.0
    )
    stale = sup.ActiveProcess(
        "stale", stale_process, (0,), tmp_path / "stale.log", 1.0
    )
    scheduler.active[owned_process.pid] = owned
    kill_calls: list[tuple[int, int]] = []

    def fake_killpg(pgid, signum):
        kill_calls.append((pgid, signum))
        owned_process.returncode = -signum

    monkeypatch.setattr(sup.os, "killpg", fake_killpg)
    scheduler._signal_owned_process_group(stale, sup.signal.SIGTERM)
    assert kill_calls == []
    scheduler._signal_owned_process_group(owned, sup.signal.SIGTERM)
    assert kill_calls == [(owned_process.pid, sup.signal.SIGTERM)]
    assert scheduler._terminate_and_reap((owned,)) == []
    assert scheduler.active == {}


def test_wait_escalates_and_cleans_registry_when_child_ignores_sigterm(
    tmp_path: Path, monkeypatch
) -> None:
    scheduler = sup.GpuScheduler(_settings(tmp_path), lambda: None)

    class StubbornProcess(_FakeSchedulerProcess):
        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if self.returncode is not None:
                return self.returncode
            if timeout is not None:
                raise sup.subprocess.TimeoutExpired(["stubborn"], timeout)
            raise AssertionError("unbounded wait occurred before SIGKILL")

    process = StubbornProcess(4106)
    active = sup.ActiveProcess(
        "stubborn", process, (0,), tmp_path / "stubborn.log", 1.0
    )
    scheduler.active[process.pid] = active
    kill_calls: list[tuple[int, int]] = []

    def fake_killpg(pgid, signum):
        kill_calls.append((pgid, signum))
        if signum == sup.signal.SIGKILL:
            process.returncode = -signum

    monkeypatch.setattr(sup.os, "killpg", fake_killpg)
    scheduler.interrupt()
    with pytest.raises(sup.Interrupted, match="supervisor interrupted"):
        scheduler._wait(active)
    assert kill_calls == [
        (process.pid, sup.signal.SIGTERM),
        (process.pid, sup.signal.SIGTERM),
        (process.pid, sup.signal.SIGKILL),
    ]
    assert process.wait_calls == [
        sup.CHILD_WAIT_POLL_SECONDS,
        sup.CHILD_TERMINATION_GRACE_SECONDS,
        None,
    ]
    assert scheduler.active == {}
    assert scheduler.snapshot() == []


def test_signal_handler_sets_spawn_barrier_before_status_io(
    tmp_path: Path, monkeypatch
) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path))
    handlers = {}
    events: list[str] = []

    def fake_signal(signum, handler):
        handlers[signum] = handler
        return sup.signal.SIG_DFL

    monkeypatch.setattr(sup.signal, "signal", fake_signal)
    monkeypatch.setattr(
        supervisor.scheduler, "interrupt", lambda: events.append("interrupt")
    )
    monkeypatch.setattr(
        supervisor,
        "_set_status",
        lambda *_args, **_kwargs: events.append("status"),
    )
    supervisor._install_signals()
    handlers[sup.signal.SIGTERM](sup.signal.SIGTERM, None)
    assert events == ["interrupt", "status"]


@pytest.mark.parametrize(
    ("stage_name", "expected_steps"),
    [
        ("b0h_smoke_closed_loop", 2),
        ("bcore_smoke_closed_loop", 2),
        ("paired_validation_smoke", 2),
        ("b0h_probe", sup.MAX_STEPS[sup.TASKS[0]]),
        ("bcore_validation20", sup.MAX_STEPS[sup.TASKS[0]]),
        ("paired_validation20", sup.MAX_STEPS[sup.TASKS[0]]),
    ],
)
def test_task_queue_uses_short_smoke_but_formal_task_horizon(
    tmp_path: Path, stage_name: str, expected_steps: int
) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path))
    commands: list[list[str]] = []

    class StopAfterFirstWave(RuntimeError):
        pass

    class FakeScheduler:
        def run_wave(self, jobs):
            commands.extend(list(command) for _name, command, _gpu, _log in jobs)
            raise StopAfterFirstWave

    supervisor.scheduler = FakeScheduler()  # type: ignore[assignment]
    with pytest.raises(StopAfterFirstWave):
        supervisor._task_queue_action(
            sup.STAGES[stage_name], tmp_path / f"{stage_name}.json"
        )
    assert commands
    for command in commands:
        task = command[command.index("--task") + 1]
        expected = 2 if "smoke" in stage_name else sup.MAX_STEPS[task]
        assert int(command[command.index("--max-steps") + 1]) == expected
    assert int(commands[0][commands[0].index("--max-steps") + 1]) == expected_steps


def _smoke_validation_result(
    tmp_path: Path,
    *,
    steps: int,
    success: bool,
    progress_success: bool | None = None,
) -> dict[str, object]:
    tasks: dict[str, object] = {}
    for index, task in enumerate(sup.TASKS):
        seed = 900_000 + index
        progress = tmp_path / task / "progress.jsonl"
        progress.parent.mkdir(parents=True)
        progress.write_text(
            "".join(
                json.dumps(
                    {
                        "task": task,
                        "seed": seed,
                        "step": step,
                        "max_steps": sup.SMOKE_INTERFACE_STEPS,
                        "success": (
                            success if progress_success is None else progress_success
                        ),
                        "action_clipped": False,
                        "policy_output_clipping": False,
                        "executed_gripper_oob_count": 0,
                    },
                    sort_keys=True,
                )
                + "\n"
                for step in range(1, steps + 1)
            ),
            encoding="utf-8",
        )
        episode_row = {
            "task": task,
            "seed": seed,
            "steps": steps,
            "max_steps": sup.SMOKE_INTERFACE_STEPS,
            "success": success,
            "action_trace_sha256": "a" * 64,
            "prediction_gripper_oob_count": 0,
            "ensemble_plan_gripper_oob_count": 0,
            "executed_gripper_oob_count": 0,
            "policy_output_clipping": False,
            "action_clipping": False,
            "state_clipping": False,
            "gripper_reparameterization": False,
        }
        receipt = tmp_path / task / "receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "status": "PASSED",
                    "task": task,
                    "episodes": 1,
                    "completed": 1,
                    "max_steps": sup.SMOKE_INTERFACE_STEPS,
                    "rollout_steps": [steps],
                    "rows": [episode_row],
                    "rows_path": str(progress.resolve()),
                    "rows_sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
                    "smoke_interface_steps": sup.SMOKE_INTERFACE_STEPS,
                    "policy_output_clipping": False,
                    "action_clipping": False,
                    "state_clipping": False,
                    "gripper_reparameterization": False,
                    "executed_gripper_oob_count": 0,
                    "prediction_gripper_oob_count": 0,
                    "ensemble_plan_gripper_oob_count": 0,
                }
            ),
            encoding="utf-8",
        )
        tasks[task] = {
            "episodes": 1,
            "completed": 1,
            "successes": int(success),
            "max_steps": sup.SMOKE_INTERFACE_STEPS,
            "rollout_steps": [steps],
            "paired": False,
            "progress_receipt": str(receipt.resolve()),
            "progress_receipt_sha256": hashlib.sha256(
                receipt.read_bytes()
            ).hexdigest(),
            "smoke_interface_steps": sup.SMOKE_INTERFACE_STEPS,
            "policy_output_clipping": False,
            "action_clipping": False,
            "state_clipping": False,
            "gripper_reparameterization": False,
            "executed_gripper_oob_count": 0,
            "prediction_gripper_oob_count": 0,
            "ensemble_plan_gripper_oob_count": 0,
            "artifacts": [
                {
                    "path": str(progress.resolve()),
                    "sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
                    "kind": "validation_progress",
                },
                {
                    "path": str(receipt.resolve()),
                    "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    "kind": "progress_receipt",
                },
            ],
        }
    return {"tasks": tasks}


@pytest.mark.parametrize(("steps", "success"), [(2, False), (1, True)])
def test_smoke_receipt_accepts_horizon_or_early_success(
    tmp_path: Path, steps: int, success: bool
) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    supervisor._validate_task_results(
        _smoke_validation_result(tmp_path / "evidence", steps=steps, success=success),
        1,
        require_success=False,
        max_steps_by_task=sup.SMOKE_MAX_STEPS,
    )


def test_smoke_receipt_rejects_unsuccessful_early_stop(tmp_path: Path) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    with pytest.raises(sup.InvalidArtifact, match="before horizon without success"):
        supervisor._validate_task_results(
            _smoke_validation_result(tmp_path / "evidence", steps=1, success=False),
            1,
            require_success=False,
            max_steps_by_task=sup.SMOKE_MAX_STEPS,
        )


def test_smoke_receipt_rejects_shallow_success_mismatch(tmp_path: Path) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    result = _smoke_validation_result(
        tmp_path / "evidence", steps=2, success=False
    )
    task_rows = result["tasks"]
    assert isinstance(task_rows, dict)
    first_row = task_rows[sup.TASKS[0]]
    assert isinstance(first_row, dict)
    first_row["successes"] = 1
    with pytest.raises(sup.InvalidArtifact, match="success summary differs"):
        supervisor._validate_task_results(
            result,
            1,
            require_success=False,
            max_steps_by_task=sup.SMOKE_MAX_STEPS,
        )


def test_smoke_receipt_rejects_progress_success_mismatch(tmp_path: Path) -> None:
    supervisor = sup.Supervisor(_settings(tmp_path / "settings"))
    result = _smoke_validation_result(
        tmp_path / "evidence",
        steps=1,
        success=True,
        progress_success=False,
    )
    with pytest.raises(sup.InvalidArtifact, match="progress final success differs"):
        supervisor._validate_task_results(
            result,
            1,
            require_success=False,
            max_steps_by_task=sup.SMOKE_MAX_STEPS,
        )


def _evaluation_args(tmp_path: Path, **values) -> argparse.Namespace:
    for name in ("repo", "benchmark", "dataset", "dino"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    defaults = {
        "repo": tmp_path / "repo",
        "benchmark_repo": tmp_path / "benchmark",
        "dataset": tmp_path / "dataset",
        "run": tmp_path / "run",
        "dino_model": tmp_path / "dino",
        "result": tmp_path / "result.json",
        "config_sha256": "a" * 64,
        "auto_resume": False,
        "record_progress": True,
        "seed_start": None,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_b0h_smoke_requires_exactly_one_episode(tmp_path: Path) -> None:
    args = _evaluation_args(
        tmp_path,
        operation="smoke-closed-loop",
        task=sup.TASKS[0],
        episodes=2,
        max_steps=sup.SMOKE_INTERFACE_STEPS,
    )
    with pytest.raises(ValueError, match="requires 1 episodes"):
        run_b0h_evaluation(args)


@pytest.mark.parametrize(
    ("operation", "max_steps", "expected"),
    [
        ("smoke-paired", sup.MAX_STEPS[sup.TASKS[0]], 2),
        ("validation20-paired", 2, sup.MAX_STEPS[sup.TASKS[0]]),
    ],
)
def test_paired_evaluator_separates_smoke_and_formal_horizons(
    tmp_path: Path, operation: str, max_steps: int, expected: int
) -> None:
    args = _evaluation_args(
        tmp_path,
        operation=operation,
        task=sup.TASKS[0],
        episodes=1 if operation == "smoke-paired" else 20,
        max_steps=max_steps,
    )
    with pytest.raises(ValueError, match=rf"must be {expected}"):
        run_paired_evaluation(args)


def _write_paired_progress_fixture(
    path: Path,
    *,
    task: str,
    mode: str,
    max_steps: int,
    seed_steps: list[tuple[int, list[int]]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task": task,
                    "seed": seed,
                    "mode": mode,
                    "step": step,
                    "max_steps": max_steps,
                    "action_clipped": False,
                },
                sort_keys=True,
            )
            + "\n"
            for seed, steps in seed_steps
            for step in steps
        ),
        encoding="utf-8",
    )


def test_combined_paired_progress_accepts_seed_local_step_reset(
    tmp_path: Path,
) -> None:
    task = sup.TASKS[0]
    path = tmp_path / "combined.jsonl"
    _write_paired_progress_fixture(
        path,
        task=task,
        mode="care",
        max_steps=4,
        seed_steps=[(7001, [1, 2]), (7002, [1, 2, 3])],
    )

    sup._validate_paired_progress(
        path,
        mode="care",
        task=task,
        max_steps=4,
        seed_steps=((7001, 2), (7002, 3)),
        context="two-episode combined CARE progress",
    )


def test_combined_paired_progress_rejects_cross_seed_global_step_sequence(
    tmp_path: Path,
) -> None:
    task = sup.TASKS[0]
    path = tmp_path / "combined.jsonl"
    _write_paired_progress_fixture(
        path,
        task=task,
        mode="selector_off",
        max_steps=4,
        seed_steps=[(7001, [1, 2]), (7002, [3, 4])],
    )

    with pytest.raises(sup.InvalidArtifact, match="seed/step sequence differs"):
        sup._validate_paired_progress(
            path,
            mode="selector_off",
            task=task,
            max_steps=4,
            seed_steps=((7001, 2), (7002, 2)),
            context="two-episode combined selector-off progress",
        )


def test_care_grid_has_twelve_main_and_three_oof_jobs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    supervisor = sup.Supervisor(settings)
    commands: list[tuple[str, list[str], int]] = []

    class FakeScheduler:
        def run_wave(self, jobs):
            commands.extend((name, list(command), gpu) for name, command, gpu, _log in jobs)

    supervisor.scheduler = FakeScheduler()  # type: ignore[assignment]
    supervisor._aggregate_worker_results = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    supervisor._care_grid_action(sup.STAGES["belief_train"], tmp_path / "belief.json")
    assert len(commands) == 15
    assert sum("--oof-shadow-fold" in command for _name, command, _gpu in commands) == 3
    assert [
        int(command[command.index("--oof-shadow-fold") + 1])
        for _name, command, _gpu in commands
        if "--oof-shadow-fold" in command
    ] == [0, 1, 2]
    assert all(gpu in range(4) for _name, _command, gpu in commands)


def test_scheduler_children_use_benchmark_cwd(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    supervisor = sup.Supervisor(settings)
    captured = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(command, *, cwd, env, **kwargs):
        captured["cwd"] = cwd
        captured["env"] = env
        captured["start_new_session"] = kwargs["start_new_session"]
        return FakeProcess()

    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sup, "_utc_now", lambda: "now")
    active = supervisor.scheduler._spawn(
        "cwd-test", ["python", "-c", "pass"], (0,), tmp_path / "run.log"
    )
    assert active.process.pid == 4242
    assert captured["cwd"] == settings.benchmark_repo
    assert captured["start_new_session"] is True
