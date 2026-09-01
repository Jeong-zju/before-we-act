from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deployment.bicoord_care.config import TASKS
from deployment.bicoord_care import seed_discovery as seed_stage
from deployment.bicoord_care import supervisor as supervisor_stage
from deployment.bicoord_care.asset_runtime import (
    OVERLAY_ENV,
    REQUIRED_ENV,
)
from deployment.bicoord_care.seed_discovery import (
    OFFICIAL_SEED_MULTIPLIER,
    RepeatedStructuralSeedError,
    SEED_PROGRESS_SCHEMA,
    STRUCTURAL_ERROR_STREAK_LIMIT,
    discover,
    run,
)
from deployment.bicoord_care.stage_common import (
    RESULT_SCHEMA,
    atomic_json,
    read_json,
    sha256_file,
)


TASK = TASKS[0]


class UnStableError(Exception):
    """Fixture spelling used by the official BiCoord environment."""


class _ErrorEnv:
    plan_success = False
    eval_success = False
    stage_eval_score = 0.0

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.closed = False

    def play_once(self) -> None:
        # Both numeric fields legitimately vary between candidates.  They must
        # not turn one structural call site into a different signature/seed.
        raise IndexError(
            f"003_plate contact point 2 is absent for seed {self.seed} "
            f"at {hex(self.seed)}"
        )

    def check_success(self) -> bool:
        raise AssertionError("unreachable")

    def close_env(self) -> None:
        self.closed = True


class _OutcomeEnv:
    eval_success = False
    stage_eval_score = 0.0

    def __init__(self, seed: int, outcome: str) -> None:
        self.seed = seed
        self.outcome = outcome
        self.plan_success = False

    def play_once(self) -> dict[str, Any]:
        if self.outcome == "error":
            raise IndexError(f"contact point 2 absent for seed {self.seed}")
        if self.outcome == "planner_exception":
            raise RuntimeError(
                f"motion planning failed: no feasible path for seed {self.seed}"
            )
        self.plan_success = self.outcome == "success"
        return {"official_expert": True}

    def check_success(self) -> bool:
        return self.outcome == "success"

    def close_env(self) -> None:
        pass


def _success_env(seed: int) -> _OutcomeEnv:
    return _OutcomeEnv(seed, "success")


def test_repeated_structural_exception_fails_closed_after_three_atomic_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots: list[dict[str, Any]] = []
    real_atomic_json = seed_stage.atomic_json

    def recording_atomic_json(path: str | Path, value: object) -> Path:
        if Path(path).name == f"progress_{TASK}.json":
            snapshots.append(json.loads(json.dumps(value)))
        return real_atomic_json(path, value)

    monkeypatch.setattr(seed_stage, "atomic_json", recording_atomic_json)
    monkeypatch.setattr(
        seed_stage,
        "_make_env",
        lambda _root, _task, seed: _ErrorEnv(seed),
    )

    with pytest.raises(
        RepeatedStructuralSeedError,
        match=r"failed closed after 3 consecutive IndexError exceptions",
    ):
        discover(
            tmp_path,
            episodes=1,
            max_attempts=5_000,
            task=TASK,
            progress_dir=tmp_path,
        )

    # One initialization write plus exactly one atomic replacement per seed.
    assert [row["attempts_completed"] for row in snapshots] == [0, 1, 2, 3]
    receipt = read_json(tmp_path / f"progress_{TASK}.json")
    assert receipt["schema"] == SEED_PROGRESS_SCHEMA
    assert receipt["status"] == "FAILED"
    assert receipt["attempts_completed"] == STRUCTURAL_ERROR_STREAK_LIMIT
    assert receipt["failure"]["reason"] == "repeated_structural_exception"
    assert receipt["exception_type_counts"] == {"IndexError": 3}
    assert receipt["exception_counts"][0]["count"] == 3
    signatures = {row["error_signature"] for row in receipt["attempts"]}
    assert signatures == {receipt["failure"]["error_signature"]}
    assert [row["seed"] for row in receipt["attempts"]] == [
        OFFICIAL_SEED_MULTIPLIER + offset for offset in range(3)
    ]
    assert receipt["consecutive_structural_error"]["count"] == 3
    seed_receipts = sorted((tmp_path / "attempts" / TASK).glob("seed_*.json"))
    assert len(seed_receipts) == 3
    for index, path in enumerate(seed_receipts, start=1):
        value = read_json(path)
        assert value["attempt_index"] == index
        assert value["row"]["error_signature"] == receipt["failure"]["error_signature"]
    assert not list(tmp_path.glob(".progress_*.tmp"))


def test_planner_failure_continues_search_and_breaks_exception_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        100_000: "error",
        100_001: "error",
        100_002: "planner_failure",
        100_003: "error",
        100_004: "error",
        100_005: "success",
    }
    monkeypatch.setattr(
        seed_stage,
        "_make_env",
        lambda _root, _task, seed: _OutcomeEnv(seed, outcomes[seed]),
    )

    manifest = discover(
        tmp_path,
        episodes=1,
        max_attempts=len(outcomes),
        task=TASK,
        progress_dir=tmp_path,
    )

    assert manifest["valid_seeds"] == {TASK: [100_005]}
    rows = manifest["attempts"][TASK]
    assert len(rows) == len(outcomes)
    planner_row = rows[2]
    assert planner_row["valid"] is False
    assert planner_row["plan_success"] is False
    assert planner_row["structural_error"] is False
    assert "error_signature" not in planner_row
    assert manifest["exception_type_counts"] == {TASK: {"IndexError": 4}}
    assert manifest["exception_counts"][TASK][0]["count"] == 4
    assert len(manifest["seed_receipts"][TASK]) == len(outcomes)
    assert manifest["seed_receipts_sha256"][TASK] == [
        sha256_file(path) for path in manifest["seed_receipts"][TASK]
    ]

    receipt = read_json(tmp_path / f"progress_{TASK}.json")
    assert receipt["status"] == "PASSED"
    assert receipt["valid_seeds"] == [100_005]
    assert receipt["consecutive_structural_error"] is None
    assert "failure" not in receipt


def test_repeated_normal_planner_exceptions_are_counted_but_do_not_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        100_000: "planner_exception",
        100_001: "planner_exception",
        100_002: "planner_exception",
        100_003: "planner_exception",
        100_004: "success",
    }
    monkeypatch.setattr(
        seed_stage,
        "_make_env",
        lambda _root, _task, seed: _OutcomeEnv(seed, outcomes[seed]),
    )

    manifest = discover(
        tmp_path,
        episodes=1,
        max_attempts=len(outcomes),
        task=TASK,
        progress_dir=tmp_path,
    )

    assert manifest["valid_seeds"] == {TASK: [100_004]}
    assert manifest["exception_type_counts"] == {TASK: {"RuntimeError": 4}}
    assert manifest["structural_exception_type_counts"] == {TASK: {}}
    for row in manifest["attempts"][TASK][:-1]:
        assert row["structural_error"] is False
        assert len(row["error_signature"]) == 64


def test_three_official_unstable_seeds_continue_to_a_later_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make_env(_root: Path, _task: str, seed: int) -> _OutcomeEnv:
        if seed < 100_003:
            raise UnStableError(f"Objects is unstable in seed({seed})")
        return _success_env(seed)

    monkeypatch.setattr(seed_stage, "_make_env", make_env)

    manifest = discover(
        tmp_path,
        episodes=1,
        max_attempts=4,
        task=TASK,
        progress_dir=tmp_path,
    )

    assert manifest["valid_seeds"] == {TASK: [100_003]}
    unstable_rows = manifest["attempts"][TASK][:-1]
    assert len(unstable_rows) == STRUCTURAL_ERROR_STREAK_LIMIT
    assert all(row["structural_error"] is False for row in unstable_rows)
    assert all(row["expected_seed_rejection"] is True for row in unstable_rows)
    assert len({row["error_signature"] for row in unstable_rows}) == 1
    assert manifest["exception_type_counts"] == {TASK: {"UnStableError": 3}}
    assert manifest["structural_exception_type_counts"] == {TASK: {}}


def test_required_plate_overlay_missing_is_a_structural_failure_with_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plate_task = "place_plate_and_cup"
    benchmark = tmp_path / "benchmark"
    task_config = benchmark / "task_config"
    robot = benchmark / "robot"
    task_config.mkdir(parents=True)
    robot.mkdir()
    (task_config / "demo_clean.yml").write_text("embodiment: [fake]\n")
    (task_config / "_embodiment_config.yml").write_text(
        "fake:\n  file_path: robot\n"
    )
    (robot / "config.yml").write_text("{}\n")

    class FakeOfficialTask:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.setup_called = False
            self.closed = False
            self.__class__.instances.append(self)

        def setup_demo(self, **_config: Any) -> None:
            self.setup_called = True

        def close_env(self) -> None:
            self.closed = True

    real_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> Any:
        if name == f"envs.{plate_task}":
            return SimpleNamespace(**{plate_task: FakeOfficialTask})
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setenv(REQUIRED_ENV, "1")
    monkeypatch.delenv(OVERLAY_ENV, raising=False)

    with pytest.raises(
        RepeatedStructuralSeedError,
        match=r"consecutive RuntimeAssetError exceptions",
    ):
        discover(
            benchmark,
            episodes=1,
            max_attempts=10,
            task=plate_task,
            progress_dir=tmp_path,
        )

    assert len(FakeOfficialTask.instances) == STRUCTURAL_ERROR_STREAK_LIMIT
    assert all(env.setup_called and env.closed for env in FakeOfficialTask.instances)
    receipt = read_json(tmp_path / f"progress_{plate_task}.json")
    assert receipt["exception_type_counts"] == {"RuntimeAssetError": 3}
    for row in receipt["recent_attempts"]:
        assert row["asset_overlay"]["applied"] is False
        assert row["asset_overlay"]["overlay"] is None
        assert row["asset_overlay"]["contact_points_pose_sha256"] is None


def test_plate_attempt_records_applied_overlay_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plate_task = "place_plate_and_cup"
    overlay = {
        "task": plate_task,
        "applied": True,
        "overlay": "/audited/003_plate/model_data0.json",
        "contact_points_pose_sha256": "b" * 64,
        "copied_fields": ["contact_points_pose"],
    }
    env = _success_env(100_000)
    env._bicoord_asset_overlay = overlay
    monkeypatch.setattr(seed_stage, "_make_env", lambda *_args: env)

    manifest = discover(
        tmp_path,
        episodes=1,
        max_attempts=1,
        task=plate_task,
        progress_dir=tmp_path,
    )

    assert manifest["attempts"][plate_task][0]["asset_overlay"] == overlay
    seed_receipt = read_json(manifest["seed_receipts"][plate_task][0])
    assert seed_receipt["row"]["asset_overlay"] == overlay


def test_task_worker_result_hash_binds_progress_for_supervisor_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    benchmark = tmp_path / "benchmark"
    dataset = tmp_path / "dataset"
    dino = tmp_path / "dino"
    for path in (repo, benchmark, dataset, dino):
        path.mkdir()
    run_root = tmp_path / "run"
    config_sha256 = "a" * 64
    dependency = run_root / "stage_results" / "bcore_smoke_closed_loop.json"
    atomic_json(
        dependency,
        {
            "schema": RESULT_SCHEMA,
            "stage": "bcore_smoke_closed_loop",
            "status": "PASSED",
            "benchmark_adapter": "BiCoord",
            "config_sha256": config_sha256,
        },
    )
    monkeypatch.setattr(
        seed_stage,
        "_make_env",
        lambda _root, _task, seed: _success_env(seed),
    )
    args = argparse.Namespace(
        operation="smoke-discover",
        repo=repo,
        benchmark_repo=benchmark,
        dataset=dataset,
        run=run_root,
        dino_model=dino,
        result=run_root / "worker_results" / "seed_discovery_smoke" / f"{TASK}.json",
        config_sha256=config_sha256,
        auto_resume=True,
        seed_bucket=0,
        max_attempts=5_000,
        smoke=False,
        task=TASK,
        episodes=1,
    )

    result = run(args)

    assert result["stage"] == "seed_discovery_worker"
    assert result["task"] == TASK
    assert result["completed"] == 1
    progress = Path(result["progress_receipt"])
    assert progress.is_file()
    assert result["progress_receipt_sha256"] == sha256_file(progress)
    assert result["progress_receipts"] == {TASK: str(progress)}
    assert result["progress_receipts_sha256"] == {TASK: sha256_file(progress)}
    progress_artifacts = [
        row
        for row in result["artifacts"]
        if row.get("kind") == "expert_seed_progress"
    ]
    assert progress_artifacts == [
        {
            "path": str(progress),
            "sha256": sha256_file(progress),
            "kind": "expert_seed_progress",
        }
    ]
    manifest = read_json(result["seed_manifest"])
    assert manifest["progress_receipts"] == {TASK: str(progress)}
    assert manifest["progress_receipt_sha256"] == {
        TASK: sha256_file(progress)
    }


def test_supervisor_aggregates_and_revalidates_structural_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    benchmark = tmp_path / "benchmark"
    dataset = tmp_path / "dataset"
    dino = tmp_path / "dino"
    for path in (repo, benchmark, dataset, dino):
        path.mkdir()
    run_root = tmp_path / "run"
    settings = supervisor_stage.Settings(
        repo=repo,
        benchmark_repo=benchmark,
        dataset=dataset,
        run=run_root,
        dino_model=dino,
        python="python3",
        care_source_revision="a" * 40,
        modules=dict(supervisor_stage.DEFAULT_MODULES),
    )
    supervisor = supervisor_stage.Supervisor(settings)
    dependency = run_root / "stage_results" / "bcore_smoke_closed_loop.json"
    atomic_json(
        dependency,
        {
            "schema": RESULT_SCHEMA,
            "stage": "bcore_smoke_closed_loop",
            "status": "PASSED",
            "benchmark_adapter": "BiCoord",
            "config_sha256": supervisor.config_hash,
        },
    )

    overlay = (
        run_root
        / "artifacts"
        / "asset_contract"
        / "overlay"
        / "003_plate"
        / "model_data0.json"
    )
    overlay.parent.mkdir(parents=True)
    overlay.write_text("{}\n", encoding="utf-8")
    contact_sha256 = "b" * 64
    shovel_overlay = (
        run_root
        / "artifacts"
        / "asset_contract"
        / "overlay"
        / "082_smallshovel"
        / "model_data3.json"
    )
    shovel_overlay.parent.mkdir(parents=True)
    shovel_overlay.write_text("{}\n", encoding="utf-8")
    asset_receipt = run_root / "artifacts" / "asset_contract" / "asset_contract.json"
    atomic_json(
        asset_receipt,
        {
            "plate_overlay": {
                "overlay_metadata": str(overlay.resolve()),
                "target_contact_points_pose_sha256": contact_sha256,
            }
        },
    )
    atomic_json(
        supervisor.result_path("asset_contract"),
        {"asset_contract": str(asset_receipt.resolve())},
    )
    receipt_sha256 = sha256_file(asset_receipt)
    overlay_expectations = {
        "place_plate_and_cup": {
            "overlay": str(overlay.resolve()),
            "contact_points_pose_sha256": contact_sha256,
            "contact_points_pose_count": 4,
            "receipt": str(asset_receipt.resolve()),
            "receipt_sha256": receipt_sha256,
        },
        "sweep_block": {
            "overlay": str(shovel_overlay.resolve()),
            "contact_points_pose_sha256": (
                supervisor_stage.SHOVEL_CONTACT_POINTS_POSE_SHA256
            ),
            "contact_points_pose_count": 1,
            "receipt": str(asset_receipt.resolve()),
            "receipt_sha256": receipt_sha256,
        },
    }
    monkeypatch.setattr(
        supervisor,
        "_asset_runtime_expectations",
        lambda: overlay_expectations,
    )

    def make_env(_root: Path, task: str, seed: int) -> _OutcomeEnv:
        env = _OutcomeEnv(seed, "error" if seed == 100_000 else "success")
        if task == "place_plate_and_cup":
            env._bicoord_asset_overlay = {
                "task": task,
                "applied": True,
                "overlay": str(overlay.resolve()),
                "contact_points_pose_sha256": contact_sha256,
                "receipt": str(asset_receipt.resolve()),
                "receipt_sha256": receipt_sha256,
                "actors": {
                    name: {
                        "after_sha256": contact_sha256,
                        "contact_points_pose_count": 4,
                        "scale_preserved": True,
                        "changed_fields": ["contact_points_pose"],
                    }
                    for name in ("plate", "plate_2")
                },
                "copied_fields": ["contact_points_pose"],
                "task_source_modified": False,
            }
        elif task == "sweep_block":
            shovel_contact_sha256 = (
                supervisor_stage.SHOVEL_CONTACT_POINTS_POSE_SHA256
            )
            env._bicoord_asset_overlay = {
                "task": task,
                "applied": True,
                "overlay": str(shovel_overlay.resolve()),
                "contact_points_pose_sha256": shovel_contact_sha256,
                "receipt": str(asset_receipt.resolve()),
                "receipt_sha256": receipt_sha256,
                "actors": {
                    "shovel": {
                        "after_sha256": shovel_contact_sha256,
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
            }
        return env

    monkeypatch.setattr(seed_stage, "_make_env", make_env)
    worker_root = run_root / "worker_results" / "seed_discovery_smoke"
    for task in TASKS:
        run(
            argparse.Namespace(
                operation="smoke-discover",
                repo=repo,
                benchmark_repo=benchmark,
                dataset=dataset,
                run=run_root,
                dino_model=dino,
                result=worker_root / f"{task}.json",
                config_sha256=supervisor.config_hash,
                auto_resume=True,
                seed_bucket=0,
                max_attempts=2,
                smoke=False,
                task=task,
                episodes=1,
            )
        )

    class NoOpScheduler:
        def run_wave(self, _jobs: object) -> None:
            pass

    supervisor.scheduler = NoOpScheduler()  # type: ignore[assignment]
    monkeypatch.setattr(supervisor, "_base_command", lambda *_args, **_kwargs: [])
    spec = supervisor_stage.STAGES["seed_discovery_smoke"]
    candidate = run_root / "stage_results" / ".seed.candidate.json"
    supervisor._seed_task_queue_action(spec, candidate)

    aggregate = read_json(candidate)
    expected_types = {task: {"IndexError": 1} for task in TASKS}
    assert aggregate["structural_exception_type_counts"] == expected_types
    assert all(
        rows[0]["count"] == 1
        for rows in aggregate["structural_exception_counts"].values()
    )
    supervisor._validate_result(spec, candidate)

    missing = dict(aggregate)
    missing.pop("structural_exception_counts")
    missing_candidate = candidate.with_name(".seed.missing.candidate.json")
    atomic_json(missing_candidate, missing)
    with pytest.raises(
        supervisor_stage.InvalidArtifact,
        match="result differs from manifest at structural_exception_counts",
    ):
        supervisor._validate_result(spec, missing_candidate)

    worker_path = worker_root / f"{TASK}.json"
    worker = read_json(worker_path)
    worker["structural_exception_type_counts"] = {TASK: {}}
    atomic_json(worker_path, worker)
    with pytest.raises(
        supervisor_stage.InvalidArtifact,
        match="seed worker diagnostics differs at structural_exception_type_counts",
    ):
        supervisor._seed_task_queue_action(
            spec,
            candidate.with_name(".seed.tampered.candidate.json"),
        )
