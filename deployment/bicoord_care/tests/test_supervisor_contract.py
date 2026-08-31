from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.bicoord_care import supervisor as sup
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
        return FakeProcess()

    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sup, "_utc_now", lambda: "now")
    active = supervisor.scheduler._spawn(
        "cwd-test", ["python", "-c", "pass"], (0,), tmp_path / "run.log"
    )
    assert active.process.pid == 4242
    assert captured["cwd"] == settings.benchmark_repo
