from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.bicoord_care import supervisor as sup


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
