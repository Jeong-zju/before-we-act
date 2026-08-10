from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _runtime():
    path = ROOT / "scripts/before_we_act/r11_runtime.py"
    spec = importlib.util.spec_from_file_location("r11_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pid_identity_requires_the_same_proc_start_time():
    runtime = _runtime()
    ticks = runtime.process_start_ticks(os.getpid())
    assert runtime.pid_identity_alive(os.getpid(), ticks)
    assert not runtime.pid_identity_alive(os.getpid(), ticks + 1)
    assert not runtime.pid_identity_alive(0, 0)


def test_watchdog_refuses_a_reused_or_missing_process(tmp_path):
    runtime = _runtime()
    args = SimpleNamespace(
        run_root=tmp_path,
        candidate="A",
        stage="F0",
        pid=os.getpid(),
        pid_start_time_ticks=runtime.process_start_ticks(os.getpid()) + 1,
    )
    with pytest.raises(ProcessLookupError):
        runtime.watchdog(args)
    assert not (tmp_path / "A/status/pipeline_heartbeat.json").exists()


def test_terminal_pass_fail_is_authoritative_only_from_complete_acceptance():
    runtime = _runtime()
    state, alerts = runtime._authoritative_state("PASSED", {})
    assert state == "UNKNOWN"
    assert alerts == ["TERMINAL_WITHOUT_COMPLETE_ACCEPTANCE"]
    acceptance = {
        "complete": True,
        "passed": False,
        "checks": [{"id": str(index), "passed": index != 3} for index in range(7)],
    }
    assert runtime._authoritative_state("TRAINING", acceptance)[0] == "FAILED"


def test_status_schema_carries_exact_branch_commit_and_process_identity(tmp_path):
    runtime = _runtime()
    args = SimpleNamespace(
        run_root=tmp_path,
        candidate="D",
        state="PREFLIGHT",
        stage="F1",
        program="run_r11_candidate.sh",
        detail="full path",
        branch="feat/r11-lawam-latent-subgoal",
        commit="1" * 40,
        upstream_commit="2" * 40,
        pid=os.getpid(),
        pid_start_time_ticks=runtime.process_start_ticks(os.getpid()),
        child_pid=0,
        child_pid_start_time_ticks=0,
        log="/tmp/not-a-real-log",
        exit_code=None,
    )
    runtime.update_status(args)
    payload = json.loads((tmp_path / "D/status/runtime.json").read_text())
    assert payload["branch"] == args.branch
    assert payload["commit"] == args.commit
    assert payload["pid_start_time_ticks"] == args.pid_start_time_ticks
