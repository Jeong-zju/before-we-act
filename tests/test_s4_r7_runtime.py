from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import scripts.s4_r7_runtime as s4_r7_runtime
from scripts.s4_r7_runtime import (
    FORMAT_VERSION,
    initialize_run,
    render_monitor,
    update_shared_status,
    update_status,
)


ROOT = Path(__file__).resolve().parents[1]


def _initialize(tmp_path: Path) -> Path:
    p0 = tmp_path / "p0"
    p1 = tmp_path / "p1"
    p0.mkdir()
    p1.mkdir()
    root = tmp_path / "run"
    initialize_run(
        root,
        run_id="s4-r7-test",
        session="ssh_tmux",
        window_prefix="s4-r7-test",
        monitor_window="s4-r7-test-monitor",
        base_repo=tmp_path,
        parent_commit="a" * 40,
        worktrees=[f"P0={p0}", f"P1={p1}"],
    )
    return root


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_r7_monitor_reports_runtime_heartbeat_progress_and_derived_gates(
    tmp_path: Path,
) -> None:
    root = _initialize(tmp_path)
    update_shared_status(
        root,
        phase="complete",
        program="prepare_s4_r7_shared.sh",
        detail="shared datasets and ancestors ready",
        pid=100,
        child_pid=101,
        condition=None,
        task=None,
        episode=None,
        episodes_total=None,
        step=None,
        steps_total=None,
        micro_batch=None,
        gradient_accumulation=None,
        effective_batch=None,
        update=None,
        total_updates=None,
        team_windows_seen=None,
        agent_windows_seen=None,
        milestone=None,
        flow_unfreeze_state=None,
        loss=None,
        grad_norm=None,
        learning_rate=None,
        preflight="PASS",
        exit_code=0,
    )
    (root / "prepare.log").write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "future_cache_progress",
                    "worker": worker,
                    "gpu": str(worker),
                    "episode": 30 + worker,
                    "episodes": 375,
                    "task_id": "lift_barrier",
                    "episode_index": 58 + worker,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            for worker in (0, 1)
        )
        + "\n",
        encoding="utf-8",
    )
    for candidate, gpu in (("P0", 0), ("P1", 1)):
        update_status(
            root,
            candidate=candidate,
            phase="validating",
            program="evaluate_s4_r7_causal.py",
            detail="paired intervention Gate20",
            pid=200 + gpu,
            child_pid=300 + gpu,
            gpu_pid=400 + gpu,
            gpu_index=gpu,
            condition="world_evidence_gate=0",
            task="take_photo",
            episode=7,
            episodes_total=20,
            step=125,
            steps_total=1500,
            micro_batch=4,
            gradient_accumulation=3,
            effective_batch=12,
            update=15_000,
            total_updates=30_000,
            team_windows_seen=180_000,
            agent_windows_seen=576_000,
            milestone="15k",
            flow_unfreeze_state="unfrozen",
            loss=0.125,
            grad_norm=0.75,
            learning_rate="flow=2e-5,router=3e-4",
            preflight="PASS",
            exit_code=None,
        )
        candidate_root = root / "candidates" / candidate.lower()
        _write_json(
            candidate_root / "legacy_scaled_zero_shuffle_gate20.json",
            {
                "normal_macro": 0.45,
                "legacy_macro": 0.39,
                "world_evidence_gate_zero_macro": 0.41,
                "joint_shuffle_macro": 0.40,
                # The monitor must derive the gate and not trust this field.
                "passed": False,
            },
        )
        _write_json(
            candidate_root / "parameter_gradient_audit.json",
            {
                "normal_gradients_present": True,
                "forbidden_gradients_zero": True,
                "passed": False,
            },
        )
        _write_json(
            candidate_root / "source_shuffle_gate20.json",
            {"own_gap": 0.03, "peer_gap": 0.01, "shared_gap": -0.01},
        )
        _write_json(candidate_root / "module_exposure.json", {"complete": True})
        (candidate_root / "forced_evidence_errors.npz").touch()

    _write_json(
        root / "pair_exact.json",
        {"passed": True, "same_indices": True, "legacy_exact": True},
    )
    _write_json(
        root / "candidates/p1/router_utility_spearman.json",
        {"spearman": 0.22, "bootstrap_ci95_lower": 0.08, "passed": False},
    )
    _write_json(root / "acceptance.json", {"passed": True, "decision": "select_p1"})

    rendered = render_monitor(root)
    assert "WAM S4-R7 monitor" in rendered
    assert "Beijing time | current=" in rendered
    assert "Beijing ETA | P0-train=complete; P1-train=complete" in rendered
    assert "paired-train=" in rendered
    assert "producer every 20s; STALE strictly after 75s" in rendered
    assert "future cache workers | GPU0=30/375 task=lift_barrier" in rendered
    assert "GPU1=31/375 task=lift_barrier" in rendered
    assert "pid=200 child_pid=300 gpu_pid=400" in rendered
    assert "condition=world_evidence_gate=0 task=take_photo episode=7/20 step=125/1500" in rendered
    assert "micro/accum/effective=4/3/12" in rendered
    assert "update=15000/30000 (50.0%)" in rendered
    assert "agent_windows=576000/1152000 (50.0%)" in rendered
    assert "milestone=15k flow=unfrozen preflight=PASS" in rendered
    assert "loss=0.125 grad=0.75 lr=flow=2e-5,router=3e-4" in rendered
    assert "generic passed=true is insufficient" in rendered
    assert "pair structure | PASS 2/2" in rendered
    assert "P0 causal | PASS normal=0.45 legacy=0.39" in rendered
    assert "P1 utility calibration | PASS spearman=0.22 ci95_lower=0.08" in rendered
    assert "decision=select_p1 winner=?" in rendered
    assert "metrics above remain authoritative" in rendered


def test_r7_monitor_reports_live_beijing_training_and_full_run_eta(
    tmp_path: Path,
) -> None:
    root = _initialize(tmp_path)
    for candidate, update, rate in (("P0", 1_000, 0.95), ("P1", 1_200, 1.0)):
        progress = root / "candidates" / candidate.lower() / "train" / "progress.jsonl"
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(
            json.dumps(
                {
                    "event": "optimizer_step",
                    "update": update,
                    "updates": 30_000,
                    "updates_per_second": rate,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    rendered = render_monitor(root)
    assert "Beijing ETA | P0-train=" in rendered
    assert "(0.95 update/s); P1-train=" in rendered
    assert "(1 update/s)" in rendered
    assert "paired-train=" in rendered
    assert "normal≈" in rendered
    assert "core4≈" in rendered
    assert "full-R7≈" in rendered
    assert "validation=historical S3-R6 five-task Gate20 5h52m/condition" in rendered


def test_r7_monitor_uses_named_pair_checks_not_expected_false_observations(
    tmp_path: Path,
) -> None:
    root = _initialize(tmp_path)
    _write_json(
        root / "pair_exact.json",
        {
            "passed": True,
            "checks": {
                "same_dataset_index_sequence": True,
                "no_oom": True,
            },
            "preflight": {
                "P0": {"oom": False, "formal_budget_complete": False},
                "P1": {"oom": False, "formal_budget_complete": False},
                "required_fallback": None,
            },
        },
    )

    rendered = render_monitor(root)
    assert "pair structure | PASS 2/2" in rendered


def test_r7_monitor_exposes_complete_normal_results_before_final_report(
    tmp_path: Path,
) -> None:
    root = _initialize(tmp_path)
    tasks = (
        "lift_barrier",
        "long_pipeline_delivery",
        "take_photo",
        "three_robots_stack_cube",
        "camera_alignment",
    )
    summary: dict[str, object] = {
        "task_order": list(tasks),
    }
    for task in tasks:
        summary[task] = {
            "episodes": [
                {"seed": 900 + index, "success": index < 9}
                for index in range(20)
            ]
        }
    _write_json(
        root
        / "candidates/p0/validation/gate20/normal/gate_summary.json",
        summary,
    )

    rendered = render_monitor(root)
    assert "P0 validation priority | core=1/4 diagnostic=0/4" in rendered
    assert "P0 causal | pending normal=0.45" in rendered
    assert "P0 normal Gate20 by task | lift_barrier=9/20" in rendered


def test_r7_monitor_replaces_zero_gpu_pid_sentinel_with_live_process(
    tmp_path: Path, monkeypatch: object
) -> None:
    root = _initialize(tmp_path)
    update_status(
        root,
        candidate="P0",
        phase="training",
        program="train_s4_r7_world_utility.py",
        detail="optimizer active",
        pid=123,
        child_pid=124,
        gpu_pid=0,
        gpu_index=0,
        condition=None,
        task=None,
        episode=None,
        episodes_total=None,
        step=None,
        steps_total=None,
        micro_batch=4,
        gradient_accumulation=3,
        effective_batch=12,
        update=1,
        total_updates=30_000,
        team_windows_seen=12,
        agent_windows_seen=38,
        milestone=None,
        flow_unfreeze_state=None,
        loss=1.0,
        grad_norm=1.0,
        learning_rate="1e-4",
        preflight="PASS",
        exit_code=None,
    )
    monkeypatch.setattr(
        s4_r7_runtime, "_gpu_processes_by_index", lambda: {0: [456]}
    )

    rendered = render_monitor(root)
    assert "gpu_pid=456" in rendered


def test_r7_monitor_marks_nonterminal_heartbeat_stale_after_75_seconds(
    tmp_path: Path,
) -> None:
    root = _initialize(tmp_path)
    update_status(
        root,
        candidate="P0",
        phase="training",
        program="train_s4_r7_world_utility.py",
        detail="optimizer active",
        pid=123,
        child_pid=124,
        gpu_pid=125,
        gpu_index=0,
        condition=None,
        task=None,
        episode=None,
        episodes_total=None,
        step=None,
        steps_total=None,
        micro_batch=4,
        gradient_accumulation=3,
        effective_batch=12,
        update=10,
        total_updates=30_000,
        team_windows_seen=120,
        agent_windows_seen=384,
        milestone=None,
        flow_unfreeze_state=None,
        loss=1.0,
        grad_norm=1.0,
        learning_rate="1e-4",
        preflight="PASS",
        exit_code=None,
    )
    heartbeat = root / "candidates/p0/heartbeat.json"
    value = json.loads(heartbeat.read_text())
    value["updated_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=76)
    ).isoformat()
    heartbeat.write_text(json.dumps(value))

    rendered = render_monitor(root)
    assert "phase=training heartbeat=STALE:1m16s" in rendered
    assert "STALE | last_program=train_s4_r7_world_utility.py" in rendered
    assert "gpu_pid=125" in rendered


def test_r7_shell_contracts_are_permanent_tmux_scoped_and_cli_driven() -> None:
    launcher_path = ROOT / "scripts/launch_s4_r7_2gpu_tmux.sh"
    stopper_path = ROOT / "scripts/stop_s4_r7_2gpu_tmux.sh"
    for path in (launcher_path, stopper_path):
        subprocess.run(["bash", "-n", str(path)], check=True)
    launcher = launcher_path.read_text(encoding="utf-8")
    stopper = stopper_path.read_text(encoding="utf-8")

    for source in (launcher, stopper):
        assert "set -e" not in source
        assert "set -u" not in source
        assert "set -o pipefail" not in source
        assert "new-session" not in source
        assert "attach-session" not in source
        assert "kill-session" not in source
    assert 'SESSION="ssh_tmux"' in launcher
    assert "remain-on-exit on" in launcher
    assert 'CANDIDATE_WINDOWS=("${WINDOW_PREFIX}-p0" "${WINDOW_PREFIX}-p1")' in launcher
    assert "status --porcelain)" in launcher
    assert "--untracked-files=no" not in launcher
    assert "bash \"${CANDIDATE_REL}\"" in launcher
    for option in (
        "--candidate",
        "--run-root",
        "--ready-file",
        "--failed-file",
        "--config",
        "--gpu-index",
        "--heartbeat-seconds",
    ):
        assert option in launcher
    assert 'grep -Fzqx "S4_R7_RUN_ROOT=${RUN_ROOT}"' in stopper
    assert "TARGET_WINDOWS=(" in stopper
    assert "Preserved the permanent tmux session" in stopper


def test_launcher_dry_run_rejects_an_untracked_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "launch_s4_r7_2gpu_tmux.sh",
        "s4_r7_runtime.py",
        "s0_hf_token_fifo.sh",
    ):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    for name in ("prepare_s4_r7_shared.sh", "run_s4_r7_candidate.sh"):
        path = scripts / name
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    subprocess.run(["git", "init", "-b", "feat/model-improvements"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "scripts"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(tmp_path / "unused.git")],
        cwd=repo,
        check=True,
    )
    (repo / "untracked.override").write_text("must be rejected", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tmux = fake_bin / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == has-session ]]; then exit 0; fi\n"
        "if [[ \"$1\" == list-windows ]]; then exit 0; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == -L ]]; then printf 'GPU 0: fake\\nGPU 1: fake\\n'; fi\n",
        encoding="utf-8",
    )
    nvidia.chmod(0o755)
    env = os.environ.copy()
    env.pop("TMUX", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(scripts / "launch_s4_r7_2gpu_tmux.sh"), "--run-id", "dirty", "--dry-run"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "untracked.override" in result.stderr
    assert "tracked or untracked files are both rejected" in result.stderr
    assert not (repo / "outputs/s4_r7_runs/dirty").exists()


def test_stop_dry_run_resolves_only_four_manifest_windows(tmp_path: Path) -> None:
    root = _initialize(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tmux = fake_bin / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == has-session ]]; then exit 0; fi\n"
        "if [[ \"$1\" == list-windows ]]; then\n"
        "  printf 's4-r7-test-prepare|@1\\ns4-r7-test-p0|@2\\n'\n"
        "  printf 's4-r7-test-p1|@3\\ns4-r7-test-monitor|@4\\nother|@9\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    nvidia.chmod(0o755)
    env = os.environ.copy()
    env.pop("TMUX", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stop_s4_r7_2gpu_tmux.sh"),
            "--run-id",
            "s4-r7-test",
            "--run-root",
            str(root),
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    for suffix in ("prepare", "p0", "p1", "monitor"):
        assert f"window: s4-r7-test-{suffix}" in result.stdout
    assert "window: other" not in result.stdout
    assert "no process was signaled and no tmux window was closed" in result.stdout


def test_runtime_manifest_records_exact_stop_identity(tmp_path: Path) -> None:
    root = _initialize(tmp_path)
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["round_id"] == "s4-r7"
    assert manifest["run_root"] == str(root.resolve())
    assert manifest["tmux_session"] == "ssh_tmux"
    assert manifest["tmux_windows"] == {
        "prepare": "s4-r7-test-prepare",
        "P0": "s4-r7-test-p0",
        "P1": "s4-r7-test-p1",
        "monitor": "s4-r7-test-monitor",
    }
