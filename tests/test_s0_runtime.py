from __future__ import annotations

import json
from pathlib import Path
import subprocess

import yaml

from scripts.s0_runtime import (
    collect_candidate,
    initialize_run,
    read_latest_jsonl,
    render_monitor,
    update_status,
)


ROOT = Path(__file__).resolve().parents[1]


def test_s0_runtime_tracks_training_and_paired_validation(tmp_path):
    run_root = tmp_path / "run"
    worktrees = [
        f"{candidate}={tmp_path / candidate.lower()}"
        for candidate in ("B0", "B1", "B2", "B3")
    ]
    initialize_run(
        run_root,
        run_id="fixture",
        session="wam-s0-fixture",
        base_repo=ROOT,
        worktrees=worktrees,
        window_prefix="fixture",
        monitor_window="fixture-monitor",
    )
    manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tmux_mode"] == "shared_existing_session"
    assert manifest["tmux_window_prefix"] == "fixture"
    assert manifest["tmux_monitor_window"] == "fixture-monitor"
    update_status(
        run_root,
        candidate="B0",
        phase="training",
        detail="fixture",
        gpu_index=0,
        total_updates=80000,
        exit_code=None,
    )
    candidate = run_root / "candidates/b0"
    progress = candidate / "train/progress.jsonl"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(
        json.dumps({"update": 100, "updates": 80000, "loss": 1.25}) + "\n",
        encoding="utf-8",
    )
    validation = candidate / "validation/gate_fixture/lift_barrier"
    validation.mkdir(parents=True)
    (validation / "rollout_episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"seed": 900, "success": True}),
                json.dumps({"seed": 901, "success": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    value = collect_candidate(run_root, "B0")

    assert value["gpu"] == "0"
    assert value["phase"] == "training"
    assert "100/80000" in value["training"]
    assert "loss=1.25" in value["training"]
    assert "lift=1/2" in value["validation"]
    assert read_latest_jsonl(progress)["update"] == 100
    rendered = render_monitor(run_root)
    assert "wam-s0-fixture" in rendered
    assert "fixture-monitor" in rendered
    assert "tmux select-window" in rendered
    assert "tmux attach" not in rendered
    assert "B0" in rendered


def test_s0_round_manifest_freezes_shared_data_and_uniform_loss():
    path = ROOT / "experiments/wam_flow/s0/round_manifest.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert value["source_parent_commit"] == (
        "07fd136fadd8831d6c111a7be5e9908056b6743d"
    )
    assert value["training"]["frozen_model_start_commit"] == (
        "a1e8814dc46bdb5de3b28f8001cdfc563d532f6e"
    )
    assert value["training"]["optimizer_updates"] == 80000
    assert value["training"]["seed"] == 101
    assert value["training"]["active_agent_loss_weighting"] is False
    assert value["candidates"] == ["B0", "B1", "B2", "B3"]
    assert len(value["data"]["manifests"]) == 2


def test_s0_launcher_dry_run_has_four_gpu_assignments(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/launch_s0_4gpu_tmux.sh"),
            "--run-id",
            "fixture",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "S0_RUN_ROOT": str(tmp_path / "run")},
    )

    assert "B0 GPU0" in result.stdout
    assert "B1 GPU1" in result.stdout
    assert "B2 GPU2" in result.stdout
    assert "B3 GPU3" in result.stdout
    assert "HF token: will be requested with hidden interactive input" in result.stdout
    assert "Branches: will be fetched from origin" in result.stdout
    assert "Shared dataset:" in result.stdout
    assert "S0_HF_TOKEN_FIFO=" in result.stdout
    assert "current or only existing permanent session" in result.stdout
    assert "window=fixture-b0" in result.stdout
    assert "no worktrees, files, tmux windows or GPU jobs were changed" in result.stdout
    assert not (tmp_path / "run").exists()


def test_s0_launcher_prompts_for_secret_without_exporting_it():
    launcher = (ROOT / "scripts/launch_s0_4gpu_tmux.sh").read_text(
        encoding="utf-8"
    )
    prepare = (ROOT / "scripts/prepare_s0_shared.sh").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_lpd_single_5090.sh").read_text(
        encoding="utf-8"
    )
    retry = (ROOT / "scripts/hf_download_retry.sh").read_text(encoding="utf-8")
    dino = (ROOT / "scripts/prepare_dinov3_encoder.py").read_text(
        encoding="utf-8"
    )
    runbook = (
        ROOT / "docs/runbooks/20260728_S0_4GPU_TMUX_ZH.md"
    ).read_text(encoding="utf-8")

    assert "read -r -s HF_TOKEN_INPUT" in launcher
    assert "mkfifo" in launcher
    assert "chmod 600" in launcher
    assert "env \\\n    -u HF_TOKEN" in launcher
    assert "export HF_TOKEN" not in launcher
    assert "export HF_TOKEN" not in prepare
    assert "export HF_TOKEN='hf_...'" not in runbook
    assert "HfApi().whoami(token=os.environ[\"HF_TOKEN\"])" in runner
    assert runner.count("hf_download_with_retry") == 4
    assert '"DINOv3 weights"' in runner
    assert '"RoboFactory assets over HTTP"' in runner
    assert 'HF_TOKEN="${HF_TOKEN}"' in retry
    assert 'HF_HUB_DISABLE_XET="${disable_xet}"' in retry
    assert "HF_DOWNLOAD_ATTEMPTS:-5" in retry
    assert "sleep" in retry
    assert "token=True" not in dino
    assert "token=token" in dino
    assert "config --get-all remote.origin.fetch" in launcher
    assert "config --add remote.origin.fetch" in launcher
    assert "git -C \"${FE_ROOT}\" fetch --no-tags origin" in launcher
    assert "tmux new-session" not in launcher
    assert "tmux attach" not in launcher
    assert "tmux attach -t" not in runbook
    assert "tmux new-window" in launcher
    assert "resolve_existing_tmux_session" in launcher
    assert "Resuming partially created S0 windows" in launcher
    assert "Preserving existing candidate window" in launcher
    assert 'unlink "${HF_TOKEN_FIFO}" 2>/dev/null || true' in launcher
    assert 'unlink "${S0_HF_TOKEN_FIFO}" 2>/dev/null || true' in prepare


def test_hf_retry_uses_exact_token_disables_xet_and_recovers(
    tmp_path: Path,
) -> None:
    command = tmp_path / "fake-hf-command"
    calls = tmp_path / "calls"
    command.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s|%s\\n' "${HF_TOKEN}" "${HF_HUB_DISABLE_XET}" >>"${CALL_LOG}"
count="$(wc -l <"${CALL_LOG}")"
if (( count < 3 )); then
  exit 29
fi
""",
        encoding="utf-8",
    )
    command.chmod(0o755)
    token = "hf_unit_test_secret"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; hf_with_retry "fixture asset" 1 "$2"',
            "bash",
            str(ROOT / "scripts/hf_download_retry.sh"),
            str(command),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HF_TOKEN": token,
            "HF_DOWNLOAD_ATTEMPTS": "4",
            "HF_DOWNLOAD_INITIAL_BACKOFF_SECONDS": "0",
            "CALL_LOG": str(calls),
        },
    )

    assert calls.read_text(encoding="utf-8").splitlines() == [
        f"{token}|1",
        f"{token}|1",
        f"{token}|1",
    ]
    assert "attempt 1/4" in result.stdout
    assert "attempt 3/4" in result.stdout
    assert result.stderr.count("retrying in 0 seconds") == 2
    assert token not in result.stdout
    assert token not in result.stderr
