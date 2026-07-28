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
    )
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
    assert "no worktrees, files, tmux sessions or GPU jobs were changed" in result.stdout
    assert not (tmp_path / "run").exists()
