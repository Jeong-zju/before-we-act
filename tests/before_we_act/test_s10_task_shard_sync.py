from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/before_we_act/sync_s10_task_shard.sh"


def test_task_shard_dry_run_is_exact_and_stateful(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "rsync-args.txt"
    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >\"$BWA_RSYNC_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_rsync.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["BWA_RSYNC_CAPTURE"] = str(capture)
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "--source-host", "root@example.test",
            "--source-port", "45132",
            "--dest-root", str(workspace),
            "--task", "long_pipeline_delivery",
            "--start", "100",
            "--end", "101",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8")
    assert "long_pipeline_delivery" in arguments
    assert "episode_000100.hdf5" in arguments
    assert "episode_000101.hdf5" in arguments
    assert "episode_000099.hdf5" not in arguments
    state_path = workspace / (
        "bwa_runs/shared/"
        "s10_asset_shard_long_pipeline_delivery_100_101/state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "STOPPED"
    assert state["detail"] == "dry_run_complete"
    assert state["task"] == "long_pipeline_delivery"
    assert (state_path.parent / "heartbeat").exists()


def test_task_shard_rejects_unknown_task_and_bad_range(tmp_path):
    for task, first, last in (
        ("unknown", 100, 101),
        ("take_photo", 150, 150),
        ("take_photo", 101, 100),
    ):
        result = subprocess.run(
            [
                "bash", str(SCRIPT),
                "--source-host", "root@example.test",
                "--source-port", "45132",
                "--dest-root", str(tmp_path / f"workspace-{task}-{first}-{last}"),
                "--task", task,
                "--start", str(first),
                "--end", str(last),
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
