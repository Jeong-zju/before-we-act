from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/before_we_act/sync_s10_dataset_tail.sh"


def test_tail_sync_dry_run_is_bounded_parallel_and_stateful(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "rsync-args.txt"
    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$BWA_RSYNC_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_rsync.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["BWA_RSYNC_CAPTURE"] = str(capture)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--source-host",
            "root@example.test",
            "--source-port",
            "45132",
            "--dest-root",
            str(tmp_path / "workspace"),
            "--start",
            "125",
            "--end",
            "126",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert all("--dry-run" in line for line in lines)
    assert all("episode_000125.hdf5" in line for line in lines)
    assert all("episode_000126.hdf5" in line for line in lines)
    assert all("episode_000124.hdf5" not in line for line in lines)
    state_path = (
        tmp_path
        / "workspace/bwa_runs/shared/s10_asset_tail_125_126/state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "STOPPED"
    assert state["detail"] == "dry_run_complete"
    assert state["start_episode"] == 125
    assert state["end_episode"] == 126
    assert (state_path.parent / "heartbeat").exists()


def test_tail_sync_rejects_out_of_range_or_reversed_shards(tmp_path):
    for start, end in ((150, 150), (10, 9)):
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--source-host",
                "root@example.test",
                "--source-port",
                "45132",
                "--dest-root",
                str(tmp_path / f"workspace-{start}-{end}"),
                "--start",
                str(start),
                "--end",
                str(end),
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
