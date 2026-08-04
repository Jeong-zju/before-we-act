from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER = ROOT / "scripts/before_we_act/download_r10_hf_assets.sh"
AUDITOR = ROOT / "scripts/before_we_act/audit_r10_hdf5_assets.py"


def test_hf_asset_dry_run_uses_s0_contract_and_fixed_revisions(tmp_path):
    fake_hf = tmp_path / "hf"
    capture = tmp_path / "hf-args.txt"
    fake_hf.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$BWA_HF_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_hf.chmod(0o755)
    env = dict(os.environ)
    env["BWA_HF_CAPTURE"] = str(capture)
    run_root = tmp_path / "run"
    result = subprocess.run(
        [
            "bash", str(DOWNLOADER),
            "--run-root", str(run_root),
            "--data-root", str(tmp_path / "datasets"),
            "--hf-home", str(tmp_path / "hf-home"),
            "--hf-cli", str(fake_hf),
            "--anonymous", "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert all(line.startswith("download zeno-ai/") for line in lines)
    assert all("--repo-type dataset" in line for line in lines)
    assert all("--revision " in line and "--local-dir " in line for line in lines)
    assert all("--dry-run" in line for line in lines)
    assert all("--max-workers" not in line for line in lines)
    assert all("snapshot_download" not in line for line in lines)
    assert "6ab620091677e69370412f08cd7adecacc28c146" in lines[0]
    assert "fee628311ff52a3ae0ddfddf82379c63d28f7533" in lines[1]
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "STOPPED"
    assert state["download_contract"]["xet"] == "enabled"
    assert state["download_contract"]["workers"] == "CLI default (8)"
    assert (run_root / "heartbeat").exists()


def test_hf_asset_downloader_requires_fifo_without_anonymous(tmp_path):
    result = subprocess.run(
        [
            "bash", str(DOWNLOADER),
            "--run-root", str(tmp_path / "run"),
            "--data-root", str(tmp_path / "datasets"),
            "--hf-home", str(tmp_path / "hf-home"),
            "--hf-cli", "/bin/true",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 3
    assert "protected token FIFO" in result.stderr


def test_hdf5_asset_auditor_reads_both_endpoints_and_fails_closed(tmp_path):
    task_root = tmp_path / "data/lift_barrier/hdf5"
    task_root.mkdir(parents=True)
    episode = task_root / "episode_000000.hdf5"
    with h5py.File(episode, "w") as handle:
        data = handle.create_group("data")
        action = data.create_group("action/agents/panda_0")
        action.create_dataset("commanded", data=np.zeros((2, 8), dtype=np.float32))
        observation = data.create_group("observation")
        agent = observation.create_group("agents/panda_0")
        agent.create_dataset("qpos", data=np.zeros((2, 9), dtype=np.float32))
        images = observation.create_group("images")
        images.create_dataset("global", data=np.zeros((2, 480, 640, 3), dtype=np.uint8))
        images.create_dataset("agent_0", data=np.zeros((2, 480, 640, 3), dtype=np.uint8))
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable, str(AUDITOR), "--data-root", str(tmp_path / "data"),
            "--expected-files", "1", "--output", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    result = subprocess.run(
        [
            sys.executable, str(AUDITOR), "--data-root", str(tmp_path / "data"),
            "--expected-files", "2", "--output", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
