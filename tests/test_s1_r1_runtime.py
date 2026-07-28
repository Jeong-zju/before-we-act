from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from scripts.s1_r1_runtime import (
    collect_candidate,
    initialize_run,
    render_monitor,
    update_status,
)


ROOT = Path(__file__).resolve().parents[1]


def test_s1_r1_runtime_tracks_both_candidates_and_shared_data(tmp_path: Path):
    run_root = tmp_path / "run"
    initialize_run(
        run_root,
        run_id="fixture",
        session="permanent",
        base_repo=ROOT,
        worktrees=[f"F0={tmp_path / 'f0'}", f"F1={tmp_path / 'f1'}"],
        window_prefix="fixture",
        monitor_window="fixture-monitor",
    )
    update_status(
        run_root,
        candidate="F1",
        phase="training",
        detail="flow matching",
        gpu_index=1,
        total_updates=80000,
        exit_code=None,
    )
    progress = run_root / "candidates/f1/train/progress.jsonl"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps({"update": 4000, "updates": 80000, "loss": 0.25}) + "\n",
        encoding="utf-8",
    )
    episodes = (
        run_root
        / "candidates/f1/validation/gate_fixture/lift_barrier/"
        "rollout_episodes.jsonl"
    )
    episodes.parent.mkdir(parents=True)
    episodes.write_text(
        "\n".join(
            (
                json.dumps({"success": True}),
                json.dumps({"success": False}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    value = collect_candidate(run_root, "F1")
    rendered = render_monitor(run_root)

    assert manifest["round_id"] == "s1-r1"
    assert manifest["worktrees"] == {
        "F0": str((tmp_path / "f0").resolve()),
        "F1": str((tmp_path / "f1").resolve()),
    }
    assert value["gpu"] == "1"
    assert "4000/80000" in value["training"]
    assert "loss=0.25" in value["training"]
    assert "lift=1/2" in value["validation"]
    assert "WAM S1-R1 monitor" in rendered
    assert "shared data:" in rendered
    assert "permanent" in rendered
    assert "tmux attach" not in rendered


def test_s1_r1_launcher_dry_run_assigns_two_gpus_without_mutation(
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/launch_s1_r1_2gpu_tmux.sh"),
            "--run-id",
            "fixture",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "S1_R1_RUN_ROOT": str(run_root),
        },
    )

    assert "F0 GPU0" in result.stdout
    assert "F1 GPU1" in result.stdout
    assert "s1/r1-f0-legacy" in result.stdout
    assert "s1/r1-f1-flow-cold" in result.stdout
    assert "one base copy" in result.stdout
    assert "never kills it" in result.stdout
    assert "mode-0600 FIFO" in result.stdout
    assert not run_root.exists()


def _fake_stop_commands(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "tmux.calls"
    tmux = fake_bin / "tmux"
    tmux.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >>"${TMUX_CALLS}"
case "$1" in
  display-message)
    if [[ "$*" == *"#S"* ]]; then printf 'permanent\\n'; else printf 'bash\\n'; fi
    ;;
  list-sessions)
    printf 'permanent\\n'
    ;;
  list-windows)
    state="${TMUX_STATE}"
    if [[ ! -f "${state}" ]]; then
      cat >"${state}" <<'EOF'
bash|@0
s1-r1-prepare|@1
s1-r1-f0|@2
s1-r1-f1|@3
s1-r1-monitor|@4
unrelated|@5
EOF
    fi
    cat "${state}"
    ;;
  send-keys)
    ;;
  kill-window)
    target="${3}"
    awk -F'|' -v target="${target}" '$2 != target' "${TMUX_STATE}" \
      >"${TMUX_STATE}.next"
    mv "${TMUX_STATE}.next" "${TMUX_STATE}"
    ;;
  *)
    printf >&2 'unexpected fake tmux call: %s\\n' "$*"
    exit 9
    ;;
esac
""",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    return fake_bin, calls


def _stop_environment(
    tmp_path: Path, fake_bin: Path, calls: Path
) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "TMUX": "/tmp/fake,1,0",
        "TMUX_CALLS": str(calls),
        "TMUX_STATE": str(tmp_path / "tmux.state"),
    }


def test_s1_r1_stop_closes_only_four_run_windows(tmp_path: Path):
    fake_bin, calls = _fake_stop_commands(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "s1-r1",
                "round_id": "s1-r1",
                "tmux_session": "permanent",
                "tmux_window_prefix": "s1-r1",
                "tmux_monitor_window": "s1-r1-monitor",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stop_s1_r1_2gpu_tmux.sh"),
            "--run-id",
            "s1-r1",
            "--run-root",
            str(run_root),
            "--grace-seconds",
            "0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=_stop_environment(tmp_path, fake_bin, calls),
    )

    log = calls.read_text(encoding="utf-8")
    assert log.count("send-keys") == 4
    assert log.count("kill-window") == 4
    assert "kill-session" not in log
    assert "unrelated" not in result.stdout
    assert "permanent tmux session" in result.stdout
    assert "shared dataset" in result.stdout
