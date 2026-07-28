from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
P1 = (
    ROOT
    / "docs/plans/"
    "20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_ZH.md"
)


def _fake_commands(tmp_path: Path) -> tuple[Path, Path]:
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
    if [[ "$*" == *"#S"* ]]; then printf 'ssh_tmux\\n'; else printf 'bash\\n'; fi
    ;;
  list-sessions)
    printf 'ssh_tmux\\n'
    ;;
  list-windows)
    state="${TMUX_STATE}"
    if [[ ! -f "${state}" ]]; then
      cat >"${state}" <<'EOF'
bash|@0
s0-round1-prepare|@1
s0-round1-b0|@2
s0-round1-b1|@3
s0-round1-b2|@4
s0-round1-b3|@5
s0-round1-monitor|@6
unrelated|@7
EOF
    fi
    cat "${state}"
    ;;
  send-keys)
    if [[ "${TMUX_FAIL_SEND_KEYS:-0}" == "1" ]]; then exit 8; fi
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


def _manifest(run_root: Path) -> None:
    run_root.mkdir(parents=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "s0-round1",
                "tmux_session": "ssh_tmux",
                "tmux_window_prefix": "s0-round1",
                "tmux_monitor_window": "s0-round1-monitor",
            }
        ),
        encoding="utf-8",
    )


def _environment(tmp_path: Path, fake_bin: Path, calls: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "TMUX": "/tmp/fake,1,0",
        "TMUX_CALLS": str(calls),
        "TMUX_STATE": str(tmp_path / "tmux.state"),
    }


def test_stop_s0_dry_run_is_read_only_and_scoped(tmp_path: Path) -> None:
    fake_bin, calls = _fake_commands(tmp_path)
    run_root = tmp_path / "run"
    _manifest(run_root)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stop_s0_4gpu_tmux.sh"),
            "--run-id",
            "s0-round1",
            "--run-root",
            str(run_root),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(tmp_path, fake_bin, calls),
    )

    log = calls.read_text(encoding="utf-8")
    assert "send-keys" not in log
    assert "kill-window" not in log
    assert "no process was signaled" in result.stdout
    assert "unrelated" not in result.stdout


def test_stop_s0_terminates_only_tagged_run_and_closes_six_windows(
    tmp_path: Path,
) -> None:
    fake_bin, calls = _fake_commands(tmp_path)
    run_root = tmp_path / "run"
    _manifest(run_root)
    target_environment = {
        **os.environ,
        "S0_RUN_ROOT": str(run_root.resolve()),
    }
    target = subprocess.Popen(
        ["sleep", "60"],
        env=target_environment,
        start_new_session=True,
    )
    unrelated = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/stop_s0_4gpu_tmux.sh"),
                "--run-id",
                "s0-round1",
                "--run-root",
                str(run_root),
                "--grace-seconds",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=_environment(tmp_path, fake_bin, calls),
        )
        target.wait(timeout=5)

        assert target.returncode is not None
        assert unrelated.poll() is None
        log = calls.read_text(encoding="utf-8")
        assert log.count("send-keys") == 6
        assert log.count("kill-window") == 6
        assert "kill-session" not in log
        assert "unrelated" not in log
        assert "six windows are closed" in result.stdout
        assert "permanent tmux session" in result.stdout
    finally:
        if target.poll() is None:
            target.terminate()
            target.wait(timeout=5)
        if unrelated.poll() is None:
            unrelated.terminate()
            unrelated.wait(timeout=5)


def test_stop_s0_tolerates_dead_or_disappearing_panes(tmp_path: Path) -> None:
    fake_bin, calls = _fake_commands(tmp_path)
    run_root = tmp_path / "run"
    _manifest(run_root)
    environment = _environment(tmp_path, fake_bin, calls)
    environment["TMUX_FAIL_SEND_KEYS"] = "1"

    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/stop_s0_4gpu_tmux.sh"),
            "--run-id",
            "s0-round1",
            "--run-root",
            str(run_root),
            "--grace-seconds",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )

    log = calls.read_text(encoding="utf-8")
    assert log.count("send-keys") == 6
    assert log.count("kill-window") == 6


def test_p1_documents_complete_zero_start_and_scoped_stop_commands() -> None:
    document = P1.read_text(encoding="utf-8")

    assert "Vast.ai 四卡从零一键部署与运行" in document
    assert "git clone \\" in document
    assert "--single-branch" in document
    assert "launch_s0_4gpu_tmux.sh \\" in document
    assert "--run-id s0-round1" in document
    assert "当前 S0 run 一键终止与窗口关闭" in document
    assert "stop_s0_4gpu_tmux.sh \\" in document
    assert "--dry-run" in document
    assert "tmux kill-session" in document
    assert "不会删除数据集" in document
