from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts/prepare_s4_r8_shared.sh"
S0_ASSETS = ROOT / "scripts/prepare_s4_r8_assets_from_s0.sh"


def test_s4_prepare_bridge_is_valid_bash_without_errexit_bundle() -> None:
    result = subprocess.run(
        ["bash", "-n", str(PREPARE)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    source = PREPARE.read_text(encoding="utf-8")
    assert re.search(r"^\s*set\s+-", source, flags=re.MULTILINE) is None
    assert "set -euo pipefail" not in source


def test_s4_asset_bootstrap_keeps_s0_download_and_secret_contract() -> None:
    result = subprocess.run(
        ["bash", "-n", str(S0_ASSETS)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    source = S0_ASSETS.read_text(encoding="utf-8")
    assert 'chmod 600' not in source  # launcher creates the mode-0600 FIFO
    assert 'IFS= read -r HF_TOKEN_INPUT <"${S4_R8_HF_TOKEN_FIFO}"' in source
    assert 'HF_TOKEN="${HF_TOKEN_INPUT}" hf_download_with_retry' in source
    assert '"${slug} training dataset" 0 "${repo}"' in source
    assert "--max-workers" not in source.split("status dinov3", maxsplit=1)[0]
    assert "prepare_dinov3_encoder.py" in source
    assert "run_lpd_single_5090.sh prepare" in source
    assert "snapshot_download" not in source


def test_s4_prepare_passes_complete_asset_fifo_environment_without_token_leak(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    fifo = run_root / ".hf_token.fifo"
    os.mkfifo(fifo, mode=0o600)
    capture = tmp_path / "bridge_environment.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        "for name in S4_R8_RUN_ROOT S4_R8_HF_TOKEN_FIFO UV_CACHE_DIR "
        "UV_PROJECT_ENVIRONMENT S4_R8_ROBOFACTORY_ROOT S4_R8_RF_PYTHON; do\n"
        '  if printenv "${name}" >/dev/null 2>&1; then\n'
        "    printf '%s=set\\n' \"${name}\"\n"
        "  else\n"
        "    printf '%s=missing\\n' \"${name}\"\n"
        "  fi\n"
        "done\n"
        'if [ "${S4_R8_RUN_ROOT:-}" = "${S4_TEST_EXPECTED_ROOT:-}" ]; then\n'
        "  printf 'run_root_match=yes\\n'\n"
        "else\n"
        "  printf 'run_root_match=no\\n'\n"
        "fi\n"
        "if printenv HF_TOKEN >/dev/null 2>&1; then\n"
        "  printf 'HF_TOKEN=present\\n'\n"
        "else\n"
        "  printf 'HF_TOKEN=absent\\n'\n"
        "fi\n"
        "if printenv HUGGING_FACE_HUB_TOKEN >/dev/null 2>&1; then\n"
        "  printf 'HUGGING_FACE_HUB_TOKEN=present\\n'\n"
        "else\n"
        "  printf 'HUGGING_FACE_HUB_TOKEN=absent\\n'\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)
    fake_sleep = fake_bin / "sleep"
    fake_sleep.write_text("#!/bin/sh\nexec /bin/sleep 0.05\n", encoding="utf-8")
    fake_sleep.chmod(0o755)

    sentinel = "synthetic-secret-must-never-be-logged"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "S4_R8_RUN_ROOT": str(run_root),
            "S4_R8_READY_FILE": str(run_root / "shared.ready"),
            "S4_R8_FAILED_FILE": str(run_root / "shared.failed"),
            "S4_R8_USE_S0_PREP": "1",
            "S4_R8_HF_TOKEN_FIFO": str(fifo),
            "S4_R8_ROBOFACTORY_ROOT": str(tmp_path / "RoboFactory"),
            "S4_R8_RF_PYTHON": str(tmp_path / "RoboFactory/.venv/bin/python"),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "UV_PROJECT_ENVIRONMENT": str(tmp_path / "uv-env"),
            "S4_TEST_EXPECTED_ROOT": str(run_root),
            "HF_TOKEN": sentinel,
            "HUGGING_FACE_HUB_TOKEN": sentinel,
        }
    )
    # The fake bridge writes only presence bits, never environment values.
    environment["S4_TEST_CAPTURE"] = str(capture)
    fake_bash.write_text(
        fake_bash.read_text(encoding="utf-8").replace(
            "#!/bin/sh\n", '#!/bin/sh\nexec >"${S4_TEST_CAPTURE}"\n'
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["/bin/bash", str(PREPARE)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0  # The fake bridge does not prepare parent artifacts.
    observed = capture.read_text(encoding="utf-8")
    for name in (
        "S4_R8_RUN_ROOT",
        "S4_R8_HF_TOKEN_FIFO",
        "UV_CACHE_DIR",
        "UV_PROJECT_ENVIRONMENT",
        "S4_R8_ROBOFACTORY_ROOT",
        "S4_R8_RF_PYTHON",
    ):
        assert f"{name}=set" in observed
    assert "run_root_match=yes" in observed
    assert "HF_TOKEN=absent" in observed
    assert "HUGGING_FACE_HUB_TOKEN=absent" in observed
    combined = result.stdout + result.stderr + observed
    log = run_root / "prepare.log"
    if log.is_file():
        combined += log.read_text(encoding="utf-8")
    assert sentinel not in combined
