from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_s4_r7_candidate.sh"


def _embedded_python() -> list[str]:
    source = RUNNER.read_text(encoding="utf-8")
    return [
        remainder.partition("\nPY\n")[0]
        for remainder in source.split("<<'PY'\n")[1:]
    ]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_candidate_runner_is_valid_bash_without_errexit_bundle() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    source = RUNNER.read_text(encoding="utf-8")
    assert re.search(r"^\s*set\s+-", source, flags=re.MULTILINE) is None
    assert "set -euo" not in source


def test_candidate_runner_embedded_python_is_syntax_valid() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    remainders = source.split("<<'PY'\n")[1:]
    blocks = _embedded_python()
    assert blocks
    for index, (block, remainder) in enumerate(zip(blocks, remainders, strict=True), 1):
        _, marker, _ = remainder.partition("\nPY\n")
        assert marker, f"unterminated Python heredoc {index}"
        compile(block, f"run_s4_r7_candidate-heredoc-{index}", "exec")


def test_minimal_oom_preflight_is_terminal_but_not_a_pass(tmp_path: Path) -> None:
    verifier = next(
        block for block in _embedded_python() if "preflight report must be a JSON object" in block
    )
    report = tmp_path / "preflight.json"
    report.write_text(
        json.dumps(
            {
                "format_version": "wam.robofactory.s4_r7.preflight/1",
                "identity": {
                    "round_id": "s4-r7",
                    "candidate_id": "P0",
                    "model_kind": "s4_r7_token_preserving",
                },
                "updates": 200,
                "completed": False,
                "oom": True,
                "micro_team_batch": 2,
                "gradient_accumulation": 6,
                "effective_team_batch": 12,
                "peak_memory_bytes": 0,
                "gpu_total_memory_bytes": 32 * 1024**3,
            }
        ),
        encoding="utf-8",
    )
    common = [
        "python3",
        "-",
        str(report),
        "P0",
        "s4_r7_token_preserving",
        "a" * 64,
        "2",
        "6",
    ]
    terminal = subprocess.run(
        [*common, "terminal"],
        input=verifier,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    strict_pass = subprocess.run(
        [*common, "pass"],
        input=verifier,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert terminal.returncode == 0, terminal.stderr
    assert strict_pass.returncode != 0


def test_checkpoint_verifier_requires_canonical_hash_and_resume_format(
    tmp_path: Path,
) -> None:
    verifier = next(
        block
        for block in _embedded_python()
        if "checkpoint/resume must contain a mapping" in block
    )
    digests = {
        "legacy_r6l_policy_sha256": "a" * 64,
        "active_flow_checkpoint_sha256": "b" * 64,
        "local_future_checkpoint_sha256": "c" * 64,
        "team_future_checkpoint_sha256": "d" * 64,
        "pca_artifact_sha256": "e" * 64,
    }
    config = tmp_path / "s4_r7.yaml"
    config.write_text(
        "parent:\n"
        f"  expected_legacy_r6l_policy_sha256: {digests['legacy_r6l_policy_sha256']}\n"
        f"  expected_active_flow_sha256: {digests['active_flow_checkpoint_sha256']}\n"
        f"  expected_local_future_sha256: {digests['local_future_checkpoint_sha256']}\n"
        f"  expected_team_future_sha256: {digests['team_future_checkpoint_sha256']}\n"
        f"  expected_pca_sha256: {digests['pca_artifact_sha256']}\n",
        encoding="utf-8",
    )
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    method = {
        "round_id": "s4-r7",
        "candidate_id": "P0",
        "model_kind": "s4_r7_token_preserving",
    }
    git_commit = "f" * 40

    def invoke(path: Path, mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "python",
                "-",
                str(path),
                "P0",
                "s4_r7_token_preserving",
                mode,
                "125000",
                str(config),
                config_sha,
                git_commit,
            ],
            cwd=ROOT,
            input=verifier,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": "wam.robofactory.s4_r7.world_utility.checkpoint/1",
            "update": 125000,
            "method": method,
            "parent_identity": digests,
            "source": {"config_sha256": config_sha, "git_commit": git_commit},
        },
        checkpoint,
    )
    assert invoke(checkpoint, "complete").returncode == 0
    assert invoke(checkpoint, "resume").returncode != 0

    missing_hash = tmp_path / "missing_hash.pt"
    torch.save(
        {
            "format_version": "wam.robofactory.s4_r7.world_utility.checkpoint/1",
            "update": 125000,
            "method": method,
            "parent_identity": digests,
            "source": {},
        },
        missing_hash,
    )
    assert invoke(missing_hash, "complete").returncode != 0

    resume = tmp_path / "resume.pt"
    torch.save(
        {
            "format_version": "wam.robofactory.s4_r7.world_utility.resume/1",
            "update": 1000,
            "identity": {
                **method,
                "config_sha256": config_sha,
                "parent_identity": digests,
            },
        },
        resume,
    )
    assert invoke(resume, "resume").returncode == 0


def test_candidate_runner_parses_the_complete_launcher_contract() -> None:
    help_result = _run("--help")
    assert help_result.returncode == 0
    for option in (
        "--candidate",
        "--run-id",
        "--run-root",
        "--ready-file",
        "--failed-file",
        "--config",
        "--gpu-index",
        "--heartbeat-seconds",
    ):
        assert option in help_result.stdout


def test_candidate_runner_rejects_wrong_gpu_before_touching_run_state() -> None:
    result = _run(
        "--candidate",
        "P0",
        "--run-id",
        "unit",
        "--run-root",
        "/path/that/does/not/exist",
        "--ready-file",
        "/path/that/does/not/exist/shared.ready",
        "--failed-file",
        "/path/that/does/not/exist/shared.failed",
        "--config",
        "/path/that/does/not/exist/config.yaml",
        "--gpu-index",
        "1",
        "--heartbeat-seconds",
        "20",
    )
    assert result.returncode == 2
    assert "P0 must use physical GPU 0" in result.stderr


def test_candidate_runner_rejects_noncanonical_heartbeat_interval() -> None:
    result = _run(
        "--candidate",
        "P1",
        "--run-id",
        "unit",
        "--run-root",
        "/path/that/does/not/exist",
        "--ready-file",
        "/path/that/does/not/exist/shared.ready",
        "--failed-file",
        "/path/that/does/not/exist/shared.failed",
        "--config",
        "/path/that/does/not/exist/config.yaml",
        "--gpu-index",
        "1",
        "--heartbeat-seconds",
        "21",
    )
    assert result.returncode == 2
    assert "fixed at 20 seconds" in result.stderr


def test_candidate_runner_orders_pair_exact_before_formal_training() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    preflight_call = source.index("--preflight-only --preflight-updates")
    pair_lock = source.index('flock -x "${PAIR_FD}"')
    pair_validator = source.index("--p0-preflight")
    formal_training = source.index('run_stage training train_s4_r7_world_utility.py')
    evaluation = source.index('run_stage validating evaluate_s4_r7_causal.py')
    acceptance_lock = source.index('flock -x "${ACCEPT_FD}"')
    assert preflight_call < pair_lock < pair_validator < formal_training
    assert formal_training < evaluation < acceptance_lock


def test_candidate_runner_is_fail_closed_for_pair_fallback_and_old_artifacts() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'PAIR_FALLBACK}" == "micro1_accum12"' in source
    assert "no one-sided auto-change is allowed" in source
    assert "verify_preflight" in source
    assert "verify_pair_exact" in source
    assert "verify_checkpoint" in source
    assert "verify_candidate_report" in source
    assert "verify_acceptance" in source
    assert "preflight_provenance" in source
    assert "local-oom-await-paired-fallback" in source
    assert source.count("ensure_digest") >= 10


def test_candidate_runner_publishes_exact_program_pid_and_wait_heartbeats() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '--phase "${CURRENT_PHASE}" --program "${CURRENT_PROGRAM}"' in source
    assert '--detail "${CURRENT_DETAIL}" --pid "$$" --child-pid "${child_value}"' in source
    assert '--gpu-index "${GPU_INDEX}" --gpu-pid "${gpu_pid_value}"' in source
    assert '--candidate "${CANDIDATE}" --pid "$$"' in source
    assert 'sleep "${HEARTBEAT_SECONDS}"' in source
    for phase in (
        "waiting_shared",
        "preflight",
        "waiting_peer_preflight",
        "pair_validation",
        "training",
        "validating",
        "waiting_peer_report",
        "accepting",
        "complete",
        "failed",
    ):
        assert phase in source


def test_candidate_runner_exports_isolated_rollout_runtime() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'S4_R7_ROBOFACTORY_ROOT="${ROBOFACTORY_ROOT}"' in source
    assert 'S4_R7_RF_PYTHON="${RF_PYTHON}"' in source
    assert "LPD_POLICY_KIND=s4_flow" in source
    assert 'LPD_PORT="$((8872 + GPU_INDEX))"' in source
    assert 'LPD_RUN_ID="${RUN_ID}_${SLUG}"' in source
