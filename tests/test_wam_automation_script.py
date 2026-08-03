from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/wam_automation.sh"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    process_env.pop("HF_TOKEN", None)
    if env:
        process_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=process_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_automation_script_is_valid_bash_and_lists_core_actions() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    listed = _run("--list")
    assert listed.returncode == 0, listed.stderr
    assert listed.stdout.splitlines() == [
        "code",
        "robofactory",
        "env",
        "robofactory-env",
        "assets",
        "hf-auth",
        "hf-download",
        "hf-upload",
        "vision",
        "doctor",
        "data-check",
        "test",
        "train-smoke",
        "train",
        "validate-smoke",
        "validate",
        "snapshot",
        "bootstrap",
        "full-smoke",
        "full",
    ]


def test_full_smoke_dry_run_is_ordered_non_mutating_and_secret_safe(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "fresh-workspace"
    secret = "hf_unit_test_secret_must_not_be_logged"
    result = _run(
        "--dry-run",
        "full-smoke",
        env={
            "WORKSPACE_ROOT": str(workspace),
            "HF_DATASET_REPO": "unit-test/liftbarrier",
            "HF_TOKEN": secret,
            "REQUIRE_CUDA": "false",
        },
    )

    assert result.returncode == 0, result.stderr
    assert not workspace.exists()
    combined = result.stdout + result.stderr
    assert secret not in combined

    expected = [
        "code",
        "robofactory",
        "env",
        "robofactory-env",
        "assets",
        "hf-auth",
        "hf-download",
        "vision",
        "doctor",
        "data-check",
        "test",
        "train-smoke",
        "validate-smoke",
        "snapshot",
    ]
    offsets = [
        combined.index(f"START action {index}:{action}")
        for index, action in enumerate(expected)
    ]
    assert offsets == sorted(offsets)
    assert "hf download unit-test/liftbarrier --type dataset" in combined
    assert "train_liftbarrier_m1_scratch.py" in combined
    assert "serve_robofactory_m1_rollout.py" in combined
    assert "run_robofactory_m1_inference.py" in combined
    assert (
        "--checkpoint "
        f"{workspace}/before-we-act/checkpoints/preflight/"
        "m1_liftbarrier_automation_smoke"
    ) in combined


def test_comma_separated_actions_preserve_order_and_unknown_action_fails(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "comma-workspace"
    ordered = _run(
        "--dry-run",
        "code,robofactory,env",
        env={"WORKSPACE_ROOT": str(workspace)},
    )
    assert ordered.returncode == 0, ordered.stderr
    assert "actions=code,robofactory,env" in ordered.stdout
    assert not workspace.exists()

    unknown = _run("--dry-run", "code,not-an-action")
    assert unknown.returncode == 2
    assert "unknown action: not-an-action" in unknown.stderr


def test_code_action_clones_local_repo_and_resume_skips_it(tmp_path: Path) -> None:
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-b", "main", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Automation Test"],
        check=True,
    )
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )

    workspace = tmp_path / "workspace"
    environment = {
        "WORKSPACE_ROOT": str(workspace),
        "FE_REPO_URL": str(source),
        "FE_REF": "main",
    }
    initial = _run("code", env=environment)
    assert initial.returncode == 0, initial.stdout + initial.stderr
    clone = workspace / "before-we-act"
    assert (clone / "README.md").read_text(encoding="utf-8") == "fixture\n"

    resumed = _run("--resume", "code", env=environment)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "SKIP completed action 0:code" in resumed.stdout
