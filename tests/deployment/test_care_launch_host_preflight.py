"""Host preflight must report every problem, not just the first one."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.care_launch.host_preflight import (
    GIB,
    CheckResult,
    check_disk_headroom,
    check_git_revision,
    check_paths_exist,
    check_token_file,
    run_checks,
    write_report,
)


def test_disk_headroom_passes_when_the_volume_can_hold_the_run(tmp_path: Path) -> None:
    result = check_disk_headroom(tmp_path, required_bytes=1, label="scratch")
    assert result.passed
    assert result.data["free_bytes"] > 0


def test_disk_headroom_fails_before_the_run_fills_the_volume(tmp_path: Path) -> None:
    result = check_disk_headroom(tmp_path, required_bytes=1 << 62, label="scratch")
    assert not result.passed
    assert "need" in result.detail


def test_disk_headroom_resolves_a_path_that_does_not_exist_yet(tmp_path: Path) -> None:
    """The output directory is usually created by the stage, not the preflight."""
    result = check_disk_headroom(
        tmp_path / "runs" / "care" / "branches", required_bytes=1, label="output"
    )
    assert result.passed
    assert result.data["path"] == str(tmp_path)


def test_token_file_must_exist_and_be_private(tmp_path: Path) -> None:
    token = tmp_path / "hf_token"
    assert not check_token_file(token).passed

    token.write_text("secret", encoding="utf-8")
    token.chmod(0o644)
    result = check_token_file(token)
    assert not result.passed and "world accessible" in result.detail

    token.chmod(0o600)
    assert check_token_file(token).passed


def test_empty_token_file_is_rejected(tmp_path: Path) -> None:
    token = tmp_path / "hf_token"
    token.write_text("   \n", encoding="utf-8")
    token.chmod(0o600)
    assert not check_token_file(token).passed


def test_git_revision_reports_drift(tmp_path: Path) -> None:
    result = check_git_revision(tmp_path, "0" * 40, label="benchmark")
    assert not result.passed and "not a checkout" in result.detail


def test_missing_paths_are_reported_individually(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.mkdir()
    rows = check_paths_exist({"dataset": present, "weights": tmp_path / "absent"})

    assert [row.passed for row in rows] == [True, False]
    assert rows[1].name == "path:weights"


def test_run_checks_collects_every_failure() -> None:
    report = run_checks(
        [
            lambda: CheckResult("first", False, "no"),
            lambda: CheckResult("second", True, "yes"),
            lambda: CheckResult("third", False, "also no"),
        ]
    )

    assert report["status"] == "FAILED"
    assert report["failures"] == ["first", "third"]
    assert len(report["checks"]) == 3


def test_a_check_that_raises_becomes_a_failure_not_a_crash() -> None:
    def broken() -> CheckResult:
        raise RuntimeError("nvidia-smi exploded")

    report = run_checks([broken, lambda: CheckResult("ok", True, "fine")])

    assert report["status"] == "FAILED"
    assert "check raised RuntimeError" in report["checks"][0]["detail"]
    assert report["checks"][1]["passed"]


def test_all_passing_checks_report_passed() -> None:
    report = run_checks([lambda: CheckResult("only", True, "fine")])
    assert report["status"] == "PASSED"
    assert report["failures"] == []


def test_report_is_written_atomically(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "preflight.json"
    report = run_checks([lambda: CheckResult("only", True, "fine")])
    write_report(report, output)

    assert json.loads(output.read_text())["status"] == "PASSED"
    assert not list(output.parent.glob(".*tmp"))


def test_gib_constant_is_binary() -> None:
    assert GIB == 1024**3


def test_a_floored_standard_deviation_is_rejected() -> None:
    """The existing audits assert std >= 1e-4, which is the floor itself.

    A dimension that is constant in the corpus lands exactly on the floor and
    passes that check, then divides by it during normalization.
    """
    from deployment.care_launch.host_preflight import (
        check_normalization_is_non_degenerate,
    )

    healthy = check_normalization_is_non_degenerate(
        {"qpos_std": [0.5, 0.3], "action_std": [0.2, 0.4]}
    )
    assert healthy.passed

    floored = check_normalization_is_non_degenerate(
        {"qpos_std": [0.5, 1e-4], "action_std": [0.2]}
    )
    assert not floored.passed
    assert floored.data["floored_dimensions"] == {"qpos_std": [1]}
    # The old guard would have accepted this.
    assert min([0.5, 1e-4]) >= 1e-4


def test_normalization_guard_accepts_both_key_conventions() -> None:
    from deployment.care_launch.host_preflight import (
        check_normalization_is_non_degenerate,
    )

    # MARS names them q_std/a_std; DuoBench and BiCoord use qpos_std/action_std.
    assert check_normalization_is_non_degenerate({"q_std": [0.5], "a_std": [0.3]}).passed
    assert not check_normalization_is_non_degenerate(
        {"q_std": [0.5], "a_std": [1e-4]}
    ).passed


def test_non_finite_normalization_is_rejected() -> None:
    from deployment.care_launch.host_preflight import (
        check_normalization_is_non_degenerate,
    )

    result = check_normalization_is_non_degenerate(
        {"qpos_std": [0.5, float("nan")]}
    )
    assert not result.passed and "non-finite" in result.detail


def test_corpus_completeness_is_checked_before_any_gpu_work() -> None:
    """Each pipeline enforces its episode count deep inside training.

    The requirement is right -- a partial corpus produces a number that is not
    comparable to the published baselines -- but reaching it takes hours, so an
    incomplete download is named only after the GPU time is already spent.
    """
    from deployment.care_launch.host_preflight import check_corpus_completeness

    expected = {"ball_maze": 50, "bin_sort": 50, "carry_pot": 50}
    assert check_corpus_completeness(expected, expected).passed

    partial = check_corpus_completeness(
        {"ball_maze": 50, "bin_sort": 47, "carry_pot": 50}, expected
    )
    assert not partial.passed
    assert partial.data["mismatched"] == {"bin_sort": {"observed": 47, "expected": 50}}


def test_a_missing_task_counts_as_zero_rather_than_passing() -> None:
    from deployment.care_launch.host_preflight import check_corpus_completeness

    result = check_corpus_completeness({"ball_maze": 50}, {"ball_maze": 50, "bin_sort": 50})
    assert not result.passed
    assert result.data["mismatched"]["bin_sort"]["observed"] == 0
