from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from deployment.bicoord_care.preflight import _source_report


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_checkout(root: Path, *, care: bool) -> str:
    root.mkdir()
    if care:
        source = root / "before_we_act"
        source.mkdir()
        for name in (
            "temporal_history_policy.py",
            "predictive_team_belief_policy.py",
            "care_belief.py",
        ):
            (source / name).write_text(f"# {name}\n", encoding="utf-8")
    else:
        for name in ("envs", "task_config", "policy"):
            directory = root / name
            directory.mkdir()
            (directory / "tracked.txt").write_text(
                f"tracked {name}\n", encoding="utf-8"
            )
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "BiCoord preflight test")
    _git(root, "config", "user.email", "bicoord-preflight@example.invalid")
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return _git(root, "rev-parse", "HEAD")


def _checkouts(tmp_path: Path) -> tuple[Path, str, Path, str]:
    care = tmp_path / "care"
    benchmark = tmp_path / "benchmark"
    care_revision = _init_checkout(care, care=True)
    benchmark_revision = _init_checkout(benchmark, care=False)
    return care, care_revision, benchmark, benchmark_revision


def _pin(
    monkeypatch: pytest.MonkeyPatch, care_revision: str, benchmark_revision: str
) -> None:
    monkeypatch.setenv("BICOORD_CARE_SOURCE_REVISION", care_revision)
    monkeypatch.setenv("BICOORD_CODE_REVISION", benchmark_revision)


def test_source_report_requires_clean_tracked_trees_but_ignores_untracked_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    care, care_revision, benchmark, benchmark_revision = _checkouts(tmp_path)
    # This mirrors BiCoord's official post-checkout supplemental asset install.
    supplemental = benchmark / "assets" / "objects" / "003_plate_large"
    supplemental.mkdir(parents=True)
    (supplemental / "model_data0.json").write_text("{}\n", encoding="utf-8")
    (care / "local-run-note.txt").write_text("untracked\n", encoding="utf-8")
    _pin(monkeypatch, care_revision, benchmark_revision)

    report = _source_report(care, benchmark)

    assert report["care_revision"] == report["expected_care_revision"]
    assert report["benchmark_revision"] == report["expected_benchmark_revision"]
    assert report["care_tracked_tree_clean"] is True
    assert report["benchmark_tracked_tree_clean"] is True
    contract = report["tracked_source_contract"]
    assert contract["status"] == "PASSED"
    assert contract["scope"] == "tracked_files_only"
    assert contract["untracked_supplemental_assets_allowed"] is True
    for name in ("care", "benchmark"):
        assert contract[name]["tracked_tree_clean"] is True
        assert contract[name]["untracked_files_ignored"] is True
        assert contract[name]["tracked_changes"] == []


@pytest.mark.parametrize("staged", [False, True])
def test_source_report_rejects_tracked_care_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged: bool
) -> None:
    care, care_revision, benchmark, benchmark_revision = _checkouts(tmp_path)
    changed = care / "before_we_act" / "care_belief.py"
    changed.write_text("# modified tracked CARE source\n", encoding="utf-8")
    if staged:
        _git(care, "add", str(changed.relative_to(care)))
    _pin(monkeypatch, care_revision, benchmark_revision)

    with pytest.raises(RuntimeError, match="CARE tracked source tree is dirty"):
        _source_report(care, benchmark)


@pytest.mark.parametrize("staged", [False, True])
def test_source_report_rejects_tracked_benchmark_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged: bool
) -> None:
    care, care_revision, benchmark, benchmark_revision = _checkouts(tmp_path)
    changed = benchmark / "policy" / "tracked.txt"
    changed.write_text("modified tracked benchmark source\n", encoding="utf-8")
    if staged:
        _git(benchmark, "add", str(changed.relative_to(benchmark)))
    _pin(monkeypatch, care_revision, benchmark_revision)

    with pytest.raises(
        RuntimeError, match="BiCoord benchmark tracked source tree is dirty"
    ):
        _source_report(care, benchmark)


def test_source_report_requires_care_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    care, _care_revision, benchmark, benchmark_revision = _checkouts(tmp_path)
    monkeypatch.delenv("BICOORD_CARE_SOURCE_REVISION", raising=False)
    monkeypatch.setenv("BICOORD_CODE_REVISION", benchmark_revision)

    with pytest.raises(RuntimeError, match="CARE source revision must be a pinned"):
        _source_report(care, benchmark)


def test_unpinned_escape_hatch_cannot_bypass_benchmark_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    care, care_revision, benchmark, _benchmark_revision = _checkouts(tmp_path)
    monkeypatch.setenv("BICOORD_CARE_SOURCE_REVISION", care_revision)
    monkeypatch.setenv("BICOORD_CODE_REVISION", "f" * 40)
    monkeypatch.setenv("BICOORD_ALLOW_UNPINNED_SOURCE", "1")

    with pytest.raises(RuntimeError, match="BiCoord benchmark revision drift"):
        _source_report(care, benchmark)
