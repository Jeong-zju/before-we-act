"""Read-only source and hardware preflight for the pinned BiCoord CARE run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .config import DATASET_REPO_ID, DATASET_REVISION, TASKS
from .stage_common import artifact, assert_common_paths, atomic_json, common_parser, publish_result


EXPECTED_BENCHMARK_COMMIT = "c4577b8808e45c15836945ee23f01f89c8a056c3"


def _git_revision(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    try:
        value = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot inspect git revision: {path}") from error
    if len(value) != 40:
        raise RuntimeError(f"invalid git revision from {path}: {value!r}")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(
            f"non-hexadecimal git revision from {path}: {value!r}"
        ) from error
    return value


def _expected_revision(value: str | None, *, label: str) -> str:
    """Validate one immutable source pin before comparing a checkout to it."""

    if value is None or len(value) != 40:
        raise RuntimeError(
            f"{label} source revision must be a pinned 40-character commit"
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(
            f"{label} source revision is not hexadecimal: {value!r}"
        ) from error
    return value


def _tracked_tree_report(path: Path, *, label: str) -> dict[str, Any]:
    """Fail closed on staged or unstaged changes to Git-tracked source.

    BiCoord's installation contract intentionally places supplemental assets in
    untracked directories below the benchmark checkout.  Those files are
    verified by the asset/download receipts, not by the source-tree pin.  The
    explicit ``--untracked-files=no`` therefore ignores them while retaining
    staged changes, tracked worktree edits/deletions, and dirty submodules.
    """

    command = [
        "git",
        "-C",
        str(path),
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=untracked",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"cannot inspect tracked {label} source tree: {path}"
        ) from error
    changes = [row for row in completed.stdout.splitlines() if row]
    if changes:
        preview = changes[:20]
        suffix = (
            ""
            if len(changes) <= len(preview)
            else f" (+{len(changes) - len(preview)} more)"
        )
        raise RuntimeError(
            f"{label} tracked source tree is dirty at {path}: {preview!r}{suffix}"
        )
    return {
        "status": "CLEAN",
        "tracked_tree_clean": True,
        "scope": "git_index_and_worktree_tracked_files",
        "untracked_files_ignored": True,
        "submodule_untracked_files_ignored": True,
        "tracked_submodule_drift_ignored": False,
        "porcelain": "v1",
        "tracked_changes": [],
    }


def _gpu_report() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover
        raise RuntimeError("nvidia-smi hardware probe failed") from error
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    parsed = [[item.strip() for item in row.split(",", 4)] for row in rows]
    if any(len(row) != 5 for row in parsed):
        raise RuntimeError(f"nvidia-smi returned malformed rows: {rows!r}")
    count = len(parsed)
    names = [row[1] for row in parsed]
    if count != 4:
        raise RuntimeError(f"formal run requires exactly four physical GPUs, got {count}")
    if not all("5090" in name for name in names):
        raise RuntimeError(f"formal run requires RTX 5090 devices, observed {names!r}")
    return {
        "nvidia_smi_available": True,
        "device_count": count,
        "device_names": names,
        "devices": [
            {
                "index": int(row[0]),
                "name": row[1],
                "uuid": row[2],
                "memory_total_mib": int(row[3]),
                "driver_version": row[4],
            }
            for row in parsed
        ],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _source_report(repo: Path, benchmark_repo: Path) -> dict[str, Any]:
    required_repo = (
        repo / "before_we_act" / "temporal_history_policy.py",
        repo / "before_we_act" / "predictive_team_belief_policy.py",
        repo / "before_we_act" / "care_belief.py",
    )
    required_bench = (
        benchmark_repo / "envs",
        benchmark_repo / "task_config",
        benchmark_repo / "policy",
    )
    missing = [str(path) for path in (*required_repo, *required_bench) if not path.exists()]
    if missing:
        raise RuntimeError(f"source checkout is incomplete: {missing}")
    care_revision = _git_revision(repo)
    bench_revision = _git_revision(benchmark_repo)
    expected_care = _expected_revision(
        os.environ.get("BICOORD_CARE_SOURCE_REVISION"), label="CARE"
    )
    if care_revision != expected_care:
        raise RuntimeError(
            f"CARE source revision drift: {care_revision} != {expected_care}"
        )
    expected_bench = _expected_revision(
        os.environ.get("BICOORD_CODE_REVISION", EXPECTED_BENCHMARK_COMMIT),
        label="BiCoord benchmark",
    )
    if bench_revision != expected_bench:
        raise RuntimeError(
            f"BiCoord benchmark revision drift: {bench_revision} != {expected_bench}"
        )
    care_tree = _tracked_tree_report(repo, label="CARE")
    benchmark_tree = _tracked_tree_report(benchmark_repo, label="BiCoord benchmark")
    return {
        "care_repo": str(repo.resolve()),
        "care_revision": care_revision,
        "expected_care_revision": expected_care,
        "care_tracked_tree_clean": True,
        "benchmark_repo": str(benchmark_repo.resolve()),
        "benchmark_revision": bench_revision,
        "expected_benchmark_revision": expected_bench,
        "benchmark_tracked_tree_clean": True,
        "tracked_source_contract": {
            "status": "PASSED",
            "scope": "tracked_files_only",
            "untracked_supplemental_assets_allowed": True,
            "care": care_tree,
            "benchmark": benchmark_tree,
        },
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "reference_policy": "B-core/TUNE",
        "required_upstream_policy": "TemporalHistoryPolicy/PredictiveTeamBeliefPolicy",
        "model_substitution": False,
        "model_dimension_override": False,
        "normalization_override": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args)
    source = _source_report(args.repo, args.benchmark_repo)
    gpu = _gpu_report()
    report_value = {
        "schema": "before-we-act.bicoord-care-preflight/1",
        "status": "PASSED",
        "tasks": list(TASKS),
        "source": source,
        "gpu": gpu,
        "hf_token_configured": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        ),
        "destructive_instance_operations": False,
    }
    report = args.run / "artifacts" / "preflight" / "source_preflight.json"
    atomic_json(report, report_value)
    return publish_result(
        args,
        stage="source_preflight",
        artifacts=[artifact(report, kind="preflight")],
        preflight=report_value,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = common_parser(__doc__, ("source-preflight", "preflight"))
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EXPECTED_BENCHMARK_COMMIT", "main", "run"]
