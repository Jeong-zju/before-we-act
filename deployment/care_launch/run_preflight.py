"""Per-benchmark host preflight entry point.

One command that answers "can this host finish the run" before any GPU work
starts, writing a receipt the orchestrator gates on. Exits non-zero when
anything fails, but only after every check has run, so one pass lists
everything to fix.

Corpus sizes are the observed on-disk footprints of the collected branch
families plus caches; the floor exists because a volume that ends a run at
100% leaves no room for the receipts proving it finished.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Sequence

from deployment.care_launch.host_preflight import (
    GIB,
    CheckResult,
    check_disk_headroom,
    check_git_revision,
    check_gpu_inventory,
    check_no_foreign_gpu_processes,
    check_offscreen_render,
    check_paths_exist,
    check_pinned_distributions,
    check_python_imports,
    check_token_file,
    run_checks,
    write_report,
)


# Pinned third-party checkouts. A newer upstream commit is a different
# benchmark, not an upgrade.
BENCHMARK_REVISIONS = {
    "mars": "2d34fb38c80cb06550a5dbf99abac2c89f4336ed",
    "duobench": "082a57cdafea9db115029e6fe9e03691e755f93f",
    "bicoord": "c4577b8808e45c15836945ee23f01f89c8a056c3",
}
# SAPIEN aborts rather than raises when these drift, so they are ABI
# constraints rather than preferences.
PINNED_DISTRIBUTIONS = {"numpy": "1.26.4", "warp-lang": "1.4.0"}
SIMULATOR_IMPORTS = ("numpy", "torch", "h5py", "gymnasium", "sapien", "warp")
# Branch corpora plus DINO caches, measured on completed runs.
REQUIRED_FREE_GIB = {"mars": 400, "duobench": 500, "bicoord": 900}
ALLOWED_GPU_MODELS = ("5090", "RTX PRO 6000", "H200", "H100", "A100")


def build_checks(args: argparse.Namespace) -> list[Callable[[], CheckResult]]:
    benchmark = args.benchmark
    required = int(REQUIRED_FREE_GIB[benchmark]) * GIB
    checks: list[Callable[[], CheckResult]] = [
        lambda: check_disk_headroom(args.run, required_bytes=required, label="run"),
        lambda: check_gpu_inventory(
            expected_count=args.gpus, allowed_models=ALLOWED_GPU_MODELS
        ),
        check_no_foreign_gpu_processes,
        lambda: check_python_imports(SIMULATOR_IMPORTS, python=args.python),
        lambda: check_pinned_distributions(PINNED_DISTRIBUTIONS, python=args.python),
        lambda: check_git_revision(
            args.benchmark_repo, BENCHMARK_REVISIONS[benchmark], label=benchmark
        ),
    ]
    checks.extend(
        (lambda row=row: row)
        for row in check_paths_exist(
            {"repo": args.repo, "dataset": args.dataset, "dino": args.dino}
        )
    )
    if args.token is not None:
        checks.append(lambda: check_token_file(args.token))
    if not args.skip_render:
        checks.append(lambda: check_offscreen_render(python=args.python))
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_REVISIONS), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=None)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument(
        "--token",
        type=Path,
        default=Path("/workspace/.secrets/hf_token"),
        help="pass an empty value to skip the token check on an offline host",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="skip the Vulkan probe on a host that only trains",
    )
    args = parser.parse_args(argv)
    if args.token is not None and str(args.token) == "":
        args.token = None
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checks(build_checks(args))
    report["benchmark"] = args.benchmark
    write_report(report, args.output)
    for row in report["checks"]:
        marker = "ok  " if row["passed"] else "FAIL"
        print(f"{marker} {row['name']}: {row['detail']}", file=sys.stderr)
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
