"""Every command a pipeline emits must exist before the host is rented.

A stage whose argv names a flag that does not exist fails the instant it runs,
and the orchestrator then retries it until the attempt budget stops the run --
after the download and training that preceded it. The MARS validation stage
shipped in exactly that state: it omitted two required flags, and the pipeline
was checked for stage ordering and gates but never for whether its commands
could parse.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from deployment.care_launch.build_pipeline import BENCHMARKS, Layout, build


REPO = Path(__file__).resolve().parents[2]
SUBCOMMANDS = {"run", "status", "prepare", "folds", "quality", "print-dag"}


def _layout(tmp_path: Path) -> Layout:
    return Layout(
        repo=REPO,
        python=Path(sys.executable),
        run=tmp_path / "run",
        benchmark_repo=tmp_path / "bench",
        dataset=tmp_path / "data",
        dino=tmp_path / "dino",
        visual_cache=tmp_path / "cache",
    )


def _pipeline(benchmark: str, tmp_path: Path):
    return build(
        benchmark,
        _layout(tmp_path),
        candidate_family="behavior",
        intervention_steps=8,
        reference_radius=0.0239,
        primary_horizon=16,
    )


def _help_text(module: str, subcommand: str | None) -> str:
    env = dict(os.environ)
    # The same path the pipeline stages carry; the branch collector cannot
    # import without it.
    env["PYTHONPATH"] = f"{REPO / 'stereo_core'}:{REPO}"
    argv = [sys.executable, "-m", module]
    if subcommand:
        argv.append(subcommand)
    argv.append("--help")
    finished = subprocess.run(
        argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=300
    )
    return finished.stdout + finished.stderr


def _module_stages(pipeline) -> list[tuple[str, str, str | None, list[str]]]:
    rows = []
    for stage in pipeline["stages"]:
        argv = stage["argv"]
        if "-m" not in argv:
            continue
        index = argv.index("-m")
        module = argv[index + 1]
        rest = argv[index + 2 :]
        subcommand = rest[0] if rest and rest[0] in SUBCOMMANDS else None
        flags = sorted({token for token in argv if token.startswith("--")})
        rows.append((stage["name"], module, subcommand, flags))
    return rows


@pytest.mark.parametrize("benchmark", sorted(BENCHMARKS))
def test_every_stage_command_accepts_the_flags_it_is_given(
    benchmark: str, tmp_path: Path
) -> None:
    pipeline = _pipeline(benchmark, tmp_path)
    problems: list[str] = []

    for name, module, subcommand, flags in _module_stages(pipeline):
        text = _help_text(module, subcommand)
        if "usage:" not in text.lower():
            problems.append(f"{name}: {module} does not present a CLI\n{text[-400:]}")
            continue
        for flag in flags:
            if flag not in text:
                problems.append(f"{name}: {module} has no {flag}")

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("benchmark", sorted(BENCHMARKS))
def test_shell_stages_point_at_a_script_that_exists(
    benchmark: str, tmp_path: Path
) -> None:
    for stage in _pipeline(benchmark, tmp_path)["stages"]:
        argv = stage["argv"]
        if argv[0] != "bash":
            continue
        assert Path(argv[1]).is_file(), f"{stage['name']}: missing {argv[1]}"


def test_every_benchmark_is_registered() -> None:
    """The DuoBench and BiCoord prompts name these; an unregistered one fails."""
    assert set(BENCHMARKS) == {"mars", "duobench", "bicoord"}


@pytest.mark.parametrize("benchmark", sorted(BENCHMARKS))
def test_preflight_runs_before_anything_touches_a_gpu(
    benchmark: str, tmp_path: Path
) -> None:
    stages = _pipeline(benchmark, tmp_path)["stages"]
    first_gpu = next(
        (index for index, stage in enumerate(stages) if stage["gpus"]), len(stages)
    )
    assert stages[0]["name"] == "host_preflight"
    assert stages[0]["gpus"] == []
    assert first_gpu > 0
