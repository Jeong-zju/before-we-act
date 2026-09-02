"""The pipeline must gate on headroom before it pays for scorer training."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.care_launch.build_pipeline import Layout, build, main


def _layout(tmp_path: Path) -> Layout:
    return Layout(
        repo=tmp_path / "repo",
        python=tmp_path / "venv/bin/python",
        run=tmp_path / "run",
        benchmark_repo=tmp_path / "bench",
        dataset=tmp_path / "data",
        dino=tmp_path / "dino",
        visual_cache=tmp_path / "cache",
    )


def _pipeline(tmp_path: Path, **overrides):
    options = {
        "candidate_family": "behavior",
        "intervention_steps": 8,
        "reference_radius": 0.0239,
        "primary_horizon": 16,
    }
    options.update(overrides)
    return build("mars", _layout(tmp_path), **options)


def _names(pipeline) -> list[str]:
    return [stage["name"] for stage in pipeline["stages"]]


def test_reported_success_rate_is_produced_before_the_gate(tmp_path: Path) -> None:
    """A BLOCKED verdict must not withhold the number a paper reports.

    Closed-loop success does not depend on the selector: when the selector never
    fires, CARE's number is the reference policy's number. The orchestrator
    retries a failed stage forever rather than skipping it, so the evaluation
    has to be scheduled ahead of the gate.
    """
    pipeline = _pipeline(tmp_path)
    names = _names(pipeline)

    assert names.index("reference_validation20") < names.index("care_headroom")
    assert names.index("host_preflight") < names.index("reference_validation20")


def test_headroom_precedes_every_training_stage(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    names = _names(pipeline)

    assert names.index("care_headroom") < names.index("care_prepare")
    assert names.index("care_headroom") < names.index("care_oof_folds")
    assert names.index("care_branches") < names.index("care_headroom")


def test_headroom_stage_blocks_the_sweep_unless_it_passes(tmp_path: Path) -> None:
    """A BLOCKED verdict must stop the pipeline, not be logged and ignored."""
    pipeline = _pipeline(tmp_path)
    stage = next(s for s in pipeline["stages"] if s["name"] == "care_headroom")

    gates = [artifact["equals"] for artifact in stage["artifacts"] if "equals" in artifact]
    assert {"verdict": "PASS"} in gates


def test_preflight_runs_first_and_gates_on_its_receipt(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    first = pipeline["stages"][0]

    assert first["name"] == "host_preflight"
    assert first["gpus"] == []
    assert {"status": "PASSED"} in [
        artifact["equals"] for artifact in first["artifacts"] if "equals" in artifact
    ]


def test_branch_collection_is_serialized_onto_one_gpu(tmp_path: Path) -> None:
    """SAPIEN's renderer is process-global here; parallel rollouts lose the device."""
    pipeline = _pipeline(tmp_path)
    stage = next(s for s in pipeline["stages"] if s["name"] == "care_branches")
    assert stage["gpus"] == [0]


def test_branch_collection_is_declared_non_resumable(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert pipeline["non_resumable_stages"] == ["care_branches"]


def test_candidate_family_and_window_reach_the_collector(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, candidate_family="behavior", intervention_steps=8)
    argv = next(s for s in pipeline["stages"] if s["name"] == "care_branches")["argv"]

    assert "--candidate-family" in argv
    assert argv[argv.index("--candidate-family") + 1] == "behavior"
    assert argv[argv.index("--intervention-steps") + 1] == "8"


def test_reference_radius_is_passed_only_when_requested(tmp_path: Path) -> None:
    with_radius = _pipeline(tmp_path, reference_radius=0.0239)
    without = _pipeline(tmp_path, reference_radius=None)

    def argv(pipeline) -> list[str]:
        return next(s for s in pipeline["stages"] if s["name"] == "care_headroom")["argv"]

    assert "--reference-radius" in argv(with_radius)
    assert "--reference-radius" not in argv(without)


def test_every_stage_carries_the_repo_python_path(tmp_path: Path) -> None:
    """stereo_core must be importable or the branch collector cannot load."""
    pipeline = _pipeline(tmp_path)
    for stage in pipeline["stages"]:
        assert "stereo_core" in stage["env"]["PYTHONPATH"]


def test_unknown_benchmark_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no pipeline for"):
        build(
            "robocasa",
            _layout(tmp_path),
            candidate_family="behavior",
            intervention_steps=8,
            reference_radius=None,
            primary_horizon=16,
        )


def test_cli_writes_a_pipeline_the_orchestrator_can_read(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "pipeline.json"
    code = main(
        [
            "--benchmark", "mars",
            "--run", str(tmp_path / "run"),
            "--benchmark-repo", str(tmp_path / "bench"),
            "--dataset", str(tmp_path / "data"),
            "--dino", str(tmp_path / "dino"),
            "--visual-cache", str(tmp_path / "cache"),
            "--output", str(output),
        ]
    )

    assert code == 0
    pipeline = json.loads(output.read_text())
    # The orchestrator requires these keys on every stage.
    for stage in pipeline["stages"]:
        assert {"name", "argv", "cwd", "env", "gpus", "artifacts"} <= set(stage)
        assert isinstance(stage["argv"], list) and stage["argv"]


def test_stages_that_can_never_succeed_stop_instead_of_burning_the_host(
    tmp_path: Path,
) -> None:
    """The orchestrator retries forever; an impossible stage would spin all rental.

    A missing dataset, a hash audit that will not match, or a headroom verdict
    that re-measurement cannot change are deterministic failures. Each gets an
    attempt budget so the run stops and names the stage instead.
    """
    pipeline = _pipeline(tmp_path)
    budgets = {
        stage["name"]: stage.get("max_attempts") for stage in pipeline["stages"]
    }

    assert pipeline["stall_after_attempts"] == 6
    # Re-measuring the same corpus returns the same verdict.
    assert budgets["care_headroom"] == 1
    # Retrying physical collection mixes two run paths into one corpus.
    assert budgets["care_branches"] == 1
    # A few attempts absorb a briefly busy GPU.
    assert budgets["host_preflight"] == 3
