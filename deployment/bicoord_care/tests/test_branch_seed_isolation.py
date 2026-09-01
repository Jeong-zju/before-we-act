from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deployment.bicoord_care import branch_collection
from deployment.bicoord_care import supervisor as sup
from deployment.bicoord_care.config import TASKS, VALIDATION_EPISODES
from deployment.bicoord_care.seed_discovery import SEED_MANIFEST_SCHEMA
from deployment.bicoord_care.stage_common import RESULT_SCHEMA, sha256_file


def _seed_result(
    *,
    stage: str,
    bucket: int,
    count: int,
    start: int,
    manifest: str,
    digest: str,
) -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "stage": stage,
        "status": "PASSED",
        "benchmark_adapter": "BiCoord",
        "seed_bucket": bucket,
        "episodes_per_task": count,
        "tasks": list(TASKS),
        "valid_seeds": {
            task: list(range(start + task_id * 1_000, start + task_id * 1_000 + count))
            for task_id, task in enumerate(TASKS)
        },
        "seed_manifest": manifest,
        "seed_manifest_sha256": digest,
        "policy_independent": True,
        "learned_policy_used": False,
        "closed_loop_policy_results_used": False,
    }


def _isolated_results() -> tuple[dict[str, object], dict[str, object]]:
    validation = _seed_result(
        stage="seed_discovery",
        bucket=0,
        count=VALIDATION_EPISODES,
        start=100_000,
        manifest="/run/artifacts/seed_discovery/seed_manifest.json",
        digest="a" * 64,
    )
    branches = _seed_result(
        stage="branch_seed_discovery",
        bucket=1,
        count=30,
        start=200_000,
        manifest="/run/artifacts/branch_seed_discovery/seed_manifest.json",
        digest="b" * 64,
    )
    return validation, branches


def test_formal_branch_seed_protocol_has_a_separate_stage_and_bucket() -> None:
    assert sup.BRANCH_SEED_EPISODES == 30
    assert sup.BRANCH_SEED_BUCKET == 1

    branch_seed_stage = sup.STAGES["branch_seed_discovery"]
    assert branch_seed_stage.module_key == "seed_discovery"
    assert branch_seed_stage.result_kind == "seed_manifest"
    assert branch_seed_stage.gpu_plan == "seed_task_queue4"
    assert tuple(sup.STAGES).index("seed_discovery") < tuple(sup.STAGES).index(
        "branch_seed_discovery"
    )
    assert tuple(sup.STAGES).index("branch_seed_discovery") < tuple(sup.STAGES).index(
        "branch_collection"
    )
    assert "branch_seed_discovery" in sup.STAGES["branch_collection"].dependencies
    assert "seed_discovery" not in sup.STAGES["branch_collection"].dependencies


def test_seed_manifest_isolation_helper_accepts_exact_disjoint_protocol() -> None:
    validation, branches = _isolated_results()
    assert sup.Supervisor._validate_seed_manifest_disjoint(validation, branches) is None


def test_seed_manifest_isolation_helper_rejects_per_task_overlap() -> None:
    validation, branches = _isolated_results()
    task = TASKS[-1]
    branch_seeds = branches["valid_seeds"]
    validation_seeds = validation["valid_seeds"]
    assert isinstance(branch_seeds, dict) and isinstance(validation_seeds, dict)
    branch_seeds[task][0] = validation_seeds[task][-1]

    with pytest.raises(sup.InvalidArtifact, match="overlap|disjoint"):
        sup.Supervisor._validate_seed_manifest_disjoint(validation, branches)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    (
        ("validation", "stage", "branch_seed_discovery"),
        ("validation", "seed_bucket", 1),
        ("validation", "episodes_per_task", VALIDATION_EPISODES - 1),
        ("branch", "stage", "seed_discovery"),
        ("branch", "seed_bucket", 0),
        ("branch", "episodes_per_task", 29),
    ),
)
def test_seed_manifest_isolation_helper_rejects_wrong_protocol_identity(
    target: str, field: str, value: object
) -> None:
    validation, branches = _isolated_results()
    selected = validation if target == "validation" else branches
    selected[field] = value

    with pytest.raises(sup.InvalidArtifact, match="seed|manifest|stage|bucket|episode"):
        sup.Supervisor._validate_seed_manifest_disjoint(validation, branches)


@pytest.mark.parametrize("identity_field", ("seed_manifest", "seed_manifest_sha256"))
def test_seed_manifest_isolation_helper_rejects_shared_manifest_identity(
    identity_field: str,
) -> None:
    validation, branches = _isolated_results()
    branches[identity_field] = validation[identity_field]

    with pytest.raises(sup.InvalidArtifact, match="manifest|identity|distinct"):
        sup.Supervisor._validate_seed_manifest_disjoint(validation, branches)


def test_formal_family_specs_bind_one_unique_branch_seed_per_family() -> None:
    for task_id, task in enumerate(TASKS):
        seeds = list(range(200_000 + task_id * 1_000, 200_030 + task_id * 1_000))
        specs = [
            branch_collection._family_spec(task, local, 30, seeds)
            for local in range(30)
        ]
        assert [row["seed"] for row in specs] == seeds
        assert len({row["seed"] for row in specs}) == 30
        assert [row["family_id"] for row in specs] == list(
            range(task_id * 30, (task_id + 1) * 30)
        )

        with pytest.raises(ValueError, match="30|unique|seed|coverage"):
            branch_collection._family_spec(task, 20, 30, seeds[:20])


def test_formal_branch_seed_loader_reads_only_branch_discovery_manifest(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    config_sha256 = "c" * 64
    task = TASKS[0]
    branch_seeds = list(range(200_000, 200_030))
    manifest = run / "artifacts" / "branch_seed_discovery" / "seed_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": SEED_MANIFEST_SCHEMA,
                "status": "PASSED",
                "stage": "branch_seed_discovery",
                "policy_independent": True,
                "seed_bucket": 1,
                "episodes_per_task": 30,
                "tasks": list(TASKS),
                "valid_seeds": {name: list(branch_seeds) for name in TASKS},
            }
        ),
        encoding="utf-8",
    )
    result = run / "stage_results" / "branch_seed_discovery.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "stage": "branch_seed_discovery",
                "status": "PASSED",
                "benchmark_adapter": "BiCoord",
                "config_sha256": config_sha256,
                "seed_bucket": 1,
                "episodes_per_task": 30,
                "valid_seeds": {name: list(branch_seeds) for name in TASKS},
                "seed_manifest": str(manifest.resolve()),
                "seed_manifest_sha256": sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        run=run,
        config_sha256=config_sha256,
        operation="formal",
    )

    assert branch_collection._official_branch_seeds(args, task, count=30) == branch_seeds

