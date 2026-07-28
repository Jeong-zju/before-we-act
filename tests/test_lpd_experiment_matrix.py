from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.build_lpd_gate_summary import build_gate_summary
from scripts.summarize_lpd_experiment_matrix import (
    FORMAT_VERSION,
    _load_candidate,
    compare_candidates,
    main,
    render_markdown,
)


def test_gate_summary_binds_checkpoint_config_and_episode_records(tmp_path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("name: fixture\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    client = {
        "checkpoint": str(checkpoint),
        "checkpoint_format": "fixture/1",
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }
    lift = _rollout([True, True, False, True], client=client)
    lpd = _rollout([True, False, True, True], client=client)

    summary = build_gate_summary(
        mode="formal",
        experiment="fixture",
        policy_kind="static_act",
        config=config,
        checkpoint=checkpoint,
        source_commit="0" * 40,
        seed_start=900,
        episodes=4,
        lift=lift,
        lpd=lpd,
    )

    assert summary["format_version"] == (
        "wam.robofactory.lpd_fixed_seed_gate/2"
    )
    assert summary["candidate"]["checkpoint_sha256"] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()
    assert summary["candidate"]["config_sha256"]
    assert summary["lift_barrier"]["successes"] == 3
    assert summary["long_pipeline_delivery"]["successes"] == 3
    assert summary["passed"] is False


def test_gate_summary_rejects_cross_task_checkpoint_drift(tmp_path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("name: fixture\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    lift = _rollout(
        [True, True],
        client={
            "checkpoint": str(checkpoint),
            "checkpoint_format": "fixture/1",
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        },
    )
    lpd = _rollout(
        [True, True],
        client={
            "checkpoint": str(checkpoint),
            "checkpoint_format": "fixture/1",
            "checkpoint_sha256": "2" * 64,
        },
    )

    with pytest.raises(ValueError, match="different inference client identities"):
        build_gate_summary(
            mode="gate",
            experiment="fixture",
            policy_kind="static_act",
            config=config,
            checkpoint=checkpoint,
            source_commit="0" * 40,
            seed_start=900,
            episodes=2,
            lift=lift,
            lpd=lpd,
        )


def test_gate_summary_rejects_checkpoint_not_used_by_client(tmp_path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("name: fixture\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    client = {
        "checkpoint": str(checkpoint),
        "checkpoint_format": "fixture/1",
        "checkpoint_sha256": "1" * 64,
    }

    with pytest.raises(ValueError, match="differs from gate checkpoint"):
        build_gate_summary(
            mode="gate",
            experiment="fixture",
            policy_kind="static_act",
            config=config,
            checkpoint=checkpoint,
            source_commit="0" * 40,
            seed_start=900,
            episodes=2,
            lift=_rollout([True, True], client=client),
            lpd=_rollout([True, True], client=client),
        )


def test_comparison_uses_exact_task_seed_pairs(tmp_path):
    baseline_path = _gate(
        tmp_path / "baseline",
        lift=[True, False, False, True],
        lpd=[False, False, True, False],
    )
    candidate_path = _gate(
        tmp_path / "candidate",
        lift=[True, True, False, True],
        lpd=[False, True, True, True],
    )
    baseline = _load_candidate(f"branch_point={baseline_path}")
    candidate = _load_candidate(f"act={candidate_path}")

    comparison = compare_candidates(
        (baseline, candidate),
        baseline="branch_point",
    )

    assert comparison["format_version"] == FORMAT_VERSION
    act = comparison["candidates"]["act"]["tasks"]
    assert act["lift_barrier"]["successes"] == 3
    assert act["lift_barrier"]["delta_vs_baseline"] == pytest.approx(0.25)
    assert act["long_pipeline_delivery"]["successes"] == 3
    assert act["long_pipeline_delivery"]["delta_vs_baseline"] == pytest.approx(
        0.5
    )
    assert (
        act["long_pipeline_delivery"]["paired_mcnemar_vs_baseline"][
            "first_only_success"
        ]
        == 2
    )
    markdown = render_markdown(comparison)
    assert "| act | lift_barrier | 3/4 | 75.0%" in markdown
    assert "identical task seeds" in markdown


def test_comparison_rejects_unpaired_seed_schedules(tmp_path):
    baseline_path = _gate(
        tmp_path / "baseline",
        lift=[True, False],
        lpd=[False, True],
        seed_start=900,
    )
    candidate_path = _gate(
        tmp_path / "candidate",
        lift=[True, True],
        lpd=[True, True],
        seed_start=920,
    )

    with pytest.raises(ValueError, match="seed schedule differs"):
        compare_candidates(
            (
                _load_candidate(f"baseline={baseline_path}"),
                _load_candidate(f"candidate={candidate_path}"),
            ),
            baseline="baseline",
        )


def test_cli_writes_auditable_json_and_markdown_without_overwrite(tmp_path):
    baseline = _gate(
        tmp_path / "baseline",
        lift=[True, False],
        lpd=[False, True],
    )
    candidate = _gate(
        tmp_path / "candidate",
        lift=[True, True],
        lpd=[True, True],
    )
    output_json = tmp_path / "comparison.json"
    output_markdown = tmp_path / "comparison.md"
    argv = [
        "--candidate",
        f"baseline={baseline}",
        "--candidate",
        f"candidate={candidate}",
        "--baseline",
        "baseline",
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_markdown),
    ]

    assert main(argv) == 0
    assert json.loads(output_json.read_text())["baseline"] == "baseline"
    assert output_markdown.read_text().startswith(
        "# LPD fixed-seed experiment comparison"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


def _gate(
    root: Path,
    *,
    lift: list[bool],
    lpd: list[bool],
    seed_start: int = 900,
) -> Path:
    assert len(lift) == len(lpd)
    root.mkdir()
    payload = {
        "format_version": "wam.robofactory.lpd_fixed_seed_gate/2",
        "mode": "gate",
        "experiment": root.name,
        "candidate": {
            "source_commit": "0" * 40,
            "checkpoint_sha256": "1" * 64,
        },
        "seed_protocol": {
            "seed_start": seed_start,
            "episodes_per_task": len(lift),
            "identical_across_tasks": True,
        },
        "lift_barrier": {
            "successes": sum(lift),
            "success_rate": sum(lift) / len(lift),
            "episodes": _episodes(lift, seed_start=seed_start),
        },
        "long_pipeline_delivery": {
            "successes": sum(lpd),
            "success_rate": sum(lpd) / len(lpd),
            "episodes": _episodes(lpd, seed_start=seed_start),
        },
    }
    path = root / "gate_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _episodes(values: list[bool], *, seed_start: int) -> list[dict[str, object]]:
    return [
        {"seed": seed_start + offset, "success": success}
        for offset, success in enumerate(values)
    ]


def _rollout(
    values: list[bool],
    *,
    client: dict[str, object],
    seed_start: int = 900,
) -> dict[str, object]:
    return {
        "completed": True,
        "fatal_error": None,
        "episodes_completed": len(values),
        "direct_model_action_coverage": 1.0,
        "successes": sum(values),
        "success_rate_wilson_95": [0.0, 1.0],
        "client": client,
        "episodes": [
            {
                "seed": seed_start + offset,
                "success": success,
                "stop_reason": "success" if success else "timeout",
                "steps": 10,
                "inference_latency_ms": {"mean": 1.0},
            }
            for offset, success in enumerate(values)
        ],
    }
