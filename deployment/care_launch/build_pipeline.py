"""Emit a headroom-first CARE pipeline for the orchestrator to drive.

``deployment.vla_baselines.orchestrator`` is the only driver here that survives
an unattended run: a stage failure is recorded and retried with exponential
backoff instead of ending the process, and a stage whose command and artifacts
are unchanged is skipped on the next sweep. This module writes the stage list it
consumes.

The ordering differs from the shell pipeline it replaces in one decisive way.
The old order collected branches and then spent twelve scorer runs plus a
calibration pass before anything reported whether the candidate family could be
selected at all -- and on every corpus measured so far the answer was no, at a
cost of the entire run. Here ``care_headroom`` sits immediately after branch
collection and gates the training stages, so a family with no room to be
selected costs one collection pass instead of a full pipeline.

Physical branch collection is marked non-resumable. Retrying it silently mixes
evidence from two different physical run paths into one corpus, which is worse
than failing: infrastructure stages retry freely, this one does not.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence


PIPELINE_VERSION = "before-we-act.care-launch-pipeline/1"
# Rendering in SAPIEN is process-global in these containers: two renderers on
# separate CUDA contexts can still lose the device. Rollout stages take one GPU.
RENDER_GPUS = [0]


@dataclass(frozen=True)
class Layout:
    """Absolute paths on the run host."""

    repo: Path
    python: Path
    run: Path
    benchmark_repo: Path
    dataset: Path
    dino: Path
    visual_cache: Path
    gpus: tuple[int, ...] = (0, 1, 2, 3)

    @property
    def logs(self) -> Path:
        return self.run / "logs"

    @property
    def contract(self) -> Path:
        return self.run / "contract"

    def env(self) -> dict[str, str]:
        return {
            "PYTHONPATH": f"{self.repo / 'stereo_core'}:{self.repo}",
            "HF_HUB_DOWNLOAD_TIMEOUT": "600",
            "HF_HUB_ETAG_TIMEOUT": "60",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MUJOCO_GL": "egl",
        }


@dataclass
class Stage:
    name: str
    argv: list[str]
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    gpus: list[int] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self, layout: Layout) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "argv": self.argv,
            "cwd": self.cwd or str(layout.repo),
            "env": {**layout.env(), **self.env},
            "gpus": self.gpus,
            "artifacts": self.artifacts,
        }
        if self.note:
            row["note"] = self.note
        return row


def _json_artifact(path: Path, **equals: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "kind": "json"}
    if equals:
        row["equals"] = equals
    return row


def mars_stages(
    layout: Layout,
    *,
    candidate_family: str,
    intervention_steps: int,
    reference_radius: float | None,
    primary_horizon: int,
) -> list[Stage]:
    python = str(layout.python)
    run = layout.run
    families = run / "care_families"
    headroom = run / "care_headroom.json"
    preflight = run / "host_preflight.json"
    prepared = run / "care_prepared.pt"

    collection = [
        python,
        "-m",
        "before_we_act.mars_care_branch_collector",
        "--manifest",
        str(layout.contract / "care_family_manifest.json"),
        "--checkpoint",
        str(run / "reference_checkpoint.pt"),
        "--output-root",
        str(families),
        "--robofactory-root",
        str(layout.benchmark_repo),
        "--device",
        "cuda:0",
        "--render-device",
        "cuda:0",
        "--candidate-family",
        candidate_family,
        "--intervention-steps",
        str(intervention_steps),
    ]

    measure = [
        python,
        "-m",
        "scripts.before_we_act.measure_care_headroom",
        "--families",
        str(families),
        "--output",
        str(headroom),
        "--primary-horizon",
        str(primary_horizon),
    ]
    if reference_radius is not None:
        measure += ["--reference-radius", str(reference_radius)]

    return [
        Stage(
            name="host_preflight",
            argv=[
                python,
                "-m",
                "deployment.care_launch.run_preflight",
                "--benchmark",
                "mars",
                "--repo",
                str(layout.repo),
                "--benchmark-repo",
                str(layout.benchmark_repo),
                "--dataset",
                str(layout.dataset),
                "--dino",
                str(layout.dino),
                "--run",
                str(run),
                "--output",
                str(preflight),
            ],
            artifacts=[_json_artifact(preflight, status="PASSED")],
            note="collects every host problem in one pass before any GPU work",
        ),
        Stage(
            name="care_branches",
            argv=collection,
            gpus=RENDER_GPUS,
            artifacts=[{"path": str(families), "kind": "dir"}],
            note=(
                "non-resumable: a retry would mix two physical run paths into "
                "one corpus. Serial Vulkan on one GPU."
            ),
        ),
        Stage(
            name="care_headroom",
            argv=measure,
            artifacts=[_json_artifact(headroom, verdict="PASS")],
            note=(
                "gate: fails the sweep unless the collected family leaves room "
                "for the selector, before any scorer training is paid for"
            ),
        ),
        Stage(
            name="care_prepare",
            argv=[
                python,
                "-m",
                "scripts.before_we_act.prepare_mars_care_training",
                "prepare",
                "--family-root",
                str(families),
                "--quality-root",
                str(run / "care_quality"),
                "--reference-checkpoint",
                str(run / "reference_checkpoint.pt"),
                "--output",
                str(prepared),
                "--manifest-output",
                str(run / "care_prepared_manifest.json"),
            ],
            artifacts=[{"path": str(prepared), "kind": "file"}],
        ),
        Stage(
            name="care_oof_folds",
            argv=[
                python,
                "-m",
                "scripts.before_we_act.run_mars_care_oof_v3",
                "folds",
                "--prepared",
                str(prepared),
                "--output",
                str(run / "oof" / "folds.json"),
            ],
            artifacts=[_json_artifact(run / "oof" / "folds.json")],
        ),
    ]


BENCHMARKS = {"mars": mars_stages}


def build(
    benchmark: str,
    layout: Layout,
    *,
    candidate_family: str,
    intervention_steps: int,
    reference_radius: float | None,
    primary_horizon: int,
) -> dict[str, Any]:
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f"no pipeline for {benchmark}; available: {sorted(BENCHMARKS)}"
        )
    stages = BENCHMARKS[benchmark](
        layout,
        candidate_family=candidate_family,
        intervention_steps=intervention_steps,
        reference_radius=reference_radius,
        primary_horizon=primary_horizon,
    )
    return {
        "schema": PIPELINE_VERSION,
        "benchmark": benchmark,
        "candidate_family": candidate_family,
        "intervention_steps": intervention_steps,
        "non_resumable_stages": ["care_branches"],
        "stages": [stage.to_dict(layout) for stage in stages],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--repo", type=Path, default=Path("/workspace/repos/before-we-act"))
    parser.add_argument("--python", type=Path, default=Path("/workspace/venvs/mars/bin/python"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--candidate-family", default="behavior")
    parser.add_argument("--intervention-steps", type=int, default=8)
    parser.add_argument("--primary-horizon", type=int, default=16)
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=None,
        help="calibration radius to judge headroom against; omit to use the "
        "irreducible radius implied by matched repeats",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = Layout(
        repo=args.repo,
        python=args.python,
        run=args.run,
        benchmark_repo=args.benchmark_repo,
        dataset=args.dataset,
        dino=args.dino,
        visual_cache=args.visual_cache,
    )
    pipeline = build(
        args.benchmark,
        layout,
        candidate_family=args.candidate_family,
        intervention_steps=args.intervention_steps,
        reference_radius=args.reference_radius,
        primary_horizon=args.primary_horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(pipeline['stages'])} stages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
