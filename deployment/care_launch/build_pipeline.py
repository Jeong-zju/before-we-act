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

The gate deliberately sits *after* the reference policy's closed-loop
evaluation. Reported success rate comes from that evaluation and does not
depend on the selector at all -- when the selector never fires, CARE's number
is the reference policy's number. Blocking headroom must therefore not block
the number a paper reports; it only stops the CARE-specific stages, which
produce nothing when no candidate can be selected. The orchestrator retries a
failed stage indefinitely rather than skipping it, so anything that must
survive a BLOCKED verdict has to be scheduled ahead of the gate.

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
# The evaluator takes one task per invocation.
MARS_TASKS = (
    "place_cube_in_cup",
    "strike_cube_hard",
    "three_robots_place_shoes",
    "four_robots_stack_cube",
)


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
    max_attempts: int | None = None

    def to_dict(self, layout: Layout) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "argv": self.argv,
            "cwd": self.cwd or str(layout.repo),
            "env": {**layout.env(), **self.env},
            "gpus": self.gpus,
            "artifacts": self.artifacts,
        }
        if self.max_attempts is not None:
            row["max_attempts"] = int(self.max_attempts)
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
            max_attempts=3,
            note=(
                "collects every host problem in one pass before any GPU work. "
                "A few attempts absorb a briefly busy GPU; beyond that the fix "
                "needs a person."
            ),
        ),
        Stage(
            name="reference_pipeline",
            argv=[
                "bash",
                str(layout.repo / "scripts/before_we_act/run_mars_care_official_pipeline.sh"),
            ],
            gpus=list(layout.gpus),
            artifacts=[
                _json_artifact(run / "pipeline_status.json", stage="CARE_BRANCHES")
            ],
            note=(
                "the existing shell pipeline: B0-H, the action-context cache, "
                "three B-core seeds, selection, and the reference Validation20 "
                "that produces the reported success rate. Wrapped rather than "
                "re-expressed, so its own resume and gates keep working."
            ),
        ),
    ] + [
        Stage(
            name=f"reference_validation20_{task}",
            argv=[
                python,
                "-m",
                "before_we_act.evaluate_mars_temporal_policy",
                "--checkpoint",
                str(run / "reference_checkpoint.pt"),
                "--dino-model",
                str(layout.dino),
                "--task",
                task,
                "--robofactory-root",
                str(layout.benchmark_repo),
                "--episodes",
                "20",
                "--output",
                str(run / "validation20" / "reference" / f"{task}.json"),
            ],
            gpus=RENDER_GPUS,
            artifacts=[
                {"path": str(run / "validation20" / "reference" / f"{task}.json"),
                 "kind": "json"}
            ],
            note=(
                "the evaluator takes one task per invocation, and the renderer "
                "is process-global, so tasks run as separate serial stages"
            ),
        )
        for task in MARS_TASKS
    ] + [
        Stage(
            name="care_branches",
            argv=collection,
            gpus=RENDER_GPUS,
            artifacts=[{"path": str(families), "kind": "dir"}],
            max_attempts=1,
            note=(
                "non-resumable: a retry would mix two physical run paths into "
                "one corpus. Serial Vulkan on one GPU."
            ),
        ),
        Stage(
            name="care_headroom",
            argv=measure,
            artifacts=[_json_artifact(headroom, verdict="PASS")],
            max_attempts=1,
            note=(
                "gate: stops the CARE-specific stages unless the collected "
                "family leaves room for the selector. Everything a reported "
                "success rate depends on has already run. One attempt only -- "
                "re-measuring the same corpus returns the same verdict, so a "
                "retry loop would burn the host to learn nothing."
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


def _preflight_stage(layout: Layout, benchmark: str, *, normalization: Path | None) -> Stage:
    argv = [
        str(layout.python),
        "-m",
        "deployment.care_launch.run_preflight",
        "--benchmark",
        benchmark,
        "--repo",
        str(layout.repo),
        "--benchmark-repo",
        str(layout.benchmark_repo),
        "--dataset",
        str(layout.dataset),
        "--dino",
        str(layout.dino),
        "--run",
        str(layout.run),
        "--output",
        str(layout.run / "host_preflight.json"),
    ]
    if normalization is not None:
        argv += ["--normalization", str(normalization)]
    return Stage(
        name="host_preflight",
        argv=argv,
        artifacts=[_json_artifact(layout.run / "host_preflight.json", status="PASSED")],
        max_attempts=3,
        note=(
            "collects every host problem in one pass before any GPU work. A few "
            "attempts absorb a briefly busy GPU; beyond that the fix needs a person."
        ),
    )


def _native_supervisor_stages(
    layout: Layout,
    benchmark: str,
    *,
    module: str,
    normalization: Path | None,
) -> list[Stage]:
    """Wrap a benchmark's own supervisor instead of re-expressing its DAG.

    BiCoord and DuoBench each own a resumable stage DAG that already covers
    download, caching, both training stages, selection, the reference
    Validation20, branch collection, and the CARE tail. Re-expressing thirty
    stages here would duplicate their receipt and resume logic and add a second
    place for the protocol to drift.

    The headroom decision does not need a stage of its own either: both DAGs
    have a ``branch_signal_gate`` between preparation and scorer training, and
    that gate now judges headroom. Running the supervisor as one stage keeps
    the gate where it belongs.
    """

    return [
        _preflight_stage(layout, benchmark, normalization=normalization),
        Stage(
            name=f"{benchmark}_pipeline",
            argv=[str(layout.python), "-m", module, "run"],
            gpus=list(layout.gpus),
            artifacts=[
                _json_artifact(layout.run / "status.json", stage="COMPLETE")
            ],
            note=(
                "the benchmark's own resumable DAG. Its branch_signal_gate "
                "stops the run before scorer training when the collected "
                "candidate family leaves no room for the selector; the "
                "reference Validation20 that produces the reported success "
                "rate has already completed by then."
            ),
        ),
    ]


def duobench_stages(
    layout: Layout,
    *,
    candidate_family: str,
    intervention_steps: int,
    reference_radius: float | None,
    primary_horizon: int,
) -> list[Stage]:
    return _native_supervisor_stages(
        layout,
        "duobench",
        module="deployment.duo_dino_reference.supervisor",
        normalization=layout.run / "prepared" / "manifest.json",
    )


def bicoord_stages(
    layout: Layout,
    *,
    candidate_family: str,
    intervention_steps: int,
    reference_radius: float | None,
    primary_horizon: int,
) -> list[Stage]:
    return _native_supervisor_stages(
        layout,
        "bicoord",
        module="deployment.bicoord_care.supervisor",
        normalization=layout.run / "artifacts" / "normalization.json",
    )


BENCHMARKS = {
    "mars": mars_stages,
    "duobench": duobench_stages,
    "bicoord": bicoord_stages,
}


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
        # A stage that keeps failing on a rented host is burning it. Six
        # attempts spans roughly fifteen minutes of backoff, which absorbs a
        # transient fault; past that the run stops and says which stage stalled
        # rather than retrying until the rental ends.
        "stall_after_attempts": 6,
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
