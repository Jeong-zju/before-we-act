"""Prepare the audited LiftBarrier M1 training manifest and normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.robofactory_m1_training_artifacts import (  # noqa: E402
    TRANSITION_SELECTIONS,
    prepare_robofactory_m1_training_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit RoboFactory M1 HDF5 files, assign seed-disjoint splits, and "
            "fit train-only state/action/delta normalization."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--conversion-manifest", default="manifest.json")
    parser.add_argument("--training-manifest", default="training_manifest.json")
    parser.add_argument("--normalization", default="normalization.npz")
    parser.add_argument(
        "--transition-selection",
        choices=TRANSITION_SELECTIONS,
        required=True,
        help=(
            "Select every recorded row or stop each episode immediately after its "
            "first done=true row. The choice is recorded in every output artifact."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--std-floor", type=float, default=1e-3)
    parser.add_argument("--expected-episodes", type=int, default=150)
    parser.add_argument("--expected-state-dim", type=int, default=36)
    parser.add_argument("--expected-action-dim", type=int, default=16)
    parser.add_argument("--expected-task-id", default="lift_barrier")
    parser.add_argument(
        "--expected-camera",
        action="append",
        default=None,
        help="Expected exported camera name; repeat to bind multiple cameras.",
    )
    parser.add_argument("--expected-fps", type=float, default=20.0)
    parser.add_argument(
        "--action-codec",
        type=Path,
        default=ROOT / "configs/action_codecs/liftbarrier_pd_joint_pos_16d.json",
        help=(
            "Affine raw-controller/canonical action contract. The default binds "
            "the dual-Panda pd_joint_pos simulator bounds."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing training artifacts.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_episodes <= 0:
        raise ValueError("--expected-episodes must be positive")
    if args.expected_state_dim <= 0 or args.expected_action_dim <= 0:
        raise ValueError("expected state/action dimensions must be positive")
    expected_cameras = tuple(args.expected_camera or ("global",))
    progress = None if args.no_progress else _print_progress
    artifacts = prepare_robofactory_m1_training_artifacts(
        args.dataset_dir,
        transition_selection=args.transition_selection,
        conversion_manifest_path=args.conversion_manifest,
        training_manifest_path=args.training_manifest,
        normalization_path=args.normalization,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        std_floor=args.std_floor,
        expected_episodes=args.expected_episodes,
        expected_state_dim=args.expected_state_dim,
        expected_action_dim=args.expected_action_dim,
        expected_task_id=args.expected_task_id,
        expected_cameras=expected_cameras,
        expected_fps=args.expected_fps,
        action_codec=args.action_codec,
        overwrite=args.overwrite,
        progress=progress,
    )
    manifest = artifacts.manifest
    print(
        json.dumps(
            {
                "passed": True,
                "training_manifest": str(artifacts.manifest_path),
                "training_manifest_sha256": artifacts.manifest_sha256,
                "normalization": str(artifacts.normalization_path),
                "normalization_file_sha256": (
                    artifacts.normalization_file_sha256
                ),
                "normalization_semantic_sha256": (
                    artifacts.normalization_semantic_sha256
                ),
                "transition_selection": manifest["transition_selection"],
                "split_counts": manifest["split_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _print_progress(value: Mapping[str, Any]) -> None:
    current = int(value["current"])
    total = int(value["total"])
    if current == 1 or current == total or current % 10 == 0:
        print(
            f"[{value['phase']}] {current}/{total} {value['path']}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
