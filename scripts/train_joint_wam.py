"""Train Joint WAM end to end: action-flow warm-up, joint coupling, and audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._train_action_flow import main as train_action_flow  # noqa: E402
from scripts._train_joint_coupling import main as train_joint_coupling  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/wam/joint_wam.yaml"
    )
    parser.add_argument("--world-model-checkpoint-dir", type=Path)
    parser.add_argument("--action-prior-checkpoint-dir", type=Path)
    parser.add_argument(
        "--action-flow-checkpoint-dir",
        type=Path,
        help="Reuse an existing warm-up artifact instead of training it.",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--action-flow-batch-size", type=int)
    parser.add_argument("--joint-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-action-flow-steps", type=int, default=-1)
    parser.add_argument("--max-joint-steps", type=int, default=-1)
    parser.add_argument("--max-eval-batches", type=int, default=-1)
    parser.add_argument("--max-episodes-per-split", type=int, default=-1)
    parser.add_argument("--on-policy-rounds", type=int)
    parser.add_argument("--on-policy-episodes-per-suite", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    checkpoint_dir = (
        args.checkpoint_dir or ROOT / config["checkpoint"]["directory"]
    ).resolve()
    generated_warmup = args.action_flow_checkpoint_dir is None
    warmup_dir = (
        args.action_flow_checkpoint_dir
        or checkpoint_dir / ".action_flow_warmup"
    ).resolve()

    common = ["--config", str(args.config.resolve()), "--device", args.device]
    if args.world_model_checkpoint_dir is not None:
        common += [
            "--world-model-checkpoint-dir",
            str(args.world_model_checkpoint_dir.resolve()),
        ]
    if args.action_prior_checkpoint_dir is not None:
        common += [
            "--action-prior-checkpoint-dir",
            str(args.action_prior_checkpoint_dir.resolve()),
        ]
    if args.num_workers is not None:
        common += ["--num-workers", str(args.num_workers)]
    if args.no_progress:
        common.append("--no-progress")

    if generated_warmup:
        warmup_args = [
            *common,
            "--checkpoint-dir",
            str(warmup_dir),
            "--max-steps",
            str(args.max_action_flow_steps),
            "--max-eval-batches",
            str(args.max_eval_batches),
            "--max-episodes-per-split",
            str(args.max_episodes_per_split),
        ]
        if args.action_flow_batch_size is not None:
            warmup_args += ["--batch-size", str(args.action_flow_batch_size)]
        if args.on_policy_rounds is not None:
            warmup_args += ["--on-policy-rounds", str(args.on_policy_rounds)]
        if args.on_policy_episodes_per_suite is not None:
            warmup_args += [
                "--on-policy-episodes-per-suite",
                str(args.on_policy_episodes_per_suite),
            ]
        result = train_action_flow(warmup_args)
        if result:
            return result

    joint_args = [
        *common,
        "--action-flow-checkpoint-dir",
        str(warmup_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--max-steps",
        str(args.max_joint_steps),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--max-episodes-per-split",
        str(args.max_episodes_per_split),
    ]
    if args.joint_batch_size is not None:
        joint_args += ["--batch-size", str(args.joint_batch_size)]
    result = train_joint_coupling(joint_args)
    if result == 0 and generated_warmup:
        shutil.rmtree(warmup_dir, ignore_errors=True)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Joint WAM config root must be a mapping")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
