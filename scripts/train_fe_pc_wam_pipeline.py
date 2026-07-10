from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    epochs: int
    batch_size: int
    num_workers: int
    max_train_episodes: int
    max_val_episodes: int
    max_val_batches: int
    eval_max_batches: int
    policy_episodes: int
    policy_render_video: int
    save_every: int
    wam_model_dim: int
    wam_layers: int
    wam_heads: int
    wam_ffn_dim: int
    wam_batch_size: int
    wam_grad_accum: int
    intention_model_dim: int
    intention_layers: int
    intention_heads: int
    intention_ffn_dim: int


PROFILES = {
    "smoke": Profile(
        epochs=1,
        batch_size=16,
        num_workers=0,
        max_train_episodes=6,
        max_val_episodes=2,
        max_val_batches=1,
        eval_max_batches=1,
        policy_episodes=1,
        policy_render_video=0,
        save_every=1,
        wam_model_dim=128,
        wam_layers=2,
        wam_heads=4,
        wam_ffn_dim=512,
        wam_batch_size=2,
        wam_grad_accum=1,
        intention_model_dim=128,
        intention_layers=2,
        intention_heads=4,
        intention_ffn_dim=512,
    ),
    "full": Profile(
        epochs=100,
        batch_size=256,
        num_workers=4,
        max_train_episodes=-1,
        max_val_episodes=-1,
        max_val_batches=-1,
        eval_max_batches=50,
        policy_episodes=20,
        policy_render_video=0,
        save_every=10,
        wam_model_dim=1024,
        wam_layers=16,
        wam_heads=16,
        wam_ffn_dim=4096,
        wam_batch_size=16,
        wam_grad_accum=4,
        intention_model_dim=512,
        intention_layers=8,
        intention_heads=8,
        intention_ffn_dim=2048,
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def command_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not current else f"{root}{os.pathsep}{current}"
    return env


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def run_command(root: Path, command: list[str], dry_run: bool = False):
    printable = " ".join(command)
    print("+", printable)
    if dry_run:
        return
    subprocess.run(command, cwd=root, env=command_env(root), check=True)


def run_python(root: Path, args: list[str], dry_run: bool = False):
    run_command(root, [sys.executable, *args], dry_run=dry_run)


def marker_path(artifacts_dir: Path, stage: str) -> Path:
    return artifacts_dir / "pipeline" / f"{stage}.done.json"


def is_stage_done(artifacts_dir: Path, stage: str, outputs: list[Path], resume: bool) -> bool:
    if not resume:
        return False
    marker = marker_path(artifacts_dir, stage)
    return marker.exists() and all(path.exists() for path in outputs)


def write_marker(artifacts_dir: Path, stage: str, outputs: list[Path], dry_run: bool = False):
    if dry_run:
        return
    marker = marker_path(artifacts_dir, stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w") as f:
        json.dump(
            {
                "stage": stage,
                "completed_at_utc": utc_now(),
                "outputs": [str(path) for path in outputs],
            },
            f,
            indent=2,
        )


def maybe_resume_arg(out_dir: Path, resume: bool) -> list[str]:
    last = out_dir / "last.pt"
    if resume and last.exists():
        return ["--resume", str(last)]
    return []


def export_checkpoint(src_dir: Path, dst: Path, dry_run: bool = False):
    src = src_dir / "best.pt"
    if not src.exists():
        src = src_dir / "last.pt"
    if dry_run:
        print(f"export: {src_dir}/best.pt or last.pt -> {dst}")
        return
    if not src.exists():
        raise FileNotFoundError(f"No best.pt or last.pt found in {src_dir}")
    print(f"export: {src} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def validate_dataset_stage(root: Path, artifacts_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path, resume: bool, dry_run: bool):
    stage = "dataset_check"
    outputs = [marker_path(artifacts_dir, stage)]
    if is_stage_done(artifacts_dir, stage, outputs, resume):
        print(f"skip {stage}: already complete")
        return
    for data_dir in [train_dir, val_dir, test_dir]:
        run_python(root, ["data/validate_dataset.py", "--data_dir", rel(data_dir, root)], dry_run=dry_run)
    write_marker(artifacts_dir, stage, outputs, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Run FE-PC-WAM staged training and inference artifact export.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--val_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--test_dir", type=str, default="datasets/stage2/test")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--plan_codebook_size", type=int, default=64)
    parser.add_argument("--plan_latent_dim", type=int, default=64)
    parser.add_argument("--skip_policy_eval", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    profile = PROFILES[args.profile]
    train_dir = (root / args.train_dir).resolve()
    val_dir = (root / args.val_dir).resolve()
    test_dir = (root / args.test_dir).resolve()
    checkpoints_dir = (root / args.checkpoints_dir).resolve()
    artifacts_dir = (root / args.artifacts_dir).resolve()
    outputs_dir = (root / args.outputs_dir).resolve()

    plan_ckpt = artifacts_dir / "plan_tokenizer" / "plan_tokenizer.pt"
    slot_ckpt = artifacts_dir / "slot_encoder" / "slot_encoder.pt"
    wam_ckpt = artifacts_dir / "wam" / "wam.pt"
    intention_ckpt = artifacts_dir / "intention" / "intention.pt"

    validate_dataset_stage(root, artifacts_dir, train_dir, val_dir, test_dir, args.resume, args.dry_run)

    stage = "plan_tokenizer"
    out_dir = checkpoints_dir / stage
    outputs = [plan_ckpt]
    if is_stage_done(artifacts_dir, stage, outputs, args.resume):
        print(f"skip {stage}: already complete")
    else:
        run_python(
            root,
            [
                "train/train_plan_tokenizer.py",
                "--train_dir",
                rel(train_dir, root),
                "--val_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(out_dir, root),
                "--horizon",
                str(args.horizon),
                "--codebook_size",
                str(args.plan_codebook_size),
                "--latent_dim",
                str(args.plan_latent_dim),
                "--batch_size",
                str(profile.batch_size),
                "--epochs",
                str(profile.epochs),
                "--num_workers",
                str(profile.num_workers),
                "--max_train_episodes",
                str(profile.max_train_episodes),
                "--max_val_episodes",
                str(profile.max_val_episodes),
                "--save_every",
                str(profile.save_every),
                *maybe_resume_arg(out_dir, args.resume),
            ],
            dry_run=args.dry_run,
        )
        export_checkpoint(out_dir, plan_ckpt, dry_run=args.dry_run)
        run_python(
            root,
            [
                "eval/evaluate_tokenizer.py",
                "--ckpt",
                rel(plan_ckpt, root),
                "--data_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(plan_ckpt.parent, root),
                "--batch_size",
                str(profile.batch_size),
                "--num_workers",
                str(profile.num_workers),
                "--max_batches",
                str(profile.eval_max_batches),
            ],
            dry_run=args.dry_run,
        )
        write_marker(artifacts_dir, stage, outputs, dry_run=args.dry_run)

    stage = "slot_encoder"
    out_dir = checkpoints_dir / stage
    outputs = [slot_ckpt]
    if is_stage_done(artifacts_dir, stage, outputs, args.resume):
        print(f"skip {stage}: already complete")
    else:
        run_python(
            root,
            [
                "train/train_slot_encoder.py",
                "--train_dir",
                rel(train_dir, root),
                "--val_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(out_dir, root),
                "--tokenizer_ckpt",
                rel(plan_ckpt, root),
                "--history",
                str(args.history),
                "--horizon",
                str(args.horizon),
                "--plan_codebook_size",
                str(args.plan_codebook_size),
                "--batch_size",
                str(profile.batch_size),
                "--epochs",
                str(profile.epochs),
                "--num_workers",
                str(profile.num_workers),
                "--max_train_episodes",
                str(profile.max_train_episodes),
                "--max_val_episodes",
                str(profile.max_val_episodes),
                "--save_every",
                str(profile.save_every),
                *maybe_resume_arg(out_dir, args.resume),
            ],
            dry_run=args.dry_run,
        )
        export_checkpoint(out_dir, slot_ckpt, dry_run=args.dry_run)
        run_python(
            root,
            [
                "eval/evaluate_slots.py",
                "--ckpt",
                rel(slot_ckpt, root),
                "--data_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(slot_ckpt.parent, root),
                "--tokenizer_ckpt",
                rel(plan_ckpt, root),
                "--batch_size",
                str(profile.batch_size),
                "--num_workers",
                str(profile.num_workers),
                "--max_batches",
                str(profile.eval_max_batches),
            ],
            dry_run=args.dry_run,
        )
        write_marker(artifacts_dir, stage, outputs, dry_run=args.dry_run)

    stage = "wam"
    out_dir = checkpoints_dir / stage
    outputs = [wam_ckpt]
    if is_stage_done(artifacts_dir, stage, outputs, args.resume):
        print(f"skip {stage}: already complete")
    else:
        run_python(
            root,
            [
                "train/train_wam.py",
                "--train_dir",
                rel(train_dir, root),
                "--val_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(out_dir, root),
                "--slot_ckpt",
                rel(slot_ckpt, root),
                "--plan_ckpt",
                rel(plan_ckpt, root),
                "--history",
                str(args.history),
                "--horizon",
                str(args.horizon),
                "--plan_codebook_size",
                str(args.plan_codebook_size),
                "--plan_latent_dim",
                str(args.plan_latent_dim),
                "--model_dim",
                str(profile.wam_model_dim),
                "--num_layers",
                str(profile.wam_layers),
                "--num_heads",
                str(profile.wam_heads),
                "--ffn_dim",
                str(profile.wam_ffn_dim),
                "--batch_size",
                str(profile.wam_batch_size),
                "--grad_accum_steps",
                str(profile.wam_grad_accum),
                "--epochs",
                str(profile.epochs),
                "--num_workers",
                str(profile.num_workers),
                "--max_train_episodes",
                str(profile.max_train_episodes),
                "--max_val_episodes",
                str(profile.max_val_episodes),
                "--max_val_batches",
                str(profile.max_val_batches),
                "--save_every",
                str(max(1, min(5, profile.save_every))),
                *maybe_resume_arg(out_dir, args.resume),
            ],
            dry_run=args.dry_run,
        )
        export_checkpoint(out_dir, wam_ckpt, dry_run=args.dry_run)
        run_python(
            root,
            [
                "eval/evaluate_wam.py",
                "--ckpt",
                rel(wam_ckpt, root),
                "--data_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(wam_ckpt.parent, root),
                "--slot_ckpt",
                rel(slot_ckpt, root),
                "--plan_ckpt",
                rel(plan_ckpt, root),
                "--batch_size",
                str(profile.wam_batch_size),
                "--num_workers",
                str(profile.num_workers),
                "--max_batches",
                str(profile.eval_max_batches),
            ],
            dry_run=args.dry_run,
        )
        write_marker(artifacts_dir, stage, outputs, dry_run=args.dry_run)

    stage = "intention"
    out_dir = checkpoints_dir / stage
    outputs = [intention_ckpt]
    if is_stage_done(artifacts_dir, stage, outputs, args.resume):
        print(f"skip {stage}: already complete")
    else:
        run_python(
            root,
            [
                "train/train_intention.py",
                "--train_dir",
                rel(train_dir, root),
                "--val_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(out_dir, root),
                "--slot_ckpt",
                rel(slot_ckpt, root),
                "--plan_ckpt",
                rel(plan_ckpt, root),
                "--wam_ckpt",
                rel(wam_ckpt, root),
                "--history",
                str(args.history),
                "--horizon",
                str(args.horizon),
                "--plan_codebook_size",
                str(args.plan_codebook_size),
                "--plan_latent_dim",
                str(args.plan_latent_dim),
                "--model_dim",
                str(profile.intention_model_dim),
                "--num_layers",
                str(profile.intention_layers),
                "--num_heads",
                str(profile.intention_heads),
                "--ffn_dim",
                str(profile.intention_ffn_dim),
                "--batch_size",
                str(profile.batch_size),
                "--epochs",
                str(profile.epochs),
                "--num_workers",
                str(profile.num_workers),
                "--max_train_episodes",
                str(profile.max_train_episodes),
                "--max_val_episodes",
                str(profile.max_val_episodes),
                "--max_val_batches",
                str(profile.max_val_batches),
                "--save_every",
                str(profile.save_every),
                *maybe_resume_arg(out_dir, args.resume),
            ],
            dry_run=args.dry_run,
        )
        export_checkpoint(out_dir, intention_ckpt, dry_run=args.dry_run)
        run_python(
            root,
            [
                "eval/evaluate_intention.py",
                "--ckpt",
                rel(intention_ckpt, root),
                "--data_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(intention_ckpt.parent, root),
                "--slot_ckpt",
                rel(slot_ckpt, root),
                "--plan_ckpt",
                rel(plan_ckpt, root),
                "--batch_size",
                str(profile.batch_size),
                "--num_workers",
                str(profile.num_workers),
                "--max_batches",
                str(profile.eval_max_batches),
            ],
            dry_run=args.dry_run,
        )
        write_marker(artifacts_dir, stage, outputs, dry_run=args.dry_run)

    stage = "free_energy_eval"
    outputs = [artifacts_dir / "free_energy" / "metrics.json"]
    if is_stage_done(artifacts_dir, stage, outputs, args.resume):
        print(f"skip {stage}: already complete")
    else:
        run_python(
            root,
            [
                "eval/evaluate_free_energy.py",
                "--data_dir",
                rel(val_dir, root),
                "--out_dir",
                rel(artifacts_dir / "free_energy", root),
                "--wam_ckpt",
                rel(wam_ckpt, root),
                "--slot_ckpt",
                rel(slot_ckpt, root),
                "--plan_ckpt",
                rel(plan_ckpt, root),
                "--max_batches",
                str(profile.eval_max_batches),
            ],
            dry_run=args.dry_run,
        )
        write_marker(artifacts_dir, stage, outputs, dry_run=args.dry_run)

    stage = "communication_eval"
    comm_outputs = [
        artifacts_dir / "communication" / "ego_0" / "metrics.json",
        artifacts_dir / "communication" / "ego_1" / "metrics.json",
    ]
    if is_stage_done(artifacts_dir, stage, comm_outputs, args.resume):
        print(f"skip {stage}: already complete")
    else:
        for ego_id in [0, 1]:
            run_python(
                root,
                [
                    "eval/evaluate_communication.py",
                    "--data_dir",
                    rel(val_dir, root),
                    "--out_dir",
                    rel(artifacts_dir / "communication" / f"ego_{ego_id}", root),
                    "--wam_ckpt",
                    rel(wam_ckpt, root),
                    "--slot_ckpt",
                    rel(slot_ckpt, root),
                    "--plan_ckpt",
                    rel(plan_ckpt, root),
                    "--intention_ckpt",
                    rel(intention_ckpt, root),
                    "--ego_id",
                    str(ego_id),
                    "--max_batches",
                    str(profile.eval_max_batches),
                ],
                dry_run=args.dry_run,
            )
        write_marker(artifacts_dir, stage, comm_outputs, dry_run=args.dry_run)

    if not args.skip_policy_eval:
        stage = "policy_rollout_eval"
        policy_root = outputs_dir / "policy_rollouts"
        policy_outputs = [policy_root / mode / "summary.json" for mode in ["no_comm", "always_comm", "selective_comm"]]
        if is_stage_done(artifacts_dir, stage, policy_outputs, args.resume):
            print(f"skip {stage}: already complete")
        else:
            for mode in ["no_comm", "always_comm", "selective_comm"]:
                run_python(
                    root,
                    [
                        "eval/evaluate_policy.py",
                        "--mode",
                        mode,
                        "--out_dir",
                        rel(policy_root / mode, root),
                        "--wam_ckpt",
                        rel(wam_ckpt, root),
                        "--slot_ckpt",
                        rel(slot_ckpt, root),
                        "--plan_ckpt",
                        rel(plan_ckpt, root),
                        "--intention_ckpt",
                        rel(intention_ckpt, root),
                        "--num_episodes",
                        str(profile.policy_episodes),
                        "--render_video",
                        str(profile.policy_render_video),
                    ],
                    dry_run=args.dry_run,
                )
            run_python(
                root,
                [
                    "eval/compare_policies.py",
                    "--root",
                    rel(policy_root, root),
                    "--out_dir",
                    rel(outputs_dir / "policy_reports", root),
                    "--modes",
                    "no_comm,always_comm,selective_comm",
                ],
                dry_run=args.dry_run,
            )
            write_marker(artifacts_dir, stage, policy_outputs, dry_run=args.dry_run)

    print("pipeline complete")


if __name__ == "__main__":
    main()
