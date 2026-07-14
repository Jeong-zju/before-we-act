"""Run the complete Research-v2 DAG with a hardware-aware training profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.research_v2 import ResearchV2Dataset  # noqa: E402
from train.research_v2_checkpoint import (  # noqa: E402
    load_research_v2_checkpoint,
    sha256_file,
    write_runtime_bundle_manifest,
)
from train.train_research_v2 import ResearchV2TrainingConfig, train_research_v2_stage  # noqa: E402


STAGES = ("plan", "belief", "world_direct", "world_block", "proposal", "intention", "calibration")

# These are microbatch sizes.  The block world uses accumulation=2, giving an
# effective batch of 384 without making matched-branch batches spike above the
# 32 GB VRAM budget.  Step caps matter more than nominal epochs on D1's
# multi-million overlapping windows.
PROFILES = {
    "rtx5090": {
        "stride": 2,
        "num_workers": 8,
        "prefetch_factor": 2,
        "max_validation_steps": 100,
        "patience": 5,
        "ensemble_size": 3,
        "communication_price": 0.05,
        "stages": {
            "plan": (2048, 10, 1000, 1),
            "belief": (1024, 12, 1250, 1),
            "world_direct": (256, 5, 1000, 2),
            "world_block": (192, 15, 2000, 2),
            "proposal": (1024, 10, 1000, 1),
            "intention": (1024, 12, 1000, 1),
            "calibration": (1024, 1, 100, 1),
        },
    },
    "balanced": {
        "stride": 1,
        "num_workers": 4,
        "prefetch_factor": 2,
        "max_validation_steps": 100,
        "patience": 5,
        "ensemble_size": 1,
        "communication_price": 0.05,
        "stages": {
            "plan": (512, 10, 1000, 1),
            "belief": (256, 10, 1000, 1),
            "world_direct": (64, 5, 750, 2),
            "world_block": (48, 12, 1500, 2),
            "proposal": (256, 10, 750, 1),
            "intention": (256, 10, 750, 1),
            "calibration": (256, 1, 100, 1),
        },
    },
}


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.resume and args.force_retrain:
        raise ValueError("--resume and --force-retrain are mutually exclusive")
    profile = PROFILES[args.profile]
    smoke = bool(args.smoke)
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run collect_research_v2_dataset.py first"
        )
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_dataset_manifest_for_training(dataset_manifest, smoke=smoke)
    for split in ("train", "val"):
        split_dir = dataset_root / split
        if not split_dir.is_dir() or not any(split_dir.glob("episode_*.hdf5")):
            raise FileNotFoundError(f"Research-v2 {split} split is empty: {split_dir}")

    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stride = 1 if smoke else _value(args.stride, profile["stride"])
    _status(
        f"[research-v2] profile={args.profile}, device={args.device}, "
        f"precision={'fp32' if smoke else args.precision}, stride={stride}, "
        f"resume={bool(args.resume)}"
    )
    _status(f"[research-v2] indexing train/validation splits under {dataset_root}")
    # Index six-million-window D1 only once; every stage reuses these immutable
    # dataset objects and each worker owns its own bounded HDF5 cache.
    train_data = ResearchV2Dataset(
        dataset_root / "train",
        history=args.history,
        horizon=args.horizon,
        stride=stride,
    )
    val_data = ResearchV2Dataset(
        dataset_root / "val",
        history=args.history,
        horizon=args.horizon,
        stride=stride,
    )
    _status(
        f"[research-v2] indexed {len(train_data):,} train and "
        f"{len(val_data):,} validation windows"
    )

    outputs: dict[str, Path] = {}
    reports: dict[str, dict] = {}
    ensemble_members: list[Path] = []

    def train(stage: str, destination: Path, seed: int) -> Path:
        batch, epochs, max_steps, accumulation = _stage_values(args, profile, stage)
        if smoke:
            batch, epochs, max_steps, accumulation = 4, 1, 1, 1
        config = ResearchV2TrainingConfig(
            stage=stage,
            train_dir=str(dataset_root / "train"),
            val_dir=str(dataset_root / "val"),
            output_dir=str(destination),
            plan_checkpoint=str(outputs["plan"]) if "plan" in outputs else None,
            belief_checkpoint=str(outputs["belief"]) if "belief" in outputs else None,
            world_block_checkpoint=(
                str(outputs["world_block"]) if "world_block" in outputs else None
            ),
            world_ensemble_checkpoints=(
                tuple(str(path) for path in ensemble_members)
                if stage == "calibration"
                else ()
            ),
            intention_checkpoint=(
                str(outputs["intention"]) if "intention" in outputs else None
            ),
            history=args.history,
            horizon=args.horizon,
            stride=stride,
            batch_size=batch,
            epochs=epochs,
            max_steps_per_epoch=max_steps,
            max_validation_steps=(
                1
                if smoke
                else _value(args.max_validation_steps, profile["max_validation_steps"])
            ),
            patience=1 if smoke else _value(args.patience, profile["patience"]),
            relative_min_delta=args.relative_min_delta,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            num_workers=(
                0 if smoke else _value(args.num_workers, profile["num_workers"])
            ),
            pin_memory=False if smoke else args.pin_memory,
            persistent_workers=False if smoke else args.persistent_workers,
            prefetch_factor=_value(args.prefetch_factor, profile["prefetch_factor"]),
            gradient_accumulation_steps=accumulation,
            precision="fp32" if smoke else args.precision,
            tf32=bool(args.tf32),
            compile_model=bool(args.compile and not smoke),
            compile_mode=args.compile_mode,
            fused_optimizer=bool(args.fused_optimizer),
            statistics_max_steps=1 if smoke else args.statistics_max_steps,
            support_max_steps=1 if smoke else args.support_max_steps,
            communication_price=_value(
                args.communication_price, profile["communication_price"]
            ),
            seed=seed,
            device="cpu" if smoke and args.device == "auto" else args.device,
            smoke=smoke,
            resume=bool(args.resume),
            force_retrain=bool(args.force_retrain),
        )
        _status(
            f"[research-v2:{stage}] seed={seed}, batch={batch}, "
            f"accumulation={accumulation}, epochs={epochs}, "
            f"max_steps={max_steps}, output={destination}"
        )
        best = train_research_v2_stage(
            config, train_data=train_data, val_data=val_data
        )
        _status(f"[research-v2:{stage}] completed: {best}")
        return best

    for stage in ("plan", "belief", "world_direct"):
        outputs[stage] = train(stage, output_root / stage, args.seed)
        reports[stage] = _artifact_report(outputs[stage])

    ensemble_size = (
        _value(args.ensemble_size, 1)
        if smoke
        else _value(args.ensemble_size, profile["ensemble_size"])
    )
    if ensemble_size <= 0:
        raise ValueError("--ensemble-size must be positive")
    for member in range(ensemble_size):
        seed = args.seed + member
        destination = output_root / "world_block" / f"member_{member:02d}_seed_{seed}"
        member_path = train("world_block", destination, seed)
        ensemble_members.append(member_path)
        if member == 0:
            outputs["world_block"] = member_path
    ensemble_hashes = [sha256_file(path) for path in ensemble_members]
    if len(set(ensemble_hashes)) != len(ensemble_hashes):
        raise RuntimeError(
            "world ensemble contains duplicate checkpoints; independent seeds "
            "must produce distinct members"
        )
    reports["world_block"] = {
        **_artifact_report(outputs["world_block"]),
        "members": [_artifact_report(path) for path in ensemble_members],
    }

    for stage in ("proposal", "intention", "calibration"):
        outputs[stage] = train(stage, output_root / stage, args.seed)
        reports[stage] = _artifact_report(outputs[stage])

    parameter_counts = {}
    for name in ("plan", "belief", "proposal", "intention"):
        state = load_research_v2_checkpoint(outputs[name])
        parameter_counts[name] = sum(
            value.numel() for value in state["model_state_dict"].values()
        )
    world_state = load_research_v2_checkpoint(outputs["world_block"])
    world_parameters = sum(
        value.numel() for value in world_state["model_state_dict"].values()
    )
    parameter_counts["world_block_per_member"] = world_parameters
    parameter_counts["world_block_ensemble"] = world_parameters * ensemble_size
    parameter_counts["total"] = (
        sum(parameter_counts[name] for name in ("plan", "belief", "proposal", "intention"))
        + parameter_counts["world_block_ensemble"]
    )
    bundle = write_runtime_bundle_manifest(
        output_root / "runtime_bundle",
        {
            "plan": outputs["plan"],
            "belief": outputs["belief"],
            "proposal": outputs["proposal"],
            "intention": outputs["intention"],
            "world_block": outputs["world_block"],
            "calibration": outputs["calibration"],
        },
        ensemble_members=ensemble_members,
        parameter_counts=parameter_counts,
    )
    manifest = {
        "pipeline_contract": "fe_pc_wam/research_v2_training_dag",
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "profile": args.profile,
        "stride": stride,
        "stages": reports,
        "world_ensemble_size": ensemble_size,
        "world_ensemble_seeds": [args.seed + index for index in range(ensemble_size)],
        "smoke": smoke,
        "test_split_read": False,
        "runtime_bundle": {"path": str(bundle), "sha256": sha256_file(bundle)},
    }
    destination = output_root / "pipeline_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return manifest


def _stage_values(args, profile, stage):
    batch, epochs, max_steps, accumulation = profile["stages"][stage]
    family = "world" if stage.startswith("world_") else (
        "distribution" if stage in {"proposal", "intention"} else stage
    )
    batch = _value(getattr(args, f"{family}_batch_size", None), batch)
    epochs = _value(getattr(args, f"{stage}_epochs", None), epochs)
    max_steps = _value(getattr(args, f"{stage}_max_steps", None), max_steps)
    accumulation = _value(
        getattr(args, f"{family}_gradient_accumulation", None), accumulation
    )
    return (
        _value(args.batch_size, batch),
        _value(args.epochs, epochs),
        _value(args.max_steps_per_epoch, max_steps),
        _value(args.gradient_accumulation, accumulation),
    )


def _value(value, default):
    return default if value is None else value


def _artifact_report(path: Path) -> dict:
    state = load_research_v2_checkpoint(path)
    return {
        "best": str(path),
        "sha256": sha256_file(path),
        "metrics": state.get("metrics", {}),
    }


def _validate_dataset_manifest_for_training(
    manifest: dict, *, smoke: bool
) -> None:
    """Fail before indexing if formal data lacks private-event supervision."""

    if smoke:
        return
    scenario_mixture = manifest.get("formal_scenario_mixture", [])
    problems: list[str] = []
    if "private_gates" not in scenario_mixture:
        problems.append("formal_scenario_mixture lacks private_gates")
    split_reports = manifest.get("splits", {})
    for split in ("train", "val"):
        report = split_reports.get(split, {})
        quality = report.get("private_event_quality", {})
        episodes = int(report.get("episodes", 0))
        minimum_private_episodes = max(1, (episodes + 19) // 20)
        if int(quality.get("private_gate_episodes", 0)) < minimum_private_episodes:
            problems.append(f"{split} has insufficient private_gates episodes")
        for field in (
            "event_type_observation_counts",
            "informed_agent_observation_counts",
            "maneuver_observation_counts",
        ):
            counts = quality.get(field, {})
            if not counts or any(int(value) <= 0 for value in counts.values()):
                problems.append(f"{split}.{field} lacks required classes")
        if int(quality.get("active_observations", 0)) <= 0:
            problems.append(f"{split} has no active private-gate observations")
        if int(quality.get("cued_agent_observations", 0)) <= 0:
            problems.append(f"{split} has no private-event cues")
    if problems:
        raise ValueError(
            "Research-v2 dataset is not valid for formal training: "
            + "; ".join(problems)
            + ". Recollect into a new dataset root with the current collector."
        )


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Research-v2; defaults are tuned for one RTX 5090 (32 GB)"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="rtx5090")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--plan-batch-size", type=int)
    parser.add_argument("--belief-batch-size", type=int)
    parser.add_argument("--world-batch-size", type=int)
    parser.add_argument("--distribution-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    for stage in STAGES:
        parser.add_argument(f"--{stage.replace('_', '-')}-epochs", dest=f"{stage}_epochs", type=int)
        parser.add_argument(
            f"--{stage.replace('_', '-')}-max-steps",
            dest=f"{stage}_max_steps",
            type=int,
        )
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--world-gradient-accumulation", type=int)
    parser.add_argument("--distribution-gradient-accumulation", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--relative-min-delta", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--statistics-max-steps", type=int, default=256)
    parser.add_argument("--support-max-steps", type=int, default=256)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument(
        "--communication-price",
        type=float,
        help="fixed VPI request threshold (profile default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_pipeline(args), indent=2))


if __name__ == "__main__":
    main()
