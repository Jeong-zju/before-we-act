"""Run the complete decentralized FE-PC-WAM training sequence.

Example::

    python scripts/train_fe_pc_wam_pipeline.py \
        --dataset-root datasets/carry --out-dir checkpoints/carry

Use ``--smoke`` to exercise every stage with one tiny optimization step.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_decentralized import (  # noqa: E402
    ProgressReporter,
    TrainingConfig,
    format_duration,
    resolve_device,
    smoke_config,
    train_stage,
)
from train.checkpoint import (  # noqa: E402
    CONTRACT_TAG,
    file_sha256,
    load_checkpoint,
)
from data.schema import SCHEMA_VERSION  # noqa: E402
from data.decentralized_dataset import DecentralizedTransitionDataset  # noqa: E402


STAGE_ORDER = ("plan", "belief", "wam", "intention", "wam_robust")
STAGE_UPSTREAMS = {
    "plan": (),
    "belief": ("plan",),
    "wam": ("plan", "belief"),
    "intention": ("plan", "belief", "wam"),
    "wam_robust": ("plan", "belief", "wam", "intention"),
}


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    pipeline_started = time.monotonic()
    pipeline_progress = ProgressReporter(
        enabled=not args.quiet,
        log_every=args.log_every,
        prefix="[pipeline]",
    )
    dataset_paths = _resolve_dataset_paths(args)
    data_dir = dataset_paths["train"]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        stage: out_dir / f"{stage}.pt"
        for stage in STAGE_ORDER
    }
    completed: dict[str, dict] = {}

    common = dict(
        data_dir=str(data_dir),
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        codebook_size=args.codebook_size,
        plan_latent_dim=args.plan_latent_dim,
        min_code_count=args.min_code_count,
        min_active_codes=args.min_active_codes,
        min_usage_ratio=args.min_usage_ratio,
        strict_codebook_health=args.strict_codebook_health,
    )
    through_index = STAGE_ORDER.index(args.through)
    stage_epochs = {
        stage: getattr(args, f"{stage}_epochs") or args.epochs for stage in STAGE_ORDER
    }
    indexing_started = time.monotonic()
    pipeline_progress.emit(f"indexing dataset={data_dir}")
    try:
        dataset = DecentralizedTransitionDataset(
            data_dir,
            history=args.history,
            horizon=args.horizon,
            stride=args.stride,
            max_episodes=args.max_episodes,
        )
        resolved_device = resolve_device(args.device)
    except Exception as exc:
        pipeline_progress.emit(
            f"failed elapsed={format_duration(time.monotonic() - pipeline_started)} "
            f"error={type(exc).__name__}: {exc}"
        )
        raise
    pipeline_progress.emit(
        f"dataset={data_dir} episodes={len(dataset.paths)} samples={len(dataset)} "
        f"device={resolved_device.type} stages={through_index + 1} "
        f"indexing_elapsed={format_duration(time.monotonic() - indexing_started)}"
    )
    for index, stage in enumerate(STAGE_ORDER):
        if index > through_index:
            break
        stage_progress = pipeline_progress.child(
            f"[stage {index + 1}/{through_index + 1}:{stage}]"
        )
        stage_started = time.monotonic()
        config = TrainingConfig(
            stage=stage,
            output=str(outputs[stage]),
            epochs=stage_epochs[stage],
            plan_checkpoint=str(outputs["plan"]) if stage != "plan" else None,
            belief_checkpoint=(
                str(outputs["belief"])
                if stage in {"wam", "intention", "wam_robust"}
                else None
            ),
            wam_checkpoint=(
                str(outputs["wam"])
                if stage in {"intention", "wam_robust"}
                else None
            ),
            intention_checkpoint=(
                str(outputs["intention"]) if stage == "wam_robust" else None
            ),
            **common,
        )
        if args.smoke:
            config = smoke_config(config)

        stage_progress.emit(
            f"start epochs={config.epochs} batch_size={config.batch_size} "
            f"checkpoint={outputs[stage]}"
        )
        try:
            if args.resume and outputs[stage].is_file():
                checkpoint = load_checkpoint(outputs[stage], expected_stage=stage)
                _validate_resume_lineage(stage, checkpoint, outputs)
                _validate_resume_config(stage, checkpoint, config)
                status = "reused"
                stage_progress.emit(
                    f"reused checkpoint={outputs[stage]} "
                    f"elapsed={format_duration(time.monotonic() - stage_started)}"
                )
            else:
                if outputs[stage].exists() and not args.force_retrain:
                    raise FileExistsError(
                        f"{outputs[stage]} already exists; use --resume to validate/reuse it "
                        "or --force-retrain to replace it"
                    )
                train_stage(config, dataset=dataset, progress=stage_progress)
                checkpoint = load_checkpoint(outputs[stage], expected_stage=stage)
                status = "trained"
                stage_progress.emit(
                    f"completed checkpoint={outputs[stage]} "
                    f"elapsed={format_duration(time.monotonic() - stage_started)}"
                )
        except Exception as exc:
            stage_progress.emit(
                f"failed elapsed={format_duration(time.monotonic() - stage_started)} "
                f"error={type(exc).__name__}: {exc}"
            )
            raise
        completed[stage] = {
            "status": status,
            "path": str(outputs[stage]),
            "sha256": file_sha256(outputs[stage]),
            "metrics": checkpoint["metrics"],
            "training_config": asdict(config),
        }

    manifest = {
        "contract_tag": CONTRACT_TAG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "dataset_root": str(dataset_paths["root"]),
        "validation_data_dir": str(dataset_paths["val"]) if dataset_paths["val"] else None,
        "test_data_dir": str(dataset_paths["test"]) if dataset_paths["test"] else None,
        "smoke": bool(args.smoke),
        "through": args.through,
        "stage_order": list(STAGE_ORDER[: through_index + 1]),
        "stages": completed,
    }
    manifest_path = out_dir / "pipeline_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(manifest_path)
    pipeline_progress.emit(
        f"completed manifest={manifest_path} "
        f"elapsed={format_duration(time.monotonic() - pipeline_started)}"
    )
    return manifest


def _validate_resume_lineage(stage: str, checkpoint: dict, outputs: dict[str, Path]) -> None:
    for upstream in STAGE_UPSTREAMS[stage]:
        reference = checkpoint.get("upstream", {}).get(upstream)
        if not isinstance(reference, dict):
            raise ValueError(
                f"cannot resume {stage}: missing {upstream} upstream reference"
            )
        actual = file_sha256(outputs[upstream])
        if reference.get("sha256") != actual:
            raise ValueError(
                f"cannot resume {stage}: {upstream} checkpoint changed "
                f"({reference.get('sha256')} != {actual})"
            )


def _validate_resume_config(stage: str, checkpoint: dict, config: TrainingConfig) -> None:
    stored = checkpoint.get("training_config")
    if not isinstance(stored, dict):
        raise ValueError(f"cannot resume {stage}: checkpoint has no training_config")
    expected = asdict(config)
    ignored = {
        "output",
        "plan_checkpoint",
        "belief_checkpoint",
        "wam_checkpoint",
        "intention_checkpoint",
        "device",
    }
    mismatches = {}
    for key, expected_value in expected.items():
        if key in ignored:
            continue
        stored_value = stored.get(key)
        if key == "data_dir" and stored_value is not None:
            stored_value = str(Path(stored_value).resolve())
            expected_value = str(Path(expected_value).resolve())
        if stored_value != expected_value:
            mismatches[key] = {"stored": stored_value, "requested": expected_value}
    if mismatches:
        raise ValueError(
            f"cannot resume {stage}: requested configuration differs from checkpoint: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def _resolve_dataset_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    if args.dataset_root:
        root = Path(args.dataset_root).resolve()
        manifest_path = root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path} is missing; collect data with collect_fe_pc_wam_dataset.py"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"dataset manifest schema={manifest.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        train = root / "train"
        val = root / "val" if (root / "val").is_dir() else None
        test = root / "test" if (root / "test").is_dir() else None
    else:
        train = Path(args.data_dir).resolve()
        root = train.parent
        val = None
        test = None
    if not train.is_dir() or not any(train.glob("episode_*.hdf5")):
        raise FileNotFoundError(f"no training episodes found in {train}")
    return {"root": root, "train": train, "val": val, "test": test}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete decentralized FE-PC-WAM training pipeline"
    )
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--dataset-root", help="root containing dataset_manifest.json and train/")
    data.add_argument("--data-dir", help="direct training split (advanced CLI)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--through", choices=STAGE_ORDER, default="wam_robust")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--plan-epochs", type=int)
    parser.add_argument("--belief-epochs", type=int)
    parser.add_argument("--wam-epochs", type=int)
    parser.add_argument("--intention-epochs", type=int)
    parser.add_argument("--wam-robust-epochs", dest="wam_robust_epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--plan-latent-dim", type=int, default=64)
    parser.add_argument("--min-code-count", type=int, default=1)
    parser.add_argument("--min-active-codes", type=int, default=4)
    parser.add_argument("--min-usage-ratio", type=float, default=0.10)
    parser.add_argument("--strict-codebook-health", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="refresh loss/throughput postfix after N optimization steps",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="disable progress logs; final manifest JSON remains on stdout",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="explicitly replace existing stage checkpoints instead of resuming",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse an existing stage only after strict checkpoint validation",
    )
    return parser


def _json_default(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.resume and args.force_retrain:
        raise ValueError("--resume and --force-retrain are mutually exclusive")
    manifest = run_pipeline(args)
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
