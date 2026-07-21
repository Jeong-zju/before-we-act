"""Evaluate the formal Phase M1 H=8 future-latent linear-probe gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.m1_future_probe import (  # noqa: E402
    FORMAL_EVENT_SAMPLES,
    FORMAL_OBJECT_SAMPLES,
    run_future_probe,
)


CANONICAL_CONFIG = ROOT / "configs/wam_multimodal/m1_latent_wam_dinov3.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=24)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-object-samples", type=int)
    parser.add_argument("--max-event-samples", type=int)
    parser.add_argument("--skip-hdf5-hashes", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve(strict=True)
    config = _load_yaml(config_path)
    canonical_output = (
        ROOT / str(config["evaluation"]["output_directory"]) / "future_probe.json"
    ).resolve()
    output = (args.output or canonical_output).resolve()
    formal = _formal_request(args, config_path, canonical_output, output)
    if not formal and output == canonical_output:
        raise ValueError("diagnostic probe overrides require a separate --output path")
    if args.batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("batch-size and torch-threads must be positive")
    for value in (args.max_object_samples, args.max_event_samples):
        if value is not None and value <= 0:
            raise ValueError("diagnostic sample caps must be positive")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))
    device = _device(args.device)
    object_samples = dict(FORMAL_OBJECT_SAMPLES)
    event_samples = dict(FORMAL_EVENT_SAMPLES)
    if args.max_object_samples is not None:
        object_samples = {
            split: min(count, args.max_object_samples)
            for split, count in object_samples.items()
        }
    if args.max_event_samples is not None:
        event_samples = {
            split: min(count, args.max_event_samples)
            for split, count in event_samples.items()
        }

    report = run_future_probe(
        config_path,
        project_root=ROOT,
        checkpoint_root=args.checkpoint_root,
        training_summary_path=args.training_summary,
        manifest_path=args.manifest,
        train_seeds=args.seeds,
        device=device,
        batch_size=args.batch_size,
        verify_hdf5_sha256=not args.skip_hdf5_hashes,
        object_samples=object_samples,
        event_samples=event_samples,
        formal_protocol=formal,
    )
    _write_json(output, report)
    print(
        "M1 future probe: "
        f"formal={report['formal_protocol']} passed={report['passed']} "
        f"object_delta={report['comparisons']['object']['baseline_minus_model_rmse']:.6f} "
        "event_delta="
        f"{report['comparisons']['event']['model_minus_baseline_balanced_accuracy']:.6f}",
        flush=True,
    )
    return 0 if (not formal or report["passed"] is True) else 1


def _formal_request(
    args: argparse.Namespace,
    config_path: Path,
    canonical_output: Path,
    output: Path,
) -> bool:
    return bool(
        config_path == CANONICAL_CONFIG.resolve()
        and args.checkpoint_root is None
        and args.training_summary is None
        and args.manifest is None
        and args.seeds is None
        and args.max_object_samples is None
        and args.max_event_samples is None
        and not args.skip_hdf5_hashes
        and output == canonical_output
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M1 probe config must contain a mapping")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
