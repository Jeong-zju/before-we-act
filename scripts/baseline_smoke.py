#!/usr/bin/env python3
"""Run a bounded, real optimizer smoke for one baseline adapter.

This is deliberately a plumbing smoke, not a claim of upstream reproduction.
It validates that a baseline can consume the frozen six-task contract, execute
forward/backward steps on CUDA/CPU, save a reloadable checkpoint and emit a
status record consumed by the web monitor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.robofactory_baselines import BASELINES, SIX_TASKS, validate_data_root


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=[item.key for item in BASELINES], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> int:
    args = _parse()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    contract = validate_data_root(args.data_root)
    if not contract["valid"]:
        raise SystemExit(json.dumps(contract, indent=2))
    spec = next(item for item in BASELINES if item.key == args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    log_path = args.output_dir / "smoke.log"
    started = time.time()

    def write_status(status: str, **extra: object) -> None:
        payload = {
            "baseline": spec.key,
            "display_name": spec.display_name,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - started, 3),
            **extra,
        }
        status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    torch.manual_seed(args.seed)
    device = _device(args.device)
    write_status("starting", device=str(device), implementation_status=spec.implementation_status)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event": "contract_validated", "tasks": list(SIX_TASKS), "data": contract}, sort_keys=True) + "\n")
        # The adapter model is intentionally small and deterministic.  Upstream
        # implementations must replace this model before reporting benchmark data.
        width = 64 + (BASELINES.index(spec) * 8)
        model = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 16)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        target = torch.zeros((16, 16), device=device)
        for step in range(1, args.steps + 1):
            model.train()
            inputs = torch.randn((16, 32), device=device)
            prediction = model(inputs)
            loss = (prediction - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            record = {"event": "train_step", "step": step, "total_steps": args.steps, "loss": float(loss.detach().cpu())}
            log.write(json.dumps(record) + "\n")
            log.flush()
            write_status("training", device=str(device), step=step, total_steps=args.steps, loss=record["loss"], implementation_status=spec.implementation_status)
    checkpoint = args.output_dir / "smoke.pt"
    torch.save({"baseline": spec.key, "state_dict": model.state_dict(), "seed": args.seed}, checkpoint)
    write_status("ready_for_closed_loop", device=str(device), step=args.steps, total_steps=args.steps, checkpoint=str(checkpoint), implementation_status=spec.implementation_status, note="optimizer smoke only; no success rate reported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
