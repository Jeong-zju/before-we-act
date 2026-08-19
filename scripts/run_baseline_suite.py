#!/usr/bin/env python3
"""Plan, validate, and run RoboFactory baseline adapter smoke tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.robofactory_baselines import BASELINES, aggregate_validation20, build_contract, validate_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate", "smoke", "aggregate"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline", choices=[item.key for item in BASELINES], action="append")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract = build_contract(data_root=args.data_root, output_root=args.output_root, seed=args.seed)
    (args.output_root / "baseline_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.command == "plan":
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    data_report = validate_data_root(args.data_root)
    (args.output_root / "data_validation.json").write_text(json.dumps(data_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.command == "validate":
        print(json.dumps(data_report, indent=2, sort_keys=True))
        return 0 if data_report["valid"] else 2
    if args.command == "aggregate":
        report = aggregate_validation20(args.output_root / "validation20", args.output_root / "validation20_summary.json")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not data_report["valid"]:
        raise SystemExit(json.dumps(data_report, indent=2))
    selected = args.baseline or [item.key for item in BASELINES]
    for key in selected:
        run_dir = args.output_root / "smoke" / key
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).with_name("baseline_smoke.py")), "--baseline", key, "--data-root", str(args.data_root), "--output-dir", str(run_dir), "--steps", str(args.steps), "--device", args.device, "--seed", str(args.seed)]
        status = {"baseline": key, "status": "launching", "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
        (run_dir / "launcher.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            (run_dir / "status.json").write_text(json.dumps({**status, "status": "failed", "returncode": completed.returncode}, indent=2) + "\n", encoding="utf-8")
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
