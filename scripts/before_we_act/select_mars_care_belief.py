#!/usr/bin/env python3
"""Select the best official belief seed without changing its deployment weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from before_we_act.care_training_data import atomic_json, sha256_file


SEEDS = (20260815, 20260816, 20260817)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.output_root / "selection_receipt.json"
    deployment = args.output_root / "deployment_checkpoint.pt"
    if receipt.exists() and deployment.exists():
        return
    rows = []
    for seed in SEEDS:
        root = args.training_root / f"seed_{seed}"
        status = json.loads((root / "status.json").read_text())
        if status.get("status") != "COMPLETE" or int(status.get("seed", -1)) != seed:
            raise RuntimeError(f"incomplete MARS belief seed: {root}")
        checkpoint = root / "deployment_checkpoint.pt"
        rows.append((float(status["selected_validation"]["macro"]["b_core"]), seed, checkpoint, status))
    _, seed, source, status = min(rows, key=lambda row: (row[0], row[1]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = deployment.with_name(f".{deployment.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, deployment)
    atomic_json(
        receipt,
        {
            "format_version": "before-we-act.mars-belief-selection/1",
            "status": "PASSED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_seed": seed,
            "selected_update": int(status["selected_update"]),
            "selection_metric": "diagnostic macro.b_core",
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": sha256_file(source),
            "deployment_checkpoint": str(deployment.resolve()),
            "deployment_checkpoint_sha256": sha256_file(deployment),
        },
    )


if __name__ == "__main__":
    main()
