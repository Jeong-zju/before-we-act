from __future__ import annotations

import json
import os
from pathlib import Path

from .common import POLICY_CONTRACT, TEMPORAL_CONTRACT, atomic_json, sha256_file


def main():
    run = Path(os.environ.get("DUO_DP_RUN", "/workspace/runs/duobench-dp"))
    training = json.loads((run / "formal" / "status.json").read_text())
    validation = json.loads((run / "formal" / "validation20" / "summary.json").read_text())
    checkpoint = Path(training["checkpoint"])
    report = {
        "schema": "duobench.dp.final-report.v1",
        "status": "complete",
        "baseline": "Diffusion Policy",
        "benchmark": "DuoBench",
        "training": training,
        "validation20": validation,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_contract": POLICY_CONTRACT,
        "temporal_contract": TEMPORAL_CONTRACT,
        "all_550_demonstrations_no_split": True,
        "fixed_final_checkpoint_no_validation_selection": True,
    }
    atomic_json(run / "final_report.json", report)
    print(json.dumps({"status": "complete", "successes": validation["successes"], "episodes": 220}))


if __name__ == "__main__":
    main()
