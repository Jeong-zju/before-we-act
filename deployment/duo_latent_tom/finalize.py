from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import FROZEN_CONFIG, POLICY_CONTRACT, atomic_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True); args = parser.parse_args()
    status = json.loads((args.run / "formal/status.json").read_text())
    validation = json.loads((args.run / "formal/validation20/summary.json").read_text())
    checkpoint = args.run / "formal/final.pt"
    if status.get("status") != "complete" or status.get("step") != 60000: raise ValueError("formal training is incomplete")
    if validation.get("status") != "complete" or validation.get("total_episodes") != 220: raise ValueError("Validation20 is incomplete")
    report = {"schema": "duobench.latent-tom.final-report.v2", "status": "complete", "policy_contract": POLICY_CONTRACT, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "config": str(FROZEN_CONFIG), "config_sha256": sha256(FROZEN_CONFIG), "training": status, "validation20": {"total_episodes": validation["total_episodes"], "successes": validation["successes"], "macro_success_rate": validation["macro_success_rate"], "diffusion_steps": validation.get("diffusion_steps"), "replan_interval": validation.get("replan_interval"), "summary": str(args.run / "formal/validation20/summary.json"), "summary_sha256": sha256(args.run / "formal/validation20/summary.json")}, "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(args.run / "final_report.json", report); print(json.dumps(report))


if __name__ == "__main__": main()
