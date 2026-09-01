from __future__ import annotations

import json
import os
from pathlib import Path

from .common import DATASET_REVISION, OPENPI_REVISION, POLICY_CONTRACT, TEMPORAL_CONTRACT, atomic_json, sha256_tree


def main() -> None:
    run = Path(os.environ.get("DUO_PI05_RUN", "/workspace/runs/pi05_duo")); training = json.loads((run / "formal/status.json").read_text()); validation = json.loads((run / "formal/validation20/summary.json").read_text()); checkpoint = Path(training["checkpoint"])
    report = {"schema": "duobench.pi05.final-report.v1", "status": "complete", "benchmark": "DuoBench", "policy": "pi0.5_lora", "source_revisions": {"openpi": OPENPI_REVISION, "duobench": "082a57cdafea9db115029e6fe9e03691e755f93f", "rcs": "4f78aeffae3bc4d0c02e7beab993e5406261dcf6", "dataset": DATASET_REVISION}, "training": training, "validation20": validation, "checkpoint": str(checkpoint), "checkpoint_tree_sha256": sha256_tree(checkpoint), "policy_contract": POLICY_CONTRACT, "temporal_contract": TEMPORAL_CONTRACT, "all_550_sim_demonstrations_no_split": True, "fixed_final_checkpoint_no_validation_selection": True, "model_architecture_unchanged": True}
    atomic_json(run / "final_report.json", report); atomic_json(run / "final_report.sha256.json", {"sha256": sha256_tree(run / "final_report.json")}); print(json.dumps({"status": "complete", "total_episodes": validation["total_episodes"], "successes": validation["successes"]}), flush=True)


if __name__ == "__main__": main()
