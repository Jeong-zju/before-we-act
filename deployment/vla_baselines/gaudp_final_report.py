#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    root = Path("/workspace/bwa_gau_dp_runs/formal/gaudp")
    checkpoint = root / "final.ckpt"
    validation = json.loads((root / "validation20" / "summary.json").read_text())
    if validation.get("status") != "complete" or validation.get("total_episodes") != 120:
        raise RuntimeError("Validation20 is incomplete")
    payload = {
        "schema": "bwa.gaudp.formal_run.v1", "status": "complete", "baseline": "GauDP",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "validation20": validation, "dataset": "/workspace/datasets/robofactory_multitask",
        "episodes": 900, "policy_contract": "shared_weights_decentralized_local_rgb_qpos_to_local_action8",
        "gaudp_adaptation": "single-view local NoPoSplat self-mode; no peer/global observation",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    out = root / "final_report.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
