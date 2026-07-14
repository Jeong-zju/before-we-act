"""Executable information-contract audit for Research-v2 data and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.research_v2 import audit_research_v2_file  # noqa: E402
from train.research_v2_checkpoint import (  # noqa: E402
    load_research_v2_checkpoint,
    sha256_file,
)


def audit(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    files = sorted(dataset_root.glob("*/*.hdf5"))
    if not files:
        raise FileNotFoundError("no Research-v2 episodes found")
    branch_groups = 0
    for path in files:
        report = audit_research_v2_file(path)
        branch_groups += int(report["branch_groups"])
    checkpoints = {}
    for raw in args.checkpoint:
        path = Path(raw).resolve()
        state = load_research_v2_checkpoint(path)
        forbidden = [
            name
            for name in state["forward_inputs"]
            if "privileged" in name or "truth" in name or "global" in name
        ]
        if forbidden:
            raise ValueError(f"{path} exposes forbidden runtime inputs: {forbidden}")
        checkpoints[state["stage"]] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "forward_inputs": state["forward_inputs"],
        }
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    if bundle.get("privileged_runtime_inputs") != []:
        raise ValueError("bundle contains privileged runtime inputs")
    unsigned_bundle = dict(bundle)
    claimed_bundle_hash = unsigned_bundle.pop("bundle_sha256", None)
    canonical = json.dumps(unsigned_bundle, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != claimed_bundle_hash:
        raise ValueError("bundle manifest hash mismatch")
    for artifact in bundle["artifacts"].values():
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = (Path(args.bundle).resolve().parent / artifact_path).resolve()
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError("bundle artifact hash mismatch")
    return {
        "contract": "fe_pc_wam/research_v2_audit",
        "passed": True,
        "episode_files": len(files),
        "matched_branch_groups": branch_groups,
        "checkpoint_stages": sorted(checkpoints),
        "bundle_sha256": bundle["bundle_sha256"],
        "privileged_runtime_inputs": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
