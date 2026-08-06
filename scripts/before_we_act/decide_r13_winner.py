#!/usr/bin/env python3
"""Apply the frozen R13 winner rule without merging any branch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for candidate in ("p0", "p1", "p2", "p3"):
        path = root / "candidates" / candidate / "acceptance.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        worktree = Path(manifest["worktrees"][candidate])
        lock_path = worktree / "experiments" / "before_we_act" / "r13" / candidate / "component_lock.yaml"
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else {}
        checkpoint = root / "candidates" / candidate / "train" / "formal" / "checkpoints" / "checkpoint_010000.pt"
        screen = root / "candidates" / candidate / "validation" / "world_screen.json"
        reasons = [
            item.get("id")
            for item in payload.get("acceptance", [])
            if not item.get("passed")
        ]
        rows.append(
            {
                "candidate_id": candidate,
                "branch": manifest["branches"][candidate],
                "commit": manifest["commits"][candidate],
                "source_commit": lock.get("upstream_commit_sha"),
                "component_lock": str(lock_path),
                "component_patch": str(root / "candidates" / candidate / "receipts" / "upstream_adaptation.patch"),
                "config": str(worktree / "configs" / "before_we_act" / "r13_world" / f"{candidate}.yaml"),
                "data_receipt": str(root / "shared" / "cache.json"),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "report": str(screen),
                "status": payload.get("status", "MISSING"),
                "valid": bool(payload.get("passed")),
                "world_screen_score": payload.get("world_screen_score"),
                "acceptance": str(path),
                "rejection_reasons": reasons,
            }
        )
    valid = [row for row in rows if row["valid"] and row["world_screen_score"] is not None]
    ranked = sorted(valid, key=lambda row: (-float(row["world_screen_score"]), row["candidate_id"]))
    winner = ranked[0]["candidate_id"] if ranked else None
    result = {
        "schema_version": 1,
        "round": "R13",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rule": "highest pre-frozen world_screen_score among valid candidates; lexical candidate_id tie-break",
        "baseline_merge_commit": manifest["parent_commit"],
        "baseline_belief_checkpoint": manifest["belief_checkpoint"],
        "baseline_belief_checkpoint_sha256": sha256(Path(manifest["belief_checkpoint"])),
        "baseline_action_checkpoint": manifest["action_checkpoint"],
        "baseline_action_checkpoint_sha256": sha256(Path(manifest["action_checkpoint"])),
        "candidates": rows,
        "qualified_set": [row["candidate_id"] for row in ranked],
        "ranking": [row["candidate_id"] for row in ranked],
        "winner": winner,
        "unique_winner": winner is not None,
        "merge_performed": False,
        "merge_authorized": False,
        "winner_source_commit": ranked[0]["source_commit"] if ranked else None,
        "winner_checkpoint_sha256": ranked[0]["checkpoint_sha256"] if ranked else None,
        "merge_commit": None,
        "rollback_commit": manifest["parent_commit"],
        "passed": winner is not None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
