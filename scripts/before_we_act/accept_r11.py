#!/usr/bin/env python3
"""Authoritative R11 validity acceptance; representation quality has no hard threshold."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(identifier: str, passed: bool, evidence: str, detail: str = "") -> dict:
    return {"id": identifier, "passed": bool(passed), "evidence": evidence, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=("p0", "p1", "p2", "p3"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--dependency", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--action-hash", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, license_result, patch = read(args.source), read(args.license), read(args.patch)
    dependency, parity = read(args.dependency), read(args.parity)
    preflight, screen, action = read(args.preflight), read(args.screen), read(args.action_hash)
    candidate = args.candidate
    candidate_consistent = all(
        payload.get("candidate_id", payload.get("candidate", candidate)) == candidate
        for payload in (source, license_result, patch, dependency, parity, preflight, screen, action)
    )
    checks = [
        check("official_source_commit_pinned", source.get("passed") and bool(source.get("resolved_commit")), args.source),
        check("license_verified_and_preserved", license_result.get("passed"), args.license),
        check("minimal_component_patch_audited", patch.get("passed"), args.patch),
        check("no_full_repo_runtime_dependency", dependency.get("passed"), args.dependency),
        check("upstream_component_parity", parity.get("passed"), args.parity),
        check("two_update_train_save_restore", preflight.get("passed"), args.preflight),
        check("formal_10000_updates_and_validation", screen.get("checkpoint_update") == 10_000, args.screen),
        check("strictly_off_path_action_hash_equal", action.get("passed") and action.get("action_hash_equal"), args.action_hash),
        check("candidate_identity_consistent", candidate_consistent, "all R11 receipts"),
    ]
    aggregate = screen.get("aggregate", {})
    result = {
        "schema_version": 1,
        "round": "R11",
        "candidate_id": candidate,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "acceptance_rules": {
            "hard_gates": [row["id"] for row in checks],
            "quality_threshold": None,
            "gate20": "N/A when action hash is equal",
            "winner_rule": "highest pre-frozen representation_screen_score among valid candidates",
        },
        "acceptance": checks,
        "optional_diagnostics": {
            "future_feature_gain": aggregate.get("future_feature_gain"),
            "partner_action_gain": aggregate.get("partner_action_gain"),
            "shared_progress_r2": aggregate.get("shared_progress_r2"),
            "windows_per_second": aggregate.get("windows_per_second"),
        },
        "representation_screen_score": aggregate.get("representation_screen_score"),
        "gate20": "N/A (action hash equal)" if action.get("action_hash_equal") else "REQUIRED (candidate is action-affecting)",
    }
    result["passed"] = all(row["passed"] for row in checks)
    result["status"] = "PASSED" if result["passed"] else "FAILED"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": candidate, "status": result["status"], "score": result["representation_screen_score"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
