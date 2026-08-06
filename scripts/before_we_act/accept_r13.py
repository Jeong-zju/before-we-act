#!/usr/bin/env python3
"""Authoritative R13 validity acceptance; quality has no hard threshold."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gate(identifier: str, passed: bool, evidence: str, detail: str = "") -> dict:
    return {"id": identifier, "passed": bool(passed), "evidence": evidence, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=("p0", "p1", "p2", "p3"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--dependency", required=True)
    parser.add_argument("--action-effect", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--action-hash", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = read(args.source)
    license_result = read(args.license)
    patch = read(args.patch)
    dependency = read(args.dependency)
    action_effect = read(args.action_effect)
    parity = read(args.parity)
    preflight = read(args.preflight)
    screen = read(args.screen)
    action_hash = read(args.action_hash)
    cache = read(args.cache)
    candidate = args.candidate
    identity_payloads = (source, license_result, patch, parity, preflight, screen, action_hash)
    identity_consistent = all(
        row.get("candidate_id", candidate) == candidate for row in identity_payloads
    )
    checks = [
        gate("official_source_commit_pinned", source.get("passed") and bool(source.get("resolved_commit")), args.source),
        gate("license_verified_and_preserved", license_result.get("passed"), args.license),
        gate("minimal_component_patch_audited", patch.get("passed"), args.patch),
        gate("no_full_repo_runtime_dependency", dependency.get("passed"), args.dependency),
        gate("strictly_off_path_no_planner_or_rerank", action_effect.get("passed") and not action_effect.get("planner_enabled") and not action_effect.get("rerank_enabled"), args.action_effect),
        gate("upstream_component_numerical_parity", parity.get("passed"), args.parity),
        gate("two_update_train_save_strict_restore", preflight.get("passed"), args.preflight),
        gate("future_targets_never_model_inputs", preflight.get("checks", {}).get("future_target_argument_rejected") and cache.get("future_targets_are_model_inputs") is False, f"{args.preflight}; {args.cache}"),
        gate("formal_10000_updates_and_validation", screen.get("checkpoint_update") == 10_000, args.screen),
        gate("frozen_w12_action_hash_exact", action_hash.get("passed") and action_hash.get("action_hash_equal"), args.action_hash),
        gate("candidate_identity_consistent", identity_consistent, "all R13 receipts"),
    ]
    aggregate = screen.get("aggregate", {})
    score = aggregate.get("world_screen_score")
    result = {
        "schema_version": 1,
        "round": "R13",
        "candidate_id": candidate,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "acceptance_rules": {
            "hard_gates": [row["id"] for row in checks],
            "quality_threshold": None,
            "gate20": "N/A only when frozen W12 action hash is equal",
            "winner_rule": "highest pre-frozen world_screen_score among valid candidates",
        },
        "acceptance": checks,
        "world_screen_score": score,
        "screen_metrics": aggregate,
        "optional_diagnostics": screen.get("optional_diagnostics", {}),
        "gate20": "N/A (action hash equal)" if action_hash.get("action_hash_equal") else "REQUIRED",
    }
    result["passed"] = all(row["passed"] for row in checks) and score is not None
    result["status"] = "PASSED" if result["passed"] else "FAILED"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": candidate, "status": result["status"], "score": score}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
