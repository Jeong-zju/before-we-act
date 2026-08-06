#!/usr/bin/env python3
"""Apply the frozen R13 winner rule without merging any branch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    rows = []
    for candidate in ("p0", "p1", "p2", "p3"):
        path = root / "candidates" / candidate / "acceptance.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        rows.append(
            {
                "candidate_id": candidate,
                "status": payload.get("status", "MISSING"),
                "valid": bool(payload.get("passed")),
                "world_screen_score": payload.get("world_screen_score"),
                "acceptance": str(path),
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
        "candidates": rows,
        "ranking": [row["candidate_id"] for row in ranked],
        "winner": winner,
        "unique_winner": winner is not None,
        "merge_performed": False,
        "merge_authorized": False,
        "passed": winner is not None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
