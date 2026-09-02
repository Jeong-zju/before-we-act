#!/usr/bin/env python3
"""Atomically update the autonomous CARE-on-MARS pipeline status."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--status")
    parser.add_argument("--detail", default="")
    parser.add_argument("--heartbeat-only", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    previous = {}
    if args.output.exists():
        try:
            previous = json.loads(args.output.read_text())
        except Exception:
            previous = {}
    if args.heartbeat_only:
        if not previous:
            raise RuntimeError("cannot heartbeat a missing pipeline status")
        previous["heartbeat_at_utc"] = now
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
        return
    if not args.stage or not args.status:
        parser.error("--stage and --status are required unless --heartbeat-only is used")
    history = list(previous.get("history", []))
    event = {
        "time_utc": now,
        "stage": args.stage,
        "status": args.status,
        "detail": args.detail,
    }
    if not history or history[-1] != event:
        history.append(event)
    value = {
        "format_version": "before-we-act.care-mars-pipeline-status/1",
        "updated_at_utc": now,
        "heartbeat_at_utc": now,
        "stage": args.stage,
        "status": args.status,
        "detail": args.detail,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
