#!/usr/bin/env python3
"""Append a read-only pipeline health snapshot every configured interval."""

import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"error": repr(exc), "path": str(path)}


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return {"error": repr(exc)}


def snapshot(supervisor_dir: Path, formal_dir: Path):
    state = load_json(supervisor_dir / "state.json")
    active = load_json(supervisor_dir / "active.json")
    item = {
        "schema": "bwa-vla-progress-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "active": active,
        "gpu": command([
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,"
            "ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
        ]),
        "disk": command(["df", "-B1", str(formal_dir)]),
        "checkpoints": sorted(
            glob.glob(str(formal_dir / "**" / "checkpoint-*"), recursive=True)
        ),
        "receipts": sorted(glob.glob(str(supervisor_dir / "receipts" / "*.json"))),
    }
    log_path = state.get("log") if isinstance(state, dict) else None
    if log_path and os.path.isfile(log_path):
        with open(log_path, "rb") as stream:
            stream.seek(max(0, os.path.getsize(log_path) - 262144))
            tail = stream.read().decode(errors="replace")
        steps = [int(x) for x in re.findall(r"\|\s*(\d+)/300000", tail)]
        lowered = tail.lower()
        item["log_health"] = {
            "path": log_path,
            "age_seconds": round(time.time() - os.path.getmtime(log_path), 3),
            "latest_rdt_step": max(steps) if steps else None,
            "fatal_counts": {
                marker: lowered.count(marker)
                for marker in ("traceback", "cuda out of memory", "nccl error", "runtimeerror")
            },
        }
    pid = active.get("pid") if isinstance(active, dict) else None
    if pid:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            item["process_identity"] = {
                "pid": pid,
                "state": fields[2],
                "start_ticks": int(fields[21]),
                "matches_active_record": int(fields[21]) == active.get("proc_start_ticks"),
            }
        except Exception as exc:
            item["process_identity"] = {"pid": pid, "error": repr(exc)}
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--supervisor-dir", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        record = snapshot(args.supervisor_dir, args.formal_dir)
        with args.output.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
