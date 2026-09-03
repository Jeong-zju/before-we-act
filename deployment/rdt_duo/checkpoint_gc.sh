#!/bin/bash
set -euo pipefail

# Keep only the two newest completed checkpoints.  The upstream RDT launcher
# passes total_limit without enabling Accelerate's automatic naming, so its
# checkpoints would otherwise consume the whole instance disk.
root="${RDT_DUO_CHECKPOINT_DIR:-/workspace/runs/rdt_duo/formal/checkpoints}"
keep="${RDT_DUO_CHECKPOINT_KEEP:-2}"
interval="${RDT_DUO_CHECKPOINT_GC_INTERVAL:-120}"

while :; do
  /venv/main/bin/python - "$root" "$keep" <<'PY'
import pathlib, re, shutil, sys, time
root = pathlib.Path(sys.argv[1]); keep = int(sys.argv[2])
if not root.is_dir():
    raise SystemExit(0)
items = []
for p in root.iterdir():
    m = re.fullmatch(r"checkpoint-(\d+)", p.name)
    if p.is_dir() and m:
        items.append((int(m.group(1)), p))
items.sort()
now = time.time()
# Never remove the newest two, and allow ten minutes for a checkpoint write to
# finish before considering an older directory for deletion.
for step, p in items[:-keep]:
    try:
        if now - p.stat().st_mtime < 600:
            continue
        shutil.rmtree(p)
    except FileNotFoundError:
        pass
PY
  sleep "$interval"
done
