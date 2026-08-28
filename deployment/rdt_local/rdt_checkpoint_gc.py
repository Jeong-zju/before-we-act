#!/usr/bin/env python3
"""Keep the newest two complete RDT DeepSpeed checkpoints while training."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

checkpoints = Path(os.environ["RDT_OUTPUT_DIR"])
keep = int(os.environ.get("RDT_CHECKPOINTS_TO_KEEP", "2"))
parent_pid = int(os.environ["RDT_GC_PARENT_PID"])


def parent_alive() -> bool:
    return Path(f"/proc/{parent_pid}").exists()


def complete(path: Path) -> bool:
    return (path / "pytorch_model.bin").is_file() and (path / "ema/model.safetensors").is_file()


while parent_alive():
    rows = sorted((path for path in checkpoints.glob("checkpoint-*")
                   if path.is_dir() and path.name[11:].isdigit() and complete(path)),
                  key=lambda path: int(path.name[11:]))
    for old in rows[:-keep]:
        shutil.rmtree(old)
        print(f"removed {old}", flush=True)
    time.sleep(30)
