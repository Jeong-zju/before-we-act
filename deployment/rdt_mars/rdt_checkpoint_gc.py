#!/usr/bin/env python3
"""Bounded retention of complete DeepSpeed checkpoints during RDT training."""
from __future__ import annotations
import os, shutil, time
from pathlib import Path

root = Path(os.environ["RDT_OUTPUT_DIR"])
keep = max(1, int(os.environ.get("RDT_CHECKPOINTS_TO_KEEP", "2")))
parent = int(os.environ["RDT_GC_PARENT_PID"])

def complete(path: Path) -> bool:
    return (path / "pytorch_model.bin").is_file() and (path / "ema/model.safetensors").is_file()

while Path(f"/proc/{parent}").exists():
    rows = sorted((p for p in root.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit() and complete(p)), key=lambda p: int(p.name[11:]))
    for old in rows[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"removed {old}", flush=True)
    time.sleep(30)
