#!/usr/bin/env python3
"""Load an RDT checkpoint and prove every RDT-1B tensor remained trainable."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import sys

import torch
sys.path.insert(0, "/workspace/repos/rdt-1b")
from models.rdt_runner import RDTRunner


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    checkpoint = Path(os.environ["RDT_AUDIT_CHECKPOINT"])
    output = Path(os.environ["RDT_AUDIT_OUTPUT"])
    device = torch.device("cuda:0")
    model = RDTRunner.from_pretrained(str(checkpoint)).to(device=device, dtype=torch.bfloat16).eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total != trainable or total < 900_000_000:
        raise RuntimeError(f"checkpoint is not full-parameter RDT-1B: {trainable}/{total}")
    with torch.inference_mode():
        batch = 1
        loss = model(
            lang_tokens=torch.zeros(batch, 4, 4096, dtype=torch.bfloat16, device=device),
            lang_attn_mask=torch.ones(batch, 4, dtype=torch.bool, device=device),
            img_tokens=torch.zeros(batch, 4374, 1152, dtype=torch.bfloat16, device=device),
            state_tokens=torch.zeros(batch, 1, 128, dtype=torch.bfloat16, device=device),
            action_gt=torch.zeros(batch, 64, 128, dtype=torch.bfloat16, device=device),
            action_mask=torch.ones(batch, 1, 128, dtype=torch.bfloat16, device=device),
            ctrl_freqs=torch.full((batch,), 20, dtype=torch.long, device=device),
        )
    if not torch.isfinite(loss):
        raise RuntimeError("checkpoint reload forward loss is non-finite")
    atomic_json(output, {"schema": "bwa.rdt.checkpoint_audit.v1", "status": "complete",
                         "checkpoint": str(checkpoint), "parameters": total, "trainable_parameters": trainable,
                         "forward_loss": float(loss), "completed_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
