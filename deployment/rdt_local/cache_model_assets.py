#!/usr/bin/env python3
"""Download immutable copies of RDT-1B and SigLIP before distributed launch."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "rdt": "robotics-diffusion-transformer/rdt-1b",
    "siglip": "google/siglip-so400m-patch14-384",
}


def main() -> None:
    token = Path("/workspace/.secrets/hf_token").read_text().strip()
    revisions = {}
    api = HfApi(token=token)
    for name, repo in MODELS.items():
        info = api.model_info(repo)
        revisions[name] = info.sha
        snapshot_download(repo_id=repo, revision=info.sha, token=token, max_workers=8)
    output = Path("/workspace/bwa_rdt_runs/audit/model_assets.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump({"schema": "bwa.rdt.model_assets.v1", "status": "complete", "revisions": revisions,
                   "completed_at": datetime.now(timezone.utc).isoformat()}, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
