"""Host/data/codec preflight for Blackwell OpenVLA-OFT MARS run."""
from __future__ import annotations
import json, os, subprocess, shutil, sys
from pathlib import Path
from .download import REPOS, TOKEN_FILE
from datetime import datetime, timezone

def main() -> None:
    if (TOKEN_FILE.stat().st_mode & 0o077) or not TOKEN_FILE.read_text().strip(): raise RuntimeError("HF secret missing or permissions too broad")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
    if len(rows) != 4: raise RuntimeError(f"expected 4 GPUs, found {len(rows)}")
    gpus = []
    for row in rows:
        idx, name, total, free = [x.strip() for x in row.split(",", 3)]
        if "RTX PRO 6000" not in name or int(total) < 90000: raise RuntimeError(f"unexpected GPU: {row}")
        gpus.append({"index": int(idx), "name": name, "memory_total_mib": int(total), "memory_free_mib": int(free)})
    cuda = subprocess.check_output([sys.executable, "-c", "import torch; print(torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] >= 10"], text=True).strip()
    root = Path(os.environ.get("MARS_OPENVLA_DATA_ROOT", "/workspace/datasets/mars_control")); datasets = {}
    for task in REPOS:
        receipt = json.loads((root / task / "download_receipt.json").read_text())
        if receipt.get("status") != "complete" or receipt.get("formal_episodes") != 150 or len(receipt.get("formal_shards", [])) != 10: raise RuntimeError(f"incomplete dataset {task}")
        datasets[task] = {"episodes": 150, "shards": 10, "bytes": sum(x["size_bytes"] for x in receipt["formal_shards"])}
    payload = {"schema": "mars-control.openvla.preflight.v1", "status": "complete", "gpus": gpus, "cuda": cuda,
               "compute_capability": subprocess.check_output([sys.executable, "-c", "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))"], text=True).strip(),
               "python": sys.executable,
               "openvla_commit": subprocess.check_output(["git", "-C", "/workspace/repos/openvla-oft", "rev-parse", "HEAD"], text=True).strip(),
               "datasets": datasets, "episodes": 600, "workspace_free_bytes": shutil.disk_usage("/workspace").free,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    if payload["openvla_commit"] != os.environ.get("OPENVLA_COMMIT", "e4287e94541f459edc4feabc4e181f537cd569a8"): raise RuntimeError("OpenVLA source revision drift")
    out = Path(os.environ.get("MARS_OPENVLA_RUN_ROOT", "/workspace/bwa_mars_openvla_runs")) / "audit/preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
if __name__ == "__main__": main()
