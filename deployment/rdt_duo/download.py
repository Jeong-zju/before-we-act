from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from .protocol import FORMAL_DATASET_REVISION

def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--token-file", type=Path, default=Path("/workspace/.secrets/hf_token")); p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(); token = a.token_file.read_text().strip()
    if not token or a.token_file.stat().st_mode & 0o077: raise RuntimeError("HF token missing/permissions are not 0600")
    api = HfApi(token=token); info = api.dataset_info("RobotControlStack/duobench", revision=FORMAL_DATASET_REVISION, files_metadata=True)
    if info.sha != FORMAL_DATASET_REVISION: raise RuntimeError(f"dataset revision drift: {info.sha}")
    a.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="RobotControlStack/duobench", repo_type="dataset", revision=FORMAL_DATASET_REVISION, local_dir=a.output, token=token, max_workers=a.workers)
    missing, mismatches, total = [], [], 0
    for sibling in info.siblings:
        path = a.output / sibling.rfilename
        if not path.is_file(): missing.append(sibling.rfilename); continue
        expected = int(sibling.size or 0); actual = path.stat().st_size; total += actual
        if expected and expected != actual: mismatches.append({"path": sibling.rfilename, "expected": expected, "actual": actual})
    if missing or mismatches: raise RuntimeError(f"incomplete snapshot: missing={missing[:5]}, mismatches={mismatches[:5]}")
    atomic(a.output / "download_receipt.json", {"schema":"duobench.rdt.download.v1", "status":"complete", "repo_id":"RobotControlStack/duobench", "revision":FORMAL_DATASET_REVISION, "files":len(info.siblings), "bytes_total":total, "all_550_episodes_no_split":True})
    print(json.dumps({"status":"complete", "revision":FORMAL_DATASET_REVISION, "files":len(info.siblings), "bytes_total":total}))
if __name__ == "__main__": main()
