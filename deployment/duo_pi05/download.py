from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .common import DATASET_REVISION, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/workspace/datasets/duobench"))
    parser.add_argument("--token-file", type=Path, default=Path("/workspace/.secrets/hf_token"))
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if not token: raise RuntimeError("empty Hugging Face token file")
    info = HfApi(token=token).dataset_info(
        "RobotControlStack/duobench", revision=DATASET_REVISION, files_metadata=True
    )
    if info.sha != DATASET_REVISION: raise RuntimeError(f"dataset revision drift: {info.sha}")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="RobotControlStack/duobench", repo_type="dataset",
        revision=DATASET_REVISION, local_dir=args.output, token=token,
        max_workers=args.workers,
    )
    missing, mismatches, total = [], [], 0
    for sibling in info.siblings:
        path = args.output / sibling.rfilename
        if not path.is_file(): missing.append(sibling.rfilename); continue
        actual, expected = path.stat().st_size, int(sibling.size or 0)
        if expected and actual != expected: mismatches.append((sibling.rfilename, expected, actual))
        total += actual
    if missing or mismatches: raise RuntimeError(f"incomplete snapshot: {missing[:5]} {mismatches[:5]}")
    receipt = {
        "schema": "duobench.pi05.download.v1", "status": "complete",
        "repo_id": "RobotControlStack/duobench", "revision": DATASET_REVISION,
        "files": len(info.siblings), "bytes_total": total,
        "download_scope": "full_repository_including_sim_and_real",
        "formal_training_scope": "all_550_sim_demonstrations_no_split",
    }
    atomic_json(args.output / "download_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__": main()
