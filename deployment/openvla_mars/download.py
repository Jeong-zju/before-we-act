"""Resumable, pinned download of the four MARS-Control corpora."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(os.environ.get("MARS_OPENVLA_DATA_ROOT", "/workspace/datasets/mars_control"))
TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", "/workspace/.secrets/hf_token"))
REPOS = {
    "place_cube_in_cup": ("Jeong-zju/mars-control-place-cube-in-cup-rf", "3878150bec8f4830e1a57a01a13762a10abc8d52"),
    "strike_cube_hard": ("Jeong-zju/mars-control-strike-cube-hard-rf", "bc7051cb0560058bf426e792871faa1ca8a4f78f"),
    "three_robots_place_shoes": ("Jeong-zju/mars-control-three-robots-place-shoes-rf", "ad231c7eff530f71f0c5302b6c03c7164bbcc896"),
    "four_robots_stack_cube": ("Jeong-zju/mars-control-four-robots-stack-cube-rf", "3fa4833f5e34c3565da04af99c62d516e048fcfc"),
}

def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def main() -> None:
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token: raise RuntimeError("empty Hugging Face token")
    api = HfApi(token=token)
    for task, (repo, revision) in REPOS.items():
        out = ROOT / task; out.mkdir(parents=True, exist_ok=True)
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision: raise RuntimeError(f"{task}: revision drift {info.sha} != {revision}")
        siblings = {x.rfilename: x for x in info.siblings}
        names = sorted(n for n in siblings if n.startswith("motionplanning/") and n.endswith(".h5") and "/." not in n)
        if len(names) != 10: raise RuntimeError(f"{task}: expected 10 shards, found {len(names)}")
        rows = []
        for name in names:
            item = siblings[name]
            local = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=str(out), token=token))
            if local.stat().st_size != int(item.size or 0): raise RuntimeError(f"{task}: size mismatch {name}")
            got = digest(local)
            if item.lfs and item.lfs.sha256 and got != item.lfs.sha256: raise RuntimeError(f"{task}: sha mismatch {name}")
            sidecar = name[:-3] + "json"
            if sidecar in siblings: hf_hub_download(repo, sidecar, repo_type="dataset", revision=revision, local_dir=str(out), token=token)
            rows.append({"path": name, "size_bytes": int(local.stat().st_size), "sha256": got})
        atomic(out / "download_receipt.json", {"schema": "mars-control.openvla.dataset.v1", "status": "complete", "task": task,
            "repo_id": repo, "revision": revision, "formal_shards": rows, "formal_episodes": 150,
            "training_policy": "all_600_successful_episodes_no_split"})
        print(json.dumps({"task": task, "status": "complete", "shards": len(rows), "bytes": sum(x["size_bytes"] for x in rows)}), flush=True)

if __name__ == "__main__": main()
