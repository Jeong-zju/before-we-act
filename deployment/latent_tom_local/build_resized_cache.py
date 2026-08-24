from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np

from local_dataset import TASKS, _episode_rows


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def target_path(root: Path, cache: Path, source: Path, agent: int) -> Path:
    return cache / source.relative_to(root).with_suffix("") / f"agent_{agent}.npy"


def valid(path: Path, frames: int) -> bool:
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return arr.shape == (frames, 240, 320, 3) and arr.dtype == np.uint8
    except Exception:
        return False


def convert_episode(job):
    root, cache, source = map(Path, job)
    cv2.setNumThreads(1)
    written = skipped = frames_total = 0
    with h5py.File(source, "r", libver="latest", swmr=True) as handle:
        image_group = handle["data/observation/images"]
        agents = sorted(int(k.rsplit("_", 1)[1]) for k in image_group if k.startswith("agent_"))
        for agent in agents:
            source_images = image_group[f"agent_{agent}"]
            frames = len(source_images); frames_total += frames
            output = target_path(root, cache, source, agent)
            if valid(output, frames):
                skipped += 1
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_name(output.name + f".tmp-{os.getpid()}")
            arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                            shape=(frames, 240, 320, 3))
            for start in range(0, frames, 16):
                block = np.asarray(source_images[start:start + 16], dtype=np.uint8)
                for offset, frame in enumerate(block):
                    arr[start + offset] = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
            arr.flush(); del arr
            os.replace(tmp, output)
            written += 1
    return {"source": str(source), "agents_written": written, "agents_skipped": skipped,
            "frames": frames_total}


def main():
    root = Path(os.environ.get("BWA_DATASET_ROOT", "/workspace/datasets/robofactory_multitask"))
    cache = Path(os.environ.get("BWA_RESIZED_CACHE", "/workspace/datasets/robofactory_multitask_320x240"))
    workers = int(os.environ.get("BWA_CACHE_WORKERS", "16"))
    rows = _episode_rows(root)
    if len(rows) != 900:
        raise RuntimeError(f"expected 900 episodes, found {len(rows)}")
    jobs = [(str(root), str(cache), str(row[2])) for row in rows]
    frames = agents = complete = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(convert_episode, jobs, chunksize=1):
            complete += 1; frames += result["frames"]
            agents += result["agents_written"] + result["agents_skipped"]
            if complete % 10 == 0 or complete == len(jobs):
                print(json.dumps({"event":"cache_progress","episodes":complete,
                                  "agents":agents,"frames":frames}), flush=True)
    manifest = {"schema":"bwa.latent_tom.rgb_cache.v1","status":"complete",
                "episodes":complete,"agents":agents,"local_frames":frames,
                "tasks":list(TASKS),"shape":[240,320,3],"dtype":"uint8",
                "interpolation":"opencv_linear","source":"all_local_agent_rgb_only",
                "completed_at":datetime.now(timezone.utc).isoformat()}
    atomic_json(cache / "cache_manifest.json", manifest)


if __name__ == "__main__":
    main()
