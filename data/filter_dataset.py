from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from data.diagnostics import validate_episode


def read_attrs_and_stats(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {
            "success": bool(f.attrs.get("success", False)),
            "failure": bool(f.attrs.get("failure", False)),
            "failure_reason": str(f.attrs.get("failure_reason", "unknown")),
            "T": int(f["actions/joint"].shape[0]),
            "max_force_proxy": float(np.max(f["global/force_proxy"][:])),
            "max_abs_action": float(np.max(np.abs(f["actions/joint"][:]))),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--success_only", type=int, default=0)
    parser.add_argument("--failure_only", type=int, default=0)
    parser.add_argument("--max_force", type=float, default=-1.0)
    parser.add_argument("--copy", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for old in dst.glob("episode_*.hdf5"):
        old.unlink()

    paths = sorted(src.glob("episode_*.hdf5"))
    kept = []
    dropped = []

    for path in tqdm(paths, desc=f"filter {src}"):
        ok, errors = validate_episode(path, min_len=args.min_len)
        if not ok:
            dropped.append({"file": str(path), "reason": "invalid", "errors": errors})
            continue

        stats = read_attrs_and_stats(path)

        if args.success_only and not stats["success"]:
            dropped.append({"file": str(path), "reason": "not_success"})
            continue

        if args.failure_only and not stats["failure"]:
            dropped.append({"file": str(path), "reason": "not_failure"})
            continue

        if args.max_force >= 0 and stats["max_force_proxy"] > args.max_force:
            dropped.append({"file": str(path), "reason": "force_too_large", "max_force_proxy": stats["max_force_proxy"]})
            continue

        out_path = dst / f"episode_{len(kept):06d}.hdf5"
        if args.copy:
            shutil.copy2(path, out_path)
        else:
            out_path.symlink_to(path.resolve())

        kept.append({"src": str(path), "dst": str(out_path), **stats})

    manifest = {
        "src": str(src),
        "dst": str(dst),
        "num_input": len(paths),
        "num_kept": len(kept),
        "num_dropped": len(dropped),
        "kept_preview": kept[:20],
        "dropped_preview": dropped[:50],
        "copy": bool(args.copy),
    }

    with open(dst / "filter_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
