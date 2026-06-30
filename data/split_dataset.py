from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def collect_paths(sources: list[str]) -> list[Path]:
    paths = []
    for src in sources:
        root = Path(src)
        paths.extend(sorted(root.glob("episode_*.hdf5")))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, nargs="+", required=True)
    parser.add_argument("--out_root", type=str, default="datasets/stage2")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    paths = collect_paths(args.sources)

    if not paths:
        raise FileNotFoundError("No episode files found in sources.")

    paths = np.asarray(paths, dtype=object)
    paths = paths[rng.permutation(len(paths))]

    n = len(paths)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }

    out_root = Path(args.out_root)
    for split, split_paths in splits.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for old in split_dir.glob("episode_*.hdf5"):
            old.unlink()

        for idx, src in enumerate(split_paths):
            dst = split_dir / f"episode_{idx:06d}.hdf5"
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())

    manifest = {
        "num_total": int(n),
        "num_train": int(len(splits["train"])),
        "num_val": int(len(splits["val"])),
        "num_test": int(len(splits["test"])),
        "sources": args.sources,
        "copy": bool(args.copy),
    }

    with open(out_root / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
