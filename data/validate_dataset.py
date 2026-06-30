from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


REQUIRED_KEYS = [
    "obs/robot_0/proprio",
    "obs/robot_1/proprio",
    "actions/joint",
    "actions/robot_0",
    "actions/robot_1",
    "global/global_state",
    "global/object_pose",
    "global/force_proxy",
    "global/contacts",
    "global/robot_distance",
    "labels/phase",
    "labels/success",
    "labels/failure",
    "labels/failure_reason",
    "labels/communication_dummy",
    "rewards/reward",
    "meta/time",
]


def validate_file(path: Path) -> tuple[bool, list[str], dict]:
    errors = []
    stats = {}

    try:
        with h5py.File(path, "r") as f:
            for key in REQUIRED_KEYS:
                if key not in f:
                    errors.append(f"missing key: {key}")

            if errors:
                return False, errors, stats

            T = f["actions/joint"].shape[0]
            stats["T"] = T
            stats["success"] = float(bool(f.attrs.get("success", False)))
            stats["failure"] = float(bool(f.attrs.get("failure", False)))
            stats["failure_reason"] = str(f.attrs.get("failure_reason", "unknown"))

            if T <= 2:
                errors.append("episode too short")

            for key in REQUIRED_KEYS:
                arr = f[key][:]
                if arr.shape[0] != T and key not in {"meta/episode_index", "meta/seed"}:
                    errors.append(f"length mismatch: {key}, {arr.shape[0]} vs {T}")
                if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
                    errors.append(f"non-finite values: {key}")

            actions = f["actions/joint"][:]
            if np.max(np.abs(actions)) > 1.0001:
                errors.append("actions out of [-1, 1]")

            stats["max_abs_action"] = float(np.max(np.abs(actions)))
            stats["max_force_proxy"] = float(np.max(f["global/force_proxy"][:]))
            stats["mean_force_proxy"] = float(np.mean(f["global/force_proxy"][:]))

    except Exception as exc:
        errors.append(repr(exc))

    return len(errors) == 0, errors, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_print", type=int, default=10)
    args = parser.parse_args()

    paths = sorted(Path(args.data_dir).glob("episode_*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {args.data_dir}")

    ok_count = 0
    bad = []
    stats_rows = []

    for p in tqdm(paths):
        ok, errors, stats = validate_file(p)
        if ok:
            ok_count += 1
        else:
            bad.append((p, errors))
        stats_rows.append(stats)

    success = np.array([s.get("success", 0.0) for s in stats_rows], dtype=np.float32)
    lengths = np.array([s.get("T", 0) for s in stats_rows], dtype=np.float32)
    forces = np.array([s.get("max_force_proxy", 0.0) for s in stats_rows], dtype=np.float32)

    print("data_dir:", args.data_dir)
    print("num_files:", len(paths))
    print("ok_files:", ok_count)
    print("bad_files:", len(bad))
    print("success_rate:", float(success.mean()))
    print("mean_length:", float(lengths.mean()))
    print("max_length:", float(lengths.max()))
    print("mean_max_force_proxy:", float(forces.mean()))
    print("max_force_proxy:", float(forces.max()))

    if bad:
        print("\nBad examples:")
        for p, errors in bad[: args.max_print]:
            print(p)
            for e in errors:
                print("  -", e)
        raise RuntimeError(f"Dataset validation failed: {len(bad)} bad files")


if __name__ == "__main__":
    main()
