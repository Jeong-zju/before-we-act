from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


PHASE_NAMES = {
    0: "approach",
    1: "align",
    2: "grasp",
    3: "carry_to_passage",
    4: "passage",
    5: "carry_to_goal",
    6: "release",
    7: "done",
    8: "failure",
}


FAILURE_NAMES = {
    0: "none",
    1: "timeout",
    2: "force_violation",
    3: "object_out_of_bounds",
    4: "robot_out_of_bounds",
    5: "object_dropped",
    6: "robot_too_far",
    7: "desync_in_passage",
    8: "object_yaw_too_large",
    99: "unknown",
}


def list_episodes(data_dir: str | Path) -> List[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("episode_*.hdf5"))


def read_episode(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with h5py.File(path, "r") as f:
        return {
            "obs_robot_0": f["obs/robot_0/proprio"][:],
            "obs_robot_1": f["obs/robot_1/proprio"][:],
            "actions": f["actions/joint"][:],
            "global_state": f["global/global_state"][:],
            "object_pose": f["global/object_pose"][:],
            "force_proxy": f["global/force_proxy"][:].reshape(-1),
            "contacts": f["global/contacts"][:].reshape(-1),
            "robot_distance": f["global/robot_distance"][:].reshape(-1),
            "phase": f["labels/phase"][:].reshape(-1),
            "success": f["labels/success"][:].reshape(-1),
            "failure": f["labels/failure"][:].reshape(-1),
            "failure_reason": f["labels/failure_reason"][:].reshape(-1),
            "communication_dummy": f["labels/communication_dummy"][:],
            "reward": f["rewards/reward"][:].reshape(-1),
            "time": f["meta/time"][:].reshape(-1),
            "attrs": dict(f.attrs),
        }


def validate_episode(path: str | Path, min_len: int = 4) -> tuple[bool, list[str]]:
    errors = []
    path = Path(path)

    try:
        with h5py.File(path, "r") as f:
            for key in REQUIRED_KEYS:
                if key not in f:
                    errors.append(f"missing key: {key}")

            if errors:
                return False, errors

            T = f["actions/joint"].shape[0]
            if T < min_len:
                errors.append(f"too short: T={T}")

            for key in REQUIRED_KEYS:
                arr = f[key][:]
                if arr.shape[0] != T:
                    errors.append(f"length mismatch: {key}, {arr.shape[0]} vs {T}")
                if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
                    errors.append(f"non-finite: {key}")

            actions = f["actions/joint"][:]
            if actions.ndim != 2 or actions.shape[1] != 8:
                errors.append(f"bad action shape: {actions.shape}")
            if np.max(np.abs(actions)) > 1.0001:
                errors.append("action out of [-1, 1]")

            obs0 = f["obs/robot_0/proprio"][:]
            obs1 = f["obs/robot_1/proprio"][:]
            if obs0.shape[1] != 11 or obs1.shape[1] != 11:
                errors.append(f"bad obs shape: robot_0={obs0.shape}, robot_1={obs1.shape}")

    except Exception as exc:
        errors.append(repr(exc))

    return len(errors) == 0, errors


def episode_summary(path: str | Path) -> dict:
    ep = read_episode(path)
    attrs = ep["attrs"]

    failure_ids = ep["failure_reason"]
    final_failure_id = int(failure_ids[-1]) if len(failure_ids) else 99
    failure_name = str(attrs.get("failure_reason", FAILURE_NAMES.get(final_failure_id, "unknown")))

    actions = ep["actions"]
    object_pose = ep["object_pose"]

    return {
        "file": str(path),
        "T": int(actions.shape[0]),
        "success": int(bool(attrs.get("success", bool(ep["success"][-1] if len(ep["success"]) else False)))),
        "failure": int(bool(attrs.get("failure", bool(ep["failure"][-1] if len(ep["failure"]) else False)))),
        "failure_reason": failure_name,
        "total_reward": float(np.sum(ep["reward"])),
        "mean_reward": float(np.mean(ep["reward"])),
        "max_force_proxy": float(np.max(ep["force_proxy"])),
        "mean_force_proxy": float(np.mean(ep["force_proxy"])),
        "max_contacts": int(np.max(ep["contacts"])),
        "min_robot_distance": float(np.min(ep["robot_distance"])),
        "max_robot_distance": float(np.max(ep["robot_distance"])),
        "mean_abs_action": float(np.mean(np.abs(actions))),
        "max_abs_action": float(np.max(np.abs(actions))),
        "final_object_x": float(object_pose[-1, 0]),
        "final_object_y": float(object_pose[-1, 1]),
        "final_object_yaw": float(object_pose[-1, 2]),
        "num_unique_phases": int(len(np.unique(ep["phase"]))),
    }


def scan_dataset(data_dir: str | Path, min_len: int = 4) -> tuple[pd.DataFrame, dict]:
    paths = list_episodes(data_dir)
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {data_dir}")

    rows = []
    bad = []

    for path in tqdm(paths, desc=f"scan {data_dir}"):
        ok, errors = validate_episode(path, min_len=min_len)
        if not ok:
            bad.append({"file": str(path), "errors": errors})
            continue
        rows.append(episode_summary(path))

    df = pd.DataFrame(rows)

    if len(df) == 0:
        summary = {
            "data_dir": str(data_dir),
            "num_files": len(paths),
            "num_valid": 0,
            "num_bad": len(bad),
            "bad": bad,
        }
        return df, summary

    summary = {
        "data_dir": str(data_dir),
        "num_files": len(paths),
        "num_valid": int(len(df)),
        "num_bad": int(len(bad)),
        "success_rate": float(df["success"].mean()),
        "mean_T": float(df["T"].mean()),
        "median_T": float(df["T"].median()),
        "mean_total_reward": float(df["total_reward"].mean()),
        "mean_max_force_proxy": float(df["max_force_proxy"].mean()),
        "max_force_proxy": float(df["max_force_proxy"].max()),
        "mean_min_robot_distance": float(df["min_robot_distance"].mean()),
        "mean_max_robot_distance": float(df["max_robot_distance"].mean()),
        "failure_reasons": df["failure_reason"].value_counts().to_dict(),
        "bad": bad[:20],
    }
    return df, summary


def save_json(obj: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_hist(series, title: str, xlabel: str, out_path: str | Path, bins: int = 30):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_failure_hist(df: pd.DataFrame, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["failure_reason"].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.title("Failure reason histogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_action_distribution(paths: Iterable[Path], out_path: str | Path, max_files: int = 200):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    actions = []
    for idx, path in enumerate(paths):
        if idx >= max_files:
            break
        ep = read_episode(path)
        actions.append(ep["actions"])

    if not actions:
        return

    arr = np.concatenate(actions, axis=0)
    plt.figure(figsize=(10, 5))
    plt.boxplot([arr[:, i] for i in range(arr.shape[1])], tick_labels=[f"a{i}" for i in range(arr.shape[1])])
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.axhline(-1.0, linestyle="--", linewidth=1)
    plt.ylabel("action value")
    plt.title("Action distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_episode(ep: Dict[str, np.ndarray], out_dir: str | Path, prefix: str = "episode"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gs = ep["global_state"]
    obj = ep["object_pose"]
    force = ep["force_proxy"]
    dist = ep["robot_distance"]
    actions = ep["actions"]
    phase = ep["phase"]
    reward = ep["reward"]

    robot0_xy = gs[:, 0:2]
    robot1_xy = gs[:, 3:5]

    # Top-down trajectory
    plt.figure(figsize=(7, 8))
    plt.plot(obj[:, 0], obj[:, 1], label="object")
    plt.plot(robot0_xy[:, 0], robot0_xy[:, 1], label="robot_0")
    plt.plot(robot1_xy[:, 0], robot1_xy[:, 1], label="robot_1")
    plt.axvline(-0.90, linestyle="--", linewidth=1, label="passage wall")
    plt.axvline(0.90, linestyle="--", linewidth=1)
    plt.axhspan(2.77, 3.33, alpha=0.2, label="goal")
    plt.xlim(-1.6, 1.6)
    plt.ylim(-1.6, 3.6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Top-down trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_trajectory.png")
    plt.close()

    # Object pose
    plt.figure(figsize=(10, 5))
    plt.plot(obj[:, 0], label="object_x")
    plt.plot(obj[:, 1], label="object_y")
    plt.plot(obj[:, 2], label="object_yaw")
    plt.xlabel("t")
    plt.ylabel("pose")
    plt.title("Object pose over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_object_pose.png")
    plt.close()

    # Force and robot distance
    plt.figure(figsize=(10, 5))
    plt.plot(force, label="force_proxy")
    plt.plot(dist, label="robot_distance")
    plt.xlabel("t")
    plt.title("Force proxy and robot distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_force_distance.png")
    plt.close()

    # Actions
    plt.figure(figsize=(11, 6))
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], label=f"a{i}", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("action")
    plt.title("Action chunk timeline")
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_actions.png")
    plt.close()

    # Phase and reward
    plt.figure(figsize=(10, 5))
    plt.plot(phase, label="phase_id")
    plt.plot(reward / (np.std(reward) + 1e-6), label="reward_norm", alpha=0.8)
    plt.xlabel("t")
    plt.title("Phase timeline and normalized reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_phase_reward.png")
    plt.close()


def make_dataset_report(data_dir: str | Path, out_dir: str | Path, num_examples: int = 5):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, summary = scan_dataset(data_dir)
    csv_path = out_dir / f"{data_dir.name}_summary.csv"
    json_path = out_dir / f"{data_dir.name}_summary.json"

    df.to_csv(csv_path, index=False)
    save_json(summary, json_path)

    if len(df):
        plot_hist(df["T"], f"{data_dir.name}: episode length", "T", out_dir / f"{data_dir.name}_length_hist.png")
        plot_hist(df["max_force_proxy"], f"{data_dir.name}: max force proxy", "max force proxy", out_dir / f"{data_dir.name}_force_hist.png")
        plot_hist(df["max_robot_distance"], f"{data_dir.name}: max robot distance", "max robot distance", out_dir / f"{data_dir.name}_robot_distance_hist.png")
        plot_failure_hist(df, out_dir / f"{data_dir.name}_failure_hist.png")
        plot_action_distribution(list_episodes(data_dir), out_dir / f"{data_dir.name}_action_distribution.png")

        # Representative examples: first successes and first failures.
        selected = []
        selected.extend(df[df["success"] == 1]["file"].head(num_examples).tolist())
        selected.extend(df[df["success"] == 0]["file"].head(num_examples).tolist())

        for idx, file in enumerate(selected[: 2 * num_examples]):
            ep = read_episode(file)
            plot_episode(ep, out_dir / "episodes", prefix=f"example_{idx:03d}_{Path(file).stem}")

    print(json.dumps(summary, indent=2))
    print("saved:", csv_path)
    print("saved:", json_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/diagnostics")
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    make_dataset_report(args.data_dir, args.out_dir, num_examples=args.num_examples)


if __name__ == "__main__":
    main()
