from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/policy_rollouts")
    parser.add_argument("--out_dir", type=str, default="outputs/policy_reports")
    parser.add_argument("--modes", type=str, default="scripted,no_comm,always_comm,selective_comm")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_ep = []

    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        p = root / mode / "summary.json"
        ep = root / mode / "episode_metrics.csv"
        if not p.exists():
            print("missing:", p)
            continue
        m = json.loads(p.read_text())
        rows.append(m)
        if ep.exists():
            df = pd.read_csv(ep)
            df["mode"] = mode
            all_ep.append(df)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "policy_summary.csv", index=False)

    if all_ep:
        ep_df = pd.concat(all_ep, ignore_index=True)
        ep_df.to_csv(out_dir / "policy_episode_metrics.csv", index=False)

        metrics = ["success", "collision_count", "max_force", "episode_steps", "comm_rate"]
        for metric in metrics:
            if metric not in ep_df.columns:
                continue
            means = ep_df.groupby("mode")[metric].mean()
            plt.figure(figsize=(8, 5))
            means.plot(kind="bar")
            plt.ylabel(metric)
            plt.title(f"Policy comparison: {metric}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric}_comparison.png")
            plt.close()

    print(summary.to_string(index=False))
    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
