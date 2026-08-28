from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import TASKS, POLICY_CONTRACT, atomic_json
from .dataset import index_corpus


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--stats", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    streams, stats = index_corpus(args.data_root, args.stats)
    report = {"schema": "mars-control.latent-tom.audit.v1", "status": "complete", "contract": POLICY_CONTRACT, "episodes": stats["episodes"], "local_streams": stats["local_streams"], "indexed_local_timesteps": stats["indexed_local_timesteps"], "tasks": {spec.name: {"episodes": 150, "arms": spec.arms, "local_streams": len(streams[index])} for index, spec in enumerate(TASKS)}, "forbidden_inputs": ["task_id", "arm_id", "peer_rgb", "peer_qpos", "global_rgb", "joint_action", "language"]}
    atomic_json(args.output, report); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
