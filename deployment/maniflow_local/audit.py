from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

from local_dataset import TASKS, episode_rows


def main():
    root = Path(os.environ.get("BWA_DATASET_ROOT", "/workspace/datasets/robofactory_multitask"))
    if len(episode_rows(root)) != 900:
        raise RuntimeError("dataset contract requires 900 episodes")
    rows = {"status": "complete", "episodes": 900, "tasks": list(TASKS), "inputs": ["own_rgb", "own_qpos"],
            "outputs": ["own_action8"], "training_split": "all_900_episodes_ignore_manifest_split",
            "forbidden": ["global", "peer", "joint_concatenation", "privileged"]}
    source_files = [Path(__file__).with_name("local_dataset.py"), Path(__file__).with_name("train.py"), Path(__file__).with_name("validate.py")]
    texts = {p.name: p.read_text() for p in source_files}
    for p, text in texts.items(): ast.parse(text, filename=p)
    forbidden = ("head_camera_global", "peer_qpos", "peer_image", "joint_action", "planner_state", "privileged")
    hits = {p: [x for x in forbidden if x in text.lower()] for p, text in texts.items()}
    if any(hits.values()):
        raise RuntimeError(f"forbidden local-policy token: {hits}")
    rows["files_sha256"] = {p: hashlib.sha256(texts[p].encode()).hexdigest() for p in texts}
    out = Path(os.environ.get("BWA_MANIFLOW_RUN_ROOT", "/workspace/bwa_maniflow_runs")) / "audit" / "training_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "episodes": 900, "output": str(out)}))


if __name__ == "__main__": main()
