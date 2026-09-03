"""Prepare the audited DuoBench arrays consumed by the RDT adapter."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from .protocol import FORMAL_IMAGE_SIZE, TASKS

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--image-size", type=int, default=FORMAL_IMAGE_SIZE); p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    if a.image_size != FORMAL_IMAGE_SIZE: raise ValueError("DuoBench RDT requires official 224x224 streams")
    command = [sys.executable, "-m", "deployment.duo_act.prepare", "--dataset", str(a.dataset), "--output", str(a.output), "--image-size", str(a.image_size), "--jobs", str(a.jobs)]
    env = os.environ.copy(); env["PYTHONPATH"] = str(Path(__file__).parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=env)
    manifest = json.loads((a.output / "manifest.json").read_text())
    manifest.update({"schema":"duobench-rdt-prepared-v1", "rdt_tasks":list(TASKS), "rdt_action_chunk_size":64, "rdt_image_history_size":2, "rdt_state_dim":128, "rdt_local_stream_contract":"one task/episode/arm stream; obs[i] -> action[i+1]", "rdt_policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_local_absolute_action8"})
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
if __name__ == "__main__": main()
