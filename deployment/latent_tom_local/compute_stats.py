from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import h5py, numpy as np
from local_dataset import TASKS

def main():
    root = Path(os.environ.get("BWA_DATASET_ROOT", "/workspace/datasets/robofactory_multitask"))
    out = Path(os.environ.get("BWA_LATENT_TOM_OUTPUT", "/workspace/bwa_latent_tom_runs/formal")); out.mkdir(parents=True, exist_ok=True)
    q_rows, a_rows, episodes = [], [], 0
    for task in TASKS:
        manifest = json.loads((root / task / "training_manifest.json").read_text())
        if len(manifest.get("episodes", [])) != 150: raise RuntimeError(f"{task}: not 150 episodes")
        for ep in manifest["episodes"]:
            with h5py.File(root / task / ep["hdf5_path"], "r") as h:
                agents = sorted(h["data/observation/agents"].keys())
                for key in agents:
                    agent = key.rsplit("_", 1)[1]
                    q = np.asarray(h[f"data/observation/agents/{key}/qpos"][:], np.float32)
                    a = np.asarray(h[f"data/action/agents/panda_{agent}/commanded"][:], np.float32)
                    q_rows.append(q); a_rows.append(a)
            episodes += 1
    if episodes != 900: raise RuntimeError(f"expected 900, got {episodes}")
    q, a = np.concatenate(q_rows), np.concatenate(a_rows)
    def row(x):
        return {"mean": x.mean(0).astype(float).tolist(), "std": np.maximum(x.std(0), 1e-5).astype(float).tolist()}
    payload = {"schema": "bwa.latent_tom.normalization.v1", "status": "complete", "episodes": episodes, "qpos": row(q), "action": row(a), "all_episodes": True}
    path = out / "normalization.json"; tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2) + "\n"); os.replace(tmp, path)
    print(json.dumps({"status":"complete", "episodes":episodes, "qpos_rows":len(q), "action_rows":len(a), "output":str(path)}))
if __name__ == "__main__": main()
