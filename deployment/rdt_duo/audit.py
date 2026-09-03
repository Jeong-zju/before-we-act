"""Fail-closed audit for data, decentralization and upstream RDT wiring."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from .protocol import TASKS, FORMAL_DATASET_REVISION

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--rdt",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    manifest=json.loads((a.data/"manifest.json").read_text()); checks={"revision":manifest.get("dataset_revision")==FORMAL_DATASET_REVISION,"tasks":tuple(manifest.get("tasks",()))==TASKS,"episodes":manifest.get("total_episodes")==550,"frames":manifest.get("total_frames")==285988,"lag":manifest.get("recording_alignment",{}).get("action_lag_rows")==1,"image_size":manifest.get("image_size")==224}
    streams=frames=episodes=0
    for task in TASKS:
        d={k:np.load(a.data/task/f"{k}.npy",mmap_mode="r") for k in ("state","action","head","left","right","episodes")}; episodes += len(np.unique(d["episodes"])); frames += len(d["state"]); streams += 2*len(np.unique(d["episodes"]))
        checks[f"{task}_shape"] = d["state"].shape == d["action"].shape and d["state"].shape[1:]==(16,) and all(d[k].shape[0]==len(d["state"]) for k in ("head","left","right","episodes"))
        checks[f"{task}_rgb"] = all(d[k].dtype==np.uint8 and d[k].shape[1:]==(224,224,3) for k in ("head","left","right"))
        checks[f"{task}_binary_gripper"] = bool(np.isin(d["state"].reshape(-1,2,8)[...,7],(0,1)).all() and np.isin(d["action"].reshape(-1,2,8)[...,7],(0,1)).all())
    source=(a.rdt/"train/train.py").read_text(); ast.parse(source); checks["full_optimizer"] = "params_to_optimize = rdt.parameters()" in source
    checks["upstream_commit"] = subprocess.check_output(["git","-C",str(a.rdt),"rev-parse","HEAD"],text=True).strip() == "cd79363a1387e8f81c7724d070ef7e45fd23150f"
    control=json.loads((a.rdt/"configs/dataset_control_freq.json").read_text()); checks["control_frequency"] = control.get("duobench") == 30
    checks["finetune_dataset"] = json.loads((a.rdt/"configs/finetune_datasets.json").read_text()) == ["duobench"]
    checks["sample_weight"] = json.loads((a.rdt/"configs/finetune_sample_weights.json").read_text()) == [1.0]
    local_adapter=Path(__file__).with_name("hdf5_vla_dataset.py"); upstream_adapter=a.rdt/"data/hdf5_vla_dataset.py"
    checks["adapter_overlay"] = hashlib.sha256(local_adapter.read_bytes()).digest() == hashlib.sha256(upstream_adapter.read_bytes()).digest()
    checks["package_shadowing_fix"] = all((a.rdt/name/"__init__.py").is_file() for name in ("configs","data","models","train"))
    report={"schema":"duobench.rdt.audit.v1","status":"PASSED" if all(checks.values()) else "FAILED","passed":all(checks.values()),"checks":checks,"episodes":episodes,"frames":frames,"local_streams":streams,"causal_samples":frames-episodes,"all_data_no_split":True,"optimizer_scope":"all_rdt_parameters","policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_local_absolute_action8","forbidden_inputs":["peer_rgb","peer_qpos","global_state","arm_id"],"adapter_sha256":hashlib.sha256((a.rdt/"data/hdf5_vla_dataset.py").read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,sort_keys=True));
    if not report["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
