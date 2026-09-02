#!/usr/bin/env python3
import ast,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent)
 with os.fdopen(fd,"w") as f: json.dump(v,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
repo=Path("/workspace/repos/rdt-1b"); sys.path.insert(0,str(repo)); os.chdir(repo)
token=Path("/workspace/.secrets/hf_token")
if not token.is_file() or not token.read_text().strip() or token.stat().st_mode & 0o077: raise RuntimeError("HF token must be nonempty mode 0600")
gpus=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total","--format=csv,noheader,nounits"],text=True).splitlines()
if len(gpus)!=4 or any("RTX PRO 6000" not in x for x in gpus): raise RuntimeError(f"expected 4x RTX PRO 6000, got {gpus}")
tree=ast.parse((repo/"train/train.py").read_text()); full=any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="params_to_optimize" for t in n.targets) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="parameters" for n in ast.walk(tree))
if not full: raise RuntimeError("trainer is not full-parameter")
from data.hdf5_vla_dataset import HDF5VLADataset
d=HDF5VLADataset(); s=d.get_item(0)
if len(d._episodes)!=600 or len(d)!=1650 or tuple(d._tasks)!=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube"): raise RuntimeError("dataset cardinality/task contract drift")
if s["state"].shape!=(1,128) or s["actions"].shape!=(64,128) or s["cam_high"].shape!=(2,240,320,3): raise RuntimeError("tensor contract drift")
atomic(Path("/workspace/runs/rdt_mars/audit.json"),{"schema":"mars-control.rdt.audit.v1","status":"complete","episodes":600,"local_streams":1650,"tasks":d._tasks,"all_data_no_split":True,"optimizer_scope":"all_rdt_parameters","gpus":gpus,"policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_absolute_action8","forbidden_inputs":["peer_rgb","peer_qpos","global_rgb","joint_action","arm_id"],"adapter_sha256":hashlib.sha256((repo/"data/hdf5_vla_dataset.py").read_bytes()).hexdigest()})
print("MARS RDT contract audit passed")
