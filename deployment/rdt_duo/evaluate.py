"""Strict DuoBench closed-loop evaluator for one RDT-1B checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import deque
from pathlib import Path
import numpy as np, torch, gymnasium as gym
from deployment.duo_act.action_target import canonicalize_controller_action
from deployment.duo_act.protocol import VALIDATION_MAX_STEPS
from .protocol import TASKS, EVALUATOR_REVISION, TEMPORAL_ENSEMBLE_DECAY
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo
from torchvision.transforms.v2 import functional as TVF

def make_env(task):
    # Importing rcs registers its own asset paths before DuoBench is loaded.
    # Keep the two registries explicit for subprocesses used by validation.
    os.environ.setdefault("RCS_PREFIX", "/root/.rcs")
    os.environ.setdefault("DUOBENCH_PREFIX", "/workspace/datasets/duobench_assets")
    mod=__import__(f"duobench.tasks.{task}",fromlist=["*"]); cls=getattr(mod,"".join(x.title() for x in task.split("_"))+"EnvConfig"); cfg=cls().config(); cfg.headless=True; cfg.control_mode=ControlMode.JOINTS; cfg.relative_to=RelativeTo.NONE; cfg.sim_cfg=SimConfig(async_control=True,realtime=False,frequency=30); cfg.wrapper_cfg.binary_gripper=True
    return gym.make(f"duobench/{task}",cfg=cfg)

def runtime_rgb(value):
    image=torch.from_numpy(np.asarray(value,np.uint8).copy()).permute(2,0,1)
    return TVF.resize(image,(224,224),antialias=True).permute(1,2,0).contiguous().numpy()

class Policy:
    def __init__(self, checkpoint, data, device):
        repo="/workspace/repos/rdt-1b"; sys.path.insert(0,repo); os.chdir(repo)
        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        from models.rdt_runner import RDTRunner
        self.dev=torch.device(device); self.dtype=torch.bfloat16; self.vision=SiglipVisionTower("google/siglip-so400m-patch14-384",None).to(self.dev,dtype=self.dtype).eval(); self.model=RDTRunner.from_pretrained(str(checkpoint)).to(self.dev,dtype=self.dtype).eval(); self.proc=self.vision.image_processor; self.data=Path(data); self.ids=[i for i in range(7)]+[10]; self.history=[None,None]
        bg=np.asarray([int(x*255) for x in self.proc.image_mean],np.uint8); self.bg=np.broadcast_to(bg,(self.proc.size["height"],self.proc.size["width"],3)).copy()
    def reset(self): self.history=[None,None]
    def infer(self, heads, wrists, qposes, task):
        values=[]
        for arm in range(2):
            current=(heads[arm],wrists[arm]); previous=self.history[arm] if self.history[arm] is not None else current; self.history[arm]=(heads[arm].copy(),wrists[arm].copy())
            for head,wrist in (previous,current): values.extend((head,wrist,self.bg))
        px=[]
        from PIL import Image
        for im in values: px.append(self.proc.preprocess(Image.fromarray(im),return_tensors="pt")["pixel_values"][0])
        with torch.inference_mode():
            tok=self.vision(torch.stack(px).to(self.dev,dtype=self.dtype)).reshape(2,-1,self.vision.hidden_size); lang=[]
            path=self.data/task/"lang_embed.pt"; emb=torch.load(path,map_location="cpu",weights_only=False).to(self.dev,dtype=self.dtype)
            lang=emb.unsqueeze(0).expand(2,-1,-1); mask=torch.ones(lang.shape[:2],device=self.dev,dtype=torch.bool); state=torch.zeros(2,1,128,device=self.dev,dtype=self.dtype); state[:,:,self.ids]=torch.as_tensor(np.asarray(qposes,np.float32),device=self.dev,dtype=self.dtype).unsqueeze(1); sm=torch.zeros(2,1,128,device=self.dev,dtype=self.dtype); sm[:,:,self.ids]=1; freq=torch.full((2,),30,dtype=torch.long,device=self.dev); return self.model.predict_action(lang,mask,tok,state,sm,freq).float().cpu().numpy()[...,self.ids]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--task",choices=TASKS,required=True); p.add_argument("--episodes",type=int,default=20); p.add_argument("--seed-start",type=int,default=20260820); p.add_argument("--max-steps",type=int); p.add_argument("--device",default="cuda:0"); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    dev=torch.device(a.device); policy=Policy(a.checkpoint,a.data,dev); env=make_env(a.task); max_steps=a.max_steps or VALIDATION_MAX_STEPS[a.task]; tid=TASKS.index(a.task); rows=[]; journal=a.output.with_suffix(".jsonl"); recovered={}
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                row=json.loads(line)
                if row.get("evaluator_revision")==EVALUATOR_REVISION: recovered[int(row["seed"])] = row
            except Exception: pass
    try:
        for ep in range(a.episodes):
            seed=a.seed_start+tid*1000+ep
            if seed in recovered: rows.append(recovered[seed]); continue
            obs,_=env.reset(seed=seed); policy.reset(); chunks=[deque(maxlen=64),deque(maxlen=64)]; trace=hashlib.sha256(); success=False; progress=[]; t0=time.perf_counter()
            for step in range(max_steps):
                head=runtime_rgb(obs["frames"]["head"]["rgb"]["data"]); heads=[head,head]; wrists=[runtime_rgb(obs["frames"]["left_wrist"]["rgb"]["data"]),runtime_rgb(obs["frames"]["right_wrist"]["rgb"]["data"])]
                q=[]
                for key in ("left","right"):
                    joints=np.asarray(obs[key]["joints"],np.float32); grip=np.asarray(obs[key]["gripper"],np.float32).reshape(-1); q.append(np.r_[joints,float(grip[0]>0.9)])
                pred=policy.infer(heads,wrists,q,a.task)
                for arm in range(2): chunks[arm].append((step,pred[arm]))
                action={}
                for arm,key in enumerate(("left","right")):
                    candidates=[c[step-born] for born,c in chunks[arm] if step-born<len(c)]; w=np.exp(-TEMPORAL_ENSEMBLE_DECAY*np.arange(len(candidates)-1,-1,-1)); w/=w.sum(); local=canonicalize_controller_action(np.sum(np.asarray(candidates)*w[:,None],axis=0))[0:8]; action[key]={"joints":local[:7].astype(np.float32),"gripper":np.asarray([local[7]],np.float32)}; trace.update(local.astype(np.float32).tobytes())
                obs,reward,term,trunc,info=env.step(action); progress.append(float(reward)); success=bool(info.get("success",False));
                if success or bool(np.asarray(term).all()) or bool(np.asarray(trunc).all()): break
            row={"task":a.task,"seed":seed,"success":success,"steps":step+1,"max_steps":max_steps,"final_stage_progress":float(progress[-1] if progress else 0),"max_stage_progress":float(max(progress) if progress else 0),"action_trace_sha256":trace.hexdigest(),"wall_seconds":time.perf_counter()-t0,"evaluator_revision":EVALUATOR_REVISION}; rows.append(row); journal.parent.mkdir(parents=True,exist_ok=True); journal.open("a").write(json.dumps(row)+"\n"); print(json.dumps(row),flush=True)
    finally: env.close()
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({"status":"complete","schema":"duobench-rdt-validation20-task-v1","task":a.task,"episodes":len(rows),"successes":sum(int(x["success"]) for x in rows),"success_rate":float(np.mean([x["success"] for x in rows])),"max_steps":max_steps,"evaluator_revision":EVALUATOR_REVISION,"policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_local_absolute_action8","normalization":"upstream RDT unified local joints + binary gripper; SigLIP processor; controller ctrlrange canonicalization","rows":rows},indent=2)+"\n")
if __name__=="__main__": main()
