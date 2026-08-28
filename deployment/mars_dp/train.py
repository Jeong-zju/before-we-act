from __future__ import annotations

import argparse, copy, hashlib, json, math, os, random, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel
from .common import atomic_json
from .dataset import MarsDPDataset, TaskBalancedBatchSampler
from .modeling import build_policy

def atomic_torch(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent); os.close(fd)
    try: torch.save(payload, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(16*1024*1024),b""): h.update(block)
    return h.hexdigest()
def lr_scheduler(opt, total, warmup, start=0):
    def rate(step):
        if step < warmup: return max((step + 1) / max(warmup, 1), 1e-8)
        return .5 * (1 + math.cos(math.pi * min(max((step - warmup) / max(total - warmup, 1), 0), 1)))
    for group in opt.param_groups: group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.LambdaLR(opt, rate, last_epoch=start-1)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True); p.add_argument("--output",required=True); p.add_argument("--steps",type=int,default=60000); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--workers",type=int,default=16); p.add_argument("--resume",action="store_true"); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    seed=20260827; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(8)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); target=min(a.steps,10) if a.smoke else a.steps
    ds=MarsDPDataset(a.data_root,out/"normalization.json"); sampler=TaskBalancedBatchSampler(ds.task_indices,a.batch_size,target,seed)
    dl=DataLoader(ds,batch_sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,prefetch_factor=2 if a.workers else None)
    dev=torch.device("cuda:0"); model=build_policy(ds.stats,dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6); start=0; saved={}
    latest=out/"last.pt"
    if a.resume and latest.is_file():
        saved=torch.load(latest,map_location=dev,weights_only=False); model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"]); start=int(saved["step"])
    sched=lr_scheduler(opt,a.steps,500,start)
    if saved.get("scheduler"): sched.load_state_dict(saved["scheduler"])
    ema_model=copy.deepcopy(model).to(dev); ema=EMAModel(ema_model,update_after_step=0,inv_gamma=1,power=.75,min_value=0,max_value=.9999)
    if saved.get("ema_model"): ema_model.load_state_dict(saved["ema_model"]); ema.optimization_step=int(saved.get("ema_step",start))
    model.train(); t0=time.monotonic()
    for step,batch in enumerate(dl,start=1):
        absolute=start+step
        if absolute>target: break
        prepared={"obs":{"head_cam":batch["head_cam"].to(dev,non_blocking=True).float().div_(255),"agent_pos":batch["agent_pos"].to(dev,non_blocking=True)},"action":batch["action"].to(dev,non_blocking=True)}
        opt.zero_grad(set_to_none=True); loss=model.compute_loss(prepared)
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss at {absolute}")
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)); opt.step(); sched.step(); ema.step(model)
        with torch.no_grad():
            for eb,b in zip(ema_model.buffers(),model.buffers()): eb.copy_(b)
        if absolute==1 or absolute%20==0: print(json.dumps({"step":absolute,"target_updates":target,"loss":float(loss),"grad_norm":grad,"lr":sched.get_last_lr()[0],"updates_per_second":step/max(time.monotonic()-t0,1e-6)}),flush=True)
        if absolute%5000==0 or absolute==target:
            payload={"schema":"mars-control.dp.checkpoint.v2","step":absolute,"model":model.state_dict(),"ema_model":ema_model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"ema_step":ema.optimization_step,"stats":ds.stats,"contract":"shared_weights_strict_local_rgb_qpos_to_absolute_action8","config":{"obs_steps":3,"horizon":8,"action_steps":8,"diffusion_train_steps":100,"batch_size":a.batch_size,"all_600_episodes_no_split":True,"task_balanced":True,"action_clip":"robofactory_pd_joint_pos_bounds","seed":seed}}
            atomic_torch(latest,payload)
            if absolute%5000==0: atomic_torch(out/f"checkpoint_{absolute:06d}.pt",payload)
    status={"status":"complete","step":target,"checkpoint":str(latest),"checkpoint_sha256":digest(latest),"episodes":600,"local_streams":1650,"indexed_local_timesteps":len(ds),"all_episodes":True,"completed_at":datetime.now(timezone.utc).isoformat()}; atomic_json(out/("smoke_status.json" if a.smoke else "status.json"),status)
if __name__=="__main__": main()
