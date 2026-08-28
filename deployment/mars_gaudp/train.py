from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from .common import atomic_json
from .dataset import MarsGauDPDataset, TaskBalancedBatchSampler
from .model import build_model

def digest(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(16*1024*1024),b""): h.update(b)
    return h.hexdigest()
def atomic_torch(path,payload):
    path=Path(path); fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent); os.close(fd)
    try: torch.save(payload,tmp); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def schedule(opt,total,warmup,start=0):
    def rate(step):
        if step<warmup: return max((step+1)/max(warmup,1),1e-8)
        return .5*(1+math.cos(math.pi*min(max((step-warmup)/max(total-warmup,1),0),1)))
    for group in opt.param_groups: group.setdefault("initial_lr",group["lr"])
    return torch.optim.lr_scheduler.LambdaLR(opt,rate,last_epoch=start-1)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True); p.add_argument("--cache-root",required=True); p.add_argument("--output",required=True); p.add_argument("--steps",type=int,default=60000); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--workers",type=int,default=8); p.add_argument("--smoke",action="store_true"); p.add_argument("--resume",action="store_true"); a=p.parse_args()
    seed=20260827; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=True
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); target=10 if a.smoke else a.steps
    ds=MarsGauDPDataset(a.data_root,a.cache_root,out/"normalization.json"); sampler=TaskBalancedBatchSampler(ds.task_indices,a.batch_size,target,seed); dl=DataLoader(ds,batch_sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,prefetch_factor=2 if a.workers else None)
    dev=torch.device("cuda:0"); model=build_model(ds.stats,dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6); ema=copy.deepcopy(model).to(dev); start=0; latest=out/"last.pt"; saved={}
    if a.resume and latest.is_file():
        saved=torch.load(latest,map_location=dev,weights_only=False); model.load_state_dict(saved["model"]); ema.load_state_dict(saved["ema_model"]); opt.load_state_dict(saved["optimizer"]); start=int(saved["step"])
    sched=schedule(opt,a.steps,500,start)
    if saved.get("scheduler"): sched.load_state_dict(saved["scheduler"])
    model.train(); t0=time.monotonic()
    for local,batch in enumerate(dl,1):
        step=start+local
        if step>target: break
        batch={"obs":{k:v.to(dev,non_blocking=True) for k,v in batch["obs"].items()},"action":batch["action"].to(dev,non_blocking=True)}; opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss=model.compute_loss(batch)
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss at {step}")
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)); opt.step(); sched.step()
        with torch.no_grad():
            decay=0.9
            for ep,mp in zip(ema.parameters(),model.parameters()): ep.mul_(decay).add_(mp,alpha=1-decay)
            for eb,mb in zip(ema.buffers(),model.buffers()): eb.copy_(mb)
        if step==1 or step%20==0: print(json.dumps({"step":step,"target_updates":target,"loss":float(loss),"grad_norm":grad,"lr":sched.get_last_lr()[0],"updates_per_second":local/max(time.monotonic()-t0,1e-6),"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated()}),flush=True)
        if step%5000==0 or step==target:
            payload={"schema":"mars-control.gaudp.checkpoint.v1","step":step,"model":model.state_dict(),"ema_model":ema.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"stats":ds.stats,"contract":"mars-control.gaudp.shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8","config":{"obs_steps":3,"horizon":8,"action_steps":6,"diffusion_train_steps":100,"batch_size":a.batch_size,"all_600_episodes_no_split":True,"task_balanced":True,"gaussian_cache":"NoPoSplat self-coordinate local single-view","precision":"bfloat16 autocast","seed":seed}}
            atomic_torch(latest,payload)
            if step%5000==0: atomic_torch(out/f"checkpoint_{step:06d}.pt",payload)
    status={"status":"complete","step":target,"checkpoint":str(latest),"checkpoint_sha256":digest(latest),"episodes":ds.stats["episodes"],"local_streams":ds.stats["local_streams"],"indexed_local_timesteps":ds.stats["indexed_local_timesteps"],"all_episodes":True,"completed_at":datetime.now(timezone.utc).isoformat()}; atomic_json(out/("smoke_status.json" if a.smoke else "status.json"),status)
if __name__=="__main__": main()
