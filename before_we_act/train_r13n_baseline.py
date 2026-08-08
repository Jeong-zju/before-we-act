"""Train the from-scratch R13N six-task ACT baseline."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader

from before_we_act.action_generator.r13n_baseline import R13NActionGenerator, load_r13n_config
from before_we_act.data.full_episode_windows import FullEpisodeActionWindows, ExactFiveTaskFullEpisodeSampler
from before_we_act.r13n import FULL_CACHE_PROTOCOL, TASKS, TASK_SPECS, sha256
from before_we_act.train_action_generator_r4 import (
    atomic_json, atomic_torch_save, capture_rng_state, device_batch,
    restore_rng_state, robustify_source_aware_history, seed_everything,
)


def learning_rate(training, update: int) -> float:
    base = float(training["learning_rate"])
    warmup = int(training["warmup_steps"])
    decay = int(training["decay_steps"])
    floor = float(training["decay_lr_ratio"])
    if update <= warmup:
        return base * update / warmup
    progress = min(1.0, max(0.0, (update - warmup) / (decay - warmup)))
    return base * (floor + (1-floor)*0.5*(1+math.cos(math.pi*progress)))


def validate_index(payload: dict) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("round") != "R13N"
        or payload.get("protocol_variant") != FULL_CACHE_PROTOCOL
        or tuple(payload.get("tasks", ())) != TASKS
        or payload.get("step_counts", {}).get("train")
        != {task:int(TASK_SPECS[task]["train_steps"]) for task in TASKS}
    ):
        raise ValueError("R13N full cache index identity differs")
    stats = payload.get("stats", {})
    if len(stats.get("a_mean", ())) != 8 or len(stats.get("a_std", ())) != 8:
        raise ValueError("R13N action normalization differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--full-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--heartbeat", default="")
    args = parser.parse_args()
    config_path = Path(args.config).resolve(strict=True)
    config = load_r13n_config(config_path)
    target_updates = int(args.updates or config.training["updates"])
    if not 1 <= target_updates <= int(config.training["updates"]):
        raise ValueError("R13N requested updates exceed frozen budget")
    if args.workers < 0:
        raise ValueError("R13N workers cannot be negative")
    seed = int(config.training["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    index_path = Path(args.full_index).resolve(strict=True)
    index = json.loads(index_path.read_text())
    validate_index(index)
    stats = {key:torch.as_tensor(index["stats"][key],dtype=torch.float32) for key in ("a_mean","a_std")}
    dataset = FullEpisodeActionWindows(index["episodes"],stats,split="train",cache_episodes=8,tasks=TASKS)
    resume = torch.load(args.resume,map_location="cpu",weights_only=False) if args.resume else None
    start_update = int(resume["update"]) if resume else 0
    if start_update >= target_updates:
        raise ValueError("R13N resume update is not below target")
    sampler = ExactFiveTaskFullEpisodeSampler(
        dataset,updates=target_updates,rows_per_task=int(config.training["rows_per_task"]),seed=seed,start_update=start_update,
    )
    loader = DataLoader(
        dataset,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,
        persistent_workers=args.workers>0,prefetch_factor=2 if args.workers>0 else None,
    )
    model = R13NActionGenerator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(config.training["learning_rate"]),weight_decay=float(config.training["weight_decay"]))
    if resume:
        if resume.get("round") != "R13N" or resume.get("model_id") != "b6_act_six_task" or resume.get("full_index_sha256") != sha256(index_path):
            raise ValueError("R13N resume identity differs")
        model.load_state_dict(resume["model"],strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        restore_rng_state(resume["rng_state"])
    output = Path(args.output).resolve()
    checkpoints = output/"checkpoints"; checkpoints.mkdir(parents=True,exist_ok=True)
    progress_path = output/"progress.jsonl"
    identity = {
        "schema_version":1,"round":"R13N","model_id":"b6_act_six_task",
        "config":str(config_path),"config_sha256":sha256(config_path),
        "full_index":str(index_path),"full_index_sha256":sha256(index_path),
        "tasks":list(TASKS),"step_counts":index["step_counts"],
        "stats":index["stats"],"from_scratch":True,"historical_checkpoint_loaded":False,
        "candidate_native":True,"created_at_epoch":time.time(),
    }
    atomic_json(output/"training_identity.json",identity)
    heartbeat = Path(args.heartbeat).resolve() if args.heartbeat else None
    stopping = False
    def request_stop(_signum,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    last: dict[str,float|int|str] = {}
    started=time.monotonic()
    def save(update: int, name: str) -> Path:
        path=checkpoints/name
        atomic_torch_save(path,{
            "schema_version":1,"round":"R13N","model_id":"b6_act_six_task",
            "update":update,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "config":dict(config.raw),"stats":stats,"full_index_sha256":identity["full_index_sha256"],
            "candidate_native":True,"last_metrics":last,"rng_state":capture_rng_state(),
        })
        return path
    update=start_update
    for update,cpu_batch in enumerate(loader,start=start_update+1):
        batch=device_batch(cpu_batch,device)
        batch["actions"],history_metrics=robustify_source_aware_history(batch,stats,config.training,update,seed)
        lr=learning_rate(config.training,update)
        for group in optimizer.param_groups: group["lr"]=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            losses=model.training_loss(
                batch,batch["spatial_tokens"],batch["spatial_view_mask"],batch["task_index"],
                batch["joint_actions"],batch["action_step_mask"].bool(),
            )
            total=losses["loss"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite R13N loss at update {update}")
        total.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(config.training["grad_clip"]))
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError(f"non-finite R13N grad norm at {update}")
        optimizer.step()
        last={
            "update":update,"stage":"training","loss":float(total.detach()),
            "grad_norm":float(grad_norm),"learning_rate":lr,**history_metrics,
            **{key:float(value.detach()) for key,value in losses.items() if key!="loss" and isinstance(value,torch.Tensor) and value.numel()==1},
        }
        if update==start_update+1 or update%int(config.training["progress_every"])==0:
            elapsed=time.monotonic()-started; completed=update-start_update
            row={**last,"target_updates":target_updates,"updates_per_hour":completed/max(elapsed,1e-6)*3600,"eta_hours":(target_updates-update)*elapsed/max(completed,1)/3600,"gpu_memory_gb":torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else 0.0,"time":time.time()}
            with progress_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(row,sort_keys=True)+"\n")
            print(json.dumps(row,sort_keys=True),flush=True)
            if heartbeat:
                atomic_json(heartbeat,{"producer":"train_r13n_baseline","pid":os.getpid(),"update":update,"target_updates":target_updates,"loss":row["loss"],"eta_hours":row["eta_hours"],"updated_at_epoch":time.time()})
        if update%int(config.training["checkpoint_every"])==0 or update==target_updates or stopping:
            latest=save(update,"checkpoint_latest.pt")
            if update==target_updates: save(update,f"checkpoint_{update:06d}.pt")
            print(json.dumps({"saved":str(latest),"update":update}),flush=True)
        if stopping:
            raise SystemExit(130)
    print(json.dumps({"complete":True,"round":"R13N","model_id":"b6_act_six_task","update":update}),flush=True)


if __name__ == "__main__":
    main()
