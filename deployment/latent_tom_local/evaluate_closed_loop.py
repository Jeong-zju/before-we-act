from __future__ import annotations

import argparse, json, os, tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
import robofactory.tasks  # noqa: F401

from local_dataset import TASKS
from local_policy import LocalLatentToMPolicy

MAX_STEPS = {"lift_barrier":500, "camera_alignment":1500, "long_pipeline_delivery":1500,
             "take_photo":1500, "pass_shoe":500, "place_food":500}
ACTION_LOW=np.asarray([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973,-1.0],np.float32)
ACTION_HIGH=np.asarray([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973,1.0],np.float32)

def atomic_json(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent)
    with os.fdopen(fd,"w") as f:
        json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def scalar_bool(value):
    if torch.is_tensor(value): return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])

def make_env(config):
    env_id=yaml.safe_load(open(config))["task_name"]+"-rf"
    return gym.make(env_id,config=str(config),obs_mode="rgb",control_mode="pd_joint_pos",
                    render_mode="rgb_array",num_envs=1,sim_backend="cpu",
                    sensor_configs={"shader_pack":"default","width":320,"height":240},
                    human_render_camera_configs={"shader_pack":"default"},
                    viewer_camera_configs={"shader_pack":"default"})

def local_obs(obs, agent, task_id, image_hist, qpos_hist, device):
    image=obs["sensor_data"][f"head_camera_agent{agent}"]["rgb"][0].cpu().numpy()
    qpos=obs["agent"][f"panda-{agent}"]["qpos"][0,:9].cpu().numpy()
    image_hist.append(image); qpos_hist.append(qpos)
    while len(image_hist)<2: image_hist.appendleft(image_hist[0])
    while len(qpos_hist)<2: qpos_hist.appendleft(qpos_hist[0])
    onehot=np.zeros(6,np.float32); onehot[task_id]=1
    return {"image":torch.from_numpy(np.moveaxis(np.asarray(image_hist),-1,1).astype(np.float32)/255)[None].to(device),
            "qpos":torch.from_numpy(np.asarray(qpos_hist,np.float32))[None].to(device),
            "task":torch.from_numpy(onehot)[None].to(device)}

@torch.no_grad()
def evaluate(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if payload.get("contract")!="shared_weights_local_rgb_qpos_task_to_local_action8":
        raise RuntimeError("checkpoint local-only contract mismatch")
    model=LocalLatentToMPolicy().to(args.device); model.load_state_dict(payload.get("ema_model",payload["model"])); model.eval()
    rows={}
    for task in args.task or TASKS:
        result_path=out/f"{task}.json"; existing={}
        if result_path.is_file():
            try: existing=json.loads(result_path.read_text())
            except json.JSONDecodeError: pass
        episode_map={int(r["episode"]):r for r in existing.get("episodes_detail",[]) if not r.get("error")}
        for episode in range(args.episodes):
            if episode in episode_map: continue
            env=None; row={"episode":episode,"seed":args.seed+episode,"success":False,"steps":0}
            try:
                # Diffusion inference is stochastic.  Bind its noise to the
                # published rollout seed so Validation20 is exactly repeatable.
                torch.manual_seed(args.seed + episode)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed + episode)
                config=Path(args.config_root)/f"{task}.yaml"; env=make_env(config); obs,_=env.reset(seed=args.seed+episode)
                agents=len(yaml.safe_load(config.read_text())["agents"])
                images=[deque(maxlen=2) for _ in range(agents)]; qposes=[deque(maxlen=2) for _ in range(agents)]
                chunks=[None]*agents; offsets=[0]*agents; task_id=TASKS.index(task)
                for step in range(MAX_STEPS[task]):
                    # Batching is only a scheduling optimization: each row is
                    # one actor's own RGB/qpos/task and the network has no
                    # cross-row operation in eval mode.  Every actor still
                    # receives only its corresponding output row.
                    own_obs=[local_obs(obs,agent,task_id,images[agent],qposes[agent],args.device)
                             for agent in range(agents)]
                    replanners=[agent for agent in range(agents)
                                if chunks[agent] is None or offsets[agent]>=args.replan_interval]
                    if replanners:
                        batched={key:torch.cat([own_obs[agent][key] for agent in replanners],dim=0)
                                 for key in ("image","qpos","task")}
                        predicted=model.predict_chunk(batched,steps=args.diffusion_steps).float().cpu().numpy()
                        for row_index,agent in enumerate(replanners):
                            chunks[agent]=predicted[row_index]
                            offsets[agent]=1  # [t-1,t] predicts chunk indices [t-1,t,...]
                    action={}
                    for agent in range(agents):
                        action[f"panda-{agent}"]=np.clip(
                            chunks[agent][offsets[agent]],ACTION_LOW,ACTION_HIGH
                        ).astype(np.float32)
                        offsets[agent]+=1
                    obs,_,terminated,truncated,info=env.step(action)
                    row["success"]=scalar_bool(info.get("success",False)); row["steps"]=step+1
                    if row["success"] or scalar_bool(terminated) or scalar_bool(truncated): break
            except Exception as exc: row["error"]=f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None: env.close()
                torch.cuda.empty_cache()
            episode_map[episode]=row
            detail=[episode_map[k] for k in sorted(episode_map)]; successes=sum(bool(x["success"]) for x in detail)
            atomic_json(result_path,{"schema":"bwa.latent_tom.validation20.task.v1","task":task,
                        "status":"failed" if any(x.get("error") for x in detail) else ("complete" if len(detail)==args.episodes else "running"),
                        "episodes":len(detail),"target_episodes":args.episodes,"successes":successes,
                        "success_rate":successes/len(detail),"max_steps":MAX_STEPS[task],
                        "policy_contract":"shared_weights_strict_local_rgb_qpos_to_local_action8",
                        "episodes_detail":detail,"updated_at":datetime.now(timezone.utc).isoformat()})
        rows[task]=json.loads(result_path.read_text())
    errors=[e for r in rows.values() for e in r["episodes_detail"] if e.get("error")]
    summary={"schema":"bwa.latent_tom.validation20.v1","status":"failed" if errors else "complete",
             "episodes_per_task":args.episodes,"total_episodes":sum(r["episodes"] for r in rows.values()),
             "tasks":rows,"macro_success_rate":float(np.mean([r["success_rate"] for r in rows.values()])),
             "seed_base":args.seed,"sim_backend":"cpu","replan_interval":args.replan_interval,
             "policy_contract":"same_checkpoint_per_actor_strict_local_rgb_qpos_to_local_action8",
             "completed_at":datetime.now(timezone.utc).isoformat()}
    atomic_json(out/"summary.json",summary)
    if errors: raise RuntimeError(f"{len(errors)} validation episodes failed")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True)
    p.add_argument("--config-root",default="/workspace/repos/RoboFactory/robofactory/configs/table")
    p.add_argument("--task",action="append",choices=TASKS); p.add_argument("--episodes",type=int,default=20)
    p.add_argument("--seed",type=int,default=20260820); p.add_argument("--device",default="cuda:0")
    p.add_argument("--diffusion-steps",type=int,default=20); p.add_argument("--replan-interval",type=int,default=8)
    evaluate(p.parse_args())
