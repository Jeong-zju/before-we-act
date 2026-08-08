"""Candidate-native six-task closed-loop evaluator for R13N."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import robofactory  # noqa: F401

from before_we_act.action_generator.r13n_baseline import R13NActionGenerator, load_r13n_config
from before_we_act.benchmark import get_task
from before_we_act.evaluate_action_generator import TemporalChunkEnsembler, patch_means
from before_we_act.r13n import (
    TASKS,
    TASK_SPECS,
    camera_sensor_key,
    clamp_action_to_space,
    sha256,
)
from before_we_act.spatial_observation import R12SpatialObservationEncoder
from before_we_act.train_action_generator_r4 import atomic_json


def reset_reproducibly(env, seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def rgb(sensor_data, view: str) -> np.ndarray:
    value=np.asarray(sensor_data[camera_sensor_key(view)]["rgb"])
    value=value[0] if value.ndim==4 else value
    if tuple(value.shape)!=(480,640,3): raise ValueError(f"R13N {view} RGB shape differs: {tuple(value.shape)}")
    return np.asarray(value,dtype=np.uint8)


class TeamHistory:
    def __init__(self, arms, camera_order):
        self.arms=tuple(arms); self.camera_order=tuple(camera_order); self.rows=[]

    def batch(self, observation, previous_action, device):
        sensors=observation["sensor_data"]
        images=[rgb(sensors,view) for view in self.camera_order]
        visual=np.zeros((16,15),dtype=np.float32); view_mask=np.zeros(5,dtype=np.float32); raw=np.zeros((5,3,480,640),dtype=np.uint8)
        for index,image in enumerate(images):
            visual[:,index*3:(index+1)*3]=patch_means(image); view_mask[index]=1; raw[index]=image.transpose(2,0,1)
        qpos=np.zeros((4,9),dtype=np.float32); actions=np.zeros((4,8),dtype=np.float32)
        for index,arm in enumerate(self.arms):
            value=np.asarray(observation["agent"][f"panda-{arm}"]["qpos"]); qpos[index]=value[0] if value.ndim==2 else value
            if previous_action is not None: actions[index]=previous_action[f"panda-{arm}"]
        self.rows.append((visual,view_mask,qpos,actions,raw)); self.rows=self.rows[-3:]
        padded=[self.rows[0]]*(3-len(self.rows))+self.rows
        agent_mask=torch.zeros((1,4),dtype=torch.bool,device=device); agent_mask[:,:len(self.arms)]=True
        return {
            "visual":torch.from_numpy(np.stack([row[0] for row in padded])).unsqueeze(0).to(device),
            "view_mask":torch.from_numpy(np.stack([row[1] for row in padded])).unsqueeze(0).to(device),
            "qpos":torch.from_numpy(np.stack([row[2] for row in padded])).unsqueeze(0).to(device),
            "actions":torch.from_numpy(np.stack([row[3] for row in padded])).unsqueeze(0).to(device),
            "agent_mask":agent_mask,"raw_fixed_rgb":torch.from_numpy(raw).unsqueeze(0).to(device),
            "spatial_view_mask":torch.from_numpy(view_mask).unsqueeze(0).bool().to(device),
        }


def make_env(task: str):
    spec=get_task(task)
    return gym.make(spec["env_id"],config=f"/workspace/RoboFactory/{spec['config']}",obs_mode="rgb",control_mode="pd_joint_pos",render_mode="sensors",reward_mode="dense",sim_backend="cpu",sensor_configs=dict(shader_pack="default",width=640,height=480),human_render_camera_configs=dict(shader_pack="default"),viewer_camera_configs=dict(shader_pack="default"))


def terminal_info(info) -> dict:
    result={}
    for key,value in info.items():
        array=np.asarray(value)
        if array.size==1:
            item=array.reshape(-1)[0]
            if isinstance(item,(np.bool_,bool)): result[key]=bool(item)
            elif isinstance(item,(np.integer,int)): result[key]=int(item)
            elif isinstance(item,(np.floating,float)) and np.isfinite(item): result[key]=float(item)
    return result


def open_video(path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.stem}.{os.getpid()}.{time.time_ns()}.mp4")
    try: writer=imageio.get_writer(temporary,fps=15,codec="libx264",quality=7,macro_block_size=None)
    except Exception: writer=imageio.get_writer(temporary,fps=15,codec="mpeg4",quality=7,macro_block_size=None)
    return temporary,writer


@torch.inference_mode()
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--vision-artifact",required=True); parser.add_argument("--task",choices=TASKS,required=True)
    parser.add_argument("--stage",choices=("discovery","validation","formal"),required=True)
    parser.add_argument("--seed-file",required=True); parser.add_argument("--episodes",type=int,default=20)
    parser.add_argument("--device",default="cuda:0"); parser.add_argument("--output",required=True)
    parser.add_argument("--heartbeat",default=""); parser.add_argument("--resume-log",default="")
    parser.add_argument("--video-dir",default="")
    args=parser.parse_args()
    if args.episodes!=20: raise ValueError("R13N stages require exactly 20 episodes per task")
    torch.set_num_threads(12); device=torch.device(args.device); config=load_r13n_config(args.config)
    checkpoint_path=Path(args.checkpoint).resolve(strict=True); checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if checkpoint.get("round")!="R13N" or checkpoint.get("model_id")!="b6_act_six_task" or int(checkpoint.get("update",-1))!=int(config.training["updates"]): raise ValueError("R13N closed-loop checkpoint identity differs")
    model=R13NActionGenerator(config).to(device); model.load_state_dict(checkpoint["model"],strict=True); model.eval()
    spatial=R12SpatialObservationEncoder(config.observation,args.vision_artifact,inference_batch_size=5).to(device).eval()
    stats={key:torch.as_tensor(checkpoint["stats"][key],device=device) for key in ("a_mean","a_std")}
    seed_path=Path(args.seed_file).resolve(strict=True); seed_bytes=seed_path.read_bytes(); seed_manifest=json.loads(seed_bytes)
    if seed_manifest.get("round")!="R13N" or seed_manifest.get("task")!=args.task or seed_manifest.get("stage")!=args.stage: raise ValueError("R13N seed identity differs")
    requested=[int(seed) for seed in seed_manifest["seeds"]]
    if len(requested)!=20 or len(set(requested))!=20: raise ValueError("R13N seed count differs")
    recovered=[]
    if args.resume_log and Path(args.resume_log).is_file():
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if row.get("task")==args.task and row.get("stage")==args.stage and row.get("seed") in requested and isinstance(row.get("success"),bool): recovered.append(row)
    recovered=list({row["seed"]:row for row in recovered}.values()); completed={row["seed"] for row in recovered}
    heartbeat=Path(args.heartbeat).resolve() if args.heartbeat else None
    rows=[]; latencies=[]
    env=make_env(args.task); spec=TASK_SPECS[args.task]; arms=get_task(args.task)["agents"]
    video_dir=Path(args.video_dir).resolve() if args.video_dir else None
    try:
        for episode_index,seed in enumerate(requested):
            if seed in completed: continue
            observation,_=reset_reproducibly(env,seed); history=TeamHistory(arms,spec["camera_order"]); ensemble=TemporalChunkEnsembler(arms,decay=float(config.raw["evaluation"]["temporal_ensemble_decay"])); previous_action=None; success=False; info={}
            episode_clip_elements=0; episode_action_elements=0
            temporary=None; writer=None
            if video_dir:
                temporary,writer=open_video(video_dir/f"{args.task}_{args.stage}_{seed}.mp4")
            try:
                for step in range(int(spec["max_steps"])):
                    if writer is not None and step%5==0: writer.append_data(rgb(observation["sensor_data"],"global"))
                    batch=history.batch(observation,previous_action,device); task_index=torch.tensor([TASKS.index(args.task)],device=device)
                    if device.type=="cuda": torch.cuda.synchronize(device)
                    started=time.perf_counter_ns()
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                        spatial_tokens,spatial_mask=spatial(batch["raw_fixed_rgb"],batch["spatial_view_mask"])
                        proposals=model.sample(batch,spatial_tokens=spatial_tokens,spatial_view_mask=spatial_mask,task_index=task_index)
                    if device.type=="cuda": torch.cuda.synchronize(device)
                    latencies.append((time.perf_counter_ns()-started)/1e6)
                    normalized=proposals.actions[0,0,:len(arms)]; raw_actions=(normalized*stats["a_std"][None,None]+stats["a_mean"][None,None]).float().cpu().numpy(); action=ensemble.append_and_select(step,raw_actions)
                    if not all(np.isfinite(value).all() for value in action.values()): raise FloatingPointError("R13N produced non-finite action")
                    action,clipped,total=clamp_action_to_space(env.action_space,action)
                    episode_clip_elements+=clipped; episode_action_elements+=total
                    previous_action={key:value.copy() for key,value in action.items()}; observation,_,terminated,truncated,info=env.step(action); success=bool(np.asarray(info.get("success",False)).all())
                    if heartbeat and (step==0 or step%20==0): atomic_json(heartbeat,{"producer":"evaluate_r13n","pid":os.getpid(),"task":args.task,"stage":args.stage,"episode_index":episode_index,"episodes":20,"seed":seed,"step":step+1,"max_steps":int(spec["max_steps"]),"updated_at_epoch":time.time()})
                    if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()): break
            finally:
                if writer is not None: writer.close()
            video=None
            if video_dir and temporary:
                category="success" if success else "failure"; target=video_dir/f"{args.task}_{category}.mp4"
                if not target.exists(): os.replace(temporary,target); video=str(target)
                else: temporary.unlink(missing_ok=True)
            row={"schema_version":1,"round":"R13N","model_id":"b6_act_six_task","task":args.task,"stage":args.stage,"seed":seed,"success":success,"steps":step+1,"candidate_native":True,"fallback_used":False,"safety_system_success":success,"physical_action_clip_elements":episode_clip_elements,"physical_action_elements":episode_action_elements,"terminal_info":terminal_info(info),"video":video}
            rows.append(row); print(json.dumps(row,sort_keys=True),flush=True)
    finally: env.close()
    rows=recovered+rows; rows.sort(key=lambda row:requested.index(row["seed"])); values=np.asarray(latencies,dtype=np.float64)
    physical_clip_elements=sum(int(row.get("physical_action_clip_elements",0)) for row in rows)
    physical_action_elements=sum(int(row.get("physical_action_elements",0)) for row in rows)
    result={"schema_version":1,"round":"R13N","model_id":"b6_act_six_task","task":args.task,"stage":args.stage,"checkpoint":str(checkpoint_path),"checkpoint_sha256":sha256(checkpoint_path),"episodes":len(rows),"successes":sum(row["success"] for row in rows),"candidate_native_episodes":sum(row.get("candidate_native") is True for row in rows),"fallback_episodes":sum(row.get("fallback_used") is True for row in rows),"rows":rows,"latency_ms":{"samples":len(latencies),"p50":float(np.percentile(values,50)) if len(values) else None,"p95":float(np.percentile(values,95)) if len(values) else None},"normalized_clip":float(model.normalized_clip),"physical_action_clip":{"elements":physical_action_elements,"clipped_elements":physical_clip_elements,"fraction":physical_clip_elements/max(physical_action_elements,1)},"seed_protocol":{"source":str(seed_path),"sha256":hashlib.sha256(seed_bytes).hexdigest()},"policy_inputs":"manifest-selected native RGB plus causal qpos/executed-action history","privileged_inputs":False,"candidate_native":True,"temporal_aggregation":"exponential action-chunk ensemble decay=0.01"}
    if len(rows)!=20 or result["candidate_native_episodes"]!=20 or result["fallback_episodes"]!=0: raise ValueError("R13N rollout completeness/coverage differs")
    atomic_json(Path(args.output),result)
    if heartbeat: atomic_json(heartbeat,{"producer":"evaluate_r13n","status":"PASSED","task":args.task,"stage":args.stage,"successes":result["successes"],"episodes":20,"updated_at_epoch":time.time()})
    print(json.dumps(result|{"rows":"saved"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
