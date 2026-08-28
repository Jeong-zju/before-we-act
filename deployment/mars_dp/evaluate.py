from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from collections import deque
from pathlib import Path
import gymnasium as gym
import numpy as np, torch
from .common import ENVS, atomic_json
from .modeling import load_policy

# v2 follows the official 3/8/8 window:
#   obs=[t-2,t-1,t], action=[t-2,t-1,t,...,t+5].
# DiffusionUnetImagePolicy.predict_action() exposes ``action`` starting at
# horizon index n_obs_steps-1 (index 2), i.e. the action for the current env
# step t. Consuming action_pred from index zero would introduce a two-step lag.
# Keep this revision explicit so old validation files
# cannot be mistaken for results from the corrected executor.
EVALUATOR_REVISION = "dp-official-obs3-horizon8-exec6-rgb-v4-clipped-targets"
def scalar(v): return bool(np.asarray(v).reshape(-1)[0])
def local_obs(obs, arm):
    image=np.asarray(obs["sensor_data"][f"head_camera_agent{arm}"] ["rgb"]); image=image[0] if image.ndim==4 else image
    q=np.asarray(obs["agent"][f"panda-{arm}"]["qpos"]); q=q[0] if q.ndim==2 else q
    if image.shape != (240,320,3) or image.dtype != np.uint8: raise ValueError(f"RGB contract drift: {image.shape} {image.dtype}")
    if q.shape != (9,): raise ValueError(f"qpos contract drift: {q.shape}")
    return image, q
@torch.no_grad()
def episode(policy, task, rf_root, seed, device, max_steps, replan_interval):
    os.chdir(rf_root); sys.path.insert(0, str(rf_root)); import tasks  # noqa: F401
    env_id,cfg_name,_ = ENVS[task]; cfg=str(Path(rf_root)/"configs/table"/cfg_name)
    env=gym.make(env_id,config=cfg,obs_mode="rgb",control_mode="pd_joint_pos",render_mode="sensors",reward_mode="dense",sim_backend="cpu",sensor_configs={"shader_pack":"default"},human_render_camera_configs={"shader_pack":"default"},viewer_camera_configs={"shader_pack":"default"})
    obs,_=env.reset(seed=int(seed))
    # Environment construction/reset consumes Torch random numbers.  Seed the
    # diffusion sampler only after reset so repeating a published episode seed
    # yields an identical action trace.
    torch.manual_seed(int(seed))
    arms=range({"place_cube_in_cup":2,"strike_cube_hard":2,"three_robots_place_shoes":3,"four_robots_stack_cube":4}[task]); histories=[deque(maxlen=3) for _ in arms]; pending=[deque() for _ in arms]; trace=hashlib.sha256(); success=False; times=[]
    try:
        for step in range(max_steps):
            batch_images=[]; batch_q=[]
            for arm in arms:
                image,q=local_obs(obs,arm); histories[arm].append((image,q)); rows=list(histories[arm])
                while len(rows)<3: rows.insert(0,rows[0])
                batch_images.append(np.stack([x[0] for x in rows])); batch_q.append(np.stack([x[1] for x in rows]))
            if step % replan_interval == 0 or any(not queue for queue in pending):
                # SAPIEN may use CUDA RNG internally between replans.  Reset
                # both generators at every sampling boundary so the DDPM noise
                # is tied to (episode seed, replan index), not process history.
                # The renderer itself is not bit-exact across fresh envs, so
                # the full visual rollout can still differ at pixel level.
                sample_seed = int(seed) + step // max(replan_interval, 1)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(sample_seed)
                images=torch.from_numpy(np.stack(batch_images)).permute(0,1,4,2,3).to(device).float().div_(255.0); q=torch.from_numpy(np.stack(batch_q)).to(device).float()
                started=time.perf_counter()
                prediction = policy.predict_action({"head_cam":images,"agent_pos":q})
                # Dataset windows are [obs(t-2), obs(t-1), obs(t)] ->
                # action[t-2], action[t-1], action[t], ... . The policy's
                # convenience ``action`` field starts at horizon index
                # n_obs_steps-1 (index 2), which is exactly action[t]. With an
                # eight-step horizon this field contains six executable rows.
                pred=prediction["action"][:, :replan_interval].float().cpu().numpy()
                times.append(time.perf_counter()-started)
                # Standard Diffusion Policy receding horizon: sample one local
                # eight-action chunk per actor and execute it open-loop before
                # observing/replanning. Arms are only GPU-batched; no actor's
                # observation or action enters another actor's sample.
                for arm,row in zip(arms,pred): pending[arm]=deque(row[:replan_interval])
            actions={}
            for arm in arms:
                action=np.asarray(pending[arm].popleft(),np.float32)
                space=env.action_space.spaces[f"panda-{arm}"]; action=np.clip(action,space.low,space.high).astype(np.float32); trace.update(action.tobytes()); actions[f"panda-{arm}"]=action
            obs,_,terminated,truncated,info=env.step(actions); success=scalar(info.get("success",False))
            if success or scalar(terminated) or scalar(truncated): break
    finally: env.close()
    return {"seed":int(seed),"success":bool(success),"steps":step+1,"mean_inference_seconds":float(np.mean(times)) if times else None,"p95_inference_seconds":float(np.quantile(times,.95)) if times else None,"action_trace_sha256":trace.hexdigest(),"evaluator_revision":EVALUATOR_REVISION}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--task",choices=ENVS,required=True); p.add_argument("--robofactory-root",required=True); p.add_argument("--output",required=True); p.add_argument("--episodes",type=int,default=20); p.add_argument("--seed-start",type=int,required=True); p.add_argument("--max-steps",type=int,required=True); p.add_argument("--smoke",action="store_true"); p.add_argument("--inference-steps",type=int,default=20); p.add_argument("--replan-interval",type=int,default=6); a=p.parse_args()
    device=torch.device("cuda:0"); policy,payload=load_policy(a.checkpoint,device,a.inference_steps); out=Path(a.output); rows=[]; existing={}
    journal=out.with_suffix(".jsonl")
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                x=json.loads(line)
                # Never recycle outcomes produced by an older action-indexing
                # contract.  The previous implementation silently did this
                # when rerunning Validation20 into the same output directory.
                if x.get("evaluator_revision") == EVALUATOR_REVISION: existing[int(x["seed"])]=x
            except Exception: pass
    for i in range(a.episodes):
        seed=a.seed_start+i; row=existing.get(seed) or episode(policy,a.task,a.robofactory_root,seed,device,a.max_steps,a.replan_interval); rows.append(row)
        if seed not in existing: journal.parent.mkdir(parents=True,exist_ok=True); journal.open("a").write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
    result={"schema":"mars-control.dp.smoke.v1" if a.smoke else "mars-control.dp.validation20.task.v1","status":"complete","task":a.task,"episodes":len(rows),"successes":sum(int(x["success"]) for x in rows),"success_rate":sum(int(x["success"]) for x in rows)/len(rows),"rows":rows,"checkpoint":a.checkpoint,"checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),"evaluator_revision":EVALUATOR_REVISION,"rgb_preprocessing":"uint8_div_255_to_unit_float","state_action_codec":"corpus_minmax_to_minus1_plus1_and_inverse_once","policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_absolute_action8","max_steps":a.max_steps,"replan_interval":a.replan_interval}
    atomic_json(out,result)
if __name__=="__main__": main()
