#!/usr/bin/env python3
"""Strict-local RDT-1B closed-loop evaluator for MARS-Control."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
from collections import deque
import numpy as np, torch, yaml
TASKS={"place_cube_in_cup":("PlaceCubeInCup-rf","place_cube_in_cup.yaml",2,500,"Place the cube in the cup"),"strike_cube_hard":("StrikeCubeHard-rf","strike_cube_hard.yaml",2,500,"Strike the cube hard"),"three_robots_place_shoes":("ThreeRobotsPlaceShoes-rf","three_robots_place_shoes.yaml",3,1200,"Three robots place shoes"),"four_robots_stack_cube":("FourRobotsStackCube-rf","four_robots_stack_cube.yaml",4,800,"Four robots stack the cube")}
REVISION="mars-rdt-strict-local-absolute-v1"
def scalar(x): return bool(np.asarray(x).reshape(-1)[0])
class Policy:
 def __init__(self,checkpoint,data_root):
  from PIL import Image; self.Image=Image; repo="/workspace/repos/rdt-1b"; sys.path.insert(0,repo)
  for name in list(sys.modules):
   if name in {"models","configs"} or name.startswith(("models.","configs.")): del sys.modules[name]
  os.chdir(repo); from configs.state_vec import STATE_VEC_IDX_MAPPING; from models.multimodal_encoder.siglip_encoder import SiglipVisionTower; from models.rdt_runner import RDTRunner
  self.dev=torch.device("cuda:0"); self.dtype=torch.bfloat16; self.vision=SiglipVisionTower("google/siglip-so400m-patch14-384",None).to(self.dev,dtype=self.dtype).eval(); self.model=RDTRunner.from_pretrained(checkpoint).to(self.dev,dtype=self.dtype).eval(); self.root=Path(data_root); self.ids=[STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]+[STATE_VEC_IDX_MAPPING["right_gripper_open"]]; self.history={}
 def reset(self): self.history={}
 def infer(self,image,qpos,task,arm):
  proc=self.vision.image_processor; bg=np.asarray([int(x*255) for x in proc.image_mean],np.uint8); bg=np.broadcast_to(bg,(proc.size["height"],proc.size["width"],3)).copy(); previous=self.history.get(arm,image); self.history[arm]=image.copy(); ordered=[]
  for local in (previous,image): ordered.extend((local,bg,bg))
  values=[]
  for im in ordered:
   pil=self.Image.fromarray(im); w,h=pil.size
   if w!=h:
    side=max(w,h); square=self.Image.new("RGB",(side,side),tuple(int(x*255) for x in proc.image_mean)); square.paste(pil,((side-w)//2,(side-h)//2)); pil=square
   values.append(proc.preprocess(pil,return_tensors="pt")["pixel_values"][0])
  images=torch.stack(values).to(self.dev,dtype=self.dtype)
  with torch.inference_mode():
   encoded=self.vision(images).reshape(1,-1,self.vision.hidden_size); state=torch.zeros(1,1,128,device=self.dev,dtype=self.dtype); mask=torch.zeros_like(state); state[0,0,self.ids[:7]]=torch.as_tensor(qpos[:7],device=self.dev,dtype=self.dtype); state[0,0,self.ids[7]]=torch.as_tensor(qpos[7:9].mean()/0.04,device=self.dev,dtype=self.dtype); mask[0,0,self.ids]=1; lang=torch.load(self.root/task/"lang_embed.pt",map_location="cpu").to(self.dev,dtype=self.dtype).unsqueeze(0); pred=self.model.predict_action(lang,torch.ones(lang.shape[:2],device=self.dev,dtype=torch.bool),encoded,state,mask,torch.full((1,),20,device=self.dev,dtype=torch.long))[0,:,self.ids].float().cpu().numpy()
  pred[:,7]=pred[:,7]*2-1; return pred.astype(np.float32)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--task",choices=TASKS,required=True); p.add_argument("--data-root",default="/workspace/datasets/mars_control"); p.add_argument("--robofactory-root",default="/workspace/repos/RoboFactory"); p.add_argument("--episodes",type=int,default=20); p.add_argument("--seed-start",type=int,default=20260820); p.add_argument("--max-steps",type=int); p.add_argument("--output",required=True); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
 random.seed(a.seed_start); np.random.seed(a.seed_start); torch.manual_seed(a.seed_start); root=Path(a.robofactory_root); sys.path.insert(0,str(root)); os.chdir(root); import gymnasium as gym; import tasks
 env_id,cfg_name,arms,formal_max,_=TASKS[a.task]; max_steps=a.max_steps or formal_max; policy=Policy(a.checkpoint,a.data_root)
 # Policy loading imports the RDT package and temporarily changes cwd to the
 # RDT repository.  RoboFactory resolves scene meshes (e.g. assets/scenes/
 # table/table.glb) via relative paths, so restore the simulator repository
 # cwd before creating any environment.
 os.chdir(root); cfg=root/"configs/table"/cfg_name; rows=[]
 for episode in range(a.episodes):
  seed=a.seed_start+episode; env=gym.make(env_id,config=str(cfg),obs_mode="rgb",control_mode="pd_joint_pos",render_mode="sensors",reward_mode="dense",sim_backend="cpu",render_backend="gpu",sensor_configs={"shader_pack":"default"},human_render_camera_configs={"shader_pack":"default"},viewer_camera_configs={"shader_pack":"default"}); obs,_=env.reset(seed=seed); policy.reset(); histories=[[] for _ in range(arms)]; trace=hashlib.sha256(); success=False; times=[]
  try:
   for step in range(max_steps):
    action={}
    for arm in range(arms):
     im=np.asarray(obs["sensor_data"][f"head_camera_agent{arm}"]["rgb"]); q=np.asarray(obs["agent"][f"panda-{arm}"]["qpos"]); im=im[0] if im.ndim==4 else im; q=q[0] if q.ndim==2 else q
     if im.shape!=(240,320,3) or im.dtype!=np.uint8 or q.shape!=(9,): raise ValueError(f"local observation drift: {im.shape}/{im.dtype}/{q.shape}")
     started=time.perf_counter(); chunk=policy.infer(im,q,a.task,arm); times.append(time.perf_counter()-started); histories[arm].append((step,chunk)); candidates=np.asarray([c[step-born] for born,c in histories[arm] if step-born<len(c)]); weights=np.exp(-.01*np.arange(len(candidates)-1,-1,-1)); weights/=weights.sum(); row=np.sum(candidates*weights[:,None],0); space=env.action_space.spaces[f"panda-{arm}"]; row=np.clip(row,np.asarray(space.low),np.asarray(space.high)).astype(np.float32); action[f"panda-{arm}"]=row; trace.update(row.tobytes())
    obs,_,term,trunc,info=env.step(action); success=scalar(info.get("success",False))
    if success or scalar(term) or scalar(trunc): break
  finally: env.close()
  row={"episode":episode,"seed":seed,"success":success,"steps":step+1,"mean_inference_seconds":float(np.mean(times)),"action_trace_sha256":trace.hexdigest()}; rows.append(row); print(json.dumps(row),flush=True)
 out={"schema":"mars-control.rdt.smoke.v1" if a.smoke else "mars-control.rdt.validation20.task.v1","status":"complete","task":a.task,"episodes":len(rows),"successes":sum(int(x["success"]) for x in rows),"success_rate":sum(int(x["success"]) for x in rows)/len(rows),"max_steps":max_steps,"evaluator_revision":REVISION,"normalization":{"rgb":"SigLIP processor after mean-color square padding","state":"raw joints plus finger mean/0.04 into RDT unified vector","action":"RDT unified absolute joints plus gripper [0,1] decoded to [-1,1], then env clip"},"policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_absolute_action8","rows":rows}; path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__": main()
