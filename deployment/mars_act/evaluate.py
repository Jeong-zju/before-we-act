#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
import numpy as np, torch, yaml
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'stereo_core'))
from stereo_core.train_act import ACT
TASKS={'place_cube_in_cup':('PlaceCubeInCup-rf','place_cube_in_cup.yaml',2,500),'strike_cube_hard':('StrikeCubeHard-rf','strike_cube_hard.yaml',2,500),'three_robots_place_shoes':('ThreeRobotsPlaceShoes-rf','three_robots_place_shoes.yaml',3,1200),'four_robots_stack_cube':('FourRobotsStackCube-rf','four_robots_stack_cube.yaml',4,800)}
EVALUATOR_REVISION='rgb-uint8-to-unit-float-v2'
def scalar(x): return bool(np.asarray(x).reshape(-1)[0])
@torch.no_grad()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--task',choices=TASKS,required=True); p.add_argument('--robofactory-root',required=True); p.add_argument('--episodes',type=int,default=20); p.add_argument('--seed-start',type=int,default=20260820); p.add_argument('--max-steps',type=int); p.add_argument('--output',required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
 import gymnasium as gym; os.chdir(a.robofactory_root); import tasks # register official branch
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=saved['config']; st={k:np.asarray(v,np.float32) for k,v in saved['stats'].items()}; dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 model=ACT(c['state_dim'],c['action_dim'],c.get('horizon',100),c.get('d_model',384),c.get('enc_layers',4),c.get('dec_layers',7),vision_backbone=c.get('vision_backbone','resnet18'),dino_model=c.get('dino_model','facebook/dinov3-vitb16-pretrain-lvd1689m')).to(dev); model.load_state_dict(saved['model']); model.eval()
 env_id,cfg_name,arms,default_max=TASKS[a.task]; cfg=str(Path(a.robofactory_root)/'configs/table'/cfg_name); env=gym.make(env_id,config=cfg,obs_mode='rgb',control_mode='pd_joint_pos',render_mode='sensors',reward_mode='dense',sim_backend='cpu',sensor_configs={'shader_pack':'default'},human_render_camera_configs={'shader_pack':'default'},viewer_camera_configs={'shader_pack':'default'})
 rows=[]; max_steps=a.max_steps or default_max
 for ep in range(a.episodes):
  seed=a.seed_start+ep; obs,_=env.reset(seed=seed); history=[]; success=False
  histories=[[] for _ in range(arms)]
  for step in range(max_steps):
   images=[]; qposes=[]
   for arm in range(arms):
    im=np.asarray(obs['sensor_data'][f'head_camera_agent{arm}']['rgb']); im=im[0] if im.ndim==4 else im; q=np.asarray(obs['agent'][f'panda-{arm}']['qpos']); q=q[0] if q.ndim==2 else q
    # Match the training path exactly: train.py converts uint8 RGB to [0, 1]
    # before the ACT visual backbone.  Feeding raw [0, 255] values here shifts
    # every visual feature far outside the training distribution.
    image=torch.from_numpy(im.copy()).permute(2,0,1)[None].to(dev).float().div_(255)
    if image.max() > 1.001 or image.min() < -0.001:
     raise ValueError(f'RGB preprocessing contract violated: range=({image.min().item():.4f},{image.max().item():.4f})')
    q=((torch.from_numpy(q).to(dev)-torch.from_numpy(st['q_mean']).to(dev))/torch.from_numpy(st['q_std']).to(dev))[None]
    images.append(image); qposes.append(q)
   # Arms remain independent samples with identical shared weights. Batching
   # only removes N serial GPU launches; it introduces no cross-arm inputs.
   pred,_,_=model(torch.cat(images),torch.cat(qposes),None)
   chunks=pred.float().cpu().numpy()*st['a_std'][None,None,:]+st['a_mean'][None,None,:]
   actions={}
   for arm,ch in enumerate(chunks):
    histories[arm].append((step,ch))
    candidates=np.asarray([old[step-born] for born,old in histories[arm] if step-born<len(old)])
    weights=np.exp(-0.01*np.arange(len(candidates)-1,-1,-1)); weights/=weights.sum()
    row=np.sum(candidates*weights[:,None],axis=0); space=env.action_space.spaces[f'panda-{arm}']; actions[f'panda-{arm}']=np.clip(row,space.low,space.high).astype(np.float32)
   obs,_,term,trunc,info=env.step(actions); success=scalar(info.get('success',False))
   if success or scalar(term) or scalar(trunc): break
  rows.append({'episode':ep,'seed':seed,'success':success,'steps':step+1}); print(json.dumps(rows[-1]),flush=True)
 env.close(); out={'schema':'mars-control.act.validation20.v1' if not a.smoke else 'mars-control.act.smoke.v1','status':'complete','task':a.task,'episodes':len(rows),'successes':sum(int(x['success']) for x in rows),'max_steps':max_steps,'rows':rows,'evaluator_revision':EVALUATOR_REVISION,'rgb_preprocessing':'uint8_div_255_to_unit_float','policy_contract':'shared_weights_decentralized_local_rgb_qpos_to_local_action8'}; Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
