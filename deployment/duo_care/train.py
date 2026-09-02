from __future__ import annotations
import argparse,json,os,random,time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from .dataset import DuoLocalDataset,TaskSampler
from deployment.mars_care.model import CAREPolicy,ModelConfig

def save(x,p): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); torch.save(x,t); os.replace(t,p)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=20000); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,default=8); p.add_argument('--save-every',type=int,default=1000); p.add_argument('--init-checkpoint',type=Path); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
 rank=int(os.environ.get('RANK',0)); distributed='RANK' in os.environ
 if distributed: torch.distributed.init_process_group('nccl'); local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local)
 else: local=0
 device=torch.device('cuda',local); random.seed(20260829+rank); np.random.seed(20260829+rank); torch.manual_seed(20260829+rank)
 ds=DuoLocalDataset(a.data,a.data/'manifest.json',image_size=224); replicas=torch.distributed.get_world_size() if distributed else 1; sampler=TaskSampler(ds,rank,replicas); loader=DataLoader(ds,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,drop_last=True,prefetch_factor=3 if a.workers else None)
 manifest=json.loads((a.data/'manifest.json').read_text()); normalization=manifest['normalization']; action_encoding=normalization.get('action_encoding','anchor_joint_residual_gripper_absolute')
 cfg=ModelConfig(qpos_dim=8,action_dim=8,tasks=11,image_size=224,history=16,candidates=6); model=CAREPolicy(cfg,pretrained=(rank==0)).to(device)
 if a.init_checkpoint and a.init_checkpoint.is_file(): model.load_state_dict(torch.load(a.init_checkpoint,map_location='cpu',weights_only=False)['model'])
 if distributed: model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=3e-6); start=0; latest=a.output/'latest.pt'
 if latest.is_file() and not a.smoke:
  saved=torch.load(latest,map_location='cpu',weights_only=False); (model.module if distributed else model).load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); sch.load_state_dict(saved['scheduler']); start=int(saved['step'])
 it=iter(loader); started=time.time()
 for step in range(start+1,a.steps+1):
  try:b=next(it)
  except StopIteration:sampler.set_epoch(step);it=iter(loader);b=next(it)
  b={k:v.to(device,non_blocking=True) for k,v in b.items()}; opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): out=model(b['image'],b['qpos'],b['task_id'],b['history'],b['history_mask']); loss,pieces=model.module.loss_from_output(out,b) if distributed else model.loss_from_output(out,b)
  if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss step {step}')
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
  if rank==0 and (step==1 or step%100==0):
   row={'step':step,'total_steps':a.steps,'lr':sch.get_last_lr()[0],'elapsed_seconds':time.time()-started,**{k:float(v) for k,v in pieces.items()}}; a.output.mkdir(parents=True,exist_ok=True); (a.output/'progress.jsonl').open('a').write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
  if rank==0 and (step%a.save_every==0 or step==a.steps):
   m=model.module if distributed else model; save({'format':'duobench-care-v2','step':step,'model_config':m.config_dict(),'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'normalization':normalization,'action_encoding':action_encoding,'policy_contract':'shared_weights_decentralized_shared_head_rgb_own_wrist_rgb_own_qpos_history_to_own_anchor_relative_action8','training_contract':{'all_11_tasks':True,'all_50_demos_per_task':True,'task_balanced':True,'global_batch_size':a.batch_size*replicas,'updates':a.steps,'local_samples_seen':a.steps*a.batch_size*replicas,'equivalent_dataset_traversals':a.steps*a.batch_size*replicas/len(ds)}},a.output/'latest.pt')
   if step==a.steps: save(torch.load(a.output/'latest.pt',weights_only=False),a.output/'final.pt')
 if distributed: torch.distributed.destroy_process_group()
if __name__=='__main__':main()
