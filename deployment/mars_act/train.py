#!/usr/bin/env python3
"""ACT trainer for MARS-Control; all demonstrations, no split.

The reported policy settings live in a versioned JSON artifact.  The few
legacy CLI flags are retained only for launcher compatibility and are checked
against that artifact instead of silently overriding it.
"""
import argparse,json,glob,hashlib,os,random,time
from pathlib import Path
import h5py,numpy as np, torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from stereo_core.train_act import ACT

TASKS=('place_cube_in_cup','strike_cube_hard','three_robots_place_shoes','four_robots_stack_cube')
ARMS={'place_cube_in_cup':2,'strike_cube_hard':2,'three_robots_place_shoes':3,'four_robots_stack_cube':4}
DEFAULT_CONFIG=Path(__file__).resolve().parents[2]/'configs/act/mars_control_full_data_v1.json'

def load_config(path):
 with open(path) as f: cfg=json.load(f)
 if cfg.get('policy')!='ACT' or cfg.get('benchmark')!='MARS-Control': raise ValueError('wrong ACT/MARS-Control config')
 if cfg.get('status')!='frozen_reported_run': raise ValueError('configuration is not frozen')
 return cfg

def sha256_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

class MarsDataset(Dataset):
 def __init__(self,root,horizon=100):
  self.horizon=horizon; self.entries=[]; self.handles={}; self.task_ids=[]; qs=[]; acts=[]
  for tid,task in enumerate(TASKS):
   for path in sorted(glob.glob(str(Path(root)/task/'motionplanning'/'*.shard*.h5'))):
    with h5py.File(path,'r') as f:
     for tr in sorted(f.keys(),key=lambda x:int(x.split('_')[-1])):
      n=min(len(f[tr][f'actions/panda-{a}']) for a in range(ARMS[task]))
      for arm in range(ARMS[task]):
       q=np.asarray(f[tr]['obs/agent'][f'panda-{arm}']['qpos'][:n],np.float32); x=np.asarray(f[tr]['actions'][f'panda-{arm}'][:n],np.float32)
       qs.append(q); acts.append(x)
       for t in range(n): self.entries.append((path,tr,arm,t,tid)); self.task_ids.append(tid)
  q=np.concatenate(qs); x=np.concatenate(acts); self.qmean=q.mean(0); self.qstd=np.maximum(q.std(0),1e-4); self.amean=x.mean(0); self.astd=np.maximum(x.std(0),1e-4)
 def __getstate__(self):
  state=dict(self.__dict__); state['handles']={}; return state
 def __len__(self): return len(self.entries)
 def __getitem__(self,i):
  path,tr,arm,t,tid=self.entries[i];
  if path not in self.handles:self.handles[path]=h5py.File(path,'r',swmr=True)
  g=self.handles[path][tr]; im=np.asarray(g['obs/sensor_data'][f'head_camera_agent{arm}']['rgb'][t],np.uint8); q=np.asarray(g['obs/agent'][f'panda-{arm}']['qpos'][t],np.float32); ds=g['actions'][f'panda-{arm}']; end=min(len(ds),t+self.horizon); x=np.asarray(ds[t:end],np.float32)
  valid=len(x); pad=np.repeat(x[-1:],self.horizon-valid,axis=0) if valid<self.horizon else np.empty((0,8),np.float32); x=np.concatenate((x,pad),0); mask=np.zeros(self.horizon,np.float32); mask[:valid]=1
  return torch.from_numpy(im).permute(2,0,1),torch.from_numpy((q-self.qmean)/self.qstd),torch.from_numpy((x-self.amean)/self.astd),torch.from_numpy(mask)
def save(payload,path): path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default=str(DEFAULT_CONFIG)); p.add_argument('--data-root',required=True); p.add_argument('--output',required=True); p.add_argument('--updates',type=int); p.add_argument('--batch-size',type=int); p.add_argument('--workers',type=int); p.add_argument('--horizon',type=int); p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',action='store_true'); a=p.parse_args()
 cfg=load_config(a.config); dc=cfg['data']; sc=cfg['sampling_and_loader']; mc=cfg['model']; oc=cfg['optimization']; lc=cfg['objective']; rc=cfg['runtime']
 def frozen(name,supplied,value):
  if supplied is not None and supplied!=value: raise ValueError(f'{name} is frozen at {value}, got {supplied}')
  return value
 updates=frozen('updates',a.updates,oc['updates']); batch=frozen('batch-size',a.batch_size,sc['global_batch_size']); workers=frozen('workers',a.workers,sc['workers']); horizon=frozen('horizon',a.horizon,dc['chunk_horizon'])
 torch.set_num_threads(rc['torch_cpu_threads']); random.seed(rc['seed_python']); np.random.seed(rc['seed_numpy']); torch.manual_seed(rc['seed_torch'])
 ds=MarsDataset(a.data_root,horizon); counts=np.bincount(ds.task_ids,minlength=4); weights=np.asarray([1/counts[x] for x in ds.task_ids]); sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),num_samples=len(ds),replacement=sc['replacement']); dl=DataLoader(ds,batch_size=batch,sampler=sampler,drop_last=sc['drop_last'],num_workers=workers,pin_memory=sc['pin_memory'],persistent_workers=sc['persistent_workers'] if workers else False,multiprocessing_context=sc['multiprocessing_context'] if workers else None,prefetch_factor=sc['prefetch_factor'] if workers else None)
 dev=torch.device(rc['device']); model=ACT(mc['state_dimension'],mc['action_dimension'],horizon,mc['hidden_dimension'],mc['posterior_encoder_layers'],mc['action_decoder_layers'],latent_dim=mc['latent_dimension'],vision_backbone=mc['vision_backbone']).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=oc['learning_rate'],betas=tuple(oc['betas']),eps=oc['epsilon'],weight_decay=oc['weight_decay'],amsgrad=oc['amsgrad']); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,updates,eta_min=oc['scheduler_minimum_learning_rate']); out=Path(a.output); start=0
 if a.resume and (out/'last.pt').is_file():
  s=torch.load(out/'last.pt',map_location=dev,weights_only=False); model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); sched.load_state_dict(s['scheduler']); start=int(s['updates'])
 it=iter(dl); started=time.time(); target=oc['smoke_updates'] if a.smoke else updates; log=out/'progress.jsonl'; out.mkdir(parents=True,exist_ok=True)
 for step in range(start+1,target+1):
  try:b=next(it)
  except StopIteration:it=iter(dl);b=next(it)
  im,q,x,m=[z.to(dev,non_blocking=True) for z in b]; opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): pred,mu,lv=model(im.float().div(255),q,x); mse=((pred-x).square().mean(-1)*m).sum()/m.sum().clamp_min(1); kl=-.5*(1+lv-mu.square()-lv.exp()).sum(-1).mean(); loss=mse+lc['kl_weight_beta']*kl
  if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss step {step}')
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),oc['maximum_gradient_norm']); opt.step(); sched.step()
  if step==1 or step%20==0:
   row={'step':step,'target_updates':target,'loss':float(loss),'mse':float(mse),'kl':float(kl),'lr':sched.get_last_lr()[0],'elapsed_s':time.time()-started}; log.open('a').write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
  if step%5000==0 or step==target:
   payload={'schema':'mars-control.act.checkpoint.v2','model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'updates':step,'stats':{'q_mean':ds.qmean,'q_std':ds.qstd,'a_mean':ds.amean,'a_std':ds.astd},'config':{'state_dim':mc['state_dimension'],'action_dim':mc['action_dimension'],'horizon':horizon,'d_model':mc['hidden_dimension'],'enc_layers':mc['posterior_encoder_layers'],'dec_layers':mc['action_decoder_layers'],'latent_dim':mc['latent_dimension'],'vision_backbone':mc['vision_backbone'],'arms':ARMS,'tasks':TASKS,'training_data_policy':'all_600_episodes_no_split','global_batch_size':batch,'optimizer':'AdamW','learning_rate':oc['learning_rate'],'beta':lc['kl_weight_beta'],'seed':rc['seed_python']},'training_config':cfg,'training_config_path':str(Path(a.config)),'training_config_sha256':sha256_file(a.config)}; save(payload,out/'last.pt')
   if step%5000==0: save(payload,out/f'checkpoint_{step:06d}.pt')
   if step==target: save(payload,out/'final.pt')
if __name__=='__main__':main()
