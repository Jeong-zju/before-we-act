"""Trainer for strict-local-wrist Stereo decoder variants."""
from __future__ import annotations
import argparse, json, glob
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_act import EpisodeBlockBatchSampler, _stats, _trajectories, seed_everything
from train_stereo_act import WristRGBDACTDataset, DEPTH_MM_TO_M
from five_task_contract import hierarchical_item_weights
from stereo_decoder_variants import StereoFFNMoE, StereoARCA


def one_loss(model, rgb, depth, qpos, actions, mask, beta, router_aux_weight):
    rgb = rgb.float().div_(255)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred, mu, logvar, aux = model(rgb, depth, qpos, actions)
        mse = ((pred-actions).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1)
        kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()
        total = mse + beta*kl + router_aux_weight*(aux-1.0)
    return total, mse, kl, aux


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant', choices=('ffn_moe','arca'), required=True)
    p.add_argument('--data', required=True); p.add_argument('--shared-arms',default='0,1,2,3'); p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--workers',type=int,default=8); p.add_argument('--updates',type=int,default=80000)
    p.add_argument('--cache-episodes',type=int,default=0); p.add_argument('--episode-block-updates',type=int,default=64)
    p.add_argument('--save-updates',default='40000,80000'); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--beta',type=float,default=1e-3)
    p.add_argument('--router-aux-weight',type=float,default=1e-2); p.add_argument('--experts',type=int,default=4); p.add_argument('--role-rank',type=int,default=32)
    p.add_argument('--task-balanced',action='store_true'); p.add_argument('--seed',type=int,default=20260726)
    a=p.parse_args(); arms=tuple(int(x) for x in a.shared_arms.split(','))
    paths=sorted({path for pat in a.data.split(',') for path in glob.glob(pat)})
    trajectories=_trajectories(paths,arms)
    if len(trajectories)<10: raise ValueError('need at least 10 successful RGB-D demonstrations')
    seed_everything(a.seed); torch.backends.cudnn.benchmark=True
    lazy_cache=a.cache_episodes>0
    if lazy_cache and a.workers: raise ValueError('bounded RGB-D cache requires --workers 0')
    stats=_stats(trajectories,arms); train=WristRGBDACTDataset(trajectories,100,stats,True,preload=not lazy_cache,cache_limit=a.cache_episodes)
    counts=Counter(train.item_tasks); sampler=None
    if lazy_cache:
        sampler=EpisodeBlockBatchSampler(train,a.batch_size,a.updates,a.episode_block_updates,a.seed,a.task_balanced)
        loader=DataLoader(train,batch_sampler=sampler,num_workers=0,pin_memory=True)
    elif a.task_balanced and len(counts)>1:
        weights=torch.as_tensor(train.item_weights,dtype=torch.double)
        sampler=torch.utils.data.WeightedRandomSampler(weights,num_samples=len(weights),replacement=True)
    if not lazy_cache:
        loader=DataLoader(train,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,drop_last=True,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    sample=train[0]; kwargs=dict(horizon=100,d_model=384,enc_layers=4,dec_layers=7)
    if a.variant=='ffn_moe': model=StereoFFNMoE(len(sample[2]),len(sample[3][0]),experts=a.experts,**kwargs)
    else: model=StereoARCA(len(sample[2]),len(sample[3][0]),roles=a.experts,role_rank=a.role_rank,**kwargs)
    device=torch.device('cuda:0'); model=model.to(device)
    opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.updates)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=vars(a)|{'horizon':100,'enc_layers':4,'dec_layers':7,'d_model':384,'vision_backbone':'stereo_act_cross_relbias','dino_model':'facebook/dinov3-vitb16-pretrain-lvd1689m','defm_model':model.defm_model_name,'camera_width':640,'camera_height':480,'patch_grid':[30,40],'fusion_layers':2,'depth_storage_unit':'millimeters','depth_to_meters_scale':DEPTH_MM_TO_M,'arms':arms,'state_dim':len(sample[2]),'action_dim':len(sample[3][0]),'files':paths,'episodes':len(trajectories),'train_task_item_counts':dict(counts),'policy_variant':'stereo_'+a.variant,'strict_policy_input':'current local panda_hand wrist RGB-D and local qpos only; no task/agent ID, peer/global/right-camera/language input'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2)); torch.save({'stats':stats},out/'normalization.pt')
    milestones={int(x) for x in a.save_updates.split(',') if x}; update=0; totals={'loss':0.,'mse':0.,'kl':0.,'router_aux':0.,'n':0}
    while update<a.updates:
      model.train()
      for rgb,depth,qpos,actions,mask in loader:
        rgb,depth,qpos,actions,mask=(x.to(device,non_blocking=True) for x in (rgb,depth,qpos,actions,mask))
        opt.zero_grad(set_to_none=True); total,mse,kl,aux=one_loss(model,rgb,depth,qpos,actions,mask,a.beta,a.router_aux_weight)
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();update+=1
        n=len(rgb)
        for name,val in (('loss',total),('mse',mse),('kl',kl),('router_aux',aux)): totals[name]+=float(val.detach())*n
        totals['n']+=n
        if update in milestones: torch.save({'model':model.state_dict(),'stats':stats,'config':cfg,'update':update},out/f'checkpoint_{update:06d}.pt')
        if update%100==0: print(json.dumps({'update':update,**{k:v/totals['n'] for k,v in totals.items() if k!='n'}}),flush=True)
        if update>=a.updates: break

if __name__=='__main__': main()
