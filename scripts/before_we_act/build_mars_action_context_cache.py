#!/usr/bin/env python3
"""Cache frozen MARS B0-H decoded contexts for official CARE belief training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract_hash,
    normalization_stats_hash,
    validate_checkpoint_action_contract,
)
from before_we_act.mars_temporal_data import (
    MarsVisualCache,
    clip_pd_action,
    load_mars_episodes,
    local_task_text,
    validate_mars_normalization,
)
from before_we_act.temporal_history_data import ACTION_HORIZON, HISTORY_STEPS, task_text_tensor, sha256_file
from before_we_act.temporal_history_policy import TemporalHistoryPolicy


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    os.replace(temporary,path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream: np.save(stream,value,allow_pickle=False)
    os.replace(temporary,path)


def episode_inputs(episode, rows, stats, visual_cache, handle, device):
    group=handle[episode.trajectory]; cached=visual_cache.load(episode)
    q_sources={arm:np.asarray(group[f"obs/agent/panda-{arm}/qpos"][:episode.length],np.float32) for arm in episode.arms}
    a_sources={arm:clip_pd_action(np.asarray(group[f"actions/panda-{arm}"][:episode.length],np.float32)) for arm in episode.arms}
    count=len(rows)
    visual=torch.zeros(count,HISTORY_STEPS,2,768,dtype=torch.float16)
    qpos=torch.zeros(count,HISTORY_STEPS,9); action=torch.zeros(count,HISTORY_STEPS,8)
    hmask=torch.zeros(count,HISTORY_STEPS,dtype=torch.bool)
    amask=torch.zeros(count,HISTORY_STEPS,dtype=torch.bool)
    images=[]; resets=[]; texts=[]
    q_mean=torch.as_tensor(stats["q_mean"]); q_std=torch.as_tensor(stats["q_std"])
    a_mean=torch.as_tensor(stats["a_mean"]); a_std=torch.as_tensor(stats["a_std"])
    for row,(t,arm) in enumerate(rows):
        first=max(0,t-HISTORY_STEPS+1); obs=list(range(first,t+1)); offset=HISTORY_STEPS-len(obs)
        afirst=max(0,t-HISTORY_STEPS); past=list(range(afirst,t)); aoffset=HISTORY_STEPS-len(past)
        own=torch.from_numpy(cached[f"agent_{arm}"][obs])
        visual[row,offset:,0]=own; visual[row,offset:,1]=own
        qpos[row,offset:]=(torch.from_numpy(q_sources[arm][obs])-q_mean)/q_std; hmask[row,offset:]=True
        if past:
            action[row,aoffset:]=(torch.from_numpy(a_sources[arm][past])-a_mean)/a_std; amask[row,aoffset:]=True
        images.append(np.asarray(group[f"obs/sensor_data/head_camera_agent{arm}/rgb"][t]))
        resets.append(t==0); texts.append(task_text_tensor(local_task_text(episode.task,arm)))
    rgb=torch.as_tensor(np.stack(images)).permute(0,3,1,2).float().div_(255).to(device)
    return {
        "global_rgb":rgb,"local_rgb":rgb,
        "history_visual_raw":visual.to(device),"history_qpos":qpos.to(device),
        "history_action":action.to(device),"history_mask":hmask.to(device),
        "action_history_mask":amask.to(device),
        "task_bytes":torch.stack([x[0] for x in texts]).to(device),
        "task_text_mask":torch.stack([x[1] for x in texts]).to(device),
        "episode_reset":torch.tensor(resets,dtype=torch.bool,device=device),
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root",type=Path,required=True)
    parser.add_argument("--normalization",type=Path,required=True)
    parser.add_argument("--visual-cache",type=Path,required=True)
    parser.add_argument("--temporal-checkpoint",type=Path,required=True)
    parser.add_argument("--dino-model",default="",help="fallback for legacy/smoke checkpoints")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--episodes-per-task",type=int,default=0,help="smoke-only prefix per task; zero means all 150")
    args=parser.parse_args()
    rank=int(os.environ.get("RANK","0")); world=int(os.environ.get("WORLD_SIZE","1"))
    local=int(os.environ.get("LOCAL_RANK","0")); device=torch.device("cuda",local)
    torch.cuda.set_device(device)
    # This job is an embarrassingly parallel cache build.  Do not initialize a
    # NCCL process group: ranks have highly skewed episode lengths (one MARS
    # trajectory is thousands of steps), so a collective barrier would make the
    # fast ranks time out while the slow rank is still writing its files.
    episodes=load_mars_episodes(args.raw_root)
    if args.episodes_per_task:
        if not 1 <= args.episodes_per_task < 150: raise ValueError("invalid smoke episode limit")
        episodes=[episode for task in sorted({e.task for e in episodes})
                  for episode in [e for e in episodes if e.task==task][:args.episodes_per_task]]
    stats=json.loads(args.normalization.read_text())
    validate_mars_normalization(stats)
    payload=torch.load(args.temporal_checkpoint,map_location="cpu",weights_only=False)
    checkpoint_contract=validate_checkpoint_action_contract(payload)
    normalization_semantic_sha=normalization_stats_hash(stats)
    if checkpoint_contract.get("annotations",{}).get("normalization_sha256")!=normalization_semantic_sha:
        raise ValueError("B0-H checkpoint and cache normalization differ")
    config=payload["config"]
    if config.get("action_contract_version") != ACTION_CONTRACT_VERSION: raise ValueError("wrong B0-H action contract")
    model=TemporalHistoryPolicy(
        variant="hidden_residual",dino_model=str(config.get("dino_model") or args.dino_model),
        image_height=int(config["image_height"]),image_width=int(config["image_width"]),
    ).to(device)
    model.load_state_dict(payload["model"],strict=True); model.eval().requires_grad_(False)
    visual_cache=MarsVisualCache(args.visual_cache,limit=4)
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=sha256_file(args.temporal_checkpoint)
    normalization_sha=sha256_file(args.normalization)
    if rank==0:
        metadata={"format_version":"before-we-act.mars-action-context-metadata/1","episodes":[
            {"cache_key":e.cache_key,"task":e.task,"length":e.length,"arms":list(e.arms)} for e in episodes]}
        atomic_json(args.output/"metadata.json",metadata)
        task_tokens={}
        with torch.inference_mode():
            for episode in episodes:
                for arm in episode.arms:
                    key=f"{episode.task}:{arm}"
                    if key in task_tokens: continue
                    text,mask=task_text_tensor(local_task_text(episode.task,arm))
                    task_tokens[key]=model._task_token(text[None].to(device),mask[None].to(device))[0].float().cpu().tolist()
        atomic_json(args.output/"task_tokens.json",task_tokens)
    completed=0; samples=0; started=time.time()
    for ordinal,episode in enumerate(episodes[rank::world],start=1):
        directory=args.output/episode.task; decoded_path=directory/f"{episode.cache_key}.decoded.npy"
        base_path=directory/f"{episode.cache_key}.base_action.npy"; marker=directory/f"{episode.cache_key}.complete.json"
        if marker.is_file() and decoded_path.is_file() and base_path.is_file():
            completed+=1; samples+=episode.length*len(episode.arms); continue
        shape=(episode.length,len(episode.arms),ACTION_HORIZON,384)
        decoded=np.empty(shape,dtype=np.float16)
        base=np.empty((episode.length,len(episode.arms),ACTION_HORIZON,8),dtype=np.float16)
        rows=[(t,arm) for t in range(episode.length) for arm in episode.arms]
        with h5py.File(episode.path,"r") as handle:
            for first in range(0,len(rows),args.batch_size):
                selected=rows[first:first+args.batch_size]
                inputs=episode_inputs(episode,selected,stats,visual_cache,handle,device)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    context=model._decode_action_context(**inputs,actions=None)
                    prediction=model.out(context.decoded)
                    history=context.history_summary[:,None].expand(-1,ACTION_HORIZON,-1)
                    prediction=prediction+model.hidden_residual(torch.cat((context.decoded,history),-1))
                for local_row,(t,arm) in enumerate(selected):
                    decoded[t,arm]=context.decoded[local_row].float().cpu().numpy().astype(np.float16)
                    base[t,arm]=prediction[local_row].float().cpu().numpy().astype(np.float16)
        atomic_npy(decoded_path,decoded); atomic_npy(base_path,base)
        atomic_json(marker,{"status":"PASSED","cache_key":episode.cache_key,"task":episode.task,
            "samples":episode.length*len(episode.arms),"b0h_checkpoint_sha256":checkpoint_sha})
        completed+=1; samples+=episode.length*len(episode.arms)
        if ordinal==1 or ordinal%5==0:
            print(json.dumps({"rank":rank,"episodes":completed,"assigned":len(episodes[rank::world]),
                "samples":samples,"episodes_per_hour":completed/max(time.time()-started,1e-6)*3600}),flush=True)
    atomic_json(args.output/f"rank_{rank:02d}_receipt.json",{"rank":rank,"world_size":world,
        "episodes":completed,"samples":samples,"b0h_checkpoint_sha256":checkpoint_sha,
        "normalization_sha256":normalization_sha,
        "normalization_semantic_sha256":normalization_semantic_sha,
        "action_contract_sha256":action_contract_hash()})
    if rank==0:
        # Ranks finish independently.  Rank 0 acts as a filesystem-based
        # coordinator rather than entering a distributed collective, and waits
        # for every rank receipt plus every episode marker.  This is robust to
        # long-tail trajectories and to resuming a partially built cache.
        expected=sum(e.length*len(e.arms) for e in episodes)
        expected_by_rank={r: len(episodes[r::world]) for r in range(world)}
        expected_samples_by_rank={r: sum(e.length*len(e.arms) for e in episodes[r::world]) for r in range(world)}
        deadline=time.time()+2*60*60
        while True:
            receipts=[]
            for r in range(world):
                path=args.output/f"rank_{r:02d}_receipt.json"
                if not path.is_file(): break
                try: value=json.loads(path.read_text())
                except (OSError, json.JSONDecodeError): break
                if (int(value.get("rank",-1))!=r
                        or int(value.get("world_size",-1))!=world
                        or int(value.get("episodes",-1))!=expected_by_rank[r]
                        or int(value.get("samples",-1))!=expected_samples_by_rank[r]
                        or value.get("b0h_checkpoint_sha256")!=checkpoint_sha
                        or value.get("normalization_sha256")!=normalization_sha
                        or value.get("normalization_semantic_sha256")!=normalization_semantic_sha
                        or value.get("action_contract_sha256")!=action_contract_hash()): break
                receipts.append(value)
            markers=list(args.output.glob("*/*.complete.json"))
            if len(receipts)==world and len(markers)==len(episodes):
                total=sum(int(json.loads(p.read_text())["samples"]) for p in markers)
                if total==expected: break
            if time.time()>=deadline:
                raise RuntimeError(f"action cache coordinator timeout: receipts={len(receipts)}/{world}, markers={len(markers)}/{len(episodes)}")
            time.sleep(5)
        total=expected
        atomic_json(args.output/"cache_receipt.json",{
            "format_version":"before-we-act.mars-action-context-cache/1","status":"PASSED",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"episodes":len(episodes),"samples":total,
            "decoded_shape_per_sample":[ACTION_HORIZON,384],"base_action_shape_per_sample":[ACTION_HORIZON,8],
            "dtype":"float16","action_encoding":"absolute_pd_joint_pos",
            "action_contract_version":ACTION_CONTRACT_VERSION,
            "action_contract_sha256":action_contract_hash(),
            "b0h_checkpoint":str(args.temporal_checkpoint.resolve()),"b0h_checkpoint_sha256":checkpoint_sha,
            "normalization_sha256":normalization_sha,
            "normalization_semantic_sha256":normalization_semantic_sha,
            "world_size":world})


if __name__ == "__main__": main()
