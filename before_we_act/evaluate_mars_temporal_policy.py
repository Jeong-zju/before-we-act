"""Closed-loop MARS validation for the strict-local DINO temporal policy."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

from before_we_act.mars_temporal_data import TASK_TEXT, local_task_text
from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    canonicalize_action,
    normalization_stats_hash,
    validate_action_space_bounds,
    validate_action_stats,
    validate_checkpoint_action_contract,
)
from before_we_act.temporal_history_data import HISTORY_STEPS, TASK_TEXT_BYTES, task_text_tensor
from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from deployment.mars_care.common import TASK_BY_NAME, local_observation, make_env


def scalar(value: Any) -> bool:
    return bool(np.asarray(value).all())


def load_policy(checkpoint: Path, dino_model: str, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = validate_checkpoint_action_contract(saved)
    stats_payload = saved.get("stats")
    validate_action_stats(stats_payload)
    annotations = contract.get("annotations", {})
    expected_stats_hash = normalization_stats_hash(stats_payload)
    if annotations.get("normalization_sha256") != expected_stats_hash:
        raise ValueError("checkpoint action statistics hash differs")
    config = saved["config"]
    if config.get("action_contract_version") != ACTION_CONTRACT_VERSION:
        raise ValueError("checkpoint config is missing the shared MARS action contract")
    if config.get("strict_local") is not True or config.get("vision") != "dinov3_vitb16_frozen":
        raise ValueError("checkpoint is not the strict-local MARS DINO reference policy")
    model = TemporalHistoryPolicy(
        variant="hidden_residual", dino_model=dino_model,
        image_height=int(config["image_height"]), image_width=int(config["image_width"]),
    ).to(device)
    model.load_state_dict(saved["model"], strict=True); model.eval()
    stats = {key: torch.as_tensor(saved["stats"][key], device=device, dtype=torch.float32)
             for key in ("q_mean", "q_std", "a_mean", "a_std")}
    return model, stats, config


class LocalHistory:
    def __init__(self, arms):
        self.arms = tuple(arms)
        self.visual = {arm: deque(maxlen=HISTORY_STEPS-1) for arm in self.arms}
        self.qpos = {arm: deque(maxlen=HISTORY_STEPS-1) for arm in self.arms}
        self.action = {arm: deque(maxlen=HISTORY_STEPS) for arm in self.arms}

    def batch(self, current_qpos: torch.Tensor, task: str, device: torch.device, role_context: bool):
        n=len(self.arms); visual=torch.zeros(n,HISTORY_STEPS,2,768,device=device)
        qpos=torch.zeros(n,HISTORY_STEPS,9,device=device)
        action=torch.zeros(n,HISTORY_STEPS,8,device=device)
        hmask=torch.zeros(n,HISTORY_STEPS,dtype=torch.bool,device=device)
        amask=torch.zeros(n,HISTORY_STEPS,dtype=torch.bool,device=device)
        reset=[]
        for row,arm in enumerate(self.arms):
            vv=list(self.visual[arm]); qq=list(self.qpos[arm]); aa=list(self.action[arm])
            if len(vv)!=len(qq): raise RuntimeError("local history drift")
            if vv:
                first=HISTORY_STEPS-1-len(vv); visual[row,first:-1]=torch.stack(vv).to(device)
                qpos[row,first:-1]=torch.stack(qq).to(device); hmask[row,first:-1]=True
            qpos[row,-1]=current_qpos[row]; hmask[row,-1]=True
            if aa:
                first=HISTORY_STEPS-len(aa); action[row,first:]=torch.stack(aa).to(device); amask[row,first:]=True
            reset.append(not vv and not aa)
        texts=[task_text_tensor(local_task_text(task,arm) if role_context else TASK_TEXT[task]) for arm in self.arms]
        text=torch.stack([x[0] for x in texts]); text_mask=torch.stack([x[1] for x in texts])
        return {"history_visual_raw":visual,"history_qpos":qpos,"history_action":action,
                "history_mask":hmask,"action_history_mask":amask,
                "task_bytes":text.to(device),
                "task_text_mask":text_mask.to(device),
                "episode_reset":torch.tensor(reset,dtype=torch.bool,device=device)}

    def append_observation(self, visual: torch.Tensor, qpos: torch.Tensor):
        for row,arm in enumerate(self.arms):
            self.visual[arm].append(visual[row].detach().float().cpu())
            self.qpos[arm].append(qpos[row].detach().float().cpu())

    def append_action(self, action: dict[int, torch.Tensor]):
        for arm in self.arms: self.action[arm].append(action[arm].detach().float().cpu())


class ChunkEnsembler:
    def __init__(self, arms, decay=.01): self.arms=tuple(arms); self.decay=decay; self.chunks=[[] for _ in arms]
    def select(self, step: int, chunks: np.ndarray):
        result={}
        for row,arm in enumerate(self.arms):
            self.chunks[row].append((step,chunks[row])); self.chunks[row]=[(s,c) for s,c in self.chunks[row] if step-s<len(c)]
            candidates=np.asarray([c[step-s] for s,c in self.chunks[row]])
            weights=np.exp(-self.decay*np.arange(len(candidates)-1,-1,-1)); weights/=weights.sum()
            result[arm]=np.sum(candidates*weights[:,None],axis=0)
        return result


@torch.inference_mode()
def run_episode(model, stats, task, robofactory_root: Path, seed: int, device: torch.device, max_steps: int, ensemble_decay: float, action_encoding: str):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    root=str(robofactory_root.resolve())
    if root not in sys.path: sys.path.insert(0,root)
    env=make_env(task,robofactory_root); observation,_=env.reset(seed=seed)
    arms=tuple(range(task.arms)); history=LocalHistory(arms); ensemble=ChunkEnsembler(arms,ensemble_decay)
    role_context=action_encoding.endswith("+role")
    action_encoding=action_encoding.removesuffix("+role")
    inference=[]; trace=hashlib.sha256(); success=False
    movement=np.zeros(len(arms),dtype=np.float64); ever_flags={}
    try:
        for step in range(max_steps):
            images=[]; qposes=[]
            for arm in arms:
                image,qpos=local_observation(observation,arm)
                if image.shape!=(240,320,3): raise ValueError(f"MARS validation RGB drift: {image.shape}")
                images.append(image); qposes.append(qpos.reshape(-1))
            rgb=torch.as_tensor(np.stack(images),device=device).permute(0,3,1,2).float().div_(255)
            qpos=torch.as_tensor(np.stack(qposes),device=device).float(); qnorm=(qpos-stats["q_mean"])/stats["q_std"]
            temporal=history.batch(qnorm,task.name,device,role_context); started=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                chunks,_mu,_logvar,current_visual=model(rgb,rgb,**temporal,return_current_visual=True)
            history.append_observation(current_visual,qnorm)
            decoded=(chunks*stats["a_std"]+stats["a_mean"]).float().cpu().numpy()
            rows=ensemble.select(step,decoded); action={}; normalized={}
            for arm,row in rows.items():
                if action_encoding not in ("absolute", "absolute_pd_joint_pos"):
                    raise ValueError(f"official MARS CARE requires absolute pd_joint_pos, got {action_encoding}")
                space=env.action_space.spaces[f"panda-{arm}"]
                validate_action_space_bounds(space)
                row=canonicalize_action(row)
                action[f"panda-{arm}"]=row; trace.update(row.tobytes())
                movement[arms.index(arm)]+=float(np.linalg.norm(row[:7]-qposes[arms.index(arm)][:7]))
                normalized[arm]=(torch.as_tensor(row,device=device)-stats["a_mean"])/stats["a_std"]
            history.append_action(normalized); inference.append(time.perf_counter()-started)
            observation,_reward,terminated,truncated,info=env.step(action)
            success=scalar(info.get("success",False))
            for key,value in info.items():
                try:
                    flag=scalar(value)
                except Exception:
                    continue
                ever_flags[str(key)]=bool(ever_flags.get(str(key),False) or flag)
            if success or scalar(terminated) or scalar(truncated): break
    finally: env.close()
    return {"task":task.name,"seed":seed,"success":success,"steps":step+1,
            "mean_inference_seconds":float(np.mean(inference)),"action_trace_sha256":trace.hexdigest(),
            "mean_command_delta_by_arm":(movement/(step+1)).tolist(),"ever_info_flags":ever_flags}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--dino-model",required=True); p.add_argument("--task",choices=tuple(TASK_BY_NAME),required=True)
    p.add_argument("--robofactory-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=20); p.add_argument("--seed-start",type=int,default=20260827)
    p.add_argument("--ensemble-decay",type=float,default=.01)
    p.add_argument("--max-steps",type=int); p.add_argument("--device",default="cuda:0"); args=p.parse_args()
    device=torch.device(args.device); model,stats,config=load_policy(args.checkpoint,args.dino_model,device); task=TASK_BY_NAME[args.task]
    log=args.output.with_suffix(".jsonl"); recovered={}
    if log.is_file():
        for line in log.read_text().splitlines():
            try: row=json.loads(line); recovered[int(row["seed"])]=row
            except Exception: pass
    rows=[]
    for seed in range(args.seed_start,args.seed_start+args.episodes):
        encoding=str(config.get("action_encoding","absolute"))
        if config.get("role_context")=="own_base_xy_in_task_context": encoding += "+role"
        row=recovered.get(seed) or run_episode(model,stats,task,args.robofactory_root,seed,device,args.max_steps or task.max_steps,args.ensemble_decay,encoding)
        rows.append(row)
        if seed not in recovered:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with log.open("a") as stream: stream.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
    result={"status":"complete","policy":"care_dino_temporal_reference","strict_local":True,
            "task":task.name,"episodes":len(rows),"successes":sum(int(x["success"]) for x in rows),
            "success_rate":float(np.mean([x["success"] for x in rows])),"rows":rows,"config":config}
    tmp=args.output.with_suffix(".tmp"); tmp.write_text(json.dumps(result,indent=2)+"\n"); os.replace(tmp,args.output)


if __name__=="__main__": main()
