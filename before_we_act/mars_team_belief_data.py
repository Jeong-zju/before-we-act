"""MARS benchmark projection for the unchanged official CARE belief core."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from before_we_act.mars_action_contract import action_contract_hash, normalization_stats_hash
from before_we_act.mars_temporal_data import (
    EFFECTIVE_BATCH,
    MARS_TASKS,
    MarsTemporalEpisode,
    MarsVisualCache,
    clip_pd_action,
    validate_mars_normalization,
)
from before_we_act.raw_team_signal_data import TeamSampleRequest
from before_we_act.temporal_history_data import ACTION_HORIZON, HISTORY_STEPS
from before_we_act.team_belief.predictive_core import FUTURE_OFFSETS_STEPS


TEAMMATE_ACTION_HORIZON=16
SAMPLES_PER_TASK=EFFECTIVE_BATCH//len(MARS_TASKS)


class MarsActionContextCache:
    def __init__(self,root: str|Path,stats: Mapping,limit: int=4):
        self.root=Path(root); self.limit=limit
        receipt=json.loads((self.root/"cache_receipt.json").read_text())
        if receipt.get("format_version")!="before-we-act.mars-action-context-cache/1" or receipt.get("status")!="PASSED":
            raise ValueError("MARS action-context cache is incomplete")
        if receipt.get("action_contract_sha256")!=action_contract_hash():
            raise ValueError("MARS action-context cache action contract differs")
        if receipt.get("normalization_semantic_sha256")!=normalization_stats_hash(stats):
            raise ValueError("MARS action-context cache normalization differs")
        self.receipt=receipt
        self.values: OrderedDict[str,tuple[np.ndarray,np.ndarray]]=OrderedDict()
        self.task_tokens={key:torch.tensor(value,dtype=torch.float32) for key,value in
            json.loads((self.root/"task_tokens.json").read_text()).items()}

    def load(self,episode: MarsTemporalEpisode)->tuple[np.ndarray,np.ndarray]:
        key=episode.cache_key
        if key in self.values:
            self.values.move_to_end(key); return self.values[key]
        directory=self.root/episode.task
        decoded=np.load(directory/f"{key}.decoded.npy",mmap_mode="r")
        base=np.load(directory/f"{key}.base_action.npy",mmap_mode="r")
        expected=(episode.length,len(episode.arms),ACTION_HORIZON)
        if decoded.shape!=(*expected,384) or base.shape!=(*expected,8):
            raise ValueError(f"MARS action context shape drift: {key}")
        if decoded.dtype!=np.float16 or base.dtype!=np.float16:
            raise ValueError("MARS action context must be float16")
        self.values[key]=(decoded,base)
        while len(self.values)>self.limit: self.values.popitem(last=False)
        return decoded,base


class MarsTeamBeliefDataset(Dataset):
    """Strict-local runtime input plus removable, permutation-invariant team teacher."""

    RUNTIME_FIELDS=frozenset({
        "runtime_visual_tokens","runtime_visual_mask","history_qpos","history_action",
        "history_mask","action_history_mask","task_token","episode_reset_mask",
        "decoded_action_hidden","base_action",
    })
    TEACHER_FIELDS=frozenset({
        "teacher_current_visual_tokens","teacher_current_visual_mask",
        "teacher_future_visual_tokens","teacher_future_visual_mask",
        "teacher_future_anchor_mask","teacher_agent_state","teacher_agent_mask",
        "teacher_relative_agent_role",
    })

    def __init__(self,episodes: Sequence[MarsTemporalEpisode],stats: Mapping,
                 visual_cache_root: str|Path,action_context_root: str|Path):
        validate_mars_normalization(stats)
        self.episodes=list(episodes); self.visual=MarsVisualCache(visual_cache_root,limit=8)
        self.context=MarsActionContextCache(action_context_root,stats,limit=2)
        self.q_mean=torch.tensor(stats["q_mean"],dtype=torch.float32)
        self.q_std=torch.tensor(stats["q_std"],dtype=torch.float32)
        self.a_mean=torch.tensor(stats["a_mean"],dtype=torch.float32)
        self.a_std=torch.tensor(stats["a_std"],dtype=torch.float32)
        self._handles: dict[str,h5py.File]={}

    def __getstate__(self):
        value=dict(self.__dict__); value["_handles"]={}; return value

    def _group(self,episode: MarsTemporalEpisode):
        if episode.path not in self._handles:
            self._handles[episode.path]=h5py.File(episode.path,"r",swmr=True)
        return self._handles[episode.path][episode.trajectory]

    def __len__(self): return sum(e.length*len(e.arms) for e in self.episodes)

    def __getitem__(self,request: TeamSampleRequest|tuple)->dict:
        if not isinstance(request,TeamSampleRequest): request=TeamSampleRequest(*request)
        episode=self.episodes[request.episode_index]
        if request.task!=episode.task or request.arm not in episode.arms: raise ValueError("MARS belief request drift")
        t=int(request.time_index); ego=int(request.arm); teammates=[arm for arm in episode.arms if arm!=ego]
        group=self._group(episode); cached=self.visual.load(episode)

        first=max(0,t-HISTORY_STEPS+1); obs_idx=list(range(first,t+1)); offset=HISTORY_STEPS-len(obs_idx)
        afirst=max(0,t-HISTORY_STEPS); action_idx=list(range(afirst,t)); aoffset=HISTORY_STEPS-len(action_idx)
        runtime=torch.zeros(HISTORY_STEPS,2,1,768); hmask=torch.zeros(HISTORY_STEPS,dtype=torch.bool)
        qhist=torch.zeros(HISTORY_STEPS,9); ahist=torch.zeros(HISTORY_STEPS,8)
        amask=torch.zeros(HISTORY_STEPS,dtype=torch.bool); reset=torch.zeros(HISTORY_STEPS,dtype=torch.bool)
        own=torch.from_numpy(np.array(cached[f"agent_{ego}"][obs_idx],dtype=np.float32,copy=True))
        runtime[offset:,0,0]=own; runtime[offset:,1,0]=own; hmask[offset:]=True; reset[offset]=True
        q=np.asarray(group[f"obs/agent/panda-{ego}/qpos"][obs_idx],np.float32)
        qhist[offset:]=(torch.from_numpy(q)-self.q_mean)/self.q_std
        if action_idx:
            past=clip_pd_action(np.asarray(group[f"actions/panda-{ego}"][action_idx],np.float32))
            ahist[aoffset:]=(torch.from_numpy(past)-self.a_mean)/self.a_std; amask[aoffset:]=True

        current=torch.zeros(3,1,768); current_mask=torch.zeros(3,1,dtype=torch.bool)
        current[0,0]=torch.from_numpy(np.array(cached[f"agent_{ego}"][t],dtype=np.float32,copy=True))
        current[1,0]=torch.stack([torch.from_numpy(np.array(cached[f"agent_{arm}"][t],dtype=np.float32,copy=True)) for arm in teammates]).mean(0)
        current_mask[:2]=True
        future=torch.zeros(len(FUTURE_OFFSETS_STEPS),3,1,768)
        future_mask=torch.zeros(len(FUTURE_OFFSETS_STEPS),3,1,dtype=torch.bool)
        anchor_mask=torch.zeros(len(FUTURE_OFFSETS_STEPS),dtype=torch.bool)
        teammate_delta=torch.zeros(len(FUTURE_OFFSETS_STEPS),9)
        teammate_now=torch.stack([torch.from_numpy(np.asarray(group[f"obs/agent/panda-{arm}/qpos"][t],np.float32)) for arm in teammates]).mean(0)
        for slot,delta in enumerate(FUTURE_OFFSETS_STEPS):
            target=t+delta
            if target>=episode.length: continue
            future[slot,0,0]=torch.from_numpy(np.array(cached[f"agent_{ego}"][target],dtype=np.float32,copy=True))
            future[slot,1,0]=torch.stack([torch.from_numpy(np.array(cached[f"agent_{arm}"][target],dtype=np.float32,copy=True)) for arm in teammates]).mean(0)
            future_mask[slot,:2]=True; anchor_mask[slot]=True
            future_team=torch.stack([torch.from_numpy(np.asarray(group[f"obs/agent/panda-{arm}/qpos"][target],np.float32)) for arm in teammates]).mean(0)
            teammate_delta[slot]=(future_team-teammate_now)/self.q_std

        ego_state=(torch.from_numpy(np.asarray(group[f"obs/agent/panda-{ego}/qpos"][t],np.float32))-self.q_mean)/self.q_std
        team_state=(teammate_now-self.q_mean)/self.q_std
        agent_state=torch.stack((ego_state,team_state))
        end=min(episode.length,t+TEAMMATE_ACTION_HORIZON)
        teammate_action=torch.zeros(TEAMMATE_ACTION_HORIZON,8); teammate_action_mask=torch.zeros(TEAMMATE_ACTION_HORIZON,dtype=torch.bool)
        team_source=torch.stack([torch.from_numpy(clip_pd_action(np.asarray(group[f"actions/panda-{arm}"][t:end],np.float32))) for arm in teammates]).mean(0)
        teammate_action[:len(team_source)]=(team_source-self.a_mean)/self.a_std; teammate_action_mask[:len(team_source)]=True

        action_end=min(episode.length,t+ACTION_HORIZON)
        source=torch.from_numpy(clip_pd_action(np.asarray(group[f"actions/panda-{ego}"][t:action_end],np.float32)))
        normalized=(source-self.a_mean)/self.a_std
        action=torch.empty(ACTION_HORIZON,8); action[:len(source)]=normalized; action[len(source):]=normalized[-1]
        action_mask=torch.zeros(ACTION_HORIZON,dtype=torch.bool); action_mask[:len(source)]=True
        decoded,base=self.context.load(episode)
        phase=float(t/max(episode.length-1,1))
        return {
            "runtime_visual_tokens":runtime,"runtime_visual_mask":hmask[:,None,None].expand(-1,2,1).clone(),
            "history_qpos":qhist,"history_action":ahist,"history_mask":hmask,
            "action_history_mask":amask,"episode_reset_mask":reset,
            "task_token":self.context.task_tokens[f"{episode.task}:{ego}"].clone(),
            "decoded_action_hidden":torch.from_numpy(np.array(decoded[t,ego],dtype=np.float32,copy=True)),
            "base_action":torch.from_numpy(np.array(base[t,ego],dtype=np.float32,copy=True)),
            "teacher_current_visual_tokens":current,"teacher_current_visual_mask":current_mask,
            "teacher_future_visual_tokens":future,"teacher_future_visual_mask":future_mask,
            "teacher_future_anchor_mask":anchor_mask,"teacher_agent_state":agent_state,
            "teacher_agent_mask":torch.ones(2,dtype=torch.bool),
            "teacher_relative_agent_role":torch.tensor((0,1),dtype=torch.long),
            "teammate_delta":teammate_delta,"teammate_action":teammate_action,
            "teammate_action_mask":teammate_action_mask,"action":action,"action_mask":action_mask,
            "pair_id":torch.tensor(request.episode_index*1_000_000+t),
            "phase_bin":torch.tensor(min(3,int(phase*4))),
            "task_index":torch.tensor(MARS_TASKS.index(episode.task)),
            "sample_key":request.sample_key,
        }


class MarsPairedSituationBatchSampler(Sampler[list[TeamSampleRequest]]):
    """Six paired situations per MARS task; every policy row remains one local arm."""
    def __init__(self,episodes: Sequence[MarsTemporalEpisode],*,updates:int,data_seed:int,start_update:int=0):
        if SAMPLES_PER_TASK!=12 or EFFECTIVE_BATCH!=48: raise ValueError("MARS belief batch contract drift")
        self.episodes=list(episodes); self.updates=updates; self.data_seed=data_seed; self.start_update=start_update
        self.by_task=defaultdict(list)
        for i,e in enumerate(episodes): self.by_task[e.task].append(i)
        counts={len(self.by_task[t]) for t in MARS_TASKS}
        if len(counts)!=1 or next(iter(counts))<1: raise ValueError("MARS belief task episode balance drift")
    def __len__(self): return self.updates-self.start_update
    def requests_for_update(self,update:int)->list[TeamSampleRequest]:
        rng=random.Random(self.data_seed+1_000_003*update); pairs=[]
        for task in MARS_TASKS:
            used=set()
            for _ in range(SAMPLES_PER_TASK//2):
                while True:
                    index=rng.choice(self.by_task[task]); episode=self.episodes[index]; t=rng.randrange(episode.length)
                    if (index,t) not in used: used.add((index,t)); break
                arms=rng.sample(list(episode.arms),2)
                pair=[]
                for arm in arms:
                    identity=f"{episode.cache_key}:{arm}:{t}:mars-n2"
                    pair.append(TeamSampleRequest(index,arm,t,hashlib.sha256(identity.encode()).hexdigest(),task))
                pairs.append(pair)
        rng.shuffle(pairs); rows=[row for pair in pairs for row in pair]
        if Counter(x.task for x in rows)!=Counter({task:SAMPLES_PER_TASK for task in MARS_TASKS}): raise AssertionError("MARS belief task balance")
        return rows
    def __iter__(self)->Iterable[list[TeamSampleRequest]]:
        for update in range(self.start_update+1,self.updates+1): yield self.requests_for_update(update)
    def cursor_receipt(self,completed_update:int)->dict:
        next_update=completed_update+1
        keys=[x.sample_key for x in self.requests_for_update(next_update)] if next_update<=self.updates else []
        return {"format_version":"before-we-act.mars-n2-cursor/1","data_seed":self.data_seed,
            "completed_update":completed_update,"next_update":next_update if keys else None,
            "next_sample_keys":keys,"effective_batch":EFFECTIVE_BATCH,"samples_per_task":SAMPLES_PER_TASK}


def fixed_diagnostic_requests(episodes: Sequence[MarsTemporalEpisode])->list[TeamSampleRequest]:
    rows=[]
    for task in MARS_TASKS:
        candidates=[(i,e) for i,e in enumerate(episodes) if e.task==task]
        ordinals=np.unique(np.linspace(0,len(candidates)-1,num=min(4,len(candidates)),dtype=np.int64)).tolist()
        for ordinal in ordinals:
            index,episode=candidates[ordinal]; t=int(round((ordinal%4+1)*episode.length/5)); arms=(episode.arms[0],episode.arms[-1])
            for arm in arms:
                rows.append(TeamSampleRequest(index,arm,min(t,episode.length-1),f"diagnostic:{episode.cache_key}:{arm}:{t}",task))
    return rows


__all__=["MarsActionContextCache","MarsPairedSituationBatchSampler","MarsTeamBeliefDataset",
    "TEAMMATE_ACTION_HORIZON","fixed_diagnostic_requests"]
