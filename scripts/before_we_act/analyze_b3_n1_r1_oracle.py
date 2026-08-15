#!/usr/bin/env python3
"""Evaluate and classify R1-2 oracle value plus same-situation pairing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from before_we_act.b3_n1_r1 import (
    FrozenR1Backbones,
    R1OracleDataset,
    R1OracleProbeSet,
    R1_SEEDS,
    fixed_requests,
    load_split,
    split_by_episode_key,
)
from before_we_act.step2_temporal_data import SIX_TASKS, sha256_file
from before_we_act.train_b3_n1_r1_fair_probe import device_batch
from before_we_act.train_b3_n1_r1_oracle import evaluate_oracle, fixed_loader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--oracle-contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize(metrics):
    h = float(metrics["macro"]["h"]); oracle = float(metrics["macro"]["h_oracle"])
    per_task = {}
    for index, task in enumerate(SIX_TASKS):
        base = float(metrics["per_task"]["h"][str(index)])
        value = float(metrics["per_task"]["h_oracle"][str(index)])
        per_task[task] = {"h": base, "h_oracle": value, "absolute_h_minus_oracle": base-value, "relative_improvement": (base-value)/max(abs(base),1e-12)}
    return {"macro": metrics["macro"], "absolute_h_minus_oracle": h-oracle, "relative_improvement": (h-oracle)/max(abs(h),1e-12), "per_task": per_task, "rows": metrics["rows"]}


@torch.no_grad()
def pair_audit(dataset, split, backbones, device):
    loader = fixed_loader(dataset, split, "test")
    rows = []
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = backbones(batch).h.float()
        for index in range(len(h)):
            rows.append(
                {
                    "h": h[index].cpu().numpy(),
                    "task": int(batch["task_index"][index]),
                    "arm": int(batch["agent_slot"][index]),
                    "phase_bin": int(batch["phase_bin"][index]),
                    "episode": int(batch["episode_label"][index]),
                    "ego": batch["action"][index].float().cpu().numpy(),
                    "ego_mask": batch["action_mask"][index].cpu().numpy(),
                    "teammate": batch["oracle_teammate_action"][index].float().cpu().numpy(),
                    "teammate_mask": batch["oracle_teammate_action_mask"][index].cpu().numpy(),
                }
            )
    result = {}
    for task_index, task in enumerate(SIX_TASKS):
        indices = [index for index,row in enumerate(rows) if row["task"] == task_index]
        ego_distances=[]; teammate_distances=[]; similarities=[]
        for index in indices:
            row=rows[index]
            candidates=[other for other in indices if rows[other]["arm"]==row["arm"] and rows[other]["phase_bin"]==row["phase_bin"] and rows[other]["episode"]!=row["episode"]]
            if not candidates: continue
            normalized=row["h"]/max(np.linalg.norm(row["h"]),1e-8)
            other=max(candidates,key=lambda item: float(normalized @ (rows[item]["h"]/max(np.linalg.norm(rows[item]["h"]),1e-8))))
            peer=rows[other]
            valid_ego=row["ego_mask"] & peer["ego_mask"]
            valid_team=row["teammate_mask"] & peer["teammate_mask"]
            if not valid_ego.any() or not valid_team.any(): continue
            ego_distances.append(float(np.square(row["ego"][valid_ego]-peer["ego"][valid_ego]).mean()))
            teammate_distances.append(float(np.square(row["teammate"][valid_team]-peer["teammate"][valid_team]).mean()))
            similarities.append(float(normalized @ (peer["h"]/max(np.linalg.norm(peer["h"]),1e-8))))
        ego=np.asarray(ego_distances); teammate=np.asarray(teammate_distances)
        correlation=float(np.corrcoef(ego,teammate)[0,1]) if len(ego)>2 and ego.std()>0 and teammate.std()>0 else 0.0
        rng=np.random.default_rng(20260815+task_index)
        bootstrap=[]
        for _ in range(10_000):
            chosen=rng.integers(0,len(ego),size=len(ego))
            x=ego[chosen]; y=teammate[chosen]
            bootstrap.append(float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else 0.0)
        null=[]
        for _ in range(10_000):
            y=rng.permutation(teammate)
            null.append(float(np.corrcoef(ego,y)[0,1]) if ego.std()>0 and y.std()>0 else 0.0)
        ci=[float(value) for value in np.quantile(bootstrap,[0.025,0.975])]
        null_q=float(np.quantile(null,0.975))
        positive=correlation>null_q and ci[0]>0
        result[task]={"pairs":len(ego),"mean_hidden_cosine":float(np.mean(similarities)),"correlation":correlation,"bootstrap_ci95":ci,"permutation_97_5":null_q,"positive":positive}
    return result


def main():
    args=parse_args()
    parent=json.loads(args.parent_contract.read_text())
    oracle_contract=json.loads(args.oracle_contract.read_text())
    split_payload=load_split(args.scenario_split); split=split_by_episode_key(split_payload)
    statuses={}
    for seed in R1_SEEDS:
        statuses[str(seed)]=json.loads((args.run_root/"r1_2_oracle"/f"seed_{seed}"/"status.json").read_text())
    sufficient=all(row["status"] in {"PLATFORM_REACHED","SATURATED_BY_OVERFIT"} for row in statuses.values())
    if not sufficient:
        payload={"format_version":"before-we-act.b3-n1-r1-oracle-conclusion/1","stage":"R1-2-OLD-DATA-IDENTIFIABILITY","status":"INCONCLUSIVE_TRAINING_NOT_CONVERGED","training_status":statuses,"test_opened":False,"completed_at_utc":utc_now(),"n2_authorized":False,"human_summary":"队友 oracle 至少一个 seed 还没到平台，旧数据是否有答案仍不能判断。"}
        atomic_json(args.output,payload); print(json.dumps({"status":payload["status"]})); return
    dataset=R1OracleDataset(args.cache); device=torch.device("cuda:0"); torch.cuda.set_device(device)
    evaluation={}; first_backbones=None
    for seed in R1_SEEDS:
        selected=int(statuses[str(seed)]["selected_update"])
        checkpoint=torch.load(args.run_root/"r1_2_oracle"/f"seed_{seed}"/f"checkpoint_{selected:06d}.pt",map_location="cpu",weights_only=False)
        probes=R1OracleProbeSet().to(device); probes.load_state_dict(checkpoint["probes"])
        n1=Path(parent["old_n1_read_only"]["representation_checkpoints"][str(seed)]["path"])
        backbones=FrozenR1Backbones(b0h_checkpoint=Path(parent["b0h"]["checkpoint"]),n1_checkpoint=n1,visual_mean=dataset.visual_mean,visual_std=dataset.visual_std).to(device)
        evaluation[str(seed)]={"selected_update":selected}
        for name in ("validation","test"):
            evaluation[str(seed)][name]=summarize(evaluate_oracle(backbones,probes,fixed_loader(dataset,split,name),device))
        if seed==R1_SEEDS[0]: first_backbones=backbones
        else: del backbones
        del probes; torch.cuda.empty_cache()
    pairing=pair_audit(dataset,split,first_backbones,device); del first_backbones; torch.cuda.empty_cache()
    every_seed=all(evaluation[str(seed)][name]["absolute_h_minus_oracle"]>0 for seed in R1_SEEDS for name in ("validation","test"))
    task_medians={}; positive_tasks=0
    for task in SIX_TASKS:
        val=float(np.median([evaluation[str(seed)]["validation"]["per_task"][task]["relative_improvement"] for seed in R1_SEEDS]))
        test=float(np.median([evaluation[str(seed)]["test"]["per_task"][task]["relative_improvement"] for seed in R1_SEEDS]))
        positive=val>0 and test>0; positive_tasks+=int(positive); task_medians[task]={"validation_relative_median":val,"test_relative_median":test,"positive_both":positive}
    controls=all(evaluation[str(seed)][name]["macro"]["h_oracle"]<evaluation[str(seed)][name]["macro"][control] for seed in R1_SEEDS for name in ("validation","test") for control in ("h_oracle_shuffle","h_matched_capacity"))
    pair_positive=sum(int(row["positive"]) for row in pairing.values())
    passed=every_seed and positive_tasks>=4 and controls and pair_positive>=4
    status="OLD_DATA_TEAMMATE_ORACLE_VALUE_IDENTIFIED" if passed else "DATA_OR_TASK_HAS_NO_IDENTIFIABLE_TEAMMATE_ACTION_VALUE"
    payload={"format_version":"before-we-act.b3-n1-r1-oracle-conclusion/1","stage":"R1-2-OLD-DATA-IDENTIFIABILITY","status":status,"completed_at_utc":utc_now(),"contract_sha256":sha256_file(args.oracle_contract),"training_status":statuses,"test_opened":True,"evaluation":evaluation,"same_situation_pair_audit":pairing,"task_medians":task_medians,"gate":{"oracle_beats_h_every_seed_both_splits":every_seed,"positive_tasks":positive_tasks,"controls_clean":controls,"pair_audit_positive_tasks":pair_positive,"passed":passed},"n2_authorized":False,"human_summary":"旧 720 条数据里，真的知道队友接下来做什么会让 ego 动作更好预测，而且相似局面配对也支持这种关联；旧 belief 没学出来更像监督问题。" if passed else "连训练期偷看真实队友动作都不能稳定改善 ego 动作，或这种增量可被容量/打乱解释；继续换 belief 架构没有依据。"}
    atomic_json(args.output,payload); print(json.dumps({"status":status,"positive_tasks":positive_tasks,"pair_positive_tasks":pair_positive,"controls_clean":controls},sort_keys=True))


if __name__=="__main__": main()
