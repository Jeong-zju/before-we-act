"""Train the unchanged official predictive CARE belief on MARS adapters."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract_hash,
    normalization_stats_hash,
    validate_checkpoint_action_contract,
)
from before_we_act.mars_temporal_data import (
    MARS_TASKS,
    load_mars_episodes,
    validate_mars_normalization,
)
from before_we_act.mars_team_belief_data import (
    MarsPairedSituationBatchSampler,
    MarsTeamBeliefDataset,
    fixed_diagnostic_requests,
)
from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.predictive_team_belief_training import TeamBeliefExperiment, paired_permutation
from before_we_act.team_belief.losses import TeamBeliefLossWeights, compute_team_belief_losses
from before_we_act.team_belief.predictive_core import TeamBeliefConfig
from before_we_act.train_predictive_team_belief import masked_action_mse, row_action_mse, shuffle_permutation
from before_we_act.temporal_history_data import sha256_file


MAX_UPDATES=120_000
EVAL_EVERY=5_000
LR_DROP=80_000
DATA_SEED=20260815
CONFIG=TeamBeliefConfig(n_belief_tokens=16,n_evidence_queries=4,event_capacity=4,temporal_layers=2)
WEIGHTS=TeamBeliefLossWeights(
    action=1.0,action_posterior_kl=0.0,teacher_alignment=0.1,future_latent=0.01,
    teacher_reconstruction=0.01,teammate_delta=0.1,teammate_action=0.1,
    exchange_consistency=0.05,anti_collapse=0.01,action_pairing=1.0,
    action_pairing_margin_fraction=0.1,action_pairing_margin_cap=0.01,
)


def atomic_json(path:Path,value:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)


def atomic_save(value:object,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value,tmp); os.replace(tmp,path)


def device_batch(raw,device): return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in raw.items()}


@torch.no_grad()
def evaluate(model,loader,device)->dict:
    model.eval(); values={key:[] for key in ("b0h","b_core","b_shuffle","direct_reactive")}; auxiliary={}
    residual_targets=[]; residual_outputs=[]
    for raw in loader:
        batch=device_batch(raw,device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(batch); permutation=shuffle_permutation(batch["task_index"],batch["phase_bin"])
            shuffled_residual,_=model.belief_residual(
                batch["decoded_action_hidden"],output.candidate.belief.mu[permutation],
                output.candidate.belief.sigma[permutation],output.candidate.belief.reliability[permutation])
            shuffled=batch["base_action"]+shuffled_residual
        for name,prediction in (("b0h",batch["base_action"]),("b_core",output.candidate.prediction),
                                ("b_shuffle",shuffled),("direct_reactive",output.direct_prediction)):
            values[name].extend(row_action_mse(prediction,batch["action"],batch["action_mask"]).cpu().tolist())
        residual_targets.append((batch["action"]-batch["base_action"])[batch["action_mask"]].float().cpu())
        residual_outputs.append(output.candidate.belief_residual[batch["action_mask"]].float().cpu())
        losses=compute_team_belief_losses(output.candidate,batch["action"],batch["action_mask"],
            batch["teammate_delta"],batch["teacher_future_anchor_mask"],batch["teammate_action"],
            batch["teammate_action_mask"],WEIGHTS)
        for key,value in losses.items(): auxiliary.setdefault(key,[]).append(float(value))
    return {
        "macro":{key:float(np.mean(value)) for key,value in values.items()},
        "auxiliary":{key:float(np.mean(value)) for key,value in auxiliary.items()},
        "residual_target_rms":float(torch.cat(residual_targets).square().mean().sqrt()),
        "residual_output_rms":float(torch.cat(residual_outputs).square().mean().sqrt()),
        "rows":len(values["b0h"]),
    }


def export_deployment(selected_path:Path,base_path:Path,output:Path)->None:
    selected=torch.load(selected_path,map_location="cpu",weights_only=False)
    base=torch.load(base_path,map_location="cpu",weights_only=False); config=base["config"]
    base_contract=validate_checkpoint_action_contract(base)
    validate_mars_normalization(base["stats"])
    expected_normalization=normalization_stats_hash(base["stats"])
    if base_contract.get("annotations",{}).get("normalization_sha256")!=expected_normalization:
        raise ValueError("B0-H checkpoint normalization/action contract differs")
    if selected.get("provenance",{}).get("action_contract_sha256")!=action_contract_hash():
        raise ValueError("selected N2 checkpoint predates the shared action contract")
    if selected.get("provenance",{}).get("normalization_semantic_sha256")!=expected_normalization:
        raise ValueError("selected N2 checkpoint normalization/action contract differs")
    policy=PredictiveTeamBeliefPolicy(
        CONFIG,state_dim=9,action_dim=8,horizon=100,d_model=384,enc_layers=4,dec_layers=7,
        roles=4,role_rank=32,history_layers=2,dino_model=str(config["dino_model"]),
        image_height=int(config["image_height"]),image_width=int(config["image_width"]),
        include_teacher=False,residual_safety={"enabled":False},
    )
    incompatible=policy.load_state_dict(base["model"],strict=False)
    allowed={key for key in policy.state_dict() if key.startswith(("belief_core.","direct_belief_residual."))}
    if set(incompatible.missing_keys)!=allowed or incompatible.unexpected_keys: raise RuntimeError(incompatible)
    state=selected["model"]
    runtime={key.removeprefix("belief_core."):value for key,value in state.items()
             if key.startswith("belief_core.") and not key.startswith("belief_core.teacher_branch.")}
    policy.belief_core.load_state_dict(runtime,strict=True)
    residual={key.removeprefix("belief_residual."):value for key,value in state.items() if key.startswith("belief_residual.")}
    policy.direct_belief_residual.load_state_dict(residual,strict=True)
    payload={"format_version":"before-we-act.mars-n2-deployment-checkpoint/1",
        "model":policy.deployment_state_dict(),"stats":base["stats"],"update":int(selected["update"]),
        "action_contract":base_contract,
        "config":{**config,"policy_variant":"mars_predictive_team_belief","n2_config":CONFIG.__dict__,
            "teacher_present":False,"residual_safety":{"enabled":False},
            "action_contract_version":ACTION_CONTRACT_VERSION,
            "action_contract_sha256":action_contract_hash(),
            "normalization_sha256":expected_normalization,
            "source_b0h_checkpoint":str(base_path.resolve()),"source_b0h_checkpoint_sha256":sha256_file(base_path),
            "source_training_checkpoint":str(selected_path.resolve()),"source_training_checkpoint_sha256":sha256_file(selected_path)}}
    atomic_save(payload,output)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root",type=Path,required=True); parser.add_argument("--normalization",type=Path,required=True)
    parser.add_argument("--visual-cache",type=Path,required=True); parser.add_argument("--action-context-cache",type=Path,required=True)
    parser.add_argument("--b0h-checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True); parser.add_argument("--updates",type=int,default=MAX_UPDATES)
    parser.add_argument("--workers",type=int,default=2); parser.add_argument("--save-every",type=int,default=EVAL_EVERY)
    parser.add_argument("--log-every",type=int,default=100); parser.add_argument("--evaluate-at-end",action="store_true")
    parser.add_argument("--episodes-per-task",type=int,default=0,help="smoke-only subset; zero means all 150")
    args=parser.parse_args()
    if not 1<=args.updates<=MAX_UPDATES: raise ValueError("invalid belief budget")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device("cuda:0"); torch.cuda.set_device(device); torch.set_num_threads(min(12,os.cpu_count() or 12))
    random.seed(args.seed); np.random.seed(args.seed%(2**32)); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    episodes=load_mars_episodes(args.raw_root); stats=json.loads(args.normalization.read_text())
    validate_mars_normalization(stats)
    b0h_payload=torch.load(args.b0h_checkpoint,map_location="cpu",weights_only=False)
    b0h_contract=validate_checkpoint_action_contract(b0h_payload)
    normalization_semantic_sha256=normalization_stats_hash(stats)
    if b0h_contract.get("annotations",{}).get("normalization_sha256")!=normalization_semantic_sha256:
        raise ValueError("B0-H checkpoint and requested normalization differ")
    if args.episodes_per_task:
        if not 1<=args.episodes_per_task<150: raise ValueError("invalid smoke episode subset")
        episodes=[episode for task in MARS_TASKS
                  for episode in [e for e in episodes if e.task==task][:args.episodes_per_task]]
    elif int(dataset_receipt := json.loads((args.action_context_cache/"cache_receipt.json").read_text())["episodes"]) != 600:
        raise RuntimeError(f"formal MARS belief requires the 600-episode action cache, got {dataset_receipt}")
    dataset=MarsTeamBeliefDataset(episodes,stats,args.visual_cache,args.action_context_cache)
    model=TeamBeliefExperiment(CONFIG).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0 if step<LR_DROP else 0.1)
    args.output.mkdir(parents=True,exist_ok=True); latest=args.output/"checkpoint_latest.pt"
    saved=torch.load(latest,map_location="cpu",weights_only=False) if latest.is_file() else None
    provenance={"seed":args.seed,"b0h_checkpoint_sha256":sha256_file(args.b0h_checkpoint),
        "normalization_sha256":sha256_file(args.normalization),
        "normalization_semantic_sha256":normalization_semantic_sha256,
        "action_contract_version":ACTION_CONTRACT_VERSION,
        "action_contract_sha256":action_contract_hash(),
        "action_context_cache_receipt_sha256":sha256_file(args.action_context_cache/"cache_receipt.json"),
        "policy_episode_count":len(episodes),"policy_training_split":"all"}
    start=0; evaluations=[]
    if saved:
        if saved["provenance"]!=provenance: raise RuntimeError("MARS N2 resume provenance drift")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start=int(saved["update"]); evaluations=list(saved["evaluations"])
    sampler=MarsPairedSituationBatchSampler(episodes,updates=MAX_UPDATES,data_seed=DATA_SEED,start_update=start)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,
        persistent_workers=args.workers>0,prefetch_factor=2 if args.workers>0 else None)
    requests=fixed_diagnostic_requests(episodes); batches=[requests[i:i+32] for i in range(0,len(requests),32)]
    diagnostic=DataLoader(dataset,batch_sampler=batches,num_workers=0,pin_memory=True)
    atomic_json(args.output/"status.json",{"status":"TRAINING","seed":args.seed,"update":start,"target_updates":args.updates})
    started=time.time(); last={}
    for update,raw in enumerate(loader,start=start+1):
        if update>args.updates: break
        step_seed=args.seed+10_000_019*update; random.seed(step_seed); np.random.seed(step_seed%(2**32))
        torch.manual_seed(step_seed); torch.cuda.manual_seed_all(step_seed)
        batch=device_batch(raw,device); optimizer.zero_grad(set_to_none=True); model.train()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(batch); partner=paired_permutation(batch["pair_id"])
            swapped=replace(output.candidate,belief=replace(output.candidate.belief,mu=output.candidate.belief.mu[partner]))
            negative=shuffle_permutation(batch["task_index"],batch["phase_bin"])
            cf_residual,_=model.belief_residual(batch["decoded_action_hidden"],output.candidate.belief.mu[negative],
                output.candidate.belief.sigma[negative],output.candidate.belief.reliability[negative])
            residual_target=batch["action"]-batch["base_action"]
            losses=compute_team_belief_losses(output.candidate,batch["action"],batch["action_mask"],
                batch["teammate_delta"],batch["teacher_future_anchor_mask"],batch["teammate_action"],
                batch["teammate_action_mask"],WEIGHTS,swapped_output=swapped,
                counterfactual_prediction=batch["base_action"]+cf_residual,
                counterfactual_residual_target=residual_target[negative],counterfactual_action_mask=batch["action_mask"][negative])
            direct=masked_action_mse(output.direct_prediction,batch["action"],batch["action_mask"]); loss=losses["total"]+direct
        if not torch.isfinite(loss): raise FloatingPointError(update)
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad): raise FloatingPointError(f"gradient {update}")
        optimizer.step(); scheduler.step(); last={key:float(value.detach()) for key,value in losses.items()}
        last.update(direct_reactive_action=float(direct),combined=float(loss),grad_norm=float(grad))
        if update==start+1 or update%args.log_every==0 or update==args.updates:
            elapsed=time.time()-started; row={"update":update,"target_updates":args.updates,**last,
                "learning_rate":scheduler.get_last_lr()[0],"eta_hours":(args.updates-update)*elapsed/max(update-start,1)/3600}
            print(json.dumps(row,sort_keys=True),flush=True); atomic_json(args.output/"heartbeat.json",row)
        should_eval=(args.updates==MAX_UPDATES and update%EVAL_EVERY==0) or (args.evaluate_at_end and update==args.updates)
        if should_eval:
            metrics=evaluate(model,diagnostic,device); evaluations.append({"update":update,"validation":metrics})
            print(json.dumps({"evaluation":evaluations[-1]},sort_keys=True),flush=True)
        if update==args.updates or update%args.save_every==0:
            checkpoint={"format_version":"before-we-act.mars-n2-training-checkpoint/1","model":model.state_dict(),
                "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"update":update,
                "evaluations":evaluations,"last_losses":last,"sample_cursor":sampler.cursor_receipt(update),
                "provenance":provenance,"config":CONFIG.__dict__}
            atomic_save(checkpoint,latest); atomic_save(checkpoint,args.output/f"checkpoint_{update:06d}.pt")
    if args.updates<MAX_UPDATES:
        atomic_json(args.output/"status.json",{"status":"PASSED_SMOKE","seed":args.seed,"update":args.updates,
            "resume_start_update":start,
            "selected_validation":evaluations[-1]["validation"] if evaluations else None}); return
    candidates=[row for row in evaluations if int(row["update"])>=100_000]
    selected=min(candidates,key=lambda row:row["validation"]["macro"]["b_core"])
    selected_path=args.output/f"checkpoint_{int(selected['update']):06d}.pt"
    export_deployment(selected_path,args.b0h_checkpoint,args.output/"deployment_checkpoint.pt")
    atomic_json(args.output/"status.json",{"status":"COMPLETE","seed":args.seed,"update":MAX_UPDATES,
        "selected_update":selected["update"],"selected_validation":selected["validation"],
        "deployment_checkpoint_sha256":sha256_file(args.output/"deployment_checkpoint.pt"),
        "completed_at_utc":datetime.now(timezone.utc).isoformat()})


if __name__=="__main__": main()
