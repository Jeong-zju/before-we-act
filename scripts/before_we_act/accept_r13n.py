#!/usr/bin/env python3
"""Authoritative completeness acceptance for the R13N B6 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
import time

from before_we_act.r13n import TASKS, sha256
from before_we_act.train_action_generator_r4 import atomic_json


STAGES=("discovery","validation","formal")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--offline",type=Path,required=True); parser.add_argument("--seed-protocol",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    checkpoint=args.checkpoint.resolve(strict=True); offline=json.loads(args.offline.resolve(strict=True).read_text()); seeds=json.loads(args.seed_protocol.resolve(strict=True).read_text())
    checks={
        "full_checkpoint_130k":checkpoint.name=="checkpoint_130000.pt",
        "full_validation_complete":offline.get("validation_rows")==offline.get("expected_validation_rows") and offline.get("all_outputs_finite") is True,
        "seed_sets_disjoint":seeds.get("all_seeds_disjoint") is True and seeds.get("total_unique_seeds")==360,
    }
    manifest_path=args.run_root/"run_manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    expected_clip=manifest.get("normalized_clip") if manifest.get("run_variant")=="normalized_clip_physical_bounds_fix_v2" else None
    results={}; totals={}; all_rows=[]; runtime_bounds=[]
    for stage in STAGES:
        results[stage]={}; totals[stage]=0
        for task in TASKS:
            path=args.run_root/"evaluation"/stage/f"{task}.json"; payload=json.loads(path.resolve(strict=True).read_text()); rows=payload.get("rows",[])
            runtime=payload.get("physical_action_clip",{})
            runtime_valid=(
                expected_clip is None
                or (
                    payload.get("normalized_clip")==expected_clip
                    and isinstance(runtime.get("elements"),int)
                    and runtime.get("elements",0)>0
                    and isinstance(runtime.get("clipped_elements"),int)
                    and 0<=runtime.get("clipped_elements",-1)<=runtime.get("elements",0)
                    and isinstance(runtime.get("fraction"),(int,float))
                    and math.isfinite(float(runtime["fraction"]))
                    and 0<=float(runtime["fraction"])<=1
                )
            )
            valid=payload.get("round")=="R13N" and payload.get("model_id")=="b6_act_six_task" and payload.get("task")==task and payload.get("stage")==stage and payload.get("episodes")==20 and len(rows)==20 and payload.get("candidate_native_episodes")==20 and payload.get("fallback_episodes")==0 and all(row.get("candidate_native") is True and row.get("fallback_used") is False for row in rows) and runtime_valid
            checks[f"{stage}_{task}_complete_native"]=valid
            successes=int(payload.get("successes",0)); totals[stage]+=successes; results[stage][task]={"successes":successes,"episodes":20,"success_rate":successes/20,"p95_latency_ms":payload.get("latency_ms",{}).get("p95"),"physical_action_clip_fraction":runtime.get("fraction"),"result":str(path)}; all_rows.extend(rows)
            if runtime_valid and expected_clip is not None: runtime_bounds.append(runtime)
    checks["all_360_rollouts_unique"]=len({(row.get("stage"),row.get("task"),row.get("seed")) for row in all_rows})==360
    checks["candidate_native_coverage_100_percent"]=len(all_rows)==360 and all(row.get("candidate_native") is True for row in all_rows)
    checks["no_fallback_result_reuse"]=all(row.get("fallback_used") is False for row in all_rows)
    if expected_clip is not None:
        checks["clipfix_runtime_contract_complete"]=len(runtime_bounds)==18
    logs=list((args.run_root/"logs").glob("*.log")); alerts=[]
    for path in logs:
        text=path.read_text(errors="replace")
        for token in ("CUDA out of memory","FloatingPointError","non-finite","Traceback (most recent call last)"):
            if token in text: alerts.append({"log":str(path),"token":token})
    checks["no_runtime_alerts"]=not alerts
    videos={task:{kind:str(args.run_root/"videos"/f"{task}_{kind}.mp4") for kind in ("success","failure") if (args.run_root/"videos"/f"{task}_{kind}.mp4").is_file()} for task in TASKS}
    physical_elements=sum(row.get("elements",0) for row in runtime_bounds)
    physical_clipped=sum(row.get("clipped_elements",0) for row in runtime_bounds)
    result={"schema_version":1,"round":"R13N","model_id":"b6_act_six_task","run_variant":manifest.get("run_variant","original_v1"),"status":"PASSED" if all(checks.values()) else "FAILED","passed":all(checks.values()),"checks":checks,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"offline":offline,"closed_loop":results,"stage_totals":{stage:{"successes":totals[stage],"episodes":120,"success_rate":totals[stage]/120} for stage in STAGES},"candidate_native_episodes":len(all_rows),"fallback_episodes":sum(row.get("fallback_used") is True for row in all_rows),"normalized_clip":expected_clip,"physical_action_clip":{"elements":physical_elements,"clipped_elements":physical_clipped,"fraction":physical_clipped/max(physical_elements,1)} if expected_clip is not None else None,"alerts":alerts,"videos":videos,"created_at_epoch":time.time()}
    atomic_json(args.output,result); print(json.dumps(result|{"offline":"saved","closed_loop":"saved"},sort_keys=True)); raise SystemExit(0 if result["passed"] else 10)


if __name__=="__main__": main()
