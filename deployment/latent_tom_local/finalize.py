from __future__ import annotations
import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch
from local_dataset import TASKS

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(16*1024*1024),b""): h.update(b)
    return h.hexdigest()

def commit(root):
    return subprocess.check_output(["git","-c",f"safe.directory={root}","-C",root,"rev-parse","HEAD"],text=True).strip()

def main():
    data=Path("/workspace/datasets/robofactory_multitask"); run=Path("/workspace/bwa_latent_tom_runs/formal")
    validation=json.loads((run/"validation20/summary.json").read_text())
    train_status=json.loads((run/"status.json").read_text())
    receipts={t:json.loads((data/t/"download_receipt.json").read_text()) for t in TASKS}
    checkpoint_path=run/"last.pt"
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False,mmap=True)
    code_root=Path("/workspace/repos/before-we-act/deployment/latent_tom_local")
    code_files=("local_dataset.py","local_policy.py","train_local.py","evaluate_closed_loop.py",
                "checkpoint_smoke.py","build_resized_cache.py","audit_runtime_isolation.py",
                "finalize.py","verify_delivery.py")
    supervisor_receipts=Path("/workspace/bwa_latent_tom_runs/supervisor/receipts")
    stage_receipts={p.stem:{"path":str(p),"sha256":digest(p)}
                    for p in sorted(supervisor_receipts.glob("*.json"))}
    report={"schema":"bwa.latent_tom.final_report.v1","status":"complete","episodes":sum(x["episodes_total"] for x in receipts.values()),
            "local_agent_policy":"same_checkpoint_strict_local_rgb_qpos_task_to_action8",
            "checkpoint":{"path":str(checkpoint_path),"sha256":digest(checkpoint_path),
                          "step":int(checkpoint["step"]),"contract":checkpoint["contract"],
                          "raw_weights":bool(checkpoint.get("model")),
                          "ema_weights":bool(checkpoint.get("ema_model")),
                          "ema_optimization_step":int(checkpoint.get("ema_optimization_step",0))},
            "training":{"status":train_status["status"],"all_episodes":train_status["all_episodes"],
                        "indexed_local_timesteps":train_status["indexed_local_timesteps"],
                        "effective_dataset_passes":train_status["effective_dataset_passes"],
                        "steps":300000,"batch_size":512,"gradient_accumulation":1,
                        "workers":16,"precision":"bfloat16","warmup_steps":500,
                        "lr_schedule":"linear_warmup_cosine","optimizer":"AdamW",
                        "learning_rate":1e-4,"rgb_shape":[240,320],"observation_steps":2,
                        "action_horizon":40,"action_dim":8,"locality_run":16},
            "validation20":{"path":str(run/"validation20/summary.json"),"sha256":digest(run/"validation20/summary.json"),
                            "episodes":validation["total_episodes"],"episodes_per_task":validation["episodes_per_task"],
                            "macro_success_rate":validation["macro_success_rate"],
                            "per_task":{t:{"episodes":validation["tasks"][t]["episodes"],
                                             "successes":validation["tasks"][t]["successes"],
                                             "success_rate":validation["tasks"][t]["success_rate"]}
                                        for t in TASKS}},
            "source":{"latent_tom_official":"a51d929027799a53d54e7d7d2ba90e2703642b4a",
                      "before_we_act":commit("/workspace/repos/before-we-act"),"robofactory":commit("/workspace/repos/RoboFactory"),
                      "pipeline":{"path":"/workspace/bwa_latent_tom_pipeline/pipeline.json",
                                  "sha256":digest("/workspace/bwa_latent_tom_pipeline/pipeline.json")},
                      "adaptation_files":{name:digest(code_root/name) for name in code_files}},
            "datasets":{t:{"repo_id":r["repo_id"],"revision":r["revision"],"episodes":r["episodes_total"],"bytes":r["bytes_total"]} for t,r in receipts.items()},
            "supervisor_stage_receipts":stage_receipts,
            "completed_at":datetime.now(timezone.utc).isoformat()}
    path=Path("/workspace/bwa_latent_tom_runs/final_report.json"); path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"complete","output":str(path),"validation_episodes":validation["total_episodes"]}))
if __name__=="__main__": main()
