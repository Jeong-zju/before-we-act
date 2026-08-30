from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from .common import TASKS,ENVS,atomic_json
from .evaluate import REVISION
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--noposplat-weight",required=True); p.add_argument("--output-root",required=True); p.add_argument("--robofactory-root",required=True); p.add_argument("--inference-steps",type=int,choices=(20,100),default=100); p.add_argument("--ensemble-decay",type=float,default=0.01); p.add_argument("--selection-file"); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True); rows={}; inference_steps=int(json.loads(Path(a.selection_file).read_text())["selected_inference_steps"]) if a.selection_file else a.inference_steps
    for task in TASKS:
        _,_,limit,seed=ENVS[task]; out=root/f"{task}.json"; cmd=[sys.executable,"-m","deployment.mars_gaudp.evaluate","--checkpoint",a.checkpoint,"--noposplat-weight",a.noposplat_weight,"--task",task,"--robofactory-root",a.robofactory_root,"--output",str(out),"--episodes",str(1 if a.smoke else 20),"--seed-start",str(990000 if a.smoke else seed),"--max-steps",str(2 if a.smoke else limit),"--inference-steps",str(inference_steps),"--ensemble-decay",str(a.ensemble_decay)];
        if a.smoke: cmd.append("--smoke")
        subprocess.run(cmd,check=True); rows[task]=json.loads(out.read_text())
    summary={"schema":"mars-control.gaudp.smoke.summary.v1" if a.smoke else "mars-control.gaudp.validation20.summary.v1","status":"complete","episodes_per_task":1 if a.smoke else 20,"total_episodes":4 if a.smoke else 80,"checkpoint":a.checkpoint,"checkpoint_sha256":next(iter({r["checkpoint_sha256"] for r in rows.values()})),"evaluator_revision":REVISION,"inference_steps":inference_steps,"temporal_ensemble_decay":a.ensemble_decay,"policy_contract":"shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8","tasks":{k:{"episodes":v["episodes"],"successes":v["successes"],"success_rate":v["success_rate"]} for k,v in rows.items()},"successes":sum(v["successes"] for v in rows.values())}; atomic_json(root/"summary.json",summary); print(json.dumps(summary))
if __name__=="__main__": main()
