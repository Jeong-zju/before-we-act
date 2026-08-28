from __future__ import annotations
import argparse, json
from pathlib import Path
from .common import TASKS, ENVS, atomic_json
from .evaluate import EVALUATOR_REVISION
import subprocess, sys
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output-root",required=True); p.add_argument("--robofactory-root",required=True); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True); rows={}
    for i,task in enumerate(TASKS):
        episodes=1 if a.smoke else 20; seed=(990000 if a.smoke else 20260820)+i*1000; out=root/f"{task}.json"; _,_,max_steps=ENVS[task]
        cmd=[sys.executable,"-m","deployment.mars_dp.evaluate","--checkpoint",a.checkpoint,"--task",task,"--robofactory-root",a.robofactory_root,"--output",str(out),"--episodes",str(episodes),"--seed-start",str(seed),"--max-steps",str(2 if a.smoke else max_steps),"--inference-steps","20","--replan-interval","6"]
        if a.smoke: cmd.append("--smoke")
        subprocess.run(cmd,check=True); rows[task]=json.loads(out.read_text())
    summary={"schema":"mars-control.dp.smoke.summary.v1" if a.smoke else "mars-control.dp.validation20.summary.v1","status":"complete","episodes_per_task":1 if a.smoke else 20,"total_episodes":4 if a.smoke else 80,"checkpoint":a.checkpoint,"checkpoint_sha256":next(iter({r["checkpoint_sha256"] for r in rows.values()})),"evaluator_revision":EVALUATOR_REVISION,"rgb_preprocessing":"uint8_div_255_to_unit_float","policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_absolute_action8","tasks":{k:{"episodes":v["episodes"],"successes":v["successes"],"success_rate":v["success_rate"]} for k,v in rows.items()},"successes":sum(v["successes"] for v in rows.values())}
    atomic_json(root/"summary.json",summary); print(json.dumps(summary))
if __name__=="__main__": main()
