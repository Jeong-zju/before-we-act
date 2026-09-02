#!/usr/bin/env python3
import argparse,json,os,subprocess,sys
from pathlib import Path
TASKS=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube"); REVISION="mars-rdt-strict-local-absolute-v1"
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output-root",required=True); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True); procs=[]
for gpu,task in enumerate(TASKS):
 out=root/f"{task}.json"; cmd=[sys.executable,"-m","deployment.rdt_mars.evaluate","--checkpoint",a.checkpoint,"--task",task,"--episodes","1" if a.smoke else "20","--seed-start",str((990000 if a.smoke else 20260820)+gpu*1000),"--output",str(out)];
 if a.smoke: cmd += ["--smoke","--max-steps","2"]
 env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); log=(root/f"{task}.log").open("ab",buffering=0); procs.append((task,out,log,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)))
failed=[]
for task,out,log,proc in procs:
 code=proc.wait(); log.close();
 if code: failed.append((task,code))
if failed: raise RuntimeError(f"validation workers failed: {failed}")
rows={task:json.loads(out.read_text()) for task,out,_,_ in procs}
if any(x.get("evaluator_revision")!=REVISION for x in rows.values()): raise RuntimeError("stale evaluator output")
summary={"schema":"mars-control.rdt.smoke.summary.v1" if a.smoke else "mars-control.rdt.validation20.summary.v1","status":"complete","episodes_per_task":1 if a.smoke else 20,"total_episodes":4 if a.smoke else 80,"checkpoint":a.checkpoint,"evaluator_revision":REVISION,"policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_absolute_action8","tasks":{k:{"episodes":v["episodes"],"successes":v["successes"],"max_steps":v["max_steps"]} for k,v in rows.items()},"successes":sum(v["successes"] for v in rows.values())}; (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary))
