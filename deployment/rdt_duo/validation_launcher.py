from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from .protocol import TASKS, EVALUATOR_REVISION

def main():
 p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--episodes",type=int,default=20); p.add_argument("--workers",type=int,default=4); p.add_argument("--max-steps",type=int); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); pending=list(TASKS); results=[]
 while pending:
  batch=pending[:a.workers]; pending=pending[a.workers:]; procs=[]
  for i,task in enumerate(batch):
   out=a.output/f"{task}.json"; cmd=[sys.executable,"-m","deployment.rdt_duo.evaluate","--checkpoint",str(a.checkpoint),"--data",str(a.data),"--output",str(out),"--task",task,"--episodes",str(a.episodes),"--device","cuda:0"]
   if a.max_steps: cmd += ["--max-steps",str(a.max_steps)]
   if a.smoke: cmd += ["--smoke"]
   log=(a.output/f"{task}.log").open("a"); env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(i); procs.append((task,out,log,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)))
  for task,out,log,proc in procs:
   rc=proc.wait(); log.close();
   if rc: raise RuntimeError(f"{task} evaluator exited {rc}")
   results.append(json.loads(out.read_text()))
 rows=[r for x in results for r in x["rows"]]; by={t:[r for r in rows if r["task"]==t] for t in TASKS}; summary={"status":"complete","schema":"duobench-rdt-validation20-v1","episodes_per_task":a.episodes,"total_episodes":len(rows),"successes":sum(int(r["success"]) for r in rows),"macro_success_rate":float(np.mean([np.mean([r["success"] for r in by[t]]) for t in TASKS])),"evaluator_revision":EVALUATOR_REVISION,"policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_local_absolute_action8","tasks":{t:{"episodes":len(by[t]),"successes":sum(int(r["success"]) for r in by[t]),"success_rate":float(np.mean([r["success"] for r in by[t]])),"max_steps":by[t][0]["max_steps"]} for t in TASKS},"rows":rows}; (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps({k:summary[k] for k in ("status","total_episodes","successes","macro_success_rate")}))
if __name__=="__main__": main()
