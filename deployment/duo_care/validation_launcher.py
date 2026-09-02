from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
from .prepare import TASKS
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--episodes',type=int,required=True); p.add_argument('--max-steps',type=int); p.add_argument('--workers',type=int,default=4); p.add_argument('--candidate-zero',action='store_true'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); pending=list(TASKS); active=[]; results=[]; free_slots=list(range(a.workers))
 while pending or active:
  while pending and len(active)<a.workers:
   task=pending.pop(0); out=a.output/f'{task}.json'
   if out.is_file() and json.loads(out.read_text()).get('total_episodes')==a.episodes: results.append(json.loads(out.read_text())); continue
   slot=free_slots.pop(0); argv=['/venv/main/bin/python','-m','deployment.duo_care.evaluate','--checkpoint',str(a.checkpoint),'--data',str(a.data),'--output',str(out),'--episodes',str(a.episodes),'--task',task,'--device','cuda:0']
   if a.max_steps: argv += ['--max-steps',str(a.max_steps)]
   if a.candidate_zero: argv += ['--candidate-zero']
   log=(a.output/f'{task}.log').open('a'); env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(slot); env['MUJOCO_EGL_DEVICE_ID']='0'; active.append((slot,task,out,subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True),log))
  slot,task,out,proc,log=active.pop(0); code=proc.wait(); log.close(); free_slots.append(slot); free_slots.sort()
  if code: raise RuntimeError(f'{task} evaluator exited {code}')
  results.append(json.loads(out.read_text()))
 rows=[row for result in results for row in result['rows']]; tasks={t:[r for r in rows if r['task']==t] for t in TASKS}; summary={'status':'complete','schema':'duobench-care-validation20-v1','episodes_per_task':a.episodes,'total_episodes':len(rows),'successes':sum(int(r['success']) for r in rows),'macro_success_rate':float(np.mean([np.mean([r['success'] for r in tasks[t]]) for t in TASKS])),'normalized_final_stage_progress':float(np.mean([r['final_stage_progress'] for r in rows])),'tasks':{t:{'episodes':len(tasks[t]),'successes':sum(int(r['success']) for r in tasks[t]),'success_rate':float(np.mean([r['success'] for r in tasks[t]])),'mean_final_stage_progress':float(np.mean([r['final_stage_progress'] for r in tasks[t]]))} for t in TASKS},'rows':rows}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps({k:summary[k] for k in ('status','total_episodes','successes','macro_success_rate')}))
if __name__=='__main__':main()
