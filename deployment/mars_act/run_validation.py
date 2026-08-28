#!/usr/bin/env python3
import argparse,json,os,subprocess,sys
from pathlib import Path
TASKS=('place_cube_in_cup','strike_cube_hard','three_robots_place_shoes','four_robots_stack_cube')
EVALUATOR_REVISION='rgb-uint8-to-unit-float-v2'
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output-root',required=True); p.add_argument('--robofactory-root',default='/workspace/repos/RoboFactory'); p.add_argument('--smoke',action='store_true'); a=p.parse_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True); rows={}
for i,task in enumerate(TASKS):
 out=root/f'{task}.json'; cmd=[sys.executable,'-m','deployment.mars_act.evaluate','--checkpoint',a.checkpoint,'--task',task,'--robofactory-root',a.robofactory_root,'--episodes','1' if a.smoke else '20','--seed-start',str((990000 if a.smoke else 20260820)+i*1000),'--output',str(out)]
 if a.smoke: cmd+=['--smoke','--max-steps','2']
 subprocess.run(cmd,check=True); rows[task]=json.loads(out.read_text())
if any(v.get('evaluator_revision') != EVALUATOR_REVISION for v in rows.values()):
 raise RuntimeError('validation task output was produced by a stale evaluator')
summary={'schema':'mars-control.act.smoke.summary.v1' if a.smoke else 'mars-control.act.validation20.summary.v1','status':'complete','episodes_per_task':1 if a.smoke else 20,'total_episodes':4 if a.smoke else 80,'checkpoint':a.checkpoint,'evaluator_revision':EVALUATOR_REVISION,'rgb_preprocessing':'uint8_div_255_to_unit_float','policy_contract':'shared_weights_decentralized_local_rgb_qpos_to_local_action8','tasks':{k:{'episodes':v['episodes'],'successes':v['successes']} for k,v in rows.items()}}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary))
