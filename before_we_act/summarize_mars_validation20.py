"""Summarize the four-task MARS validation20 run."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

TASKS=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); args=p.parse_args()
    results={}; successes=episodes=0
    for task in TASKS:
        value=json.loads((args.root/f"{task}.json").read_text())
        if value.get("status")!="complete" or value.get("episodes")!=20: raise RuntimeError(f"incomplete {task}")
        results[task]={"successes":value["successes"],"episodes":20,"success_rate":value["success_rate"]}
        successes+=int(value["successes"]); episodes+=20
    summary={"status":"complete","policy":"care_dino_temporal_reference","strict_local":True,
             "successes":successes,"episodes":episodes,"success_rate":successes/episodes,"tasks":results}
    tmp=args.root/"summary.tmp"; tmp.write_text(json.dumps(summary,indent=2)+"\n"); os.replace(tmp,args.root/"summary.json")
    print(json.dumps(summary),flush=True)
if __name__=="__main__": main()
