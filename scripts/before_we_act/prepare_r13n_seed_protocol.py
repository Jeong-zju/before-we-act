#!/usr/bin/env python3
"""Freeze disjoint Discovery/Validation/Formal seeds for six R13N tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

from before_we_act.r13n import TASKS


STAGES=("discovery","validation","formal")


def atomic_json(path: Path,payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--base-seed",type=int,default=20260808); args=parser.parse_args()
    used=set(); files={}
    for stage in STAGES:
        files[stage]={}
        for task in TASKS:
            digest=hashlib.sha256(f"R13N|{args.base_seed}|{stage}|{task}".encode()).digest(); rng=random.Random(int.from_bytes(digest[:8],"big")); seeds=[]
            while len(seeds)<20:
                value=rng.randrange(1,2**31-1)
                if value not in used: used.add(value); seeds.append(value)
            path=args.output_root/stage/f"{task}.json"; payload={"schema_version":1,"round":"R13N","protocol":"independent_six_task_discovery_validation_formal_v1","stage":stage,"task":task,"base_seed":args.base_seed,"seeds":seeds,"created_at_epoch":time.time()}; atomic_json(path,payload); files[stage][task]=str(path.resolve())
    receipt={"schema_version":1,"round":"R13N","stages":list(STAGES),"tasks":list(TASKS),"episodes_per_task":20,"total_unique_seeds":len(used),"all_seeds_disjoint":len(used)==len(STAGES)*len(TASKS)*20,"files":files,"created_at_epoch":time.time()}
    atomic_json(args.output_root/"protocol.json",receipt); print(json.dumps(receipt,sort_keys=True))


if __name__=="__main__": main()
