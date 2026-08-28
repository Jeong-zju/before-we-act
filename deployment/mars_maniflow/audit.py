#!/usr/bin/env python3
import argparse, json
from .dataset import index_corpus
from .common import atomic_json, POLICY_CONTRACT
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--output',required=True); p.add_argument('--stats',required=True); a=p.parse_args(); _,s=index_corpus(a.data_root,a.stats); r={'schema':'mars-control.maniflow.audit.v1','status':'complete','episodes':s['episodes'],'local_streams':s['local_streams'],'indexed_local_timesteps':s['indexed_local_timesteps'],'all_data_no_split':True,'policy_contract':POLICY_CONTRACT,'forbidden_inputs':['peer_rgb','peer_qpos','global_rgb','joint_action','task_id','arm_id'],'normalization':s}; atomic_json(a.output,r); print(json.dumps(r))
if __name__=='__main__': main()
