import json, os
from pathlib import Path
from .dataset import MarsGauDPDataset
from .common import atomic_json
def main():
    data=Path(os.environ["MARS_GAUDP_DATA_ROOT"]); cache=Path(os.environ["MARS_GAUDP_CACHE_ROOT"]); run=Path(os.environ["MARS_GAUDP_RUN_ROOT"]); ds=MarsGauDPDataset(data,cache,run/"audit_normalization.json")
    report={"schema":"mars-control.gaudp.audit.v1","status":"complete","episodes":ds.stats["episodes"],"local_streams":ds.stats["local_streams"],"indexed_local_timesteps":ds.stats["indexed_local_timesteps"],"all_data_no_split":True,"policy_contract":"shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8","forbidden_inputs":["peer_rgb","peer_qpos","global_rgb","joint_action","task_id","arm_id"]}; atomic_json(run/"audit.json",report); print(json.dumps(report))
if __name__=="__main__": main()
