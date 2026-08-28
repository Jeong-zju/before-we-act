from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from .common import atomic_json
def main():
    root=Path(os.environ.get("MARS_DP_RUN_ROOT","/workspace/runs/mars_dp_v2")); train=json.loads((root/"formal/status.json").read_text()); val=json.loads((root/"validation20/summary.json").read_text()); ck=Path(train["checkpoint"]); report={"schema":"mars-control.dp.final-report.v3","status":"complete","baseline":"diffusion_policy","benchmark":"MARS-Control","training":train,"validation20":val,"checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),"policy_contract":"shared_weights_decentralized_local_rgb_own_command_state_to_absolute_action8","state_contract":"official own commanded action8 feedback","temporal_contract":"official_obs3_horizon8_action8_execute6","action_targets_clipped_before_normalization":True}; atomic_json(root/"final_report.json",report); print(json.dumps({"status":"complete","successes":val["successes"],"episodes":val["total_episodes"]}))
if __name__=="__main__": main()
