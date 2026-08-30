import hashlib,json,os
from pathlib import Path
from .common import FROZEN_CONFIG, atomic_json, sha256
def main():
    run=Path(os.environ["MARS_GAUDP_RUN_ROOT"]); ck=run/"formal"/"last.pt"; val=json.loads((run/"validation20"/"summary.json").read_text()); parity=json.loads((run/"cache_parity.json").read_text()); comparison=json.loads((run/"inference_comparison.json").read_text())
    if val.get("status")!="complete" or val.get("total_episodes")!=80: raise RuntimeError("Validation20 incomplete")
    if not parity.get("passed") or not comparison.get("gate_passed"): raise RuntimeError("pre-validation gate incomplete")
    atomic_json(run/"final_report.json",{"schema":"mars-control.gaudp.final-report.v3","status":"complete","baseline":"GauDP","benchmark":"MARS-Control","frozen_config":str(FROZEN_CONFIG),"frozen_config_sha256":sha256(FROZEN_CONFIG),"checkpoint":str(ck),"checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),"cache_parity":parity,"inference_comparison":comparison,"selected_inference_steps":comparison["selected_inference_steps"],"temporal_ensemble_decay":val["temporal_ensemble_decay"],"validation20":val,"dataset":str(Path(os.environ["MARS_GAUDP_DATA_ROOT"])),"episodes":600,"local_streams":1650,"policy_contract":"shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8","gaudp_adaptation":"FP32 NoPoSplat self-coordinate local single-view cache with verified online parity and shared decentralized weights"})
if __name__=="__main__": main()
