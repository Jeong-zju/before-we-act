import hashlib,json,os
from pathlib import Path
from .common import atomic_json
def main():
    run=Path(os.environ["MARS_GAUDP_RUN_ROOT"]); ck=run/"formal"/"last.pt"; val=json.loads((run/"validation20"/"summary.json").read_text());
    if val.get("status")!="complete" or val.get("total_episodes")!=80: raise RuntimeError("Validation20 incomplete")
    atomic_json(run/"final_report.json",{"schema":"mars-control.gaudp.final-report.v1","status":"complete","baseline":"GauDP","benchmark":"MARS-Control","checkpoint":str(ck),"checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),"validation20":val,"dataset":str(Path(os.environ["MARS_GAUDP_DATA_ROOT"])),"episodes":600,"local_streams":1650,"policy_contract":"shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8","gaudp_adaptation":"NoPoSplat self-coordinate local single-view with shared decentralized weights"})
if __name__=="__main__": main()
