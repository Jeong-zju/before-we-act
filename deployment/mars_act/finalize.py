#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
root=Path('/workspace/repos/before-we-act'); config=root/'configs/act/mars_control_full_data_v1.json'; run=Path('/workspace/runs/mars_act'); ck=run/'formal/final.pt'; val=json.loads((run/'validation20/summary.json').read_text()); audit=json.loads((run/'audit.json').read_text()); h=hashlib.sha256()
with ck.open('rb') as f:
 for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
ch=hashlib.sha256(config.read_bytes()).hexdigest()
report={'schema':'mars-control.act.final.v2','status':'complete','training_updates':120000,'training_episodes':audit['episodes'],'local_streams':audit['local_streams'],'training_config':str(config),'training_config_sha256':ch,'checkpoint':str(ck),'checkpoint_sha256':h.hexdigest(),'validation20':val,'policy_contract':'one_shared_policy_per_arm_local_rgb_qpos_to_local_action8'}
(run/'final_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
