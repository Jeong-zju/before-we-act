from __future__ import annotations
import ast, json, hashlib
from pathlib import Path

ROOT=Path(__file__).parent
FILES=(ROOT/"local_dataset.py",ROOT/"local_policy.py",ROOT/"evaluate_closed_loop.py")

def main():
    texts={p.name:p.read_text() for p in FILES}
    for p in FILES: ast.parse(texts[p.name],filename=str(p))
    dataset=texts["local_dataset.py"]
    required=("data/observation/images/agent_{agent}","data/observation/agents/panda_{agent}/qpos","data/action/agents/panda_{agent}/commanded")
    if any(x not in dataset for x in required): raise RuntimeError("local dataset field whitelist drift")
    forbidden=("head_camera_global","peer_qpos","peer_image","joint_action","planner_state","privileged")
    hits={name:[x for x in forbidden if x in text.lower()] for name,text in texts.items()}
    if any(hits.values()): raise RuntimeError(f"forbidden local policy tokens: {hits}")
    policy=texts["local_policy.py"]
    if "LocalLatentToMPolicy" not in policy or "LocalLatentEncoder" not in policy: raise RuntimeError("latent policy structure absent")
    output=Path("/workspace/bwa_latent_tom_runs/audit/policy_contract.json"); output.parent.mkdir(parents=True,exist_ok=True)
    payload={"schema":"bwa.latent_tom.policy_contract.v1","status":"complete","shared_checkpoint":True,
             "inputs":["own_rgb_history","own_qpos_history","task_id"],"output":"own_commanded_action8",
             "forbidden":["global_rgb","peer_rgb","peer_qpos","peer_action","joint_action","planner_state","privileged_state"],
             "official_upstream":{"repo":"https://github.com/StanfordMSL/LatentToM","commit":"a51d929027799a53d54e7d7d2ba90e2703642b4a"},
             "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}}
    output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"complete","output":str(output)}))
if __name__=="__main__": main()
