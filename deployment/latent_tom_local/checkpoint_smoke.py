from __future__ import annotations
import argparse, json, torch
from local_policy import LocalLatentToMPolicy

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=LocalLatentToMPolicy().cuda(); model.load_state_dict(ck.get("ema_model",ck["model"])); model.eval()
    obs={"image":torch.zeros(1,2,3,240,320,device="cuda"),"qpos":torch.zeros(1,2,9,device="cuda"),"task":torch.eye(6,device="cuda")[:1]}
    with torch.no_grad(): action=model.predict_chunk(obs,steps=2)
    if action.shape!=(1,40,8) or not torch.isfinite(action).all(): raise RuntimeError("invalid reload output")
    open(a.output,"w").write(json.dumps({"status":"complete","shape":list(action.shape),"contract":ck["contract"],"weights":"ema" if "ema_model" in ck else "raw"},indent=2)+"\n")
if __name__=="__main__": main()
