from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from models.rdt_runner import RDTRunner

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    model=RDTRunner.from_pretrained(str(a.checkpoint)).to("cuda:0",dtype=torch.bfloat16).eval(); total=sum(x.numel() for x in model.parameters()); train=sum(x.numel() for x in model.parameters() if x.requires_grad)
    if total != train or total < 900_000_000: raise RuntimeError(f"not full-parameter RDT-1B: {train}/{total}")
    with torch.inference_mode(): loss=model(lang_tokens=torch.zeros(1,4,4096,dtype=torch.bfloat16,device="cuda"),lang_attn_mask=torch.ones(1,4,dtype=torch.bool,device="cuda"),img_tokens=torch.zeros(1,4374,1152,dtype=torch.bfloat16,device="cuda"),state_tokens=torch.zeros(1,1,128,dtype=torch.bfloat16,device="cuda"),action_gt=torch.zeros(1,64,128,dtype=torch.bfloat16,device="cuda"),action_mask=torch.ones(1,1,128,dtype=torch.bfloat16,device="cuda"),ctrl_freqs=torch.full((1,),30,dtype=torch.long,device="cuda"))
    if not torch.isfinite(loss): raise RuntimeError("checkpoint reload loss is non-finite")
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({"schema":"duobench.rdt.checkpoint-audit.v1","status":"complete","checkpoint":str(a.checkpoint),"parameters":total,"trainable_parameters":train,"forward_loss":float(loss)},indent=2)+"\n")
if __name__ == "__main__": main()
