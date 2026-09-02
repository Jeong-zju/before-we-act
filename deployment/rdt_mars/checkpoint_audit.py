#!/usr/bin/env python3
import json,os,sys,tempfile
from pathlib import Path
import torch
sys.path.insert(0,"/workspace/repos/rdt-1b"); from models.rdt_runner import RDTRunner
ck=Path(os.environ["RDT_AUDIT_CHECKPOINT"]); out=Path(os.environ["RDT_AUDIT_OUTPUT"]); dev=torch.device("cuda:0"); m=RDTRunner.from_pretrained(str(ck)).to(dev,dtype=torch.bfloat16).eval(); total=sum(p.numel() for p in m.parameters()); train=sum(p.numel() for p in m.parameters() if p.requires_grad)
if total!=train or total<900_000_000: raise RuntimeError(f"not full RDT: {train}/{total}")
with torch.inference_mode(): loss=m(lang_tokens=torch.zeros(1,4,4096,dtype=torch.bfloat16,device=dev),lang_attn_mask=torch.ones(1,4,dtype=torch.bool,device=dev),img_tokens=torch.zeros(1,4374,1152,dtype=torch.bfloat16,device=dev),state_tokens=torch.zeros(1,1,128,dtype=torch.bfloat16,device=dev),action_gt=torch.zeros(1,64,128,dtype=torch.bfloat16,device=dev),action_mask=torch.ones(1,1,128,dtype=torch.bfloat16,device=dev),ctrl_freqs=torch.full((1,),20,dtype=torch.long,device=dev))
if not torch.isfinite(loss): raise RuntimeError("nonfinite reload loss")
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"schema":"mars-control.rdt.checkpoint-audit.v1","status":"complete","checkpoint":str(ck),"parameters":total,"trainable_parameters":train,"forward_loss":float(loss)},indent=2)+"\n")
