#!/usr/bin/env python3
import os, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0,"/workspace/repos/rdt-1b")
from models.multimodal_encoder.t5_encoder import T5Embedder
TASK_TEXT={"place_cube_in_cup":"Place the cube in the cup","strike_cube_hard":"Strike the cube hard","three_robots_place_shoes":"Three robots place shoes","four_robots_stack_cube":"Four robots stack the cube"}
root=Path(os.environ.get("RDT_MARS_DATASET","/workspace/datasets/mars_control")); cfg=yaml.safe_load(open("configs/base.yaml")); e=T5Embedder(from_pretrained="google/t5-v1_1-xxl",model_max_length=cfg["dataset"]["tokenizer_max_length"],device=torch.device("cuda:0"))
for task,text in TASK_TEXT.items():
 out=root/task; out.mkdir(parents=True,exist_ok=True); emb,_=e.get_text_embeddings([text]); torch.save(emb[0].cpu(),out/"lang_embed.pt"); print(out/"lang_embed.pt",flush=True)
torch.save(torch.zeros((1,e.model.config.d_model),dtype=torch.bfloat16),Path("data/empty_lang_embed.pt"))
