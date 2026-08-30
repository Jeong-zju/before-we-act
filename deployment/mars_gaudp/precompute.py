from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F
from .common import TASKS, ARMS, atomic_json

CACHE_SCHEMA = "mars-control.gaudp.cache.v2"
RGB_PREPROCESSING = "uint8_div_255_then_affine_to_minus1_plus1"
ENCODER_PRECISION = "float32"
STORED_DTYPE = "float32"

def cfg_for(weight):
    from omegaconf import OmegaConf
    return OmegaConf.create({"name":"noposplat","coor_type":"self","opacity_mapping":{"initial":0.0,"final":0.0,"warm_up":1},"num_monocular_samples":32,"num_surfaces":1,"predict_opacity":False,"gaussians_per_pixel":1,"gaussian_adapter":{"gaussian_scale_min":0.5,"gaussian_scale_max":15.0,"sh_degree":4},"d_feature":128,"apply_bounds_shim":True,"gs_params_head_type":"dpt_gs","pose_free":True,"pretrained_weights":str(weight),"backbone":{"name":"croco","model":"ViTLarge_BaseDecoder","patch_embed_cls":"PatchEmbedDust3R","asymmetry_decoder":True,"intrinsics_embed_loc":"encoder","intrinsics_embed_degree":4,"intrinsics_embed_type":"token"}})

def load_encoder(weight, device):
    from model.noposplat.encoder import get_encoder
    enc=get_encoder(cfg_for(weight)); ckpt=torch.load(weight,map_location="cpu",weights_only=False); state=ckpt.get("state_dict",ckpt); state={k[8:]:v for k,v in state.items() if k.startswith("encoder.")}; enc.load_state_dict(state,strict=False); return enc.to(device).eval()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True); p.add_argument("--cache-root",required=True); p.add_argument("--weight",required=True); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--output-hw",type=int,nargs=2,default=(30,40)); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); root,cache=Path(a.data_root),Path(a.cache_root); cache.mkdir(parents=True,exist_ok=True); device=torch.device("cuda:0"); enc=load_encoder(Path(a.weight),device); allmeta={"schema":CACHE_SCHEMA,"gaussian_hw":list(a.output_hw),"rgb_preprocessing":RGB_PREPROCESSING,"encoder_precision":ENCODER_PRECISION,"stored_dtype":STORED_DTYPE,"tasks":{},"indexed_local_timesteps":0}
    for task in TASKS:
        outdir=cache/task; outdir.mkdir(parents=True,exist_ok=True); outpath=outdir/f"{task}.h5"; metap=outdir/"metadata.json"; rows=[]; total=0
        if outpath.exists() and metap.exists() and not a.smoke:
            try:
                old=json.loads(metap.read_text());
                reusable=(old.get("status")=="complete" and old.get("schema")==CACHE_SCHEMA and old.get("rgb_preprocessing")==RGB_PREPROCESSING and old.get("encoder_precision")==ENCODER_PRECISION and old.get("stored_dtype")==STORED_DTYPE and old.get("gaussian_hw")==list(a.output_hw))
                if reusable: allmeta["tasks"][task]=old["rows"]; allmeta["indexed_local_timesteps"]+=sum(int(r["length"])*ARMS[task] for r in old["rows"]); continue
            except Exception: pass
        with h5py.File(outpath,"w") as out:
            for arm in range(ARMS[task]): out.create_dataset(f"gaussian_arm{arm}",shape=(0,13,*a.output_hw),maxshape=(None,13,*a.output_hw),dtype="float32",chunks=(1,13,*a.output_hw),compression="lzf")
            shards=sorted(glob.glob(str(root/task/"motionplanning"/"*.shard*.h5")))
            for shard in shards:
                with h5py.File(shard,"r") as source:
                    for traj in sorted(source,key=lambda x:int(x.rsplit("_",1)[-1])):
                        g=source[traj]; n=min(len(g[f"actions/panda-{arm}"]) for arm in range(ARMS[task])); offsets={}
                        for arm in range(ARMS[task]):
                            offsets[str(arm)]=total; images=np.asarray(g[f"obs/sensor_data/head_camera_agent{arm}/rgb"][:n],np.uint8); pieces=[]
                            for begin in range(0,n,a.batch_size):
                                x=torch.from_numpy(images[begin:begin+a.batch_size]).permute(0,3,1,2).float().div(255.0)
                                x=F.interpolate(x,size=(256,256),mode="bilinear",align_corners=False).mul(2).sub(1).to(device)
                                with torch.inference_mode(): y=enc({"image":x[:,None]})[:,0].float()
                                y=F.interpolate(y,size=tuple(a.output_hw),mode="bilinear",align_corners=False).cpu().numpy().astype(np.float32); pieces.append(y)
                            y=np.concatenate(pieces,0); ds=out[f"gaussian_arm{arm}"]; ds.resize((total+n,13,*a.output_hw)); ds[total:total+n]=y
                        rows.append({"shard":Path(shard).name,"trajectory":traj,"length":n,"offsets":offsets}); total+=n
                        print(json.dumps({"task":task,"trajectory":traj,"frames":total}),flush=True)
                        if a.smoke and len(rows)>=1: break
                if a.smoke and len(rows)>=1: break
        atomic_json(metap,{"schema":CACHE_SCHEMA,"status":"complete","task":task,"rows":rows,"gaussian_hw":list(a.output_hw),"rgb_preprocessing":RGB_PREPROCESSING,"encoder_precision":ENCODER_PRECISION,"stored_dtype":STORED_DTYPE}); allmeta["tasks"][task]=rows; allmeta["indexed_local_timesteps"]+=sum(int(r["length"])*ARMS[task] for r in rows)
    atomic_json(cache/"metadata.json",allmeta); print(json.dumps({"status":"complete","indexed_local_timesteps":allmeta["indexed_local_timesteps"]}))
if __name__=="__main__": main()
