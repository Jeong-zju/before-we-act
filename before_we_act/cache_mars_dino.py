"""Build the frozen local-view DINOv3 cache for MARS-Control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoImageProcessor, AutoModel

from before_we_act.mars_temporal_data import MarsVisualCache, load_mars_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0")); device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1: dist.init_process_group("nccl")
    episodes = load_mars_episodes(args.raw_root)
    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    model = AutoModel.from_pretrained(args.dino_model).to(device).eval().requires_grad_(False)
    mean = torch.tensor(processor.image_mean, device=device).view(1,3,1,1)
    std = torch.tensor(processor.image_std, device=device).view(1,3,1,1)
    cache = MarsVisualCache(args.output); completed = 0; started = time.time()
    for episode in episodes[rank::world]:
        path = cache.path_for(episode)
        if path.is_file(): completed += 1; continue
        arrays = {}
        with h5py.File(episode.path, "r") as handle:
            group = handle[episode.trajectory]
            for arm in episode.arms:
                source = group[f"obs/sensor_data/head_camera_agent{arm}/rgb"]
                chunks = []
                for first in range(0, episode.length, args.batch_size):
                    raw = np.asarray(source[first:min(first+args.batch_size, episode.length)])
                    image = torch.from_numpy(raw).to(device).permute(0,3,1,2).float().div_(255)
                    image = (image-mean)/std
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        tokens = model(pixel_values=image).last_hidden_state
                        first_patch = 1 + int(getattr(model.config, "num_register_tokens", 0))
                        pooled = tokens[:, first_patch:].mean(1)
                    chunks.append(pooled.float().cpu().numpy().astype(np.float16))
                arrays[f"agent_{arm}"] = np.concatenate(chunks)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".rank{rank}.tmp.npz")
        np.savez_compressed(temporary, **arrays); os.replace(temporary, path)
        completed += 1
        if completed == 1 or completed % 10 == 0:
            print(json.dumps({"rank":rank,"episodes":completed,"elapsed":time.time()-started}), flush=True)
    if world > 1: dist.barrier()
    if rank == 0:
        files = list(args.output.glob("*/*.npz"))
        receipt = {"status":"PASSED" if len(files)==600 else "INCOMPLETE", "episodes":len(files),
                   "tasks":4, "views":"strict local duplicated at CARE dual-view boundary",
                   "image_height":240, "image_width":320, "feature_dim":768}
        (args.output/"cache_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt), flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
