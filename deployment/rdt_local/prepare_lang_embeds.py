#!/usr/bin/env python3
"""Precompute one frozen T5 embedding per RoboFactory task."""
import argparse
from pathlib import Path
import sys

import h5py
import torch
import yaml

sys.path.insert(0, "/workspace/repos/rdt-1b")
from models.multimodal_encoder.t5_encoder import T5Embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/workspace/datasets/robofactory_multitask")
    parser.add_argument("--model", default="google/t5-v1_1-xxl")
    args = parser.parse_args()
    with open("configs/base.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    embedder = T5Embedder(from_pretrained=args.model,
                          model_max_length=config["dataset"]["tokenizer_max_length"],
                          device=torch.device("cuda:0"))
    for task_root in sorted(Path(args.dataset_root).iterdir()):
        episode = task_root / "hdf5/episode_000000.hdf5"
        if not episode.is_file():
            continue
        with h5py.File(episode, "r") as handle:
            value = handle["data/task/text"][0]
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        embeddings, _ = embedder.get_text_embeddings([text])
        torch.save(embeddings[0].cpu(), task_root / "lang_embed.pt")
        print(f"saved {task_root / 'lang_embed.pt'}", flush=True)
    empty = Path("data/empty_lang_embed.pt")
    empty.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros((1, embedder.model.config.d_model), dtype=torch.bfloat16), empty)


if __name__ == "__main__":
    main()
