#!/usr/bin/env python3
"""Materialize a complete OpenVLA inference checkpoint from the LoRA adapter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", default="/workspace/models/openvla-7b")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    adapter = checkpoint / "lora_adapter"
    receipt = checkpoint / "merge_complete.json"
    if receipt.is_file() and (checkpoint / "config.json").is_file() and any(checkpoint.glob("model*.safetensors")):
        print("OpenVLA merged checkpoint already present", flush=True)
        return
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LoRA adapter missing: {adapter}")
    os.environ.setdefault("OPENVLA_ROBOFACTORY_ROOT", "/workspace/datasets/robofactory_multitask")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from peft import PeftModel
    import torch
    from transformers import AutoModelForVision2Seq

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    print(f"Loading base model for offline LoRA merge on {device}: {args.base}", flush=True)
    base = AutoModelForVision2Seq.from_pretrained(
        args.base, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
        device_map={"": device},
    )
    merged = PeftModel.from_pretrained(base, adapter)
    merged = merged.merge_and_unload()
    checkpoint.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(checkpoint, safe_serialization=True)
    receipt.write_text(json.dumps({"status": "complete", "device": device}) + "\n")
    print(f"Saved complete merged checkpoint at {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
