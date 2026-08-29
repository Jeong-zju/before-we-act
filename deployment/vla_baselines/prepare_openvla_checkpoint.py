#!/usr/bin/env python3
"""Materialize a complete OpenVLA inference checkpoint from the LoRA adapter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", default="/workspace/models/openvla-7b")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    adapter = checkpoint / "lora_adapter"
    if (checkpoint / "config.json").is_file() and any(checkpoint.glob("model*.safetensors")):
        print("OpenVLA merged checkpoint already present", flush=True)
        return
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LoRA adapter missing: {adapter}")
    os.environ.setdefault("OPENVLA_ROBOFACTORY_ROOT", "/workspace/datasets/robofactory_multitask")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq

    print(f"Loading base model for offline LoRA merge: {args.base}", flush=True)
    base = AutoModelForVision2Seq.from_pretrained(
        args.base, torch_dtype="auto", low_cpu_mem_usage=True, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, adapter)
    merged = merged.merge_and_unload()
    checkpoint.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(checkpoint, safe_serialization=True)
    print(f"Saved complete merged checkpoint at {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
