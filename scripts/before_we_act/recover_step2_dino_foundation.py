#!/usr/bin/env python3
"""Recover the frozen DINO foundation asset from its lossless W10 carrier.

The original local Hugging Face directory was removed after W10.  W10 kept the
foundation frozen, so its checkpoint is a lossless carrier.  This utility has a
strict ``vision.*`` allowlist and never loads action-policy, router, projection,
optimizer, scheduler, cursor, or RNG tensors into the recovered asset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.image_processing_dinov3_vit import (
    DINOv3ViTImageProcessor,
)
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel


W10_SHA256 = "e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574"
VISION_TENSORS = 211
VISION_PARAMETERS = 85_660_416


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.carrier) != W10_SHA256:
        raise ValueError("frozen DINO carrier checkpoint hash differs")
    if args.output.exists() and any(args.output.iterdir()):
        receipt = args.output / "foundation_receipt.json"
        if receipt.is_file() and json.loads(receipt.read_text()).get("status") == "PASSED":
            print("STEP2_DINO_FOUNDATION_ALREADY_RECOVERED")
            return
        raise FileExistsError(f"foundation output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.carrier, map_location="cpu", weights_only=False)
    source_state = saved.get("model", {})
    vision = {
        key.removeprefix("vision."): value
        for key, value in source_state.items()
        if key.startswith("vision.")
    }
    count = sum(value.numel() for value in vision.values())
    if len(vision) != VISION_TENSORS or count != VISION_PARAMETERS:
        raise ValueError(
            f"frozen DINO carrier structure differs: tensors={len(vision)}, params={count}"
        )
    config = DINOv3ViTConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        patch_size=16,
        num_register_tokens=4,
        image_size=224,
        key_bias=False,
        query_bias=True,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        use_gated_mlp=False,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        layerscale_value=1.0,
        rope_theta=100.0,
        pos_embed_rescale=2.0,
        apply_layernorm=True,
    )
    model = DINOv3ViTModel(config)
    missing, unexpected = model.load_state_dict(vision, strict=True)
    if missing or unexpected:
        raise AssertionError(f"strict DINO recovery failed: {missing}, {unexpected}")
    model.save_pretrained(args.output, safe_serialization=True)
    processor = DINOv3ViTImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    )
    processor.save_pretrained(args.output)
    model_file = args.output / "model.safetensors"
    config_file = args.output / "config.json"
    processor_file = args.output / "preprocessor_config.json"
    for path in (model_file, config_file, processor_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    receipt = {
        "format_version": "before-we-act.step2.dino_foundation_recovery/1",
        "status": "PASSED",
        "source_carrier": str(args.carrier.resolve()),
        "source_carrier_sha256": W10_SHA256,
        "source_carrier_role": "lossless carrier for the frozen foundation only",
        "loaded_key_allowlist": "vision.*",
        "loaded_tensor_count": len(vision),
        "loaded_parameter_count": count,
        "non_vision_policy_tensors_loaded": False,
        "vision_projection_loaded": False,
        "optimizer_scheduler_rng_cursor_loaded": False,
        "candidate_initialization_from_w10_policy": False,
        "model_file": str(model_file),
        "model_sha256": sha256_file(model_file),
        "config_sha256": sha256_file(config_file),
        "processor_sha256": sha256_file(processor_file),
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    atomic_json(args.output / "foundation_receipt.json", receipt)
    print("STEP2_DINO_FOUNDATION_RECOVERED")


if __name__ == "__main__":
    main()
