#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, args.upstream)
    from src.models.predictor import VisionTransformerPredictor as Official
    from before_we_act.upstream_components.vjepa2.src.models.predictor import VisionTransformerPredictor as Local
    kwargs = dict(img_size=(4, 4), patch_size=1, num_frames=4, tubelet_size=1, embed_dim=96,
                  predictor_embed_dim=96, out_embed_dim=96, depth=2, num_heads=4,
                  mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                  use_mask_tokens=True, num_mask_tokens=2, zero_init_mask_tokens=True)
    torch.manual_seed(1100)
    official = Official(**kwargs).eval()
    local = Local(**kwargs).eval()
    local.load_state_dict(official.state_dict(), strict=True)
    device = torch.device(args.device)
    official, local = official.to(device), local.to(device)
    generator = torch.Generator().manual_seed(1101)
    x = torch.randn(2, 48, 96, generator=generator).to(device)
    mx = torch.arange(48, device=device).unsqueeze(0).expand(2, -1)
    my = torch.arange(48, 64, device=device).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        left, right = official(x, mx, my), local(x, mx, my)
    result = {"schema_version": 1, "round": "R11", "candidate_id": "p0",
              "symbol": "VisionTransformerPredictor", "shape": list(left.shape),
              "exact": bool(torch.equal(left, right)),
              "max_abs": float((left - right).abs().max())}
    result["passed"] = result["exact"] and result["shape"] == [2, 16, 96]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
