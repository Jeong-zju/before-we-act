#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch


if "IPython" not in sys.modules:
    stub = types.ModuleType("IPython")
    stub.embed = lambda *args, **kwargs: None
    sys.modules["IPython"] = stub

from before_we_act.upstream_components.act.detr.models.transformer import Transformer as LocalTransformer


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("r12_act_official_transformer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    official_file = Path(args.upstream) / "detr/models/transformer.py"
    local_file = Path(__file__).parents[4] / "before_we_act/upstream_components/act/detr/models/transformer.py"
    OfficialTransformer = load_official(official_file)
    kwargs = dict(d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                  dim_feedforward=64, dropout=0.0, return_intermediate_dec=True)
    torch.manual_seed(17)
    official = OfficialTransformer(**kwargs).to(args.device).eval()
    local = LocalTransformer(**kwargs).to(args.device).eval()
    local.load_state_dict(official.state_dict(), strict=True)
    generator = torch.Generator(device=args.device).manual_seed(19)
    source = torch.randn((2, 5, 32), generator=generator, device=args.device)
    query = torch.randn((7, 32), generator=generator, device=args.device)
    position = torch.randn((5, 32), generator=generator, device=args.device)
    padding = torch.tensor([[False, False, False, False, True], [False] * 5], device=args.device)
    with torch.no_grad():
        expected = official(source, padding, query, position)
        actual = local(source, padding, query, position)
    maximum = float((expected - actual).abs().max())
    checks = {"vendored_source_exact": official_file.read_bytes() == local_file.read_bytes(),
              "transformer_forward_exact": maximum == 0.0}
    result = {"schema_version": 1, "round": "R12", "candidate_id": "p2",
              "upstream_commit": "742c753c0d4a5d87076c8f69e5628c79a8cc5488",
              "checks": checks, "max_abs": maximum, "passed": all(checks.values())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
