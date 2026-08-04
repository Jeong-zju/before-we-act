#!/usr/bin/env python3
"""R9 real-checkpoint strict-load and forward/RNG exact audit."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_class():
    upstream = ROOT / "vendor/stereo-core/stereo_core"
    sys.path.insert(0, str(upstream))
    try:
        spec = importlib.util.spec_from_file_location(
            "r9_checkpoint_reference", upstream / "no_wrist_pair_model.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load upstream no_wrist_pair_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.NoWristPAIRRoute
    finally:
        sys.path.remove(str(upstream))


def build(model_class, config, dino_model, device, state):
    model = model_class(
        config.get("state_dim", 9),
        config.get("action_dim", 8),
        horizon=config.get("horizon", 100),
        d_model=config.get("d_model", 384),
        enc_layers=config.get("enc_layers", 4),
        dec_layers=config.get("dec_layers", 7),
        roles=config.get("roles", 4),
        role_rank=config.get("role_rank", 32),
        dino_model=dino_model,
    ).to(device)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    return model


def flatten_output(output):
    return [value.detach().cpu() if isinstance(value, torch.Tensor) else value for value in output]


def compare_outputs(expected, actual):
    rows = []
    passed = len(expected) == len(actual)
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left is None or right is None:
            exact = left is right
            maximum = None
        else:
            exact = torch.equal(left, right)
            maximum = float((left.float() - right.float()).abs().max()) if left.numel() else 0.0
        rows.append({"index": index, "exact": exact, "max_abs": maximum})
        passed = passed and exact
    return passed, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from stereo_core.no_wrist_pair_model import NoWristPAIRRoute

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = saved["model"]
    config = saved["config"]
    state_hash = tensor_digest(state)
    generator = torch.Generator().manual_seed(9009)
    global_rgb = torch.rand(1, 3, 480, 640, generator=generator).to(device)
    local_rgb = torch.rand(1, 3, 480, 640, generator=generator).to(device)
    qpos = torch.rand(1, config.get("state_dim", 9), generator=generator).to(device)
    actions = torch.rand(
        1,
        config.get("horizon", 100),
        config.get("action_dim", 8),
        generator=generator,
    ).to(device)
    inputs = (global_rgb, local_rgb, qpos)
    reference_class = load_reference_class()
    reference = build(reference_class, config, args.dino_model, device, state)
    reference_hash = tensor_digest(reference.state_dict())
    reference.eval()
    with torch.no_grad():
        expected_eval = flatten_output(reference(*inputs, return_routing=True))
    reference.train()
    cpu_rng = torch.Generator().manual_seed(7211).get_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        expected_train = flatten_output(
            reference(
                *inputs,
                actions,
                return_routing=True,
                counterfactual=True,
            )
        )
    expected_cpu_after = torch.get_rng_state().clone()
    expected_cuda_after = torch.cuda.get_rng_state(device).clone() if cuda_rng is not None else None
    del reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    native = build(NoWristPAIRRoute, config, args.dino_model, device, state)
    native_hash = tensor_digest(native.state_dict())
    native.eval()
    with torch.no_grad():
        actual_eval = flatten_output(native(*inputs, return_routing=True))
        bank = native.propose_core_bank(*inputs)
    native.train()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        actual_train = flatten_output(
            native(
                *inputs,
                actions,
                return_routing=True,
                counterfactual=True,
            )
        )
    actual_cpu_after = torch.get_rng_state().clone()
    actual_cuda_after = torch.cuda.get_rng_state(device).clone() if cuda_rng is not None else None
    eval_passed, eval_rows = compare_outputs(expected_eval, actual_eval)
    train_passed, train_rows = compare_outputs(expected_train, actual_train)
    rng_passed = torch.equal(expected_cpu_after, actual_cpu_after) and (
        expected_cuda_after is None or torch.equal(expected_cuda_after, actual_cuda_after)
    )
    bank_base_exact = torch.equal(bank.chunks[:, 0].cpu(), actual_eval[0])
    result = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_file_sha256": file_digest(Path(args.checkpoint)),
        "state_dict_tensor_sha256": state_hash,
        "reference_state_dict_tensor_sha256": reference_hash,
        "native_state_dict_tensor_sha256": native_hash,
        "state_dict_exact": state_hash == reference_hash == native_hash,
        "eval_forward": {"passed": eval_passed, "outputs": eval_rows},
        "train_forward": {"passed": train_passed, "outputs": train_rows},
        "rng_after_exact": rng_passed,
        "candidate_bank": {
            "shape": list(bank.chunks.shape),
            "routes_shape": list(bank.routes.shape),
            "valid_fraction": float(bank.valid_mask.float().mean()),
            "candidate_zero_exact": bank_base_exact,
            "finite": bool(torch.isfinite(bank.chunks).all()),
        },
    }
    result["passed"] = all(
        (
            result["state_dict_exact"],
            eval_passed,
            train_passed,
            rng_passed,
            bank_base_exact,
            result["candidate_bank"]["finite"],
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
