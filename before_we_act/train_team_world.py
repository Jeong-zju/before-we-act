from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from before_we_act.data.world_windows import CachedWorldWindows, legal_model_inputs
from before_we_act.world_model.base import (
    CandidateConditionedWorldModel,
    load_r13_config,
    world_losses,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def device_batch(data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device):
    return {
        key: value.index_select(0, indices).to(device, non_blocking=True)
        for key, value in data.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--heartbeat")
    args = parser.parse_args()
    config = load_r13_config(args.config)
    target_updates = args.updates or int(config.training["updates"])
    if target_updates not in (2, int(config.training["updates"])):
        raise ValueError("R13 supports only two-update preflight or the frozen full budget")
    seed = int(config.training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal R13 training requires the assigned CUDA GPU")
    dataset = CachedWorldWindows(args.cache, "train")
    data = dict(dataset.data)
    model = CandidateConditionedWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    start_update = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("round") != "R13" or checkpoint.get("candidate_id") != config.candidate_id:
            raise ValueError("R13 resume identity differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["update"])
        generator.set_state(checkpoint["sample_generator_state"])
    output = Path(args.output).resolve()
    progress_path = output / "progress.jsonl"
    heartbeat = Path(args.heartbeat).resolve() if args.heartbeat else None
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    last: dict = {}

    def save(update: int, name: str, *, operator_stop: bool = False) -> Path:
        path = output / "checkpoints" / name
        atomic_torch_save(
            path,
            {
                "schema_version": 1,
                "round": "R13",
                "candidate_id": config.candidate_id,
                "config": dict(config.raw),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "sample_generator_state": generator.get_state(),
                "train_progress_mean": float(data["future_progress"].float().mean()),
                "last_metrics": last,
                "operator_stop": operator_stop,
                "future_targets_are_model_inputs": False,
            },
        )
        return path

    model.train()
    update = start_update
    for update in range(start_update + 1, target_updates + 1):
        indices = torch.randint(
            len(dataset),
            (int(config.training["batch_size"]),),
            generator=generator,
        )
        batch = device_batch(data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(**legal_model_inputs(batch))
            losses = world_losses(prediction, batch, config.raw["loss_weights"])
            loss = losses["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"R13 loss became non-finite at update {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.training["grad_clip"])
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError(f"R13 gradient became non-finite at update {update}")
        optimizer.step()
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = (update - start_update) / elapsed
        last = {
            "updated_at": now(),
            "candidate_id": config.candidate_id,
            "update": update,
            "target_updates": target_updates,
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "updates_per_second": rate,
            "eta_hours": (target_updates - update) / max(rate, 1e-9) / 3600,
            "gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            **{
                key: float(value.detach())
                for key, value in losses.items()
                if key != "loss"
            },
        }
        if update % int(config.training["progress_every"]) == 0 or update == target_updates:
            append_jsonl(progress_path, last)
            print(json.dumps(last, sort_keys=True), flush=True)
        if heartbeat and (
            update % int(config.training["progress_every"]) == 0 or update == target_updates
        ):
            atomic_json(
                heartbeat,
                {
                    "producer": "train_team_world",
                    "candidate": config.candidate_id,
                    "pid": os.getpid(),
                    "stage": "formal" if target_updates > 2 else "preflight",
                    "update": update,
                    "updated_at": now(),
                },
            )
        if (
            update % int(config.training["checkpoint_every"]) == 0
            or update == target_updates
            or stopping
        ):
            latest = save(update, "checkpoint_latest.pt", operator_stop=stopping)
            if update == target_updates:
                save(update, f"checkpoint_{update:06d}.pt")
            print(json.dumps({"saved": str(latest), "update": update}), flush=True)
        if stopping:
            raise SystemExit(130)
    print(json.dumps({"complete": True, "candidate": config.candidate_id, "update": update}))


if __name__ == "__main__":
    main()
