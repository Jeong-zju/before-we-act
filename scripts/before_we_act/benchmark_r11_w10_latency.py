#!/usr/bin/env python3
"""Frozen W10 action-call latency reference for the R11 score formula."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from before_we_act.r11_data import R11EpisodeDataset, R11SampleRequest, SIX_TASKS, load_r11_episodes
from before_we_act.train_r11_candidate import atomic_json, sha256_file
from stereo_core.evaluate_no_wrist_pair import load_model


WARMUP_PER_TASK = 10
REPEATS_PER_TASK = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("W10 latency checkpoint SHA256 differs")
    output = args.output.resolve()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        expected = {
            "status": "PASSED",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "warmup_per_task": WARMUP_PER_TASK,
            "repeats_per_task": REPEATS_PER_TASK,
            "tasks": list(SIX_TASKS),
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("existing immutable W10 latency receipt identity differs")
        if output.stat().st_mode & 0o777 != 0o444:
            raise ValueError("existing W10 latency receipt must be mode 0444")
        print(json.dumps(existing | {"rows": "saved", "reused": True}, sort_keys=True))
        return
    device = torch.device(args.device)
    model, stats, _ = load_model(str(checkpoint), device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    episodes = load_r11_episodes(args.manifests, split="validation")
    dataset = R11EpisodeDataset(episodes, saved["stats"])
    rows = []
    torch.set_grad_enabled(False)
    for task in SIX_TASKS:
        episode_index = next(index for index, episode in enumerate(episodes) if episode.task == task)
        episode = episodes[episode_index]
        samples = default_collate(
            [
                dataset[
                    R11SampleRequest(
                        episode_index=episode_index,
                        arm=arm,
                        time_index=0,
                        sample_key=f"latency:{task}:{arm}",
                        task=task,
                    )
                ]
                for arm in episode.arms
            ]
        )
        global_rgb = samples["current_rgb"][:, 0].float().div(255).to(device)
        local_rgb = samples["current_rgb"][:, 1].float().div(255).to(device)
        qpos = samples["qpos"].to(device)
        for index in range(WARMUP_PER_TASK + REPEATS_PER_TASK):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action = model(global_rgb, local_rgb, qpos)[0]
            torch.cuda.synchronize()
            if action.shape != (len(episode.arms), 100, 8) or not torch.isfinite(action).all():
                raise ValueError("W10 latency action contract drift")
            elapsed = (time.perf_counter() - started) * 1000
            if index >= WARMUP_PER_TASK:
                rows.append({"task": task, "batch": len(episode.arms), "latency_ms": elapsed})
    latencies = [row["latency_ms"] for row in rows]
    result = {
        "format_version": "before-we-act.r11.w10_latency/1",
        "status": "PASSED",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "warmup_per_task": WARMUP_PER_TASK,
        "repeats_per_task": REPEATS_PER_TASK,
        "tasks": list(SIX_TASKS),
        "calls": len(rows),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "rows": rows,
        "completed_at_epoch": time.time(),
    }
    atomic_json(output, result)
    output.chmod(0o444)
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True))


if __name__ == "__main__":
    main()
