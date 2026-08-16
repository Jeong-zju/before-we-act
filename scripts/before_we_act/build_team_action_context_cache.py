#!/usr/bin/env python3
"""Cache frozen B0-H decoded action contexts once for all 3-N2 seeds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist

from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from before_we_act.raw_team_signal_data import load_team_signal_metadata
from before_we_act.action_grounded_belief import _constructor_config
from before_we_act.temporal_history_data import (
    ACTION_HORIZON,
    HISTORY_STEPS,
    SIX_TASKS,
    TASK_TEXT,
    TeamTemporalDataset,
    load_temporal_episodes,
    task_text_tensor,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--signal-cache", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_stats(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("stats", payload)
    return {
        key: np.asarray(source[key], dtype=np.float32)
        for key in ("q_mean", "q_std", "a_mean", "a_std")
    }


def allocate(output: Path, n1_episodes) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for task in SIX_TASKS:
        rows = sum(row.length * 2 for row in n1_episodes if row.task == task)
        np.lib.format.open_memmap(
            output / f"{task}_decoded.npy",
            mode="w+",
            dtype=np.float16,
            shape=(rows, ACTION_HORIZON, 384),
        )
        np.lib.format.open_memmap(
            output / f"{task}_base_action.npy",
            mode="w+",
            dtype=np.float16,
            shape=(rows, ACTION_HORIZON, 8),
        )


def episode_tensors(episode, dataset, handle, rows, device):
    cache = dataset.visual_cache.load(episode)
    data = handle["data"]
    images = data["observation"]["images"]
    qpos_sources = {
        arm: np.asarray(
            data["observation"]["agents"][f"panda_{arm}"]["qpos"],
            dtype=np.float32,
        )
        for arm in episode.arms
    }
    action_sources = {
        arm: np.asarray(
            data["action"]["agents"][f"panda_{arm}"]["commanded"],
            dtype=np.float32,
        )
        for arm in episode.arms
    }
    count = len(rows)
    history_visual = torch.zeros(count, HISTORY_STEPS, 2, 768, dtype=torch.float16)
    history_qpos = torch.zeros(count, HISTORY_STEPS, 9)
    history_action = torch.zeros(count, HISTORY_STEPS, 8)
    history_mask = torch.zeros(count, HISTORY_STEPS, dtype=torch.bool)
    action_mask = torch.zeros(count, HISTORY_STEPS, dtype=torch.bool)
    global_rgb, local_rgb, reset = [], [], []
    for index, (time_index, arm) in enumerate(rows):
        first = max(0, time_index - (HISTORY_STEPS - 1))
        observation_indices = list(range(first, time_index + 1))
        observation_offset = HISTORY_STEPS - len(observation_indices)
        action_first = max(0, time_index - HISTORY_STEPS)
        action_indices = list(range(action_first, time_index))
        action_offset = HISTORY_STEPS - len(action_indices)
        local_key = f"agent_{arm}"
        if local_key not in images:
            if episode.task != "place_food":
                raise KeyError(f"missing {local_key}: {episode.path}")
            local_key = "global"
        cache_key = "view_global" if local_key == "global" else f"view_agent_{arm}"
        history_visual[index, observation_offset:, 0] = torch.from_numpy(
            cache["view_global"][observation_indices]
        )
        history_visual[index, observation_offset:, 1] = torch.from_numpy(
            cache[cache_key][observation_indices]
        )
        history_qpos[index, observation_offset:] = (
            torch.from_numpy(qpos_sources[arm][observation_indices]) - dataset.q_mean
        ) / dataset.q_std
        history_mask[index, observation_offset:] = True
        if action_indices:
            history_action[index, action_offset:] = (
                torch.from_numpy(action_sources[arm][action_indices]) - dataset.a_mean
            ) / dataset.a_std
            action_mask[index, action_offset:] = True
        global_rgb.append(np.asarray(images["global"][time_index]))
        local_rgb.append(np.asarray(images[local_key][time_index]))
        reset.append(time_index == 0)
    task_bytes, task_mask = task_text_tensor(TASK_TEXT[episode.task])
    return {
        "global_rgb": torch.as_tensor(np.stack(global_rgb)).permute(0, 3, 1, 2).float().div_(255).to(device),
        "local_rgb": torch.as_tensor(np.stack(local_rgb)).permute(0, 3, 1, 2).float().div_(255).to(device),
        "history_visual_raw": history_visual.to(device),
        "history_qpos": history_qpos.to(device),
        "history_action": history_action.to(device),
        "history_mask": history_mask.to(device),
        "action_history_mask": action_mask.to(device),
        "task_bytes": task_bytes.unsqueeze(0).expand(count, -1).to(device),
        "task_text_mask": task_mask.unsqueeze(0).expand(count, -1).to(device),
        "episode_reset": torch.tensor(reset, dtype=torch.bool, device=device),
    }


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(max(1, min(12, (os.cpu_count() or 12) // world_size)))

    stats = load_stats(args.normalization)
    episodes = load_temporal_episodes(args.manifests)
    _metadata, n1_episodes = load_team_signal_metadata(args.signal_cache)
    n1_by_hash = {episode.hdf5_sha256: episode for episode in n1_episodes}
    if {episode.hdf5_sha256 for episode in episodes} != set(n1_by_hash):
        raise RuntimeError("Step-2 and N1 episode hashes differ")
    dataset = TeamTemporalDataset(episodes, stats, args.visual_cache, cache_limit=16)
    payload = torch.load(args.temporal_checkpoint, map_location="cpu", weights_only=False)
    model = TemporalHistoryPolicy(**_constructor_config(payload)).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    if model.hidden_residual is None:
        raise RuntimeError("N2 cache requires the formal hidden-residual B0-H")

    if rank == 0:
        if (args.output / "cache_receipt.json").exists():
            raise FileExistsError("N2 action-context cache receipt already exists")
        allocate(args.output, n1_episodes)
        with torch.no_grad():
            values = []
            for task in SIX_TASKS:
                tokens, masks = task_text_tensor(TASK_TEXT[task])
                values.append(model._task_token(tokens[None].to(device), masks[None].to(device))[0].float().cpu())
            np.save(args.output / "task_tokens.npy", torch.stack(values).numpy())
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    decoded_maps = {
        task: np.lib.format.open_memmap(args.output / f"{task}_decoded.npy", mode="r+")
        for task in SIX_TASKS
    }
    base_maps = {
        task: np.lib.format.open_memmap(args.output / f"{task}_base_action.npy", mode="r+")
        for task in SIX_TASKS
    }
    completed, samples = [], 0
    started = time.time()
    assigned = list(range(rank, len(episodes), world_size))
    for ordinal, episode_index in enumerate(assigned, start=1):
        episode = episodes[episode_index]
        n1_episode = n1_by_hash[episode.hdf5_sha256]
        if episode.task != n1_episode.task or episode.length != n1_episode.length:
            raise RuntimeError("N2 action-context episode layout differs")
        with h5py.File(episode.path, "r") as handle:
            all_rows = [
                (time_index, arm)
                for time_index in range(episode.length)
                for arm in (0, 1)
            ]
            for first in range(0, len(all_rows), args.batch_size):
                rows = all_rows[first : first + args.batch_size]
                inputs = episode_tensors(episode, dataset, handle, rows, device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    context = model._decode_action_context(**inputs, actions=None)
                    base = model.out(context.decoded)
                    history = context.history_summary.unsqueeze(1).expand(-1, ACTION_HORIZON, -1)
                    base = base + model.hidden_residual(torch.cat((context.decoded, history), dim=-1))
                target_first = 2 * n1_episode.offset + 2 * rows[0][0] + rows[0][1]
                target_last = target_first + len(rows)
                decoded_maps[episode.task][target_first:target_last] = context.decoded.float().cpu().numpy().astype(np.float16)
                base_maps[episode.task][target_first:target_last] = base.float().cpu().numpy().astype(np.float16)
                samples += len(rows)
        completed.append(episode.hdf5_sha256)
        if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(assigned):
            print(json.dumps({
                "rank": rank,
                "episodes": ordinal,
                "assigned": len(assigned),
                "samples": samples,
                "episodes_per_hour": ordinal / max(time.time() - started, 1e-6) * 3600,
            }), flush=True)
    for value in (*decoded_maps.values(), *base_maps.values()):
        value.flush()
    rank_receipt = args.output / f"rank_{rank:02d}_receipt.json"
    atomic_json(rank_receipt, {
        "rank": rank,
        "world_size": world_size,
        "episode_hashes": sorted(completed),
        "episodes": len(completed),
        "samples": samples,
    })
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    if rank == 0:
        rank_receipts = [json.loads((args.output / f"rank_{index:02d}_receipt.json").read_text()) for index in range(world_size)]
        hashes = [value for receipt in rank_receipts for value in receipt["episode_hashes"]]
        if len(hashes) != 720 or len(set(hashes)) != 720:
            raise RuntimeError("N2 action-context rank coverage differs")
        samples_total = sum(int(receipt["samples"]) for receipt in rank_receipts)
        expected_samples = sum(episode.length * 2 for episode in n1_episodes)
        if samples_total != expected_samples:
            raise RuntimeError("N2 action-context sample coverage differs")
        atomic_json(args.output / "cache_receipt.json", {
            "format_version": "before-we-act.b3-n2-action-context-cache/1",
            "status": "PASSED",
            "created_at_utc": utc_now(),
            "samples": samples_total,
            "episodes": 720,
            "tasks": list(SIX_TASKS),
            "dtype": "float16",
            "decoded_shape_per_sample": [ACTION_HORIZON, 384],
            "base_action_shape_per_sample": [ACTION_HORIZON, 8],
            "b0h_checkpoint": str(args.temporal_checkpoint.resolve()),
            "b0h_checkpoint_sha256": sha256_file(args.temporal_checkpoint),
            "normalization_sha256": sha256_file(args.normalization),
            "n1_metadata_sha256": sha256_file(args.signal_cache / "metadata.json"),
            "world_size": world_size,
            "rank_receipts": [path.name for path in sorted(args.output.glob("rank_*_receipt.json"))],
        })
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
