#!/usr/bin/env python3
"""Encode successful raw RoboFactory Stack demonstrations without RGB expansion.

The legacy conversion writes current and next 480x640 RGB into one HDF5 file per
episode.  That representation is useful as a generic interchange format but is
unnecessarily large for R12/R15 action training.  This builder reads the raw
ManiSkill trajectory groups directly, applies the frozen action codec, encodes
native RGB with the frozen DINOv3 backbone, and appends immutable feature shards
to a copy of the existing full-episode index.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import h5py
import numpy as np
import torch

from before_we_act.data.full_episode_windows import FULL_EPISODE_PROTOCOL
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.spatial_observation import (
    R12SpatialObservationEncoder,
    locked_r12_full_episode_observation,
)
from data.robofactory import RoboFactoryDataset, RoboFactoryEpisode
from models.wam.action_codec import AffineActionCodec, AffineActionCodecConfig
from scripts.before_we_act.prepare_r12_full_episode_cache import (
    atomic_hdf5_save,
    patch_means,
)


TASK = "three_robots_stack_cube"
EXPECTED_ENV_ID = "ThreeRobotsStackCube-rf"
EXPECTED_CAMERAS = ("global", "agent_0", "agent_1", "agent_2")
EXPECTED_CODEC_SHA256 = (
    "38b9d91640dcb9f7a33d7f05ea0d3cab47fc843fb5cfb18859ceece3314b2eb5"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def terminal_steps(group: h5py.Group) -> int:
    """Return the first terminal transition, inclusive.

    RoboFactory motion-planning recordings may contain a short post-success tail.
    The frozen training manifests deliberately remove that tail, so supplemental
    demonstrations must reproduce the same transition contract.
    """

    terminated = np.asarray(group["terminated"], dtype=np.bool_)
    truncated = np.asarray(group["truncated"], dtype=np.bool_)
    if terminated.shape != truncated.shape or terminated.ndim != 1:
        raise ValueError("raw terminated/truncated vectors differ")
    done = terminated | truncated
    indices = np.flatnonzero(done)
    if not len(indices):
        raise ValueError("successful raw expert episode has no terminal transition")
    steps = int(indices[0]) + 1
    if "success" in group and not bool(np.asarray(group["success"][:steps]).any()):
        raise ValueError("raw expert episode terminates without a success label")
    return steps


def validate_source(source: RoboFactoryDataset, codec: AffineActionCodec) -> None:
    if source.env_id != EXPECTED_ENV_ID:
        raise ValueError(f"expert source env differs: {source.env_id!r}")
    if len(source.layout.agents) != 3:
        raise ValueError("R15 Stack expert source must contain exactly three agents")
    if [agent.action_shape for agent in source.layout.agents] != [(8,), (8,), (8,)]:
        raise ValueError("R15 Stack expert action layout differs")
    if [agent.qpos_shape for agent in source.layout.agents] != [(9,), (9,), (9,)]:
        raise ValueError("R15 Stack expert qpos layout differs")
    cameras = {camera.target_name: camera for camera in source.layout.cameras}
    if set(cameras) != set(EXPECTED_CAMERAS):
        raise ValueError(f"R15 Stack expert cameras differ: {sorted(cameras)}")
    if any(tuple(cameras[name].image_shape) != (480, 640, 3) for name in cameras):
        raise ValueError("R15 Stack expert RGB is not native 480x640")
    if codec.action_dim != 24 or codec.semantic_sha256 != EXPECTED_CODEC_SHA256:
        raise ValueError("R15 Stack expert action codec identity differs")


def selected_episodes(
    source: RoboFactoryDataset, max_episodes: int
) -> tuple[RoboFactoryEpisode, ...]:
    episodes = tuple(
        episode for episode in source.episodes if episode.success is True
    )[:max_episodes]
    if len(episodes) != max_episodes:
        raise ValueError(
            f"requested {max_episodes} successful episodes, found {len(episodes)}"
        )
    if any(episode.seed is None for episode in episodes):
        raise ValueError("R15 expert episodes require recorded seeds")
    seeds = [int(episode.seed) for episode in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("R15 expert source contains duplicate successful seeds")
    return episodes


def source_plan(
    source: RoboFactoryDataset,
    episodes: tuple[RoboFactoryEpisode, ...],
    base_index: Mapping[str, object],
) -> list[dict[str, object]]:
    existing_seeds = {
        int(row["seed"])
        for row in base_index["episodes"]
        if row["task"] == TASK
    }
    new_seeds = {int(episode.seed) for episode in episodes}
    overlap = sorted(existing_seeds & new_seeds)
    if overlap:
        raise ValueError(f"expert/base Stack seeds overlap: {overlap[:10]}")
    maximum_index = max(
        int(row["episode_index"])
        for row in base_index["episodes"]
        if row["task"] == TASK
    )
    rows = []
    for offset, episode in enumerate(episodes, 1):
        group = source._file[episode.source_key]
        rows.append(
            {
                "source_id": episode.source_id,
                "source_key": episode.source_key,
                "seed": int(episode.seed),
                "steps": terminal_steps(group),
                "episode_index": maximum_index + offset,
            }
        )
    return rows


@torch.inference_mode()
def encode_episode(
    args: argparse.Namespace,
    source: RoboFactoryDataset,
    codec: AffineActionCodec,
    encoder: R12SpatialObservationEncoder,
    row: Mapping[str, object],
    output: Path,
    source_hdf5_sha256: str,
    source_json_sha256: str,
) -> dict[str, object]:
    group = source._file[str(row["source_key"])]
    steps = int(row["steps"])
    spatial = torch.zeros((steps, 5, 48, 768), dtype=torch.float16)
    visual = torch.zeros((steps, 16, 15), dtype=torch.float16)
    view_mask = torch.zeros((steps, 5), dtype=torch.bool)
    qpos = torch.zeros((steps, 4, 9), dtype=torch.float32)
    commanded = torch.zeros((steps, 4, 8), dtype=torch.float32)
    agents = tuple(source.layout.agents)
    raw_actions = np.concatenate(
        [
            np.asarray(group[f"actions/{agent.source_name}"][:steps], dtype=np.float32)
            for agent in agents
        ],
        axis=-1,
    )
    canonical = np.asarray(codec.encode(raw_actions), dtype=np.float32).reshape(
        steps, 3, 8
    )
    commanded[:, :3] = torch.from_numpy(canonical)
    for agent_index, agent in enumerate(agents):
        qpos[:, agent_index] = torch.from_numpy(
            np.asarray(
                group[f"obs/agent/{agent.source_name}/qpos"][:steps],
                dtype=np.float32,
            )
        )
    cameras = {camera.target_name: camera.source_name for camera in source.layout.cameras}
    for start in range(0, steps, args.frame_batch_size):
        end = min(steps, start + args.frame_batch_size)
        requests: list[tuple[int, int, np.ndarray]] = []
        for frame in range(start, end):
            for view_index, view in enumerate(EXPECTED_CAMERAS):
                image = np.asarray(
                    group[f"obs/sensor_data/{cameras[view]}/rgb"][frame],
                    dtype=np.uint8,
                )
                visual[frame, :, view_index * 3 : (view_index + 1) * 3] = (
                    torch.from_numpy(patch_means(image)).to(torch.float16)
                )
                view_mask[frame, view_index] = True
                requests.append((frame, view_index, image))
        images = torch.from_numpy(
            np.stack([item[2].transpose(2, 0, 1) for item in requests])
        ).to(args.device, non_blocking=True)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=str(args.device).startswith("cuda"),
        ):
            tokens = encoder.encoder.forward_spatial_grid(
                images, grid_height=6, grid_width=8
            ).spatial_tokens
        tokens = tokens.cpu().to(torch.float16)
        for offset, (frame, view_index, _image) in enumerate(requests):
            spatial[frame, view_index] = tokens[offset]
        atomic_json(
            args.heartbeat,
            {
                "producer": "prepare_r15_expert_full_cache",
                "seed": int(row["seed"]),
                "frame": end,
                "episode_steps": steps,
                "updated_at": now(),
            },
        )
    metadata = {
        "created_at": now(),
        "protocol_variant": FULL_EPISODE_PROTOCOL,
        "extension_protocol": "r15_raw_success_expert_direct_dinov3_v1",
        "task": TASK,
        "split": "train",
        "seed": int(row["seed"]),
        "steps": steps,
        "episode_index": int(row["episode_index"]),
        "source_episode_id": int(row["source_id"]),
        "source_hdf5": str(args.raw_hdf5.resolve()),
        "source_metadata_json": str(args.raw_json.resolve()),
        "hdf5_sha256": source_hdf5_sha256,
        "source_metadata_json_sha256": source_json_sha256,
        "action_codec_sha256": codec.semantic_sha256,
        "action_semantics": "raw_pd_joint_pos encoded to canonical_unit_action; command echo",
        "terminal_policy": "first terminated_or_truncated transition inclusive",
        "observation": locked_r12_full_episode_observation(),
        "legal_inputs": "native 480x640 current fixed-view RGB plus causal qpos/executed-action history",
        "forbidden_inputs": "future RGB, simulator state, motion-planner state at policy input",
    }
    atomic_hdf5_save(
        output,
        {
            "schema_version": 1,
            "round": "R12-R4",
            "metadata": metadata,
            "visual": visual,
            "view_mask": view_mask,
            "qpos": qpos,
            "executed_actions": commanded.clone(),
            "commanded_actions": commanded,
            "agent_mask": torch.tensor([True, True, True, False]),
            "spatial_tokens": spatial,
            "spatial_view_mask": view_mask.clone(),
        },
    )
    return {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "task": TASK,
        "split": "train",
        "seed": int(row["seed"]),
        "steps": steps,
        "hdf5_sha256": source_hdf5_sha256,
        "episode_index": int(row["episode_index"]),
        "source_episode_id": int(row["source_id"]),
    }


def compose_index(
    base_index: Mapping[str, object],
    expert_rows: list[dict[str, object]],
    *,
    base_index_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    if (
        base_index.get("schema_version") != 1
        or base_index.get("round") != "R12-R4"
        or base_index.get("protocol_variant") != FULL_EPISODE_PROTOCOL
    ):
        raise ValueError("base full-episode index identity differs")
    combined = deepcopy(dict(base_index))
    existing = list(combined["episodes"])
    seed_identity = {(row["task"], int(row["seed"])) for row in existing}
    episode_identity = {
        (row["task"], row["split"], int(row["episode_index"])) for row in existing
    }
    for row in expert_rows:
        if (row["task"], int(row["seed"])) in seed_identity:
            raise ValueError("expert index duplicates an existing task/seed")
        identity = (row["task"], row["split"], int(row["episode_index"]))
        if identity in episode_identity:
            raise ValueError("expert index duplicates an existing episode identity")
        seed_identity.add((row["task"], int(row["seed"])))
        episode_identity.add(identity)
        existing.append(row)
    combined["episodes"] = sorted(
        existing, key=lambda row: (row["split"], row["task"], int(row["episode_index"]))
    )
    combined["step_counts"] = deepcopy(combined["step_counts"])
    combined["step_counts"]["train"][TASK] += sum(
        int(row["steps"]) for row in expert_rows
    )
    combined["extension"] = {
        "protocol": "r15_raw_success_expert_direct_dinov3_v1",
        "base_index": str(base_index_path.resolve()),
        "base_index_sha256": sha256(base_index_path),
        "expert_receipt": str(receipt_path.resolve()),
        "expert_receipt_sha256": sha256(receipt_path),
        "expert_episodes": len(expert_rows),
        "expert_steps": sum(int(row["steps"]) for row in expert_rows),
        "created_at": now(),
    }
    combined["created_at"] = now()
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-hdf5", type=Path, required=True)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vision-artifact", type=Path, required=True)
    parser.add_argument(
        "--action-codec",
        type=Path,
        default=Path("configs/action_codecs/robofactory_3panda_pd_joint_pos_24d.json"),
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.frame_batch_size < 1 or args.image_batch_size < 1:
        raise ValueError("episodes and cache batches must be positive")
    args.raw_hdf5 = args.raw_hdf5.resolve(strict=True)
    args.raw_json = args.raw_json.resolve(strict=True)
    args.base_index = args.base_index.resolve(strict=True)
    args.vision_artifact = args.vision_artifact.resolve(strict=True)
    args.action_codec = args.action_codec.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    args.state = (args.state or args.output_root / "state.json").resolve()
    args.heartbeat = (args.heartbeat or args.output_root / "heartbeat.json").resolve()
    base_index = json.loads(args.base_index.read_text(encoding="utf-8"))
    codec = AffineActionCodec(AffineActionCodecConfig.load(args.action_codec))
    with RoboFactoryDataset(args.raw_hdf5, metadata_path=args.raw_json) as source:
        validate_source(source, codec)
        episodes = selected_episodes(source, args.episodes)
        plan = source_plan(source, episodes, base_index)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "DRY_RUN_PASSED",
                        "episodes": len(plan),
                        "seeds": [row["seed"] for row in plan],
                        "steps": sum(int(row["steps"]) for row in plan),
                        "codec_sha256": codec.semantic_sha256,
                        "cameras": list(EXPECTED_CAMERAS),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise ValueError("expert cache output root must be new and empty")
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.state,
            {
                "state": "PREPARING",
                "stage": "source_hash",
                "episodes": len(plan),
                "updated_at": now(),
            },
        )
        source_hdf5_sha256 = sha256(args.raw_hdf5)
        source_json_sha256 = sha256(args.raw_json)
        encoder = R12SpatialObservationEncoder(
            locked_r12_full_episode_observation(),
            args.vision_artifact,
            inference_batch_size=args.image_batch_size,
        ).to(args.device).eval()
        records = []
        for index, row in enumerate(plan, 1):
            atomic_json(
                args.state,
                {
                    "state": "PREPARING",
                    "stage": "native_rgb_post_dino_expert_cache",
                    "completed_episodes": index - 1,
                    "total_episodes": len(plan),
                    "current_seed": int(row["seed"]),
                    "updated_at": now(),
                },
            )
            output = (
                args.output_root
                / "episodes"
                / "train"
                / TASK
                / f"expert_episode_{int(row['episode_index']):06d}_seed_{int(row['seed'])}.hdf5"
            )
            records.append(
                encode_episode(
                    args,
                    source,
                    codec,
                    encoder,
                    row,
                    output,
                    source_hdf5_sha256,
                    source_json_sha256,
                )
            )
    receipt_path = args.output_root / "expert_receipt.json"
    receipt = {
        "schema_version": 1,
        "round": "R15-Evolution",
        "protocol": "r15_raw_success_expert_direct_dinov3_v1",
        "source_hdf5": str(args.raw_hdf5),
        "source_hdf5_sha256": source_hdf5_sha256,
        "source_metadata_json": str(args.raw_json),
        "source_metadata_json_sha256": source_json_sha256,
        "action_codec": str(args.action_codec),
        "action_codec_sha256": codec.semantic_sha256,
        "episodes": records,
        "created_at": now(),
    }
    atomic_json(receipt_path, receipt)
    index = compose_index(
        base_index,
        records,
        base_index_path=args.base_index,
        receipt_path=receipt_path,
    )
    index_path = args.output_root / "index.json"
    atomic_json(index_path, index)
    atomic_json(
        args.state,
        {
            "state": "PASSED",
            "stage": "complete",
            "episodes": len(records),
            "steps": sum(int(row["steps"]) for row in records),
            "index": str(index_path),
            "index_sha256": sha256(index_path),
            "updated_at": now(),
        },
    )
    atomic_json(
        args.heartbeat,
        {
            "producer": "prepare_r15_expert_full_cache",
            "complete": True,
            "updated_at": now(),
        },
    )
    print(json.dumps(json.loads(args.state.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
