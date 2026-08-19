#!/usr/bin/env python3
"""扫描正式候选状态的分支前信号；不运行任何候选分支。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from before_we_act.care_branch_collector import (
    append_executed_action,
    canonicalize_policy_plans,
    current_qpos_map,
    make_env,
    new_runtime,
    policy_plan,
    scalar_bool,
    update_oracle,
)
from before_we_act.deployment_safety import ResidualSafetyConfig
from two_three_task_manifest import get_task


SUPPORTED_STAGES = {
    "A5R4-CARE-FORMAL-PREBRANCH-SCAN": {
        "contract_stage": "A4R4-CARE-FORMAL-COLLECTION",
        "row_format": "before-we-act.a5r4-care-formal-prebranch-row/1",
    },
    "A5R5-CARE-GATE-FIRST-PREBRANCH-SCAN": {
        "contract_stage": "A4R5-CARE-GATE-FIRST-COLLECTION",
        "row_format": "before-we-act.a5r5-care-gate-first-prebranch-row/1",
    },
    "A5R6-CARE-COMPACT-PREBRANCH-SCAN": {
        "contract_stage": "A4R6-CARE-COMPACT-COLLECTION",
        "row_format": "before-we-act.a5r6-care-compact-prebranch-row/1",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def partner_occlusion(base_env: Any, focal_arm: int, arms: tuple[int, ...]) -> tuple[bool, dict[str, int]]:
    sensor = base_env._sensors[f"head_camera_agent{focal_arm}"]
    segmentation = sensor.get_obs(
        rgb=False,
        depth=False,
        position=False,
        segmentation=True,
    )["segmentation"]
    values = np.asarray(segmentation.detach().cpu()).reshape(-1)
    agents_by_arm = {
        arm: base_env.agent.agents[index] for index, arm in enumerate(arms)
    }
    visible_pixels: dict[str, int] = {}
    for arm in arms:
        if arm == focal_arm:
            continue
        ids = np.asarray(
            [
                int(link._objs[0].entity.per_scene_id)
                for link in agents_by_arm[arm].robot.get_links()
            ],
            dtype=np.int64,
        )
        visible_pixels[f"panda-{arm}"] = int(np.isin(values, ids).sum())
    return any(count == 0 for count in visible_pixels.values()), visible_pixels


def has_contact(label: Mapping[str, Any]) -> bool:
    for state in label["grasp_contact_custody_state"].values():
        if state.get("contact_agents") or state.get("grasp_agents"):
            return True
    return False


def scan_one(
    *,
    candidate: Mapping[str, Any],
    env: Any,
    model: Any,
    stats: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    device: torch.device,
    contract_sha256: str,
    checkpoint_sha256: str,
    scan_manifest_sha256: str,
    stage_id: str,
    row_format: str,
) -> dict[str, Any]:
    task = str(candidate["task"])
    arms = tuple(int(value) for value in get_task(task)["agents"])
    focal = int(candidate["focal_agent"])
    safety = ResidualSafetyConfig.from_mapping(config.get("residual_safety"))
    runtime = new_runtime(arms, safety, task)
    seed = int(candidate["episode_seed"])
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    observation, info = env.reset(seed=seed)
    base_env = getattr(env, "base_env", env.unwrapped)
    label = update_oracle(base_env, task, scalar_bool(info["success"]), runtime)
    immediate_phase_change = False
    for _ in range(int(candidate["anchor_step"])):
        previous_stage = str(label["stage_id"])
        reference, base, _qpos, _diagnostics = policy_plan(
            model, stats, observation, runtime, task, device
        )
        reference, _base, _canonicalization = canonicalize_policy_plans(
            reference, base, env.action_space.spaces, runtime.arms
        )
        action = {key: value[0].copy() for key, value in reference.items()}
        append_executed_action(runtime, action, stats)
        observation, _reward, terminated, truncated, info = env.step(action)
        label = update_oracle(base_env, task, scalar_bool(info["success"]), runtime)
        immediate_phase_change = str(label["stage_id"]) != previous_stage
        if scalar_bool(terminated) or scalar_bool(truncated):
            return {
                "format_version": row_format,
                "stage_id": stage_id,
                "scan_id": candidate["scan_id"],
                "task": task,
                "valid": False,
                "invalid_reason": "episode_terminated_before_anchor",
                "candidate": dict(candidate),
                "contract_sha256": contract_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "scan_manifest_sha256": scan_manifest_sha256,
            }

    preview_runtime = deepcopy(runtime)
    reference, base, qpos, diagnostics = policy_plan(
        model, stats, observation, preview_runtime, task, device
    )
    reference, _base, _canonicalization = canonicalize_policy_plans(
        reference, base, env.action_space.spaces, preview_runtime.arms
    )
    inactive = all(
        float(np.linalg.norm(reference[f"panda-{arm}"][0, :7] - qpos[index, :7]))
        < 0.02
        for index, arm in enumerate(arms)
    )
    occluded, visible_pixels = partner_occlusion(base_env, focal, arms)
    qpos_map = current_qpos_map(observation, arms)
    return {
        "format_version": row_format,
        "stage_id": stage_id,
        "scan_id": candidate["scan_id"],
        "task": task,
        "valid": True,
        "invalid_reason": "",
        "candidate": dict(candidate),
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "scan_manifest_sha256": scan_manifest_sha256,
        "features": {
            "residual_norm": float(diagnostics["residual_norm"]),
            "belief_entropy": float(diagnostics["sigma"]),
            "reliability": float(diagnostics["reliability"]),
            "partner_occlusion": bool(occluded),
            "partner_visible_pixels": visible_pixels,
            "contact_or_phase_change": bool(has_contact(label) or immediate_phase_change),
            "contact": bool(has_contact(label)),
            "immediate_phase_change": bool(immediate_phase_change),
            "paired_inactivity": bool(inactive),
        },
        "snapshot_label": {
            "stage_id": label["stage_id"],
            "within_stage_progress": label["within_stage_progress"],
            "factorized_predicates": label["factorized_predicates"],
        },
        "qpos_sha256": hashlib.sha256(
            b"".join(np.asarray(qpos_map[f"panda-{arm}"], dtype=np.float32).tobytes() for arm in arms)
        ).hexdigest(),
    }


def main() -> None:
    from before_we_act.evaluate_predictive_team_belief import load_team_belief

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, default=Path("/workspace/RoboFactory"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    stage_id = str(manifest.get("stage_id"))
    stage = SUPPORTED_STAGES.get(stage_id)
    if stage is None or contract.get("stage_id") != stage["contract_stage"]:
        raise RuntimeError("formal scan manifest/contract stage mismatch")
    contract_sha = sha256_file(args.contract)
    checkpoint_sha = sha256_file(args.checkpoint)
    manifest_sha = sha256_file(args.manifest)
    if contract_sha != manifest["contract_sha256"] or checkpoint_sha != manifest["checkpoint_sha256"]:
        raise RuntimeError("formal scan contract or checkpoint drifted")
    selected = [
        row
        for index, row in enumerate(manifest["candidates"])
        if index % args.shard_count == args.shard_index
    ]
    if args.limit:
        selected = selected[: args.limit]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["task"])].append(row)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(min(12, os.cpu_count() or 12))
    model, stats, config = load_team_belief(str(args.checkpoint), device)
    for task in sorted(grouped):
        env = make_env(task, args.robofactory_root)
        try:
            for candidate in grouped[task]:
                output = args.output_root / task / f"{candidate['scan_id']}.json"
                if output.is_file():
                    existing = json.loads(output.read_text(encoding="utf-8"))
                    if (
                        existing.get("scan_manifest_sha256") == manifest_sha
                        and existing.get("contract_sha256") == contract_sha
                        and existing.get("checkpoint_sha256") == checkpoint_sha
                    ):
                        print(json.dumps({"scan_id": candidate["scan_id"], "reused": True}), flush=True)
                        continue
                    raise RuntimeError(f"拒绝覆盖不一致的扫描结果：{output}")
                result = scan_one(
                    candidate=candidate,
                    env=env,
                    model=model,
                    stats=stats,
                    config=config,
                    device=device,
                    contract_sha256=contract_sha,
                    checkpoint_sha256=checkpoint_sha,
                    scan_manifest_sha256=manifest_sha,
                    stage_id=stage_id,
                    row_format=str(stage["row_format"]),
                )
                atomic_json(output, result)
                print(
                    json.dumps(
                        {
                            "scan_id": candidate["scan_id"],
                            "task": task,
                            "valid": result["valid"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
