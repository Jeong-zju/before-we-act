#!/usr/bin/env python3
"""Collect reactive-only branches for the frozen A4R10 common-snapshot pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from before_we_act.care_branch_collector import (
    atomic_json,
    canonicalize_policy_plans,
    capture_snapshot,
    make_env,
    new_runtime,
    policy_plan,
    run_branch,
    sha256_file,
    sha256_tree,
    update_oracle,
    append_executed_action,
)
from before_we_act.deployment_safety import ResidualSafetyConfig
from scripts.before_we_act.audit_ssc_v7_m2 import scalar_bool


CONTRACT_STAGE = "A4R10-CARE-COMMON-SNAPSHOT-OPTION-PILOT"
MANIFEST_STAGE = "A5R9-CARE-COMMON-SNAPSHOT-OPTION-PILOT"
FORMAT_VERSION = "before-we-act.a5r9-care-common-snapshot-option-pilot-family/1"


def rebuild_snapshot(*, family, env, model, stats, config, device):
    task = str(family["task"])
    from two_three_task_manifest import get_task

    arms = tuple(int(value) for value in get_task(task)["agents"])
    runtime = new_runtime(arms, ResidualSafetyConfig.from_mapping(config.get("residual_safety")), task)
    seed = int(family["episode_seed"])
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    observation, info = env.reset(seed=seed)
    base_env = getattr(env, "base_env", env.unwrapped)
    label = update_oracle(base_env, task, scalar_bool(info["success"]), runtime)
    for _ in range(int(family["anchor_step"])):
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
        if scalar_bool(terminated) or scalar_bool(truncated):
            raise RuntimeError("episode ended before the frozen A4R10 snapshot")
    return capture_snapshot(env, observation, runtime, label)


def main() -> None:
    from before_we_act.evaluate_predictive_team_belief import load_team_belief

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, default=Path("/workspace/RoboFactory"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if manifest.get("stage_id") != MANIFEST_STAGE or manifest.get("status") != "FROZEN_BEFORE_OPTION_OUTCOMES":
        raise RuntimeError("A5R9 pilot manifest is not frozen")
    if contract.get("stage_id") != CONTRACT_STAGE or contract.get("status") != "FROZEN_BEFORE_OPTION_OUTCOMES":
        raise RuntimeError("A4R10 pilot contract is not frozen")
    contract_sha = sha256_file(args.contract)
    checkpoint_sha = sha256_file(args.checkpoint)
    if contract_sha != manifest["contract_sha256"]:
        raise RuntimeError("A4R10 contract hash drifted")
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise RuntimeError("B-core checkpoint hash drifted")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    families = [
        row
        for index, row in enumerate(manifest["families"])
        if index % args.shard_count == args.shard_index
    ]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(min(12, os.cpu_count() or 12))
    model, stats, config = load_team_belief(str(args.checkpoint), device)
    for task in sorted({str(row["task"]) for row in families}):
        env = make_env(task, args.robofactory_root)
        try:
            for family in (row for row in families if row["task"] == task):
                output = args.output_root / task / f"{family['snapshot_id']}.json"
                if output.is_file():
                    existing = json.loads(output.read_text(encoding="utf-8"))
                    if (
                        existing.get("format_version") == FORMAT_VERSION
                        and existing.get("contract_sha256") == contract_sha
                        and existing.get("checkpoint_sha256") == checkpoint_sha
                        and existing.get("branch_count") == 32
                    ):
                        print(json.dumps({"snapshot_id": family["snapshot_id"], "reused": True}), flush=True)
                        continue
                    raise RuntimeError(f"refusing to overwrite inconsistent pilot family: {output}")

                source_path = Path(str(family["source_a5r7_family"]))
                if sha256_file(source_path) != family["source_a5r7_family_sha256"]:
                    raise RuntimeError(f"A5R7 source family drifted: {source_path}")
                source = json.loads(source_path.read_text(encoding="utf-8"))
                started = time.perf_counter()
                snapshot = rebuild_snapshot(
                    family=family,
                    env=env,
                    model=model,
                    stats=stats,
                    config=config,
                    device=device,
                )
                state_sha = sha256_tree(snapshot.state)
                source_state_sha = str(source["snapshot_state_sha256"])
                source_snapshot_id = str(family["source_a5r7_snapshot_id"])
                focal = int(family["focal_agent"])
                branches = []
                for repeat_id in (0, 1):
                    reference, _ = run_branch(
                        env=env,
                        snapshot=snapshot,
                        snapshot_id=source_snapshot_id,
                        model=model,
                        stats=stats,
                        task=task,
                        focal_arm=focal,
                        candidate_id=0,
                        regime="reactive",
                        repeat_id=repeat_id,
                        teammate_reference_actions=None,
                        device=device,
                        horizon=32,
                        intervention_steps=1,
                    )
                    branches.append(reference)
                    for duration in (1, 4, 8):
                        for candidate_id in range(1, 6):
                            branch, _ = run_branch(
                                env=env,
                                snapshot=snapshot,
                                snapshot_id=source_snapshot_id,
                                model=model,
                                stats=stats,
                                task=task,
                                focal_arm=focal,
                                candidate_id=candidate_id,
                                regime="reactive",
                                repeat_id=repeat_id,
                                teammate_reference_actions=None,
                                device=device,
                                horizon=32,
                                intervention_steps=duration,
                            )
                            branches.append(branch)
                if len(branches) != 32:
                    raise RuntimeError("A4R10 pilot branch count drifted")
                result = {
                    "format_version": FORMAT_VERSION,
                    "stage_id": MANIFEST_STAGE,
                    "resource_only": True,
                    "forbidden_uses": contract["output"]["forbidden_uses"],
                    "snapshot_id": family["snapshot_id"],
                    "source_a5r7_snapshot_id": source_snapshot_id,
                    "task": task,
                    "episode_seed": int(family["episode_seed"]),
                    "anchor_step": int(family["anchor_step"]),
                    "focal_agent": focal,
                    "scenario_group_id": family["scenario_group_id"],
                    "sampling_stratum": family["sampling_stratum"],
                    "split": family["split"],
                    "contract_sha256": contract_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "source_a5r7_family": str(source_path.resolve()),
                    "source_a5r7_family_sha256": family["source_a5r7_family_sha256"],
                    "source_a5r7_quality": family["source_a5r7_quality"],
                    "source_a5r7_quality_sha256": family["source_a5r7_quality_sha256"],
                    "snapshot_state_sha256": state_sha,
                    "source_a5r7_snapshot_state_sha256": source_state_sha,
                    "snapshot_state_matches_a5r7": state_sha == source_state_sha,
                    "common_snapshot_protocol": (
                        "all duration-candidate-repeat branches restore independently "
                        "from this newly captured in-memory snapshot"
                    ),
                    "maximum_branch_steps": 32,
                    "branches": branches,
                    "branch_count": len(branches),
                    "wall_seconds": time.perf_counter() - started,
                    "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
                atomic_json(output, result)
                print(
                    json.dumps(
                        {
                            "snapshot_id": family["snapshot_id"],
                            "task": task,
                            "branch_count": len(branches),
                            "wall_seconds": result["wall_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
