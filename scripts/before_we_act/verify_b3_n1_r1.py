#!/usr/bin/env python3
"""Run R1-0/R1-1 fail-closed boundaries and deterministic save/resume F1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.b3_n1_data import N1RawSignalDataset
from before_we_act.b3_n1_r1 import (
    FrozenR1Backbones,
    R1BalancedBatchSampler,
    R1FairProbeSet,
    R1_DATA_SEED,
    R1_MAX_UPDATES,
    action_sample_mse,
    deterministic_permutations,
    load_split,
    predictions,
    split_by_episode_key,
)
from before_we_act.step2_temporal_data import SIX_TASKS, sha256_file
from before_we_act.train_b3_n1_r1_fair_probe import device_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    split_payload = load_split(args.scenario_split)
    split = split_by_episode_key(split_payload)
    dataset = N1RawSignalDataset(args.cache)
    sampler = R1BalancedBatchSampler(
        dataset.episodes,
        split,
        updates=R1_MAX_UPDATES,
        data_seed=R1_DATA_SEED,
    )
    first_requests = sampler.requests_for_update(1)
    raw = next(iter(DataLoader(dataset, batch_sampler=[first_requests[:4]], num_workers=0)))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    batch = device_batch(raw, device)
    seed = int(contract["r1_1"]["seeds"][0])
    n1 = contract["old_n1_read_only"]["representation_checkpoints"][str(seed)]
    backbones = FrozenR1Backbones(
        b0h_checkpoint=Path(contract["b0h"]["checkpoint"]),
        n1_checkpoint=Path(n1["path"]),
        visual_mean=dataset.visual_mean,
        visual_std=dataset.visual_std,
    ).to(device)
    probes = R1FairProbeSet().to(device)
    backbones.eval()
    probes.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        frozen = backbones(batch)
        output = predictions(probes, frozen, batch)
    permutation = deterministic_permutations(batch)
    split_groups = {}
    for row in split_payload["episodes"]:
        split_groups.setdefault(row["scenario_group"], set()).add(row["split"])
    checks = {
        "contract_frozen": contract.get("status") == "FROZEN_BEFORE_F0_F1",
        "old_n1_status_preserved": contract["old_n1_read_only"]["status"]
        == "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
        "old_n1_conclusion_hash_matches": sha256_file(
            Path(contract["old_n1_read_only"]["root"]) / "n1_conclusion.json"
        )
        == contract["old_n1_read_only"]["conclusion_sha256"],
        "b0h_hash_matches": sha256_file(Path(contract["b0h"]["checkpoint"]))
        == contract["b0h"]["checkpoint_sha256"],
        "n1_hash_matches": sha256_file(Path(n1["path"])) == n1["sha256"],
        "all_scenario_groups_are_split_atomic": all(
            len(values) == 1 for values in split_groups.values()
        ),
        "six_tasks_present": {episode.task for episode in dataset.episodes}
        == set(SIX_TASKS),
        "effective_batch_is_48": len(first_requests) == 48,
        "b0h_hidden_shape": tuple(frozen.h.shape) == (4, 384),
        "full_belief_token_shape": tuple(frozen.belief.shape) == (4, 16, 384),
        "all_condition_shapes": all(
            tuple(value.shape) == (4, 16, 8) for value in output.values()
        ),
        "shuffle_has_no_fixed_point": all(
            not torch.any(index == torch.arange(len(index), device=index.device)).item()
            for index in permutation.values()
        ),
        "matched_capacity_parameter_count_exact": probes.parameter_counts()["h_b"]
        == probes.parameter_counts()["h_matched_capacity"],
        "backbones_frozen": not any(
            parameter.requires_grad for parameter in backbones.parameters()
        ),
        "finite_outputs": all(torch.isfinite(value).all().item() for value in output.values()),
    }

    # F1: one update, serialize, restore into a fresh head, and compare eval output.
    probes.train()
    optimizer = torch.optim.AdamW(probes.parameters(), lr=3e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        frozen = backbones(batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = predictions(probes, frozen, batch)
        loss = torch.stack(
            [action_sample_mse(value.float(), batch).mean() for value in prediction.values()]
        ).mean()
    loss.backward()
    optimizer.step()
    probes.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = {key: value.float().cpu() for key, value in predictions(probes, frozen, batch).items()}
    with tempfile.NamedTemporaryFile(suffix=".pt") as stream:
        torch.save(
            {"probes": probes.state_dict(), "optimizer": optimizer.state_dict()},
            stream.name,
        )
        saved = torch.load(stream.name, map_location="cpu", weights_only=False)
    resumed = R1FairProbeSet().to(device)
    resumed.load_state_dict(saved["probes"], strict=True)
    resumed.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        restored = {key: value.float().cpu() for key, value in predictions(resumed, frozen, batch).items()}
    maximum = max(
        float((reference[key] - restored[key]).abs().max()) for key in reference
    )
    checks["save_resume_max_parameter_or_output_difference_zero"] = maximum == 0.0
    payload = {
        "format_version": "before-we-act.b3-n1-r1-f0-f1-receipt/1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "completed_at_utc": utc_now(),
        "checks": checks,
        "save_resume_max_abs_output_difference": maximum,
        "f1_loss": float(loss.detach()),
        "parameter_counts": probes.parameter_counts(),
        "sample_cursor_after_zero_updates": sampler.cursor_receipt(0),
        "input_boundary": {
            "runtime_fields": sorted(N1RawSignalDataset.RUNTIME_FIELDS),
            "audit_fields_not_forwarded": sorted(N1RawSignalDataset.AUDIT_ONLY_FIELDS),
            "teacher_targets_not_forwarded": sorted(N1RawSignalDataset.TEACHER_TARGET_FIELDS),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "checks": checks}, sort_keys=True))
    if payload["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
