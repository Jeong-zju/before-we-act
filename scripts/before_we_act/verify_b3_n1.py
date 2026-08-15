#!/usr/bin/env python3
"""Run 3-N1 F0 boundary checks and an exact real-data save/resume F1."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.b3_n1_data import N1BalancedBatchSampler, N1RawSignalDataset, N1Request
from before_we_act.b3_n1_model import N1RawSignalModel, representation_losses
from before_we_act.train_b3_n1_representation import atomic_json, device_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def one_update(model, optimizer, batch, seed: int):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    optimizer.zero_grad(set_to_none=True)
    inputs = {key: batch[key] for key in N1RawSignalDataset.RUNTIME_FIELDS}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**inputs)
        loss = representation_losses(output, batch)["total"]
    loss.backward(); optimizer.step()
    return float(loss)


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    if contract.get("status") != "FROZEN_BEFORE_F0_F1":
        raise RuntimeError("3-N1 contract is not frozen")
    dataset = N1RawSignalDataset(args.cache)
    audit = []
    by_task = {}
    for index, episode in enumerate(dataset.episodes):
        if episode.split == "validation" and episode.task not in by_task:
            by_task[episode.task] = index
    for task, episode_index in by_task.items():
        episode = dataset.episodes[episode_index]
        for t in (0, episode.length // 2, episode.length - 1):
            row = dataset[N1Request(episode_index, 0, t, f"f0:{task}:{t}", task)]
            audit.append(
                {
                    "task": task,
                    "time": t,
                    "history_valid": int(row["history_mask"].sum()),
                    "past_action_valid": int(row["action_history_mask"].sum()),
                    "future_valid": int(row["future_mask"].sum()),
                    "runtime_shapes_ok": row["history_visual"].shape == (16, 2, 768) and row["history_qpos"].shape == (16, 9),
                }
            )
    boundary_ok = (
        not (N1RawSignalDataset.RUNTIME_FIELDS & N1RawSignalDataset.TEACHER_TARGET_FIELDS)
        and not (N1RawSignalDataset.RUNTIME_FIELDS & N1RawSignalDataset.AUDIT_ONLY_FIELDS)
        and all(row["runtime_shapes_ok"] for row in audit)
    )
    sampler = N1BalancedBatchSampler(dataset.episodes, updates=2, data_seed=20260815)
    batches = list(DataLoader(dataset, batch_sampler=sampler, num_workers=0))
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    batches = [device_batch(batch, device) for batch in batches]
    torch.manual_seed(20260815); torch.cuda.manual_seed_all(20260815)
    model = N1RawSignalModel(dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    first_loss = one_update(model, optimizer, batches[0], 11)
    saved_model = deepcopy(model.state_dict()); saved_optimizer = deepcopy(optimizer.state_dict())
    continuous_loss = one_update(model, optimizer, batches[1], 12)
    continuous = deepcopy(model.state_dict())
    resumed = N1RawSignalModel(dropout=0.0).to(device); resumed.load_state_dict(saved_model)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=2e-4); resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_loss = one_update(resumed, resumed_optimizer, batches[1], 12)
    maximum = max(float((continuous[key] - resumed.state_dict()[key]).abs().max()) for key in continuous)
    passed = boundary_ok and np.isfinite([first_loss, continuous_loss, resumed_loss]).all() and maximum <= 1e-7
    receipt = {
        "format_version": "before-we-act.b3-n1-f0-f1/1",
        "status": "PASSED" if passed else "FAILED",
        "f0": {"boundary_ok": boundary_ok, "samples": audit, "runtime_fields": sorted(N1RawSignalDataset.RUNTIME_FIELDS), "teacher_target_fields": sorted(N1RawSignalDataset.TEACHER_TARGET_FIELDS), "audit_only_fields": sorted(N1RawSignalDataset.AUDIT_ONLY_FIELDS)},
        "f1": {"first_loss": first_loss, "continuous_second_loss": continuous_loss, "resumed_second_loss": resumed_loss, "maximum_parameter_difference": maximum, "tolerance": 1e-7},
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    atomic_json(args.output, receipt)
    if not passed:
        raise SystemExit(json.dumps(receipt, sort_keys=True))
    print(json.dumps({"status": "PASSED", "max_difference": maximum}))


if __name__ == "__main__":
    main()
