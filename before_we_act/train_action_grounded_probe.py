"""Train one seed of the frozen fair action-grounded belief probe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.raw_team_signal_data import RawTeamSignalDataset
from before_we_act.action_grounded_belief import (
    FrozenBeliefBackbones,
    ActionGroundedBatchSampler,
    ActionGroundedProbeSet,
    ACTION_GROUNDED_CONDITIONS,
    BELIEF_DATA_SEED,
    BELIEF_EVAL_EVERY,
    BELIEF_LR_DROP,
    BELIEF_MAX_UPDATES,
    BELIEF_MIN_UPDATES,
    BELIEF_SEEDS,
    action_sample_mse,
    all_conditions_platform,
    condition_platform,
    fixed_requests,
    load_split,
    predictions,
    split_by_episode_key,
)
from before_we_act.temporal_history_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
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


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def device_batch(raw: Mapping[str, object], device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in raw.items()
    }


def fixed_loader(
    dataset: RawTeamSignalDataset,
    split: Mapping[str, str],
    name: str,
) -> DataLoader:
    requests = fixed_requests(dataset.episodes, split, name)
    batches = [requests[index : index + 192] for index in range(0, len(requests), 192)]
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate_fair(
    backbones: FrozenBeliefBackbones,
    probes: ActionGroundedProbeSet,
    loader: DataLoader,
    dataset: RawTeamSignalDataset,
    group_by_key: Mapping[str, str],
    device: torch.device,
    *,
    include_samples: bool = False,
) -> dict:
    backbones.eval()
    probes.eval()
    scores = {condition: [] for condition in ACTION_GROUNDED_CONDITIONS}
    tasks = {
        condition: {task: [] for task in range(6)} for condition in ACTION_GROUNDED_CONDITIONS
    }
    groups = {condition: {} for condition in ACTION_GROUNDED_CONDITIONS}
    sample_rows: list[dict] = []
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            frozen = backbones(batch)
            output = predictions(probes, frozen, batch)
        batch_scores = {
            condition: action_sample_mse(prediction.float(), batch)
            for condition, prediction in output.items()
        }
        labels = batch["episode_label"].detach().cpu().tolist()
        task_indices = batch["task_index"].detach().cpu().tolist()
        sample_keys = list(raw["sample_key"])
        scenario_groups = [
            group_by_key[dataset.episodes[int(label)].episode_key] for label in labels
        ]
        cpu_scores = {
            condition: value.detach().float().cpu().tolist()
            for condition, value in batch_scores.items()
        }
        for condition in ACTION_GROUNDED_CONDITIONS:
            for index, score in enumerate(cpu_scores[condition]):
                value = float(score)
                scores[condition].append(value)
                tasks[condition][int(task_indices[index])].append(value)
                groups[condition].setdefault(scenario_groups[index], []).append(value)
        if include_samples:
            for index, sample_key in enumerate(sample_keys):
                sample_rows.append(
                    {
                        "sample_key": sample_key,
                        "episode_label": int(labels[index]),
                        "task_index": int(task_indices[index]),
                        "scenario_group": scenario_groups[index],
                        "scores": {
                            condition: cpu_scores[condition][index]
                            for condition in ACTION_GROUNDED_CONDITIONS
                        },
                    }
                )
    payload = {
        "macro": {
            condition: float(np.mean(values)) for condition, values in scores.items()
        },
        "per_task": {
            condition: {
                str(task): float(np.mean(values))
                for task, values in by_task.items()
            }
            for condition, by_task in tasks.items()
        },
        "per_scenario_group": {
            condition: {
                group: float(np.mean(values)) for group, values in sorted(by_group.items())
            }
            for condition, by_group in groups.items()
        },
        "rows": len(next(iter(scores.values()))),
    }
    if include_samples:
        payload["samples"] = sample_rows
    return payload


def main() -> None:
    args = parse_args()
    contract_raw = args.contract.read_bytes()
    contract = json.loads(contract_raw)
    if (
        contract.get("status") != "FROZEN_BEFORE_F0_F1"
        or contract.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF"
        or args.seed not in BELIEF_SEEDS
    ):
        raise RuntimeError("invalid or unfrozen R1 contract")
    split_payload = load_split(args.scenario_split)
    if sha256_file(args.scenario_split) != contract["scenario_split"]["sha256"]:
        raise RuntimeError("R1 scenario split hash differs")
    split = split_by_episode_key(split_payload)
    group_by_key = {
        row["episode_key"]: row["scenario_group"] for row in split_payload["episodes"]
    }
    paths = contract["old_n1_read_only"]["representation_checkpoints"][str(args.seed)]
    signal_checkpoint = Path(paths["path"])
    temporal_checkpoint = Path(contract["b0h"]["checkpoint"])
    if sha256_file(signal_checkpoint) != paths["sha256"]:
        raise RuntimeError("frozen N1 checkpoint hash differs")
    if sha256_file(temporal_checkpoint) != contract["b0h"]["checkpoint_sha256"]:
        raise RuntimeError("frozen B0-H checkpoint hash differs")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = RawTeamSignalDataset(args.cache)
    backbones = FrozenBeliefBackbones(
        temporal_checkpoint=temporal_checkpoint,
        signal_checkpoint=signal_checkpoint,
        visual_mean=dataset.visual_mean,
        visual_std=dataset.visual_std,
    ).to(device)
    probes = ActionGroundedProbeSet().to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 if step < BELIEF_LR_DROP else 0.1
    )

    latest = args.output / "checkpoint_latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.is_file() else None
    start_update = 0
    metrics: list[dict] = []
    if saved is not None:
        provenance = saved.get("provenance", {})
        expected = {
            "seed": args.seed,
            "contract_sha256": hashlib_sha256(contract_raw),
            "scenario_split_sha256": sha256_file(args.scenario_split),
            "n1_checkpoint_sha256": paths["sha256"],
            "b0h_checkpoint_sha256": contract["b0h"]["checkpoint_sha256"],
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise RuntimeError(f"R1 resume provenance differs at {key}")
        probes.load_state_dict(saved["probes"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_update = int(saved["update"])
        metrics = list(saved["evaluations"])

    sampler = ActionGroundedBatchSampler(
        dataset.episodes,
        split,
        updates=BELIEF_MAX_UPDATES,
        data_seed=BELIEF_DATA_SEED,
        start_update=start_update,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    validation = fixed_loader(dataset, split, "validation")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "status.json",
        {
            "status": "TRAINING",
            "seed": args.seed,
            "update": start_update,
            "started_at_utc": utc_now(),
            "parameter_counts": probes.parameter_counts(),
        },
    )
    started = time.time()
    completion: str | None = None
    for update, raw in enumerate(loader, start=start_update + 1):
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        batch = device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            frozen = backbones(batch)
            output = predictions(probes, frozen, batch)
            losses = {
                condition: action_sample_mse(prediction.float(), batch).mean()
                for condition, prediction in output.items()
            }
            loss = torch.stack(list(losses.values())).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite R1-1 loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if update % 100 == 0:
            atomic_json(
                args.output / "heartbeat.json",
                {
                    "status": "TRAINING",
                    "seed": args.seed,
                    "update": update,
                    "loss": float(loss),
                    "updated_at_epoch": time.time(),
                },
            )
        if update % BELIEF_EVAL_EVERY:
            continue
        validation_metrics = evaluate_fair(
            backbones,
            probes,
            validation,
            dataset,
            group_by_key,
            device,
        )
        row = {
            "update": update,
            "train": {key: float(value.detach()) for key, value in losses.items()},
            "validation": validation_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        metrics.append(row)
        with (args.output / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        payload = {
            "format_version": "before-we-act.b3-n1-r1-fair-probe-checkpoint/1",
            "probes": probes.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update": update,
            "evaluations": metrics,
            "sample_cursor": sampler.cursor_receipt(update),
            "provenance": {
                "seed": args.seed,
                "contract_sha256": hashlib_sha256(contract_raw),
                "scenario_split_sha256": sha256_file(args.scenario_split),
                "n1_checkpoint_sha256": paths["sha256"],
                "b0h_checkpoint_sha256": contract["b0h"]["checkpoint_sha256"],
            },
        }
        atomic_save(payload, latest)
        atomic_save(payload, args.output / f"checkpoint_{update:06d}.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
        if all_conditions_platform(metrics):
            completion = "PLATFORM_REACHED"
            break
        if update >= BELIEF_MIN_UPDATES and len(metrics) >= 4:
            main_scores = [
                float(item["validation"]["macro"]["h_b"]) for item in metrics[-4:]
            ]
            if all(later > earlier for earlier, later in zip(main_scores, main_scores[1:])):
                completion = "SATURATED_BY_OVERFIT"
                break
        if update >= BELIEF_MAX_UPDATES:
            completion = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
            break
    if completion is None:
        completion = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
    selected = min(metrics, key=lambda item: item["validation"]["macro"]["h_b"])
    atomic_json(
        args.output / "status.json",
        {
            "status": completion,
            "seed": args.seed,
            "update": int(metrics[-1]["update"]),
            "selected_update": int(selected["update"]),
            "selected_validation": selected["validation"],
            "condition_platform": {
                condition: condition_platform(metrics, condition)
                for condition in ACTION_GROUNDED_CONDITIONS
            },
            "parameter_counts": probes.parameter_counts(),
            "elapsed_hours": (time.time() - started) / 3600,
            "completed_at_utc": utc_now(),
        },
    )


def hashlib_sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()
