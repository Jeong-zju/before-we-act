"""Train one four-phase deployment-legal belief student and direct control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.raw_team_signal_model import masked_mse
from before_we_act.action_grounded_belief import (
    FrozenBeliefBackbones,
    ActionGroundedBatchSampler,
    ActionGroundedDataset,
    BELIEF_DATA_SEED,
    BELIEF_EVAL_EVERY,
    BELIEF_SEEDS,
    action_sample_mse,
    deterministic_permutations,
    load_split,
    split_by_episode_key,
)
from before_we_act.belief_distillation import (
    DirectReactiveControl,
    LegalBeliefStudent,
    PrivilegedBeliefTeacher,
    gaussian_nll,
    oracle_fields,
)
from before_we_act.train_action_grounded_probe import atomic_json, atomic_save, device_batch
from before_we_act.train_belief_teacher import fixed_loader, load_base_probe
from before_we_act.temporal_history_data import sha256_file


MAX_UPDATES = 80_000
CONDITIONS = (
    "h",
    "h_student",
    "h_student_shuffle",
    "h_teacher",
    "direct_reactive",
    "belief_off",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--student-contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--fair-run-root", type=Path, required=True)
    parser.add_argument("--teacher-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def phase_for_update(update: int) -> int:
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError(update)
    return min(4, (update - 1) // 20_000 + 1)


def configure_phase(
    student: LegalBeliefStudent,
    direct: DirectReactiveControl,
    phase: int,
) -> tuple[list[torch.nn.Parameter], float]:
    student.requires_grad_(False)
    direct.requires_grad_(False)
    if phase in (1, 2):
        for module in (student.reader, student.token_norm, student.logvar):
            module.requires_grad_(True)
        student.queries.requires_grad_(True)
        if phase == 2:
            for module in (
                student.teammate_action,
                student.teammate_delta,
                student.future_visual,
            ):
                module.requires_grad_(True)
    elif phase == 3:
        student.action_residual.requires_grad_(True)
        direct.requires_grad_(True)
    elif phase == 4:
        student.queries.requires_grad_(True)
        for module in (student.reader, student.token_norm, student.action_residual):
            module.requires_grad_(True)
        direct.requires_grad_(True)
    else:
        raise ValueError(phase)
    parameters = [
        parameter
        for model in (student, direct)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return parameters, (2e-5 if phase == 4 else 2e-4)


def set_train_modes(
    student: LegalBeliefStudent,
    direct: DirectReactiveControl,
    phase: int,
) -> None:
    student.eval()
    direct.eval()
    if phase in (1, 2, 4):
        student.train()
    elif phase == 3:
        student.action_residual.train()
    if phase >= 3:
        direct.train()


def load_teacher(
    run_root: Path, seed: int, device: torch.device
) -> tuple[PrivilegedBeliefTeacher, int, str]:
    root = run_root / "r1_4_teacher" / f"seed_{seed}"
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    update = int(status["selected_update"])
    checkpoint = root / f"checkpoint_{update:06d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher = PrivilegedBeliefTeacher().to(device)
    teacher.load_state_dict(payload["teacher"], strict=True)
    teacher.eval().requires_grad_(False)
    return teacher, update, sha256_file(checkpoint)


def future_visual_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    weight = mask[:, :, None, None].to(prediction.dtype)
    return ((prediction - target).square() * weight).sum() / weight.expand_as(
        prediction
    ).sum().clamp_min(1)


def token_distillation_loss(
    tokens: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    return gaussian_nll(tokens, logvar, target, mask)


@torch.no_grad()
def evaluate(
    backbones: FrozenBeliefBackbones,
    base_probe: torch.nn.Module,
    teacher: PrivilegedBeliefTeacher,
    student: LegalBeliefStudent,
    direct: DirectReactiveControl,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    student.eval()
    direct.eval()
    values = {name: [] for name in CONDITIONS}
    tasks = {name: {task: [] for task in range(6)} for name in CONDITIONS}
    token_losses: list[float] = []
    teammate_losses: list[float] = []
    delta_losses: list[float] = []
    future_losses: list[float] = []
    belief_off_max_abs = 0.0
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            frozen = backbones(batch)
            if frozen.history is None:
                raise RuntimeError("B0-H history tokens are absent")
            base = base_probe(frozen.h)
            history_mask = batch["history_mask"] | batch["action_history_mask"]
            teacher_output = teacher(base, frozen.h, oracle_fields(batch))
            student_output = student(
                base, frozen.h, frozen.history, history_mask
            )
            direct_action = direct(
                base, frozen.h, frozen.history, history_mask
            )
            permutation = deterministic_permutations(batch)["shuffle"]
            shuffled_action = student.action_residual(
                base, frozen.h, student_output.tokens[permutation]
            )
            off = student.belief_off(base)
        predictions = {
            "h": base,
            "h_student": student_output.action,
            "h_student_shuffle": shuffled_action,
            "h_teacher": teacher_output.action,
            "direct_reactive": direct_action,
            "belief_off": off,
        }
        belief_off_max_abs = max(
            belief_off_max_abs, float((off.float() - base.float()).abs().max().cpu())
        )
        scores = {
            name: action_sample_mse(value.float(), batch)
            for name, value in predictions.items()
        }
        for name in CONDITIONS:
            rows = scores[name].cpu().tolist()
            values[name].extend(rows)
            for task in range(6):
                tasks[name][task].extend(
                    scores[name][batch["task_index"] == task].cpu().tolist()
                )
        token_losses.append(
            float(
                token_distillation_loss(
                    student_output.tokens.float(),
                    student_output.token_logvar.float(),
                    teacher_output.tokens.float(),
                )
            )
        )
        teammate_losses.append(
            float(
                gaussian_nll(
                    student_output.teammate_action_mean.float(),
                    student_output.teammate_action_logvar.float(),
                    batch["oracle_teammate_action"],
                    batch["oracle_teammate_action_mask"],
                )
            )
        )
        delta_losses.append(
            float(
                masked_mse(
                    student_output.teammate_delta.float(),
                    batch["teammate_delta"],
                    batch["future_mask"],
                )
            )
        )
        future_losses.append(
            float(
                future_visual_loss(
                    student_output.future_visual.float(),
                    batch["future_visual"],
                    batch["future_mask"],
                )
            )
        )
    return {
        "macro": {name: float(np.mean(rows)) for name, rows in values.items()},
        "per_task": {
            name: {
                str(task): float(np.mean(rows)) for task, rows in by_task.items()
            }
            for name, by_task in tasks.items()
        },
        "teacher_token_gaussian_nll": float(np.mean(token_losses)),
        "teammate_action_gaussian_nll": float(np.mean(teammate_losses)),
        "teammate_delta_mse": float(np.mean(delta_losses)),
        "future_visual_mse": float(np.mean(future_losses)),
        "belief_off_max_abs": belief_off_max_abs,
        "rows": len(values["h"]),
    }


def final_platform(metrics: list[dict]) -> tuple[bool, dict]:
    phase4 = [row for row in metrics if int(row["update"]) >= 65_000]
    receipt = {"points": [int(row["update"]) for row in phase4[-4:]], "series": {}}
    if [int(row["update"]) for row in phase4[-4:]] != [65_000, 70_000, 75_000, 80_000]:
        return False, receipt
    passed = True
    for key in ("h_student", "direct_reactive"):
        scores = [float(row["validation"]["macro"][key]) for row in phase4[-4:]]
        improvements = [
            (first - second) / max(abs(first), 1e-12)
            for first, second in zip(scores, scores[1:])
        ]
        receipt["series"][key] = {
            "scores": scores,
            "relative_improvements": improvements,
            "all_three_below_one_percent": all(value < 0.01 for value in improvements),
        }
        passed &= receipt["series"][key]["all_three_below_one_percent"]
    return passed, receipt


def main() -> None:
    args = parse_args()
    parent = json.loads(args.parent_contract.read_text(encoding="utf-8"))
    contract = json.loads(args.student_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_F0_F1" or args.seed not in BELIEF_SEEDS:
        raise RuntimeError("invalid student contract")
    if contract.get("format_version") != "before-we-act.b3-n1-r1-student-contract/2":
        raise RuntimeError("student requires the owner-revised contract")
    if sha256_file(args.parent_contract) != contract["parent_contract_sha256"]:
        raise RuntimeError("student parent contract hash differs")
    split_payload = load_split(args.scenario_split)
    split = split_by_episode_key(split_payload)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = ActionGroundedDataset(args.cache)
    signal_checkpoint = parent["old_n1_read_only"]["representation_checkpoints"][str(args.seed)]
    backbones = FrozenBeliefBackbones(
        temporal_checkpoint=Path(parent["b0h"]["checkpoint"]),
        signal_checkpoint=Path(signal_checkpoint["path"]),
        visual_mean=dataset.visual_mean,
        visual_std=dataset.visual_std,
    ).to(device)
    base_probe, base_update, base_sha = load_base_probe(
        args.fair_run_root, args.seed, device
    )
    teacher, teacher_update, teacher_sha = load_teacher(
        args.teacher_run_root, args.seed, device
    )
    student = LegalBeliefStudent().to(device)
    direct = DirectReactiveControl().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "checkpoint_latest.pt"
    saved = (
        torch.load(latest, map_location="cpu", weights_only=False)
        if latest.is_file()
        else None
    )
    start = 0
    metrics: list[dict] = []
    provenance = {
        "seed": args.seed,
        "student_contract_sha256": sha256_file(args.student_contract),
        "scenario_split_sha256": sha256_file(args.scenario_split),
        "base_h_checkpoint_sha256": base_sha,
        "teacher_checkpoint_sha256": teacher_sha,
    }
    if saved:
        if saved["provenance"] != provenance:
            raise RuntimeError("student resume provenance differs")
        student.load_state_dict(saved["student"], strict=True)
        direct.load_state_dict(saved["direct"], strict=True)
        start = int(saved["update"])
        metrics = list(saved["evaluations"])

    active_phase = phase_for_update(min(start + 1, MAX_UPDATES))
    parameters, learning_rate = configure_phase(student, direct, active_phase)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    if saved and int(saved["optimizer_phase"]) == active_phase:
        optimizer.load_state_dict(saved["optimizer"])
    sampler = ActionGroundedBatchSampler(
        dataset.episodes,
        split,
        updates=MAX_UPDATES,
        data_seed=BELIEF_DATA_SEED,
        start_update=start,
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
    atomic_json(
        args.output / "status.json",
        {
            "status": "TRAINING",
            "seed": args.seed,
            "update": start,
            "phase": active_phase,
            "started_at_utc": utc_now(),
        },
    )
    started = time.time()
    weights = contract["objectives"]
    last_losses: dict[str, float] = {}
    for update, raw in enumerate(loader, start=start + 1):
        phase = phase_for_update(update)
        if phase != active_phase:
            active_phase = phase
            parameters, learning_rate = configure_phase(student, direct, phase)
            optimizer = torch.optim.AdamW(
                parameters, lr=learning_rate, weight_decay=1e-4
            )
        set_train_modes(student, direct, phase)
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed)
        np.random.seed(step_seed % 2**32)
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        batch = device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            frozen = backbones(batch)
            if frozen.history is None:
                raise RuntimeError("B0-H history tokens are absent")
            base = base_probe(frozen.h)
            teacher_output = (
                teacher(base, frozen.h, oracle_fields(batch)) if phase <= 2 else None
            )
        history_mask = batch["history_mask"] | batch["action_history_mask"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_output = student(
                base, frozen.h, frozen.history, history_mask
            )
            if phase == 1:
                assert teacher_output is not None
                alignment = token_distillation_loss(
                    student_output.tokens.float(),
                    student_output.token_logvar.float(),
                    teacher_output.tokens.float(),
                )
                loss = weights["phase1_teacher_token_gaussian_nll"] * alignment
                last_losses = {"teacher_token_gaussian_nll": float(alignment.detach())}
            elif phase == 2:
                assert teacher_output is not None
                alignment = token_distillation_loss(
                    student_output.tokens.float(),
                    student_output.token_logvar.float(),
                    teacher_output.tokens.float(),
                )
                teammate = gaussian_nll(
                    student_output.teammate_action_mean.float(),
                    student_output.teammate_action_logvar.float(),
                    batch["oracle_teammate_action"],
                    batch["oracle_teammate_action_mask"],
                )
                delta = masked_mse(
                    student_output.teammate_delta.float(),
                    batch["teammate_delta"],
                    batch["future_mask"],
                )
                future = future_visual_loss(
                    student_output.future_visual.float(),
                    batch["future_visual"],
                    batch["future_mask"],
                )
                loss = (
                    weights["phase2_teacher_token_gaussian_nll"] * alignment
                    + weights["phase2_teammate_action_gaussian_nll"] * teammate
                    + weights["phase2_teammate_delta"] * delta
                    + weights["phase2_future_dino_low_weight_auxiliary"] * future
                )
                last_losses = {
                    "teacher_token_gaussian_nll": float(alignment.detach()),
                    "teammate_action_gaussian_nll": float(teammate.detach()),
                    "teammate_delta_mse": float(delta.detach()),
                    "future_visual_mse": float(future.detach()),
                }
            else:
                direct_action = direct(
                    base, frozen.h, frozen.history, history_mask
                )
                student_action = action_sample_mse(
                    student_output.action.float(), batch
                ).mean()
                direct_action_loss = action_sample_mse(
                    direct_action.float(), batch
                ).mean()
                loss = weights["phase3_4_ego_action"] * (
                    student_action + direct_action_loss
                ) / 2
                last_losses = {
                    "student_action_mse": float(student_action.detach()),
                    "direct_action_mse": float(direct_action_loss.detach()),
                }
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite student loss at {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if update % 100 == 0:
            atomic_json(
                args.output / "heartbeat.json",
                {
                    "status": "TRAINING",
                    "seed": args.seed,
                    "update": update,
                    "phase": phase,
                    "loss": float(loss.detach()),
                    "updated_at_epoch": time.time(),
                },
            )
        if update % BELIEF_EVAL_EVERY:
            continue
        validation_metrics = evaluate(
            backbones,
            base_probe,
            teacher,
            student,
            direct,
            validation,
            device,
        )
        row = {
            "update": update,
            "phase": phase,
            "train": {"total": float(loss.detach()), **last_losses},
            "validation": validation_metrics,
            "learning_rate": learning_rate,
        }
        metrics.append(row)
        with (args.output / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        checkpoint = {
            "format_version": "before-we-act.b3-n1-r1-student-checkpoint/1",
            "student": student.state_dict(),
            "direct": direct.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_phase": phase,
            "update": update,
            "evaluations": metrics,
            "sample_cursor": sampler.cursor_receipt(update),
            "provenance": provenance,
            "base_h_selected_update": base_update,
            "teacher_selected_update": teacher_update,
        }
        atomic_save(checkpoint, latest)
        atomic_save(checkpoint, args.output / f"checkpoint_{update:06d}.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    reached, platform_receipt = final_platform(metrics)
    phase4_metrics = [row for row in metrics if int(row["update"]) >= 65_000]
    selected = min(
        phase4_metrics, key=lambda row: row["validation"]["macro"]["h_student"]
    )
    status = "PLATFORM_REACHED" if reached else "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
    training_sufficiency = {
        "format_version": "before-we-act.training-sufficiency/1",
        "status": status,
        "minimum_exposure_met": True,
        "mandatory_four_phase_schedule_completed": True,
        "validation_every": BELIEF_EVAL_EVERY,
        "maximum_updates": MAX_UPDATES,
        "selected_update": int(selected["update"]),
        "phase4_platform": platform_receipt,
    }
    atomic_json(args.output / "training_sufficiency.json", training_sufficiency)
    atomic_json(
        args.output / "status.json",
        {
            "status": status,
            "seed": args.seed,
            "update": MAX_UPDATES,
            "selected_update": int(selected["update"]),
            "selected_validation": selected["validation"],
            "training_sufficiency_sha256": sha256_file(
                args.output / "training_sufficiency.json"
            ),
            "elapsed_hours": (time.time() - started) / 3600,
            "completed_at_utc": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
