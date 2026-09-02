"""Isolated CARE-v2 paired smoke and Validation20 runtime for MARS.

This evaluator never loads a v1 scorer.  It enforces legality before selection,
uses the task's physical utility scale/correction, and exposes two distinct
coordination contracts:

``decentralized``
    Every arm independently applies its own local CARE decision.  This is the
    deployment contract requested for the shared decentralized policy.

``single_focal``
    At most one proposed arm is applied, matching the one-focal branch corpus.
    It is retained only as a paired smoke ablation because cross-arm arbitration
    is centralized.

``selector_off``
    Exact reference-policy control for paired comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.mars_action_contract import (
    canonicalize_action,
    validate_action_space_bounds,
)
from before_we_act.care_branch_collector import (
    candidate_plan,
    canonicalize_policy_plans,
    validate_candidate,
)
from before_we_act.mars_care_runtime import (
    append_action,
    current_qpos,
    environment,
    load_reference,
    policy_plan,
    privileged_task_metrics,
    scalar,
)
from before_we_act.mars_care_recorder import MarsCARERolloutRecorder
from before_we_act.care_chunk_commitment import (
    advance_chunk_commitments,
    apply_chunk_commitments,
)
from before_we_act.mars_care_v2_deployment import load_mars_care_v2
from before_we_act.mars_care_v3_deployment import load_mars_care_v3
from deployment.mars_care.common import TASK_BY_NAME


SELECTOR_MODES = ("selector_off", "single_focal", "decentralized")


def assemble_actions_v2(
    reference_plans: Mapping[str, np.ndarray],
    candidates: Sequence[np.ndarray],
    selected: Sequence[int],
    best_lower: Sequence[float],
    arms: Sequence[int],
    *,
    mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map row-local decisions to physical actions under an explicit contract."""

    if mode not in SELECTOR_MODES:
        raise ValueError(f"unsupported CARE v2 selector mode: {mode}")
    arms = tuple(int(value) for value in arms)
    if not (len(candidates) == len(selected) == len(best_lower) == len(arms)):
        raise ValueError("CARE v2 action assembly row count differs")
    action = {
        f"panda-{arm}": np.asarray(reference_plans[f"panda-{arm}"][0]).copy()
        for arm in arms
    }
    proposed: list[int] = []
    for row, candidate_id in enumerate(selected):
        candidate_id = int(candidate_id)
        if not 0 <= candidate_id < int(candidates[row].shape[0]):
            raise ValueError("CARE v2 selected candidate is out of range")
        if candidate_id != 0:
            proposed.append(row)

    if mode == "selector_off":
        applied: list[int] = []
    elif mode == "single_focal":
        # This deliberate ablation reproduces the old centralized arbitration.
        applied = (
            [max(proposed, key=lambda row: (float(best_lower[row]), -row))]
            if proposed
            else []
        )
    else:
        # No score or decision from another arm is consulted here.
        applied = list(proposed)
    for row in applied:
        arm = arms[row]
        candidate_id = int(selected[row])
        value = np.asarray(candidates[row][candidate_id, 0], dtype=np.float32)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise ValueError("CARE v2 selected physical action is invalid")
        action[f"panda-{arm}"] = value.copy()
    return action, {
        "proposed_rows": proposed,
        "applied_rows": applied,
        "simultaneous_proposals": len(proposed),
        "simultaneous_overrides": len(applied),
        "central_arbitration_suppressions": len(proposed) - len(applied),
        "strict_decentralized": mode in {"selector_off", "decentralized"},
    }


@torch.inference_mode()
def run_episode(
    reference_model: Any,
    scorer: Any,
    stats: Mapping[str, torch.Tensor],
    task: str,
    root: Path,
    seed: int,
    device: torch.device,
    max_steps: int,
    mode: str,
    render_device: str | None = None,
    recorder: MarsCARERolloutRecorder | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env = environment(task, root, render_device or str(device))
    observation, _ = env.reset(seed=seed)
    arms = tuple(range(TASK_BY_NAME[task].arms))
    from before_we_act.mars_care_runtime import new_runtime

    policy_runtime = new_runtime(arms)
    success = False
    overrides = fallbacks = 0
    illegal_rejections = safety_rejections = 0
    simultaneous_proposal_steps = simultaneous_override_steps = 0
    simultaneous_conflict_steps = arbitration_suppressions = 0
    candidate_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    per_arm_overrides: Counter[str] = Counter()
    trace = hashlib.sha256()
    inference: list[float] = []
    diagnostics: Mapping[str, Any] = {}
    last_info: Mapping[str, Any] = {}
    last_physical: Mapping[str, Any] = {}
    completed_normally = False
    commitments: dict[int, dict[str, Any]] = {}
    committed_action_steps = 0
    override_decisions = 0
    try:
        if recorder is not None:
            recorder.start(
                observation,
                metadata={
                    "selector_mode": mode,
                    "max_steps": int(max_steps),
                    "reference_policy": "B-core/TUNE",
                    "decentralized": mode == "decentralized",
                },
            )
        for step in range(max_steps):
            started = time.perf_counter()
            observation_before = observation
            qpos_before = current_qpos(observation, arms)
            reference_plans, base_plans, qpos, memory, memory_mask, diagnostics = (
                policy_plan(
                    reference_model,
                    stats,
                    observation,
                    policy_runtime,
                    task,
                    device,
                )
            )
            reference_plans, base_plans, _ = canonicalize_policy_plans(
                reference_plans, base_plans, env.action_space.spaces, arms
            )
            candidates: list[np.ndarray] = []
            legality_rows: list[list[bool]] = []
            for row, arm in enumerate(arms):
                key = f"panda-{arm}"
                current_grip = (
                    float(policy_runtime.last_action[key][7])
                    if policy_runtime.last_action is not None
                    else float(reference_plans[key][0, 7])
                )
                plans: list[np.ndarray] = []
                legality: list[bool] = []
                for candidate_id in range(6):
                    plan = candidate_plan(
                        candidate_id,
                        reference_plans[key],
                        base_plans[key],
                        qpos[row],
                        current_grip,
                    )
                    valid, _failures = validate_candidate(
                        candidate_id,
                        plan,
                        reference_plans[key],
                        base_plans[key],
                        qpos[row],
                        current_grip,
                        env.action_space.spaces[key],
                    )
                    plans.append(plan)
                    legality.append(bool(valid))
                candidates.append(np.stack(plans))
                legality_rows.append(legality)

            candidate_tensor = torch.as_tensor(np.stack(candidates), device=device)
            legality_tensor = torch.as_tensor(
                legality_rows, dtype=torch.bool, device=device
            )
            selection = scorer.score_and_select(
                memory,
                memory_mask,
                candidate_tensor,
                legality_tensor,
                task,
                selector_enabled=mode != "selector_off",
            )
            selected = [int(value) for value in selection.selected.cpu().tolist()]
            lower = [float(value) for value in selection.best_lower.float().cpu().tolist()]
            intervention_steps = int(
                getattr(scorer.model.config, "action_prefix_steps", 1)
            )
            active_rows = apply_chunk_commitments(
                candidates,
                selected,
                lower,
                commitments,
                intervention_steps=intervention_steps,
            )
            reasons = selection.reason_names()
            reason_counts.update(reasons)
            illegal_rejections += int(selection.rejected_illegal_count.sum())
            safety_rejections += int(selection.rejected_safety_count.sum())
            action, assembly = assemble_actions_v2(
                reference_plans,
                candidates,
                selected,
                lower,
                arms,
                mode=mode,
            )
            decisions, committed = advance_chunk_commitments(
                candidates,
                selected,
                lower,
                assembly["applied_rows"],
                commitments,
                active_rows,
                intervention_steps=intervention_steps,
            )
            override_decisions += decisions
            committed_action_steps += committed
            arbitration_suppressions += int(
                assembly["central_arbitration_suppressions"]
            )
            simultaneous_proposal = assembly["simultaneous_proposals"] > 1
            simultaneous_override = assembly["simultaneous_overrides"] > 1
            simultaneous_proposal_steps += int(simultaneous_proposal)
            simultaneous_override_steps += int(simultaneous_override)
            if not assembly["applied_rows"]:
                fallbacks += 1
            for row in assembly["applied_rows"]:
                arm = arms[row]
                candidate_id = selected[row]
                overrides += 1
                candidate_counts[str(candidate_id)] += 1
                per_arm_overrides[str(arm)] += 1
            action_before_canonicalize = {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in action.items()
            }
            for arm in arms:
                key = f"panda-{arm}"
                row = np.asarray(action[key], dtype=np.float32)
                validate_action_space_bounds(env.action_space.spaces[key])
                row = canonicalize_action(row)
                action[key] = row
                trace.update(row.tobytes())
            append_action(policy_runtime, action, stats)
            inference.append(time.perf_counter() - started)
            observation, _reward, terminated, truncated, info = env.step(action)
            physical = privileged_task_metrics(
                getattr(env, "base_env", env.unwrapped),
                task,
                action,
                qpos_before,
            )
            last_info = dict(info)
            last_physical = dict(physical)
            if recorder is not None:
                qpos_after = current_qpos(observation, arms)
                qpos_normalized = {
                    f"panda-{arm}": (
                        torch.as_tensor(qpos[row], dtype=torch.float32)
                        - stats["q_mean"].detach().cpu().float()
                    ).div(stats["q_std"].detach().cpu().float()).numpy()
                    for row, arm in enumerate(arms)
                }
                recorder.record_step(
                    step=step,
                    observation_before=observation_before,
                    observation_after=observation,
                    qpos_before=qpos_before,
                    qpos_after=qpos_after,
                    qpos_normalized=qpos_normalized,
                    reference_plans=reference_plans,
                    base_plans=base_plans,
                    candidates=candidates,
                    candidate_legality=legality_rows,
                    selected=selected,
                    masked_lower=selection.masked_lower,
                    best_lower=lower,
                    reason_names=reasons,
                    illegal=selection.illegal,
                    learned_unsafe=selection.learned_unsafe,
                    assembly=assembly,
                    action_before_canonicalize=action_before_canonicalize,
                    action_applied=action,
                    action_bounds={
                        f"panda-{arm}": {
                            "low": env.action_space.spaces[f"panda-{arm}"].low,
                            "high": env.action_space.spaces[f"panda-{arm}"].high,
                        }
                        for arm in arms
                    },
                    diagnostics=diagnostics,
                    physical=physical,
                    info=info,
                )
            if simultaneous_override and bool(physical["robot_conflict"]):
                simultaneous_conflict_steps += 1
            success = scalar(info.get("success", False))
            if success or scalar(terminated) or scalar(truncated):
                break
        completed_normally = True
    except BaseException as error:
        if recorder is not None:
            recorder.abort(error=error)
        raise
    finally:
        if recorder is not None and completed_normally:
            recorder.finish(
                success=success,
                final_observation=observation,
                final_info=last_info,
                final_physical=last_physical,
                status="complete",
            )
        env.close()
    steps = step + 1
    return {
        "task": task,
        "seed": seed,
        "success": success,
        "steps": steps,
        "selector_mode": mode,
        "strict_decentralized": mode in {"selector_off", "decentralized"},
        "overrides": overrides,
        "override_rate_per_arm_step": overrides / max(steps * len(arms), 1),
        "per_arm_overrides": dict(per_arm_overrides),
        "fallback_steps": fallbacks,
        "illegal_candidate_rejections": illegal_rejections,
        "safety_candidate_rejections": safety_rejections,
        "selection_reason_counts": dict(reason_counts),
        "candidate_counts": dict(candidate_counts),
        "simultaneous_proposal_steps": simultaneous_proposal_steps,
        "simultaneous_override_steps": simultaneous_override_steps,
        "simultaneous_override_conflict_steps": simultaneous_conflict_steps,
        "simultaneous_override_conflict_rate": simultaneous_conflict_steps
        / max(simultaneous_override_steps, 1),
        "central_arbitration_suppressions": arbitration_suppressions,
        "mean_inference_seconds": float(np.mean(inference)),
        "p95_inference_seconds": float(np.quantile(inference, 0.95)),
        "action_trace_sha256": trace.hexdigest(),
        "belief_diagnostics": dict(diagnostics),
        "intervention_steps": int(
            getattr(scorer.model.config, "action_prefix_steps", 1)
        ),
        "override_decisions": int(override_decisions),
        "committed_action_steps": int(committed_action_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--care-v2-checkpoint", type=Path)
    parser.add_argument(
        "--care-v3-checkpoint",
        type=Path,
        help="optional H8 CARE-v3 checkpoint; mutually exclusive with v2",
    )
    parser.add_argument("--task", choices=tuple(TASK_BY_NAME), required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20260827)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--mode", choices=SELECTOR_MODES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-device")
    parser.add_argument(
        "--record-root",
        type=Path,
        help="optional diagnostic-only root; writes per-seed local RGB MP4 and telemetry",
    )
    parser.add_argument("--record-fps", type=float, default=20.0)
    parser.add_argument(
        "--no-record-candidate-plans",
        action="store_true",
        help="omit full candidate arrays from NPZ (first actions remain in telemetry)",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("CARE v2 evaluation requires at least one episode")
    device = torch.device(args.device)
    reference_model, stats, _config = load_reference(
        args.reference_checkpoint, device
    )
    if bool(args.care_v3_checkpoint) == bool(args.care_v2_checkpoint):
        raise ValueError("provide exactly one CARE v2/v3 checkpoint")
    scorer = (
        load_mars_care_v3(args.care_v3_checkpoint, device, args.reference_checkpoint)
        if args.care_v3_checkpoint is not None
        else load_mars_care_v2(args.care_v2_checkpoint, device, args.reference_checkpoint)
    )
    rows: list[dict[str, Any]] = []
    log_path = args.output.with_suffix(".jsonl")
    recovered: dict[int, dict[str, Any]] = {}
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("selector_mode") == args.mode:
                    recovered[int(row["seed"])] = row
            except Exception:
                continue
    task_spec = TASK_BY_NAME[args.task]
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        recorder = None
        if args.record_root is not None and seed not in recovered:
            recorder = MarsCARERolloutRecorder(
                args.record_root / args.task / args.mode / f"seed_{seed}",
                task=args.task,
                seed=seed,
                arms=range(task_spec.arms),
                fps=args.record_fps,
                record_candidate_plans=not args.no_record_candidate_plans,
            )
        row = recovered.get(seed) or run_episode(
            reference_model,
            scorer,
            stats,
            args.task,
            args.robofactory_root,
            seed,
            device,
            args.max_steps or task_spec.max_steps,
            args.mode,
            args.render_device,
            recorder,
        )
        rows.append(row)
        if seed not in recovered:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    result = {
        "format_version": "before-we-act.care-mars-validation20-task-v2/1",
        "status": "complete",
        "policy": "care_mars_bench_port_v2",
        "task": args.task,
        "mode": args.mode,
        "strict_decentralized": args.mode in {"selector_off", "decentralized"},
        "promotion_scope": scorer.saved["provenance"]["promotion_scope"],
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "rows": rows,
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "care_v2_checkpoint": (
            str(args.care_v2_checkpoint.resolve())
            if args.care_v2_checkpoint is not None
            else None
        ),
        "care_v3_checkpoint": (
            str(args.care_v3_checkpoint.resolve())
            if args.care_v3_checkpoint is not None
            else None
        ),
        "safety_gate_mode": scorer.safety_gate_mode,
        "task_max_steps": int(args.max_steps or task_spec.max_steps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
