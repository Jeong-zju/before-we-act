"""Closed-loop Validation20 for the unchanged CARE selector on MARS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from before_we_act.mars_action_contract import (
    canonicalize_action,
    validate_action_space_bounds,
)
from before_we_act.care_belief import CAREBeliefConfig, CAREBeliefHead, CARECalibration, select_care_candidate
from before_we_act.care_branch_collector import candidate_plan, canonicalize_policy_plans, validate_candidate
from before_we_act.care_training_data import sha256_file
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
from deployment.mars_care.common import TASK_BY_NAME


def load_care(path: Path, device: torch.device, reference: Path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "before-we-act.care-mars-deployment-checkpoint/1":
        raise ValueError("wrong MARS CARE deployment checkpoint")
    if saved.get("reference_checkpoint_sha256") != sha256_file(reference):
        raise ValueError("MARS CARE/reference checkpoint hash mismatch")
    model = CAREBeliefHead(CAREBeliefConfig.from_mapping(saved["config"])).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, CARECalibration.from_mapping(saved["calibration"]), saved


@torch.inference_mode()
def run_episode(reference_model: Any, scorer: CAREBeliefHead, calibration: CARECalibration, stats: dict[str, torch.Tensor], task: str, root: Path, seed: int, device: torch.device, max_steps: int, selector_enabled: bool, render_device: str | None = None, recorder: MarsCARERolloutRecorder | None = None) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env = environment(task, root, render_device or str(device))
    observation, _ = env.reset(seed=seed)
    arms = tuple(range(TASK_BY_NAME[task].arms))
    from before_we_act.mars_care_runtime import new_runtime

    runtime = new_runtime(arms)
    success = False
    overrides = fallbacks = safety_rejections = 0
    candidate_counts: dict[str, int] = {}
    trace = hashlib.sha256()
    inference: list[float] = []
    diagnostics: Mapping[str, Any] = {}
    last_info: Mapping[str, Any] = {}
    last_physical: Mapping[str, Any] = {}
    completed_normally = False
    if recorder is not None:
        recorder.start(
            observation,
            metadata={
                "selector_mode": "care" if selector_enabled else "selector_off",
                "max_steps": int(max_steps),
                "reference_policy": "B-core/TUNE",
                "runtime_contract": "legacy_single_focal",
                "legacy_selector_legality_is_post_selection": True,
            },
        )
    try:
        for step in range(max_steps):
            started = time.perf_counter()
            observation_before = observation
            qpos_before = current_qpos(observation, arms)
            reference_plans, base_plans, qpos, memory, memory_mask, diagnostics = policy_plan(
                reference_model, stats, observation, runtime, task, device
            )
            reference_plans, base_plans, _ = canonicalize_policy_plans(
                reference_plans, base_plans, env.action_space.spaces, arms
            )
            candidates: list[np.ndarray] = []
            valids: list[list[bool]] = []
            for index, arm in enumerate(arms):
                key = f"panda-{arm}"
                current_grip = (
                    float(runtime.last_action[key][7])
                    if runtime.last_action is not None
                    else float(reference_plans[key][0, 7])
                )
                rows: list[np.ndarray] = []
                valid_rows: list[bool] = []
                for candidate_id in range(6):
                    plan = candidate_plan(candidate_id, reference_plans[key], base_plans[key], qpos[index], current_grip)
                    valid, _ = validate_candidate(candidate_id, plan, reference_plans[key], base_plans[key], qpos[index], current_grip, env.action_space.spaces[key])
                    rows.append(plan)
                    valid_rows.append(valid)
                candidates.append(np.stack(rows))
                valids.append(valid_rows)
            candidate_tensor = torch.as_tensor(np.stack(candidates), device=device)
            horizon = torch.full((len(arms),), scorer.config.horizons.index(calibration.primary_horizon), dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = scorer(memory, memory_mask, candidate_tensor, horizon)
            selected_tensor, lower_tensor, unsafe_tensor = select_care_candidate(output, calibration, variant=scorer.config.variant)
            selected = [int(value) for value in selected_tensor.cpu().tolist()]
            lower = [float(value) for value in lower_tensor.float().cpu().tolist()]
            unsafe = unsafe_tensor.cpu().numpy()
            component = 0 if scorer.config.variant == "replay_only" else 2
            masked_lower = output.quantiles[:, :, component, 0].float() - calibration.lower_correction
            masked_lower = masked_lower.masked_fill(unsafe_tensor, -torch.inf)
            masked_lower[:, 0] = 0.0
            safety_rejections += int(unsafe[:, 1:].sum())
            action = {key: reference_plans[key][0].copy() for key in reference_plans}
            proposed = [index for index, candidate_id in enumerate(selected) if selector_enabled and candidate_id != 0]
            focal = max(proposed, key=lambda index: lower[index]) if proposed else None
            if selector_enabled and focal is None:
                fallbacks += 1
            if focal is not None:
                candidate_id = int(selected[focal])
                if not valids[focal][candidate_id]:
                    fallbacks += 1
                else:
                    arm = arms[focal]
                    action[f"panda-{arm}"] = candidates[focal][candidate_id, 0].copy()
                    overrides += 1
                    candidate_counts[str(candidate_id)] = candidate_counts.get(str(candidate_id), 0) + 1
            applied_rows = [] if focal is None or not valids[focal][int(selected[focal])] else [focal]
            reason_names = []
            for row, candidate_id in enumerate(selected):
                if not selector_enabled:
                    reason_names.append("reference_selector_disabled")
                elif candidate_id == 0:
                    reason_names.append("reference_below_delta")
                elif row in applied_rows:
                    reason_names.append("override")
                elif not valids[row][candidate_id]:
                    reason_names.append("reference_illegal_post_selection")
                else:
                    reason_names.append("reference_central_arbitration_suppressed")
            action_before_canonicalize = {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in action.items()
            }
            for arm in arms:
                row = np.asarray(action[f"panda-{arm}"], dtype=np.float32)
                space = env.action_space.spaces[f"panda-{arm}"]
                validate_action_space_bounds(space)
                row = canonicalize_action(row)
                action[f"panda-{arm}"] = row
                trace.update(row.tobytes())
            append_action(runtime, action, stats)
            inference.append(time.perf_counter() - started)
            observation, _reward, terminated, truncated, info = env.step(action)
            physical = privileged_task_metrics(
                getattr(env, "base_env", env.unwrapped), task, action, qpos_before
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
                    candidate_legality=valids,
                    selected=selected,
                    masked_lower=masked_lower,
                    best_lower=lower,
                    reason_names=reason_names,
                    illegal=~torch.as_tensor(valids, dtype=torch.bool),
                    learned_unsafe=unsafe_tensor,
                    assembly={
                        "proposed_rows": proposed,
                        "applied_rows": applied_rows,
                        "central_arbitration_suppressions": max(len(proposed) - len(applied_rows), 0),
                        "strict_decentralized": not selector_enabled,
                        "contract": "legacy_single_focal",
                    },
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
            )
        env.close()
    return {
        "task": task,
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "overrides": overrides,
        "override_rate": overrides / max(step + 1, 1),
        "fallbacks": fallbacks,
        "safety_rejections": safety_rejections,
        "candidate_counts": candidate_counts,
        "mean_inference_seconds": float(np.mean(inference)),
        "p95_inference_seconds": float(np.quantile(inference, 0.95)),
        "action_trace_sha256": trace.hexdigest(),
        "belief_diagnostics": diagnostics,
        "final_physical_metrics": dict(last_physical),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--care-checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_BY_NAME), required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20260827)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--mode", choices=("selector_off", "care"), default="care")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-device")
    parser.add_argument(
        "--record-root",
        type=Path,
        help="optional diagnostic-only root; writes per-seed local RGB MP4 and telemetry",
    )
    parser.add_argument("--record-fps", type=float, default=20.0)
    parser.add_argument("--no-record-candidate-plans", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    reference_model, stats, _config = load_reference(args.reference_checkpoint, device)
    scorer, calibration, saved = load_care(args.care_checkpoint, device, args.reference_checkpoint)
    rows: list[dict[str, Any]] = []
    log_path = args.output.with_suffix(".jsonl")
    recovered: dict[int, dict[str, Any]] = {}
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
                recovered[int(row["seed"])] = row
            except Exception:
                continue
    task = TASK_BY_NAME[args.task]
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        recorder = None
        if args.record_root is not None and seed not in recovered:
            recorder = MarsCARERolloutRecorder(
                args.record_root / args.task / args.mode / f"seed_{seed}",
                task=args.task,
                seed=seed,
                arms=range(task.arms),
                fps=args.record_fps,
                record_candidate_plans=not args.no_record_candidate_plans,
            )
        row = recovered.get(seed) or run_episode(reference_model, scorer, calibration, stats, args.task, args.robofactory_root, seed, device, args.max_steps or task.max_steps, args.mode == "care", args.render_device, recorder)
        rows.append(row)
        if seed not in recovered:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    result = {
        "format_version": "before-we-act.care-mars-validation20-task/1",
        "status": "complete",
        "policy": "official_care_mars_bench_port",
        "strict_local": True,
        "task": args.task,
        "mode": args.mode,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "rows": rows,
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "care_checkpoint": str(args.care_checkpoint.resolve()),
        "care_selected_seed": int(saved["selected_seed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
