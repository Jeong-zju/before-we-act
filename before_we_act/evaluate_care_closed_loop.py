"""Closed-loop CARE evaluator with exact B-core fallback and one focal override."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from two_three_task_manifest import TASKS, get_task

from before_we_act.care_belief import (
    CAREBeliefConfig,
    CAREBeliefHead,
    CARECalibration,
    select_care_candidate,
)
from before_we_act.care_branch_collector import (
    ConsolidatedChunkEnsembler,
    arm_is_inactive,
    candidate_plan,
    canonicalize_policy_plans,
    validate_candidate,
)
from before_we_act.deployment_safety import (
    DeploymentProgressWatchdog,
    ResidualSafetyConfig,
)
from before_we_act.frozen_settings import load_frozen_settings
from before_we_act.evaluate_predictive_team_belief import load_team_belief
from before_we_act.evaluate_temporal_history_policy import (
    EpisodeHistory,
    prepare_current,
    reset_reproducibly,
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_care(
    path: str | Path, device: torch.device, reference_checkpoint: str | Path
) -> tuple[CAREBeliefHead, CARECalibration, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "before-we-act.a6r1-care-deployment-checkpoint/1":
        raise ValueError("wrong CARE deployment checkpoint")
    if saved["reference_checkpoint_sha256"] != sha256_file(reference_checkpoint):
        raise ValueError("CARE/B-core checkpoint hash mismatch")
    config = CAREBeliefConfig.from_mapping(saved["config"])
    model = CAREBeliefHead(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, CARECalibration.from_mapping(saved["calibration"]), saved


@torch.no_grad()
def predict_reference_and_memory(
    model: Any,
    stats: Mapping[str, torch.Tensor],
    observation: Mapping[str, Any],
    arms: Sequence[int],
    history: EpisodeHistory,
    task: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    global_rgb, local_rgb, qpos = prepare_current(observation, arms, stats, device)
    temporal = history.batch(qpos, task, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(global_rgb, local_rgb, **temporal)
    history.append_observation(output.current_visual_raw, qpos)
    reference = output.prediction * stats["a_std"] + stats["a_mean"]
    base = output.base_prediction * stats["a_std"] + stats["a_mean"]
    memory = torch.cat((output.belief.mu, output.belief.event_memory), dim=1).float()
    memory_mask = torch.cat(
        (
            torch.ones(
                output.belief.mu.shape[:2],
                dtype=torch.bool,
                device=device,
            ),
            output.belief.event_mask,
        ),
        dim=1,
    )
    diagnostics = {
        "belief_gate": float(output.residual_gate.float().mean()),
        "belief_reliability": float(output.belief.reliability.float().mean()),
        "belief_sigma": float(output.belief.sigma.float().mean()),
    }
    return (
        reference.float().cpu().numpy(),
        base.float().cpu().numpy(),
        qpos,
        memory,
        memory_mask,
        diagnostics,
    )


def make_env(task: str, robofactory_root: Path, settings: Mapping[str, Any]) -> Any:
    import gymnasium as gym
    import robofactory  # noqa: F401

    specification = settings["tasks"][task]
    return gym.make(
        specification["env_id"],
        config=str(robofactory_root / specification["config"]),
        obs_mode=settings["robofactory"]["obs_mode"],
        control_mode=settings["robofactory"]["control_mode"],
        render_mode=settings["robofactory"]["render_mode"],
        reward_mode=settings["robofactory"]["reward_mode"],
        sim_backend=settings["robofactory"]["sim_backend"],
        sensor_configs=dict(settings["robofactory"]["sensor"]),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )


def scalar_done(value: Any) -> bool:
    return bool(np.asarray(value).all())


def evaluate(
    *,
    reference_checkpoint: str,
    care_checkpoint: str,
    task: str,
    seeds: Sequence[int],
    device_name: str,
    max_steps: int,
    robofactory_root: Path,
    selector_enabled: bool,
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    torch.set_num_threads(12)
    device = torch.device(device_name)
    reference, stats, reference_config = load_team_belief(reference_checkpoint, device)
    care, calibration, care_saved = load_care(care_checkpoint, device, reference_checkpoint)
    arms = tuple(int(value) for value in settings["tasks"][task]["agents"])
    env = make_env(task, robofactory_root, settings)
    rows = []
    for seed in seeds:
        observation, _ = reset_reproducibly(env, int(seed))
        history = EpisodeHistory(arms)
        reference_ensembler = ConsolidatedChunkEnsembler(arms)
        base_ensembler = ConsolidatedChunkEnsembler(arms)
        safety = ResidualSafetyConfig.from_mapping(reference_config.get("residual_safety"))
        watchdogs = {
            arm: DeploymentProgressWatchdog(safety) for arm in arms
        }
        last_action: dict[str, np.ndarray] | None = None
        success = False
        override_steps = 0
        invalid_fallbacks = 0
        uncertainty_fallbacks = 0
        safety_rejections = 0
        candidate_counts: Counter[int] = Counter()
        focal_counts: Counter[int] = Counter()
        inference_seconds: list[float] = []
        lower_bounds: list[float] = []
        reference_output_hash = hashlib.sha256()
        for step in range(max_steps):
            started = time.perf_counter()
            (
                chunks,
                base_chunks,
                normalized_qpos,
                memory,
                memory_mask,
                _diagnostics,
            ) = predict_reference_and_memory(
                reference, stats, observation, arms, history, task, device
            )
            reference_plans = reference_ensembler.append_and_plan(step, chunks)
            base_plans = base_ensembler.append_and_plan(step, base_chunks)
            current_qpos = (
                normalized_qpos * stats["q_std"] + stats["q_mean"]
            ).detach().float().cpu().numpy()
            for local_index, arm in enumerate(arms):
                key = f"panda-{arm}"
                use_base, _reason = watchdogs[arm].choose_base(
                    candidate_inactive=arm_is_inactive(
                        reference_plans[key][0],
                        current_qpos[local_index],
                        safety.progress_inactivity_l2,
                    ),
                    base_inactive=arm_is_inactive(
                        base_plans[key][0],
                        current_qpos[local_index],
                        safety.progress_inactivity_l2,
                    ),
                )
                if use_base:
                    reference_plans[key] = base_plans[key].copy()
            reference_plans, base_plans, _canonicalization = canonicalize_policy_plans(
                reference_plans, base_plans, env.action_space.spaces, arms
            )
            per_arm_candidates: list[np.ndarray] = []
            per_arm_valid: list[list[bool]] = []
            for local_index, arm in enumerate(arms):
                key = f"panda-{arm}"
                current_grip = (
                    float(last_action[key][7])
                    if last_action is not None
                    else float(reference_plans[key][0, 7])
                )
                rows_for_arm = []
                valid_for_arm = []
                for candidate_id in range(6):
                    plan = candidate_plan(
                        candidate_id,
                        reference_plans[key],
                        base_plans[key],
                        current_qpos[local_index],
                        current_grip,
                    )
                    valid, _failures = validate_candidate(
                        candidate_id,
                        plan,
                        reference_plans[key],
                        base_plans[key],
                        current_qpos[local_index],
                        current_grip,
                        env.action_space.spaces[key],
                    )
                    rows_for_arm.append(plan)
                    valid_for_arm.append(valid)
                per_arm_candidates.append(np.stack(rows_for_arm))
                per_arm_valid.append(valid_for_arm)
            candidate_tensor = torch.as_tensor(
                np.stack(per_arm_candidates), device=device
            )
            horizon_index = torch.full(
                (len(arms),),
                care.config.horizons.index(calibration.primary_horizon),
                dtype=torch.long,
                device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = care(memory, memory_mask, candidate_tensor, horizon_index)
            selected, best_lower, unsafe = select_care_candidate(
                output, calibration, variant=care.config.variant
            )
            selected = selected.cpu().tolist()
            best_lower = best_lower.float().cpu().tolist()
            unsafe = unsafe.cpu().numpy()
            safety_rejections += int(unsafe[:, 1:].sum())
            proposed = (
                [
                    index
                    for index, candidate_id in enumerate(selected)
                    if candidate_id != 0
                ]
                if selector_enabled
                else []
            )
            focal_index = max(proposed, key=lambda index: best_lower[index]) if proposed else None
            action = {
                f"panda-{arm}": reference_plans[f"panda-{arm}"][0].copy()
                for arm in arms
            }
            if focal_index is None:
                if selector_enabled:
                    uncertainty_fallbacks += 1
            else:
                candidate_id = int(selected[focal_index])
                if not per_arm_valid[focal_index][candidate_id]:
                    invalid_fallbacks += 1
                    candidate_id = 0
                if candidate_id:
                    arm = arms[focal_index]
                    action[f"panda-{arm}"] = per_arm_candidates[focal_index][candidate_id, 0].copy()
                    override_steps += 1
                    candidate_counts[candidate_id] += 1
                    focal_counts[arm] += 1
                    lower_bounds.append(float(best_lower[focal_index]))
            for arm in arms:
                reference_output_hash.update(
                    np.asarray(reference_plans[f"panda-{arm}"][0], dtype=np.float32).tobytes()
                )
            normalized_action = {
                arm: (
                    torch.as_tensor(action[f"panda-{arm}"], device=device)
                    - stats["a_mean"]
                )
                / stats["a_std"]
                for arm in arms
            }
            history.append_action(normalized_action)
            last_action = {key: value.copy() for key, value in action.items()}
            inference_seconds.append(time.perf_counter() - started)
            observation, _reward, terminated, truncated, info = env.step(action)
            success = scalar_done(info.get("success", False))
            if success or scalar_done(terminated) or scalar_done(truncated):
                break
        row = {
            "task": task,
            "mode": "care" if selector_enabled else "selector_off",
            "seed": int(seed),
            "success": success,
            "steps": step + 1,
            "override_steps": override_steps,
            "override_rate": override_steps / (step + 1),
            "uncertainty_fallback_steps": uncertainty_fallbacks,
            "invalid_candidate_fallback_steps": invalid_fallbacks,
            "predicted_safety_rejections": safety_rejections,
            "candidate_counts": {str(key): value for key, value in sorted(candidate_counts.items())},
            "focal_arm_counts": {str(key): value for key, value in sorted(focal_counts.items())},
            "mean_selected_lower_bound": float(np.mean(lower_bounds)) if lower_bounds else None,
            "mean_inference_seconds": float(np.mean(inference_seconds)),
            "p95_inference_seconds": float(np.quantile(inference_seconds, 0.95)),
            "reference_action_trace_sha256": reference_output_hash.hexdigest(),
            "care_selected_seed": int(care_saved["selected_seed"]),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--care-checkpoint", required=True)
    parser.add_argument("--mode", choices=("care", "selector_off"), default="care")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robofactory-root", type=Path, default=Path("/workspace/RoboFactory"))
    parser.add_argument("--resume-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = load_frozen_settings()
    if args.task not in settings["tasks"]:
        raise ValueError(f"task is not in frozen settings: {args.task}")
    max_steps = int(settings["tasks"][args.task]["max_steps"])
    if args.max_steps is not None and args.max_steps != max_steps:
        raise ValueError(
            f"--max-steps drifts from frozen {args.task} setting: {args.max_steps} != {max_steps}"
        )
    seed_raw = args.seed_file.read_bytes()
    seed_manifest = json.loads(seed_raw)
    requested = [int(value) for value in seed_manifest["seeds"][: args.episodes]]
    recovered: dict[int, dict[str, Any]] = {}
    if args.resume_log and args.resume_log.is_file():
        for line in args.resume_log.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task") == args.task and row.get("mode") == args.mode:
                seed = int(row.get("seed", -1))
                if seed in requested:
                    recovered[seed] = row
    remaining = [seed for seed in requested if seed not in recovered]
    rows = list(recovered.values())
    if remaining:
        rows.extend(
            evaluate(
                reference_checkpoint=args.reference_checkpoint,
                care_checkpoint=args.care_checkpoint,
                task=args.task,
                seeds=remaining,
                device_name=args.device,
                max_steps=max_steps,
                robofactory_root=args.robofactory_root,
                selector_enabled=args.mode == "care",
                settings=settings,
            )
        )
    rows.sort(key=lambda row: requested.index(int(row["seed"])))
    result = {
        "format_version": "before-we-act.a7r1-care-validation20-task/1",
        "task": args.task,
        "mode": args.mode,
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "steps": sum(int(row["steps"]) for row in rows),
        "override_steps": sum(int(row["override_steps"]) for row in rows),
        "reference_checkpoint": str(Path(args.reference_checkpoint).resolve()),
        "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
        "care_checkpoint": str(Path(args.care_checkpoint).resolve()),
        "care_checkpoint_sha256": sha256_file(args.care_checkpoint),
        "seed_protocol": {
            "source": str(args.seed_file.resolve()),
            "sha256": hashlib.sha256(seed_raw).hexdigest(),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "rows": "saved"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
