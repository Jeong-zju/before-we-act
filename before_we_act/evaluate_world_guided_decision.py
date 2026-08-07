"""R14 closed-loop evaluator: frozen W12 proposer plus frozen W13 world."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from before_we_act.action_generator.evolution import load_r12_evolution_config
from before_we_act.benchmark import TASKS as BENCHMARK_TASKS, get_task
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.evaluate_action_generator_evolution import (
    TeamHistory,
    TemporalChunkEnsembler,
    load_specialist,
    make_env,
    reset_reproducibly,
    terminal_info,
)
from before_we_act.planner.base import WorldGuidedDecisionPlanner, load_r14_config


def write_progress(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(
    *, decision_config, action_config, action_checkpoint, belief_config,
    belief_checkpoint, world_config, world_checkpoint, vision_artifact,
    vision_batch_size, task_name, seeds, device, max_steps, progress_path=None,
):
    if task_name not in decision_config.deployment["specialist_tasks"]:
        raise ValueError("R14 live evaluator is restricted to the W12 specialist route")
    generator, belief, spatial, stats, update = load_specialist(
        action_config, action_checkpoint, belief_config, belief_checkpoint,
        vision_artifact, vision_batch_size, device,
    )
    planner = WorldGuidedDecisionPlanner(
        decision_config, world_config, world_checkpoint, device
    )
    specification = get_task(task_name)
    arms = specification["agents"]
    task_index = torch.tensor([TASKS.index(task_name)], device=device)
    env = make_env(task_name)
    rows, latencies = [], []
    try:
        for episode_index, seed in enumerate(seeds):
            observation, _ = reset_reproducibly(env, seed)
            history = TeamHistory(arms)
            ensemble = TemporalChunkEnsembler(arms)
            planner.reset()
            previous_action = None
            success, info = False, {}
            fallbacks = interventions = exceptions = timeouts = 0
            gains, planner_latencies = [], []
            reasons: dict[str, int] = {}
            write_progress(progress_path, {
                "schema_version": 1, "task": task_name, "seed": int(seed),
                "episode_index": episode_index + 1, "episodes_total": len(seeds),
                "step": 0, "max_steps": max_steps, "state": "VALIDATING",
            })
            for step in range(max_steps):
                batch = history.batch(observation, previous_action, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    belief_state = belief(batch)["belief"]
                    spatial_tokens, spatial_view_mask = spatial(
                        batch["raw_fixed_rgb"], batch["spatial_view_mask"]
                    )
                    noise_generator = torch.Generator(device=device).manual_seed(
                        int(seed) * 1_000_003 + step
                    )
                    noise = torch.randn(
                        (1, 100, 32), generator=noise_generator, device=device
                    )
                    proposals = generator.sample(
                        belief_state,
                        spatial_tokens=spatial_tokens,
                        spatial_view_mask=spatial_view_mask,
                        task_index=task_index,
                        noise=noise,
                    )
                    decision = planner.decide(
                        belief_state, proposals.actions[:, 0], seed=int(seed), step=step
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies.append((time.perf_counter_ns() - started) / 1e6)
                planner_latencies.append(decision.latency_ms)
                gains.append(decision.utility_gain)
                fallbacks += int(decision.fallback)
                interventions += int(not decision.fallback)
                exceptions += int(decision.reason.startswith("planner_exception:"))
                timeouts += int(decision.reason == "planner_deadline_exceeded")
                reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
                if step % 10 == 0:
                    write_progress(progress_path, {
                        "schema_version": 1, "task": task_name, "seed": int(seed),
                        "episode_index": episode_index + 1, "episodes_total": len(seeds),
                        "step": step + 1, "max_steps": max_steps, "state": "VALIDATING",
                        "interventions": interventions, "fallbacks": fallbacks,
                        "planner_exceptions": exceptions, "planner_timeouts": timeouts,
                    })
                normalized = decision.actions[0, : len(arms)]
                raw = normalized * stats["a_std"][None, None] + stats["a_mean"][None, None]
                action = ensemble.append_and_select(step, raw.float().cpu().numpy())
                previous_action = {key: value.copy() for key, value in action.items()}
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(np.asarray(info.get("success", False)).all())
                if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                    break
            row = {
                "task": task_name,
                "seed": int(seed),
                "success": success,
                "steps": step + 1,
                "route": "r14_world_guided_w12_stack_specialist",
                "candidate_source": "W12 transplanted ACT base / planner-refined",
                "planner_calls": step + 1,
                "interventions": interventions,
                "fallbacks": fallbacks,
                "planner_exceptions": exceptions,
                "planner_timeouts": timeouts,
                "mean_utility_gain": float(np.mean(gains)) if gains else 0.0,
                "planner_p95_ms": float(np.percentile(planner_latencies, 95)) if planner_latencies else None,
                "fallback_reasons": reasons,
                "safety_projections": 0,
                "terminal_info": terminal_info(info),
            }
            rows.append(row)
            write_progress(progress_path, {
                "schema_version": 1, "task": task_name, "seed": int(seed),
                "episode_index": episode_index + 1, "episodes_total": len(seeds),
                "step": step + 1, "max_steps": max_steps, "state": "EPISODE_COMPLETE",
                "success": success, "interventions": interventions, "fallbacks": fallbacks,
            })
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.close()
    return rows, latencies, update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--action-config", required=True)
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--world-config", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--vision-artifact", required=True)
    parser.add_argument("--vision-batch-size", type=int, default=5)
    parser.add_argument("--task", choices=tuple(BENCHMARK_TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    parser.add_argument("--progress", default="")
    args = parser.parse_args()
    decision_config = load_r14_config(args.config)
    action_config = load_r12_evolution_config(args.action_config)
    seed_path = Path(args.seed_file).resolve(strict=True)
    seed_bytes = seed_path.read_bytes()
    seeds = [int(value) for value in json.loads(seed_bytes)["seeds"]]
    if args.episodes != 20 or len(seeds) < 20:
        raise ValueError("R14 Gate20 requires exactly the first 20 frozen seeds")
    requested = seeds[:20]
    recovered = []
    if args.resume_log and Path(args.resume_log).is_file():
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("task") == args.task and row.get("seed") in requested
                and isinstance(row.get("success"), bool)
                and isinstance(row.get("steps"), int)
                and row.get("route") == "r14_world_guided_w12_stack_specialist"
            ):
                recovered.append(row)
    recovered = list({row["seed"]: row for row in recovered}.values())
    remaining = [seed for seed in requested if seed not in {row["seed"] for row in recovered}]
    if remaining:
        evaluated, latencies, update = evaluate(
            decision_config=decision_config,
            action_config=action_config,
            action_checkpoint=args.action_checkpoint,
            belief_config=args.belief_config,
            belief_checkpoint=args.belief_checkpoint,
            world_config=args.world_config,
            world_checkpoint=args.world_checkpoint,
            vision_artifact=args.vision_artifact,
            vision_batch_size=args.vision_batch_size,
            task_name=args.task,
            seeds=remaining,
            device=torch.device(args.device),
            max_steps=args.max_steps,
            progress_path=Path(args.progress) if args.progress else None,
        )
    else:
        evaluated, latencies, update = [], [], None
    rows = recovered + evaluated
    rows.sort(key=lambda row: requested.index(row["seed"]))
    values = np.asarray(latencies, dtype=np.float64)
    result = {
        "schema_version": 1,
        "round": "R14",
        "candidate_id": decision_config.candidate_id,
        "task": args.task,
        "route": "r14_world_guided_w12_stack_specialist",
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "rows": rows,
        "planner": {
            "calls": sum(row["planner_calls"] for row in rows),
            "interventions": sum(row["interventions"] for row in rows),
            "fallbacks": sum(row["fallbacks"] for row in rows),
            "exceptions": sum(row["planner_exceptions"] for row in rows),
            "timeouts": sum(row["planner_timeouts"] for row in rows),
        },
        "latency_ms": {
            "samples": len(latencies),
            "p50": float(np.percentile(values, 50)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
        },
        "action_checkpoint": str(Path(args.action_checkpoint).resolve()),
        "action_checkpoint_update": update,
        "world_checkpoint": str(Path(args.world_checkpoint).resolve()),
        "seed_protocol": {
            "source": str(seed_path),
            "sha256": hashlib.sha256(seed_bytes).hexdigest(),
        },
        "control_cadence": "one W12 proposal and one fail-closed R14 decision per environment step",
        "temporal_aggregation": "W10 exponential chunk ensemble decay=0.01",
        "privileged_inputs": False,
        "core_runtime": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
