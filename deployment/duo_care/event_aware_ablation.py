"""Standalone event-aware anchor ablation for DuoBench CARE.

This command is deliberately not imported by the formal supervisor.  It runs
the registered branch kernel twice (fixed stratified anchors and an
event/uncertainty hybrid) with identical B-core, seeds, candidates and rollout
budgets, then writes a side-by-side signal report.  It is safe to run while a
formal pipeline is in progress because it has an explicit output root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from deployment.duo_care.branch_collection_v2 import (
    KernelConfig,
    _canonical_reference,
    _encoded_executed,
    _reference_action,
    advance_to_anchor,
    collect_from_anchor,
)
from deployment.duo_care.branch_signal import HORIZONS, stratified_anchor_steps
from deployment.duo_care.care_signal_audit import audit_family_json
from deployment.duo_care.duobench_adapter import DuoBenchEnvironment, DuoBcoreProposalProvider
from deployment.duo_care.duo_dino_branch_launcher import (
    DEFAULT_SEED_START,
    TASKS,
    validate_selected_inputs,
)
from deployment.duo_care.event_aware_sampling import event_aware_hybrid_anchor_steps
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
)


SMOKE_TASKS = ("carry_pot", "transfer_gate", "pour_marbles")
# Ten families/task gives a 10-point resolution for the pre-registered
# effective-family gate while remaining much cheaper than the 330-family run.
SMOKE_FAMILIES = 10
VERSION = "before-we-act.care-duobench-event-aware-ablation/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trace_candidate_zero(
    env: DuoBenchEnvironment,
    provider: DuoBcoreProposalProvider,
    *,
    task: str,
    seed: int,
    limit: int,
) -> tuple[list[float], list[float]]:
    """Collect pre-branch event and uncertainty scores from candidate zero."""

    observation, info = env.reset(int(seed))
    runtime = provider.new_runtime(task)
    events: list[float] = []
    uncertainty: list[float] = []
    previous_progress = float(env.progress(observation, info))
    previous_action: np.ndarray | None = None
    previous_stage = float(np.asarray(info.get("stage", 0)).reshape(-1)[0]) if isinstance(info, Mapping) else 0.0
    for _step in range(int(limit)):
        proposal = provider.propose(observation, runtime, task)
        proposal.validate(agents=provider.agent_count, horizon=100)
        reference, _base, _diag = _canonical_reference(proposal, env)
        action = _reference_action(proposal, reference)
        result = env.step_absolute(action)
        physical = np.asarray(result.executed_absolute, dtype=np.float32)
        encoded = _encoded_executed(physical, proposal.qpos)
        provider.append_executed_action(runtime, encoded)
        progress = float(env.progress(result.observation, result.info))
        stage = float(np.asarray(result.info.get("stage", 0)).reshape(-1)[0]) if isinstance(result.info, Mapping) else 0.0
        action_delta = 0.0 if previous_action is None else float(np.mean(np.abs(action - previous_action)))
        gripper_delta = 0.0 if previous_action is None else float(np.mean(np.abs(action[:, 7] - previous_action[:, 7])))
        progress_delta = abs(progress - previous_progress)
        stage_change = float(stage != previous_stage)
        # These are pre-branch observables only.  No candidate outcome is used.
        events.append(float(2.0 * stage_change + 2.0 * gripper_delta + action_delta + progress_delta))
        raw_slots = np.asarray(
            proposal.diagnostics.get("valid_event_slots", 0.0), dtype=np.float64
        )
        valid_slots = float(raw_slots.mean()) if raw_slots.size else 0.0
        # B-core/base disagreement is a pre-outcome epistemic signal.
        belief_correction = float(
            np.mean(np.abs(proposal.reference_encoded - proposal.base_encoded))
        )
        uncertainty.append(
            float(action_delta + belief_correction + abs(valid_slots - 4.0) / 4.0)
        )
        previous_action = action.copy()
        previous_progress, previous_stage = progress, stage
        observation, info = result.observation, result.info
        if result.success or result.terminated or result.truncated:
            break
    return events, uncertainty


def _family_metrics(paths: Sequence[Path]) -> dict[str, Any]:
    reports = [audit_family_json(path, strict=False) for path in paths]
    families = len(reports)
    result: dict[str, Any] = {"families": families, "horizons": {}}
    for horizon in HORIZONS:
        nz = 0
        family_signal = 0
        pairwise = 0
        for path in paths:
            family = json.loads(path.read_text(encoding="utf-8"))
            by = {(int(row["candidate_id"]), str(row["regime"]), int(row["repeat_id"])): row for row in family["branches"]}
            deltas: list[float] = []
            for regime in ("reactive", "replay"):
                for repeat in (0, 1):
                    reference = float(by[(0, regime, repeat)]["outcomes"][str(horizon)]["utility_main"])
                    deltas.extend(float(by[(candidate, regime, repeat)]["outcomes"][str(horizon)]["utility_main"]) - reference for candidate in range(1, 6))
                    candidate_values = [float(by[(candidate, regime, repeat)]["outcomes"][str(horizon)]["utility_main"]) for candidate in range(1, 6)]
                    pairwise += int(np.count_nonzero(np.abs(np.subtract.outer(candidate_values, candidate_values)) > 1e-7) // 2)
            deltas = np.asarray(deltas, dtype=np.float64)
            nz += int(np.count_nonzero(np.abs(deltas) > 1e-7))
            family_signal += int(np.any(np.abs(deltas) > 1e-7))
        result["horizons"][str(horizon)] = {
            "nonzero_candidate_advantages": nz,
            "effective_family_fraction": family_signal / families if families else 0.0,
            "effective_families": family_signal,
            "pairwise_non_ties": pairwise,
        }
    return result


def _collect_policy(
    *,
    policy: str,
    output: Path,
    tasks: Sequence[str],
    families_per_task: int,
    seed_start: int,
    bcore_checkpoint: Path,
    b0h_checkpoint: Path,
    prepared_data: Path,
    visual_cache: Path,
    dino_model: Path,
) -> dict[str, Any]:
    manifest = json.loads((prepared_data / "manifest.json").read_text(encoding="utf-8"))
    provider = DuoBcoreProposalProvider(
        bcore_checkpoint,
        b0h_checkpoint=b0h_checkpoint,
        device="cuda:0",
        dino_model=str(dino_model),
        image_height=224,
        image_width=224,
    )
    paths: list[Path] = []
    input_provenance = validate_selected_inputs(
        bcore_checkpoint=bcore_checkpoint,
        b0h_checkpoint=b0h_checkpoint,
        prepared_data=prepared_data,
        visual_cache=visual_cache,
        dino_model=dino_model,
    )
    metadata: list[dict[str, Any]] = []
    for task in tasks:
        maximum = int(manifest["tasks"][task]["validation_max_steps"])
        env = DuoBenchEnvironment(task, image_size=224)
        try:
            for ordinal in range(int(families_per_task)):
                seed = int(seed_start + TASKS.index(task) * 100_000 + ordinal)
                snapshot_id = hashlib.sha256(
                    f"duobench-care-event-ablation-v1|{task}|{seed}|ordinal={ordinal}|arm={ordinal % 2}".encode()
                ).hexdigest()
                task_root = output / "families" / task
                path = task_root / f"{snapshot_id}.json"
                if path.is_file() and path.with_suffix(".npz").is_file():
                    audit = audit_family_json(path, strict=False)
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if audit["status"] == "PASSED" and saved.get("ablation_policy") == policy:
                        paths.append(path)
                        sampling = dict(saved.get("sampling_metadata", {}))
                        metadata.append({"task": task, "ordinal": ordinal, "anchor_step": int(saved["anchor_step"]), "sampling_stratum": saved["sampling_stratum"], **sampling})
                        continue
                if policy == "fixed":
                    anchors = stratified_anchor_steps(maximum, max_steps=maximum, count=families_per_task, horizon=64, critical_count=min(20, families_per_task))
                    anchor_row = anchors[ordinal]
                else:
                    events, uncertainty = _trace_candidate_zero(env, provider, task=task, seed=seed, limit=maximum - 64)
                    anchors = event_aware_hybrid_anchor_steps(maximum, max_steps=maximum, event_scores=events, uncertainty_scores=uncertainty, count=families_per_task, horizon=64)
                    anchor_row = anchors[ordinal]
                anchor_step = int(anchor_row["anchor_step"])
                # The ID intentionally excludes policy/anchor_step so fixed
                # and hybrid receive identical repeat branch seeds.
                anchor = advance_to_anchor(env, provider, task=task, episode_seed=seed, anchor_step=anchor_step, focal_agent=ordinal % 2, sampling_stratum=str(anchor_row["sampling_stratum"]), snapshot_id=snapshot_id, config=KernelConfig())
                family, arrays = collect_from_anchor(env, provider, anchor, config=KernelConfig())
                family.update({
                    "ablation_policy": policy,
                    "sampling_metadata": anchor_row,
                    "formal_protocol": False,
                    "ablation_only": True,
                    "formal_protocol_unchanged": True,
                    "collection_format": VERSION,
                    "reference_policy_family": "PredictiveTeamBeliefPolicy",
                    "base_policy_family": "TemporalHistoryPolicy",
                    "method_family": "CARE",
                    "vision": "dinov3_vitb16_frozen",
                    "vision_backbone": "dinov3_vitb16_frozen",
                    "image_preprocess_id": input_provenance["image_preprocess_id"],
                    "dino_normalization_id": input_provenance["dino_normalization_id"],
                    "strict_dino_contract": True,
                    "strictly_decentralized": True,
                    "strict_local": True,
                    "act_provider_allowed": False,
                    "source_policy_action_encoding": "absolute_joint7_binary_gripper1",
                    "action_encoding": "joint_residual7_gripper_absolute1",
                    "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
                    "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
                    "memory_semantics": input_provenance["memory_semantics"],
                    "care_memory_tokens": input_provenance["care_memory_tokens"],
                    "bcore_checkpoint_sha256": input_provenance["bcore_checkpoint_sha256"],
                    "b0h_checkpoint_sha256": input_provenance["b0h_checkpoint_sha256"],
                })
                task_root.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(family, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                np.savez_compressed(path.with_suffix(".npz"), **arrays)
                audit = audit_family_json(path, strict=False)
                if audit["status"] != "PASSED":
                    raise RuntimeError(f"ablation family audit failed: {path}: {audit['errors']}")
                paths.append(path)
                metadata.append({"task": task, "ordinal": ordinal, "anchor_step": anchor_step, "sampling_stratum": anchor_row["sampling_stratum"]})
        finally:
            env.close()
    report = {"schema": VERSION, "status": "PASSED", "policy": policy, "families": len(paths), "branches": len(paths) * 24, "formal_protocol_unchanged": True, "family_paths": [str(path.resolve()) for path in paths], "anchors": metadata, "signal": _family_metrics(paths), "final_success_rate": None, "final_success_status": "pending CARE scorer training and paired Validation20"}
    _json(output / "ablation_receipt.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcore-checkpoint", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=SMOKE_TASKS)
    parser.add_argument("--families-per-task", type=int, default=SMOKE_FAMILIES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--policy", choices=("fixed", "hybrid", "both"), default="both")
    args = parser.parse_args()
    if args.smoke and tuple(args.tasks) != SMOKE_TASKS:
        raise ValueError("ablation smoke tasks are pre-registered and cannot be changed")
    validate_selected_inputs(bcore_checkpoint=args.bcore_checkpoint, b0h_checkpoint=args.b0h_checkpoint, prepared_data=args.prepared_data, visual_cache=args.visual_cache, dino_model=args.dino_model)
    args.output.mkdir(parents=True, exist_ok=True)
    fixed = _collect_policy(policy="fixed", output=args.output / "fixed", tasks=args.tasks, families_per_task=args.families_per_task, seed_start=args.seed_start, bcore_checkpoint=args.bcore_checkpoint, b0h_checkpoint=args.b0h_checkpoint, prepared_data=args.prepared_data, visual_cache=args.visual_cache, dino_model=args.dino_model) if args.policy in ("fixed", "both") else None
    hybrid = _collect_policy(policy="hybrid", output=args.output / "hybrid", tasks=args.tasks, families_per_task=args.families_per_task, seed_start=args.seed_start, bcore_checkpoint=args.bcore_checkpoint, b0h_checkpoint=args.b0h_checkpoint, prepared_data=args.prepared_data, visual_cache=args.visual_cache, dino_model=args.dino_model) if args.policy in ("hybrid", "both") else None
    if args.policy != "both":
        result = {"schema": VERSION, "status": "PASSED", "policy": args.policy, "formal_protocol_unchanged": True, "signal": (fixed or hybrid)["signal"], "bcore_checkpoint_sha256": _sha256(args.bcore_checkpoint)}
        _json(args.output / "comparison.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    comparison = {"schema": VERSION, "status": "PASSED", "smoke": bool(args.smoke), "pre_registered_tasks": list(SMOKE_TASKS), "formal_protocol_unchanged": True, "fixed": fixed["signal"], "hybrid": hybrid["signal"], "decision_rule": "adopt_hybrid_only_if_h16_effective_family_fraction improves by >=0.10 and pairwise_non_ties does not decrease", "adopt_hybrid": bool(hybrid["signal"]["horizons"]["16"]["effective_family_fraction"] - fixed["signal"]["horizons"]["16"]["effective_family_fraction"] >= 0.10 and hybrid["signal"]["horizons"]["16"]["pairwise_non_ties"] >= fixed["signal"]["horizons"]["16"]["pairwise_non_ties"]), "bcore_checkpoint_sha256": _sha256(args.bcore_checkpoint)}
    _json(args.output / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
