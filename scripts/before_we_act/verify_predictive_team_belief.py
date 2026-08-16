#!/usr/bin/env python3
"""F0 architecture/data audit and F1 resume verification for Step 3-N2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from before_we_act.action_grounded_belief import load_split, split_by_episode_key
from before_we_act.predictive_team_belief_data import PredictiveTeamBeliefDataset, PairedSituationBatchSampler
from before_we_act.predictive_team_belief_training import TeamBeliefExperiment, paired_permutation
from before_we_act.temporal_history_data import sha256_file
from before_we_act.train_predictive_team_belief import (
    config_from_contract,
    device_batch,
    loss_weights,
    shuffle_permutation,
)
from before_we_act.team_belief.losses import compute_team_belief_losses


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def f0(args) -> None:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dataset = PredictiveTeamBeliefDataset(args.cache, args.action_context_cache)
    split = split_by_episode_key(load_split(args.scenario_split))
    sampler = PairedSituationBatchSampler(
        dataset.episodes, split, updates=1, data_seed=contract["training"]["data_seed"]
    )
    requests = sampler.requests_for_update(1)[:4]
    raw = next(iter(DataLoader(dataset, batch_sampler=[requests], num_workers=0)))
    runtime = set(dataset.RUNTIME_FIELDS)
    teacher = set(dataset.TEACHER_FIELDS)
    if runtime & teacher:
        raise RuntimeError("N2 runtime and teacher fields overlap")
    forbidden_fragments = ("future", "teammate", "success", "reward")
    illegal_runtime = sorted(
        key
        for key in runtime
        if key in {"episode_index", "time_index"}
        or any(fragment in key for fragment in forbidden_fragments)
    )
    if illegal_runtime:
        raise RuntimeError(f"N2 runtime whitelist contains privileged names: {illegal_runtime}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = TeamBeliefExperiment(config_from_contract(contract)).to(device)
    batch = device_batch(raw, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
        partner = paired_permutation(batch["pair_id"])
        negative = shuffle_permutation(batch["task_index"], batch["phase_bin"])
        counterfactual_residual, _ = model.belief_residual(
            batch["decoded_action_hidden"],
            output.candidate.belief.mu[negative],
            output.candidate.belief.sigma[negative],
            output.candidate.belief.reliability[negative],
        )
        residual_target = batch["action"] - batch["base_action"]
        losses = compute_team_belief_losses(
            output.candidate,
            batch["action"],
            batch["action_mask"],
            batch["teammate_delta"],
            batch["teacher_future_anchor_mask"],
            batch["teammate_action"],
            batch["teammate_action_mask"],
            loss_weights(contract),
            counterfactual_prediction=(
                batch["base_action"] + counterfactual_residual
            ),
            counterfactual_residual_target=residual_target[negative],
            counterfactual_action_mask=batch["action_mask"][negative],
        )
        loss = losses["total"] + (output.direct_prediction - batch["action"]).square().mean()
        occluded_mask = batch["runtime_visual_mask"].clone()
        occluded_mask[:, :, 1] = False
        occluded = model.belief_core(
            batch["runtime_visual_tokens"],
            occluded_mask,
            batch["history_qpos"],
            batch["history_action"],
            batch["history_mask"],
            batch["action_history_mask"],
            batch["task_token"],
            batch["episode_reset_mask"],
        )
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    zero_belief = torch.equal(output.candidate.prediction, batch["base_action"])
    zero_direct = torch.equal(output.direct_prediction, batch["base_action"])
    persistence = output.candidate.belief.current_visual_reference[:, None].expand_as(
        output.candidate.belief.future_latent_prediction
    )
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    checks = {
        "runtime_teacher_fields_disjoint": not bool(runtime & teacher),
        "runtime_whitelist_has_no_privileged_names": not illegal_runtime,
        "paired_rows_complete": torch.equal(partner[partner], torch.arange(len(partner), device=device)),
        "zero_init_belief_exact_base": zero_belief,
        "zero_init_direct_exact_base": zero_direct,
        "zero_init_future_exact_legal_persistence": torch.equal(
            output.candidate.belief.future_latent_prediction, persistence
        ),
        "zero_init_future_gate_has_finite_nonzero_gradient": bool(
            model.belief_core.future_predictor.horizon_gain_raw.grad is not None
            and torch.isfinite(
                model.belief_core.future_predictor.horizon_gain_raw.grad
            ).all()
            and model.belief_core.future_predictor.horizon_gain_raw.grad.abs().sum()
            > 0
        ),
        "runtime_future_has_only_two_legal_views": bool(
            output.candidate.belief.current_visual_view_mask[:, :2].all()
            and not output.candidate.belief.current_visual_view_mask[:, 2].any()
        ),
        "one_view_occlusion_increases_epistemic_uncertainty": bool(
            (
                occluded.epistemic_uncertainty
                > output.candidate.belief.epistemic_uncertainty
            ).all()
        ),
        "one_view_occlusion_reduces_reliability": bool(
            (occluded.reliability < output.candidate.belief.reliability).all()
        ),
        "action_pairing_loss_reported": "action_pairing" in losses,
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_exist_and_finite": bool(gradients) and all(torch.isfinite(value).all() for value in gradients),
        "future_anchor_mask_has_four_slots": batch["teacher_future_anchor_mask"].shape[1] == 4,
        "full_action_horizon_is_100": batch["action"].shape[1] == 100,
        "all_16_belief_tokens_preserved": model.config.n_belief_tokens == 16,
    }
    payload = {
        "format_version": "before-we-act.b3-n2-f0/1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "contract_sha256": sha256_file(args.contract),
        "checks": checks,
        "architecture": model.config.__dict__,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "forward_backward_seconds_batch4": elapsed,
        "peak_gpu_memory_gb_batch4": torch.cuda.max_memory_allocated(device) / 2**30,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
    }
    atomic_json(args.output, payload)
    if payload["status"] != "PASSED":
        raise SystemExit(1)


def f1(args) -> None:
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    if reference["provenance"] != resumed["provenance"]:
        raise RuntimeError("N2 F1 provenance differs")
    if reference["sample_cursor"] != resumed["sample_cursor"]:
        raise RuntimeError("N2 F1 sample cursor differs")
    if reference["model"].keys() != resumed["model"].keys():
        raise RuntimeError("N2 F1 model keys differ")
    maxima = {
        key: float((reference["model"][key] - resumed["model"][key]).abs().max())
        for key in reference["model"]
    }
    maximum = max(maxima.values(), default=0.0)
    checks = {
        "update_is_four": reference["update"] == resumed["update"] == 4,
        "sample_cursor_exact": reference["sample_cursor"] == resumed["sample_cursor"],
        "model_max_abs_le_1e-7": maximum <= 1e-7,
    }
    payload = {
        "format_version": "before-we-act.b3-n2-f1/1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "maximum_model_absolute_difference": maximum,
        "worst_parameters": sorted(maxima.items(), key=lambda row: row[1], reverse=True)[:10],
        "reference_sha256": sha256_file(args.reference),
        "resumed_sha256": sha256_file(args.resumed),
    }
    atomic_json(args.output, payload)
    if payload["status"] != "PASSED":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    f0_parser = subparsers.add_parser("f0")
    f0_parser.add_argument("--cache", type=Path, required=True)
    f0_parser.add_argument("--action-context-cache", type=Path, required=True)
    f0_parser.add_argument("--contract", type=Path, required=True)
    f0_parser.add_argument("--scenario-split", type=Path, required=True)
    f0_parser.add_argument("--output", type=Path, required=True)
    f1_parser = subparsers.add_parser("f1")
    f1_parser.add_argument("--reference", type=Path, required=True)
    f1_parser.add_argument("--resumed", type=Path, required=True)
    f1_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    f0(args) if args.command == "f0" else f1(args)


if __name__ == "__main__":
    main()
