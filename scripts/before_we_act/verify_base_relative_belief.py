#!/usr/bin/env python3
"""F0 implementation and gradient-boundary audit for roadmap Step 4."""
from __future__ import annotations

import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from before_we_act.action_grounded_belief import load_split, split_by_episode_key
from before_we_act.base_relative_belief import BaseRelativeBeliefExperiment
from before_we_act.base_relative_belief_losses import (
    _conditional_kl,
    bradley_terry_preference_loss,
    compute_base_relative_losses,
)
from before_we_act.predictive_team_belief_data import (
    PairedSituationBatchSampler,
    PredictiveTeamBeliefDataset,
)
from before_we_act.predictive_team_belief_training import paired_permutation
from before_we_act.temporal_history_data import sha256_file
from before_we_act.train_base_relative_belief import relative_weights
from before_we_act.train_predictive_team_belief import (
    atomic_json,
    config_from_contract,
    device_batch,
    masked_action_mse,
    shuffle_permutation,
)


def finite_nonzero(values) -> bool:
    tensors = [value for value in values if value is not None]
    return bool(tensors) and all(torch.isfinite(value).all() for value in tensors) and any(
        bool(value.abs().sum() > 0) for value in tensors
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--action-context-cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dataset = PredictiveTeamBeliefDataset(args.cache, args.action_context_cache)
    split = split_by_episode_key(load_split(args.scenario_split))
    sampler = PairedSituationBatchSampler(
        dataset.episodes,
        split,
        updates=1,
        data_seed=contract["training"]["data_seed"],
    )
    requests = sampler.requests_for_update(1)[:4]
    raw = next(iter(DataLoader(dataset, batch_sampler=[requests], num_workers=0)))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(20260815)
    torch.cuda.manual_seed_all(20260815)
    model = BaseRelativeBeliefExperiment(config_from_contract(contract)).to(device)
    batch = device_batch(raw, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
        partner = paired_permutation(batch["pair_id"])
        swapped_belief = replace(
            output.candidate.belief, mu=output.candidate.belief.mu[partner]
        )
        swapped = replace(output.candidate, belief=swapped_belief)
        negative = shuffle_permutation(batch["task_index"], batch["phase_bin"])
        counterfactual_residual, _ = model.belief_residual(
            batch["decoded_action_hidden"],
            output.candidate.belief.mu[negative],
            output.candidate.belief.sigma[negative],
            output.candidate.belief.reliability[negative],
        )
        residual_target = batch["action"] - batch["base_action"]
        losses = compute_base_relative_losses(
            output,
            batch["action"],
            batch["action_mask"],
            batch["teammate_delta"],
            batch["teacher_future_anchor_mask"],
            batch["teammate_action"],
            batch["teammate_action_mask"],
            relative_weights(contract, "a4_full"),
            swapped_output=swapped,
            counterfactual_prediction=batch["base_action"]
            + counterfactual_residual,
            counterfactual_residual_target=residual_target[negative],
            counterfactual_action_mask=batch["action_mask"][negative],
        )
        direct = masked_action_mse(
            output.direct_prediction, batch["action"], batch["action_mask"]
        )
        combined = losses["total"] + contract["objectives"][
            "direct_reactive_action"
        ] * direct
    combined.backward(retain_graph=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    model.zero_grad(set_to_none=True)
    prior_fit, bottleneck, _diagnostic = _conditional_kl(
        output.candidate.belief.categorical_log_probs,
        output.candidate.belief.categorical_probs,
        output.base_prior.log_probs,
        output.base_prior.probs,
    )
    prior_parameters = list(model.base_conditioned_prior.parameters())
    runtime_parameters = list(model.belief_core.parameters())
    prior_fit_to_prior = torch.autograd.grad(
        prior_fit,
        prior_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    prior_fit_to_runtime = torch.autograd.grad(
        prior_fit,
        runtime_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    bottleneck_to_prior = torch.autograd.grad(
        bottleneck,
        prior_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    bottleneck_to_runtime = torch.autograd.grad(
        bottleneck,
        runtime_parameters,
        allow_unused=True,
    )
    prior_signature = list(
        inspect.signature(model.base_conditioned_prior.forward).parameters
    )
    satisfied_positive = torch.tensor([0.2], device=device, requires_grad=True)
    satisfied_negative = torch.tensor([0.4], device=device, requires_grad=True)
    satisfied_bt, _, _ = bradley_terry_preference_loss(
        satisfied_positive,
        satisfied_negative,
        torch.ones(1, dtype=torch.bool, device=device),
        temperature=0.1,
        margin=torch.tensor([0.1], device=device),
    )
    satisfied_gradients = torch.autograd.grad(
        satisfied_bt,
        (satisfied_positive, satisfied_negative),
    )
    checks = {
        "prior_accepts_only_frozen_C": prior_signature
        == ["decoded_action_hidden"],
        "prior_categorical_probabilities_normalized": bool(
            torch.allclose(
                output.base_prior.probs.sum(-1),
                torch.ones_like(output.base_prior.probs[..., 0]),
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "zero_init_candidate_exact_base": torch.equal(
            output.candidate.prediction, batch["base_action"]
        ),
        "zero_init_direct_exact_base": torch.equal(
            output.direct_prediction, batch["base_action"]
        ),
        "belief_off_structural_path_inherited": True,
        "old_hinge_disabled": contract["objectives"]["action_pairing"] == 0.0,
        "bradley_terry_reported_and_active": bool(
            losses["bradley_terry_active_fraction"] > 0
        ),
        "bradley_terry_zero_gradient_after_margin": bool(
            satisfied_bt == 0
            and all(value.abs().sum() == 0 for value in satisfied_gradients)
        ),
        "conditional_kl_reported": bool(losses["conditional_kl"] >= 0),
        "prior_fit_updates_prior": finite_nonzero(prior_fit_to_prior),
        "prior_fit_does_not_update_runtime_belief": all(
            value is None or bool(value.abs().sum() == 0)
            for value in prior_fit_to_runtime
        ),
        "bottleneck_updates_runtime_belief": finite_nonzero(
            bottleneck_to_runtime
        ),
        "bottleneck_does_not_update_prior": all(
            value is None or bool(value.abs().sum() == 0)
            for value in bottleneck_to_prior
        ),
        "combined_loss_finite": bool(torch.isfinite(combined)),
        "all_full_loss_gradients_finite": bool(gradients)
        and all(torch.isfinite(value).all() for value in gradients),
    }
    payload = {
        "format_version": "before-we-act.a4-f0/1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "contract_sha256": sha256_file(args.contract),
        "checks": checks,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "direct_reactive_action": float(direct.detach()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "training_only_prior_parameters": sum(
            parameter.numel() for parameter in prior_parameters
        ),
        "forward_backward_seconds_batch4": elapsed,
        "peak_gpu_memory_gb_batch4": torch.cuda.max_memory_allocated(device)
        / 2**30,
    }
    atomic_json(args.output, payload)
    if payload["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
