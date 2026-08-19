"""Train one Step-4 base-relative control-sufficient belief seed."""
from __future__ import annotations

import argparse
from dataclasses import replace
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

from before_we_act.action_grounded_belief import (
    BELIEF_DATA_SEED,
    load_split,
    split_by_episode_key,
)
from before_we_act.base_relative_belief import BaseRelativeBeliefExperiment
from before_we_act.base_relative_belief_losses import (
    BaseRelativeLossWeights,
    compute_base_relative_losses,
)
from before_we_act.deployment_safety import calibrated_residual_safety
from before_we_act.predictive_team_belief_data import (
    PairedSituationBatchSampler,
    PredictiveTeamBeliefDataset,
)
from before_we_act.predictive_team_belief_training import paired_permutation
from before_we_act.temporal_history_data import sha256_file
from before_we_act.train_predictive_team_belief import (
    EVAL_EVERY,
    LR_DROP,
    MAX_UPDATES,
    atomic_json,
    atomic_save,
    config_from_contract,
    device_batch,
    evaluate,
    export_deployment,
    fixed_loader,
    loss_weights,
    masked_action_mse,
    shuffle_permutation,
    training_sufficiency,
)


VARIANTS = ("a4_full", "a4_no_bottleneck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--action-context-cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--evaluate-at-end", action="store_true")
    parser.add_argument(
        "--resume-audit",
        action="store_true",
        help="permit only the four-update F1 fresh-versus-resume audit",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative_weights(
    contract: Mapping, variant: str
) -> BaseRelativeLossWeights:
    values = contract["base_relative_objectives"]
    beta = float(contract["variants"][variant]["beta_b"])
    return BaseRelativeLossWeights(
        base=loss_weights(contract),
        conditional_prior_fit=float(values["conditional_prior_fit"]),
        conditional_bottleneck=beta,
        bradley_terry=float(values["bradley_terry"]),
        bradley_terry_temperature=float(values["bradley_terry_temperature"]),
        bradley_terry_margin_fraction=float(
            values["bradley_terry_margin_fraction"]
        ),
        bradley_terry_margin_cap=float(values["bradley_terry_margin_cap"]),
    )


def verify_source_receipt(contract: Mapping, root: Path) -> None:
    for relative, expected in contract["source_code"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Step-4 source receipt differs: {relative}")


def categorical_kl_rows(
    q_log: torch.Tensor, q: torch.Tensor, p_log: torch.Tensor
) -> torch.Tensor:
    return (q * (q_log - p_log)).sum(-1).mean((1, 2))


@torch.no_grad()
def evaluate_base_relative(
    model: BaseRelativeBeliefExperiment,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Report K_cond and a pre-registered small feature-offset proxy.

    The perturbation leaves cached ``C`` and ``A0`` untouched and adds the same
    low-amplitude deterministic direction to every legal DINO token.  It is a
    nuisance-sensitivity proxy, not proof of real-background invariance.
    """

    model.eval()
    k_rows: list[float] = []
    nuisance_belief: list[float] = []
    nuisance_action: list[float] = []
    runtime_entropy: list[float] = []
    prior_entropy: list[float] = []
    pattern = torch.sin(
        torch.arange(model.config.vision_dim, device=device, dtype=torch.float32)
        * 0.017
    )
    pattern = pattern / pattern.square().mean().sqrt().clamp_min(1e-12)
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        k_rows.extend(
            categorical_kl_rows(
                output.candidate.belief.categorical_log_probs.float(),
                output.candidate.belief.categorical_probs.float(),
                output.base_prior.log_probs.float(),
            ).cpu().tolist()
        )
        runtime_entropy.append(
            float(
                (
                    output.candidate.belief.categorical_entropy.float()
                    * np.log(model.config.belief_classes)
                )
                .mean()
                .cpu()
            )
        )
        prior_entropy.append(float(output.base_prior.entropy.float().mean().cpu()))

        visual = batch["runtime_visual_tokens"]
        scale = visual.float().std(dim=-1, keepdim=True, unbiased=False).clamp_min(
            1e-4
        )
        active = batch["runtime_visual_mask"].unsqueeze(-1)
        perturbed_visual = visual + (
            0.01
            * scale
            * pattern.view(1, 1, 1, 1, -1).to(visual.dtype)
            * active.to(visual.dtype)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            perturbed = model.belief_core(
                perturbed_visual,
                batch["runtime_visual_mask"],
                batch["history_qpos"],
                batch["history_action"],
                batch["history_mask"],
                batch["action_history_mask"],
                batch["task_token"],
                batch["episode_reset_mask"],
                future_action=batch["action"],
                future_action_mask=batch["action_mask"],
            )
            perturbed_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                perturbed.mu,
                perturbed.sigma,
                perturbed.reliability,
            )
        belief_scale = (
            output.candidate.belief.mu.float().var(unbiased=False).clamp_min(1e-8)
        )
        nuisance_belief.append(
            float(
                (
                    perturbed.mu.float()
                    - output.candidate.belief.mu.float()
                )
                .square()
                .mean()
                .div(belief_scale)
                .cpu()
            )
        )
        action_scale = (
            output.candidate.belief_residual.float()
            .square()
            .mean()
            .clamp_min(1e-8)
        )
        nuisance_action.append(
            float(
                (
                    perturbed_residual.float()
                    - output.candidate.belief_residual.float()
                )
                .square()
                .mean()
                .div(action_scale)
                .cpu()
            )
        )
    return {
        "conditional_kl_nats": float(np.mean(k_rows)),
        "conditional_kl_std": float(np.std(k_rows)),
        "runtime_categorical_entropy_nats": float(np.mean(runtime_entropy)),
        "base_prior_categorical_entropy_nats": float(np.mean(prior_entropy)),
        "nuisance_proxy": {
            "kind": "one-percent-fixed-DINO-feature-offset-with-frozen-C-and-A0",
            "belief_relative_mse": float(np.mean(nuisance_belief)),
            "residual_relative_mse": float(np.mean(nuisance_action)),
        },
        "rows": len(k_rows),
    }


def customize_deployment(path: Path, variant: str, contract_path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["policy_variant"] = "a4_base_relative_team_belief"
    payload["config"]["a4_training_variant"] = variant
    payload["config"]["a4_contract_sha256"] = sha256_file(contract_path)
    payload["config"]["base_conditioned_prior_present"] = False
    atomic_save(payload, path)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("format_version") != "before-we-act.a4-contract/1":
        raise RuntimeError("unsupported Step-4 contract")
    status = str(contract.get("status"))
    if status not in {"FROZEN_PILOT", "FROZEN_FORMAL"}:
        raise RuntimeError("Step-4 contract is not frozen")
    if args.seed not in contract["training"]["seeds"]:
        raise RuntimeError("Step-4 seed is outside the frozen contract")
    pilot_variants = contract["training"].get(
        "pilot_variants", ["a4_full"]
    )
    if status == "FROZEN_PILOT" and args.variant not in pilot_variants:
        raise RuntimeError(
            "loss-scale pilot variant is outside the frozen pilot arms"
        )
    if (
        status == "FROZEN_FORMAL"
        and args.updates != MAX_UPDATES
        and not (args.resume_audit and 1 <= args.updates <= 4)
    ):
        raise RuntimeError("formal Step-4 training requires 120000 updates")
    if (
        status == "FROZEN_FORMAL"
        and args.save_every != EVAL_EVERY
        and not args.resume_audit
    ):
        raise RuntimeError("formal Step-4 checkpoint interval is 5000")
    if status == "FROZEN_PILOT":
        pilot = contract["training"]["loss_scale_pilot"]
        if args.seed != int(pilot["seed"]):
            raise RuntimeError("pilot seed differs from the frozen contract")
        if args.updates != int(pilot["updates"]):
            raise RuntimeError("pilot update target differs from the frozen contract")
        if args.save_every != int(pilot["evaluation_interval"]):
            raise RuntimeError(
                "pilot checkpoint interval differs from the frozen evaluation interval"
            )
    if not 1 <= args.updates <= MAX_UPDATES:
        raise ValueError("invalid Step-4 update target")
    if args.resume_audit and (args.updates > 4 or args.evaluate_at_end):
        raise ValueError("resume audit is limited to four updates without evaluation")
    if sha256_file(args.scenario_split) != contract["inputs"][
        "scenario_split_sha256"
    ]:
        raise RuntimeError("Step-4 scenario split hash differs")
    root = Path(__file__).resolve().parents[1]
    verify_source_receipt(contract, root)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(min(12, os.cpu_count() or 12))
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = PredictiveTeamBeliefDataset(
        args.cache, args.action_context_cache
    )
    split = split_by_episode_key(load_split(args.scenario_split))
    config = config_from_contract(contract)
    weights = relative_weights(contract, args.variant)
    model = BaseRelativeBeliefExperiment(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 if step < LR_DROP else 0.1
    )
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
        "variant": args.variant,
        "contract_sha256": sha256_file(args.contract),
        "scenario_split_sha256": sha256_file(args.scenario_split),
        "n1_metadata_sha256": sha256_file(args.cache / "metadata.json"),
        "action_context_cache_receipt_sha256": sha256_file(
            args.action_context_cache / "cache_receipt.json"
        ),
    }
    if saved:
        if saved["provenance"] != provenance:
            raise RuntimeError("Step-4 resume provenance differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"])
        metrics = list(saved["evaluations"])
    sampler = PairedSituationBatchSampler(
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
            "variant": args.variant,
            "update": start,
            "target_updates": args.updates,
            "started_at_utc": utc_now(),
        },
    )
    started = time.time()
    last: dict[str, float] = saved.get("last_losses", {}) if saved else {}
    for update, raw in enumerate(loader, start=start + 1):
        if update > args.updates:
            break
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed)
        np.random.seed(step_seed % 2**32)
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        batch = device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            partner = paired_permutation(batch["pair_id"])
            swapped_belief = replace(
                output.candidate.belief,
                mu=output.candidate.belief.mu[partner],
            )
            swapped_output = replace(
                output.candidate, belief=swapped_belief
            )
            negative = shuffle_permutation(
                batch["task_index"], batch["phase_bin"]
            )
            counterfactual_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                output.candidate.belief.mu[negative],
                output.candidate.belief.sigma[negative],
                output.candidate.belief.reliability[negative],
            )
            counterfactual_prediction = (
                batch["base_action"] + counterfactual_residual
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
                weights,
                swapped_output=swapped_output,
                counterfactual_prediction=counterfactual_prediction,
                counterfactual_residual_target=residual_target[negative],
                counterfactual_action_mask=batch["action_mask"][negative],
            )
            direct = masked_action_mse(
                output.direct_prediction,
                batch["action"],
                batch["action_mask"],
            )
            loss = losses["total"] + float(
                contract["objectives"]["direct_reactive_action"]
            ) * direct
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Step-4 loss at {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite Step-4 gradient at {update}"
            )
        optimizer.step()
        scheduler.step()
        last = {name: float(value.detach()) for name, value in losses.items()}
        last["direct_reactive_action"] = float(direct.detach())
        last["combined"] = float(loss.detach())
        if (
            update == start + 1
            or update % args.log_every == 0
            or update == args.updates
        ):
            elapsed = time.time() - started
            completed = update - start
            row = {
                "update": update,
                "target_updates": args.updates,
                "variant": args.variant,
                **last,
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "updates_per_hour": completed / max(elapsed, 1e-6) * 3600,
                "eta_hours": (args.updates - update)
                * elapsed
                / max(completed, 1)
                / 3600,
                "gpu_memory_gb": round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 2
                ),
                "updated_at_epoch": time.time(),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            with (args.output / "progress.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            atomic_json(args.output / "heartbeat.json", row)
        evaluation_interval = (
            int(contract["training"]["loss_scale_pilot"]["evaluation_interval"])
            if status == "FROZEN_PILOT"
            else EVAL_EVERY
        )
        should_evaluate = update % evaluation_interval == 0 or (
            args.evaluate_at_end and update == args.updates
        )
        if should_evaluate:
            validation_metrics = evaluate(
                model, validation, device, weights.base
            )
            validation_metrics["base_relative"] = evaluate_base_relative(
                model, validation, device
            )
            evaluation = {
                "update": update,
                "train": last,
                "validation": validation_metrics,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            metrics.append(evaluation)
            with (args.output / "evaluations.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(evaluation, sort_keys=True) + "\n")
            print(
                json.dumps({"evaluation": evaluation}, sort_keys=True),
                flush=True,
            )
        if update == args.updates or update % args.save_every == 0:
            checkpoint = {
                "format_version": "before-we-act.a4-training-checkpoint/1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update": update,
                "evaluations": metrics,
                "last_losses": last,
                "sample_cursor": sampler.cursor_receipt(update),
                "provenance": provenance,
                "config": config.__dict__,
            }
            atomic_save(checkpoint, latest)
            atomic_save(
                checkpoint, args.output / f"checkpoint_{update:06d}.pt"
            )
    if args.updates < MAX_UPDATES:
        guard = contract.get("pilot_sensitivity_guard", {})
        selected_validation = metrics[-1]["validation"] if metrics else None
        guard_checks: dict[str, bool] = {}
        if guard:
            if selected_validation is None:
                guard_checks["validation_was_run"] = False
            else:
                pairing = selected_validation["action_pairing"]
                macro = selected_validation["macro"]
                guard_checks = {
                    "validation_was_run": True,
                    "output_to_target_sensitivity_bounded": float(
                        pairing["output_to_target_sensitivity"]
                    )
                    <= float(guard["output_to_target_sensitivity_max"]),
                    "output_to_residual_energy_bounded": float(
                        pairing["output_to_residual_energy"]
                    )
                    <= float(guard["output_to_residual_energy_max"]),
                    "shuffle_to_matched_ratio_bounded": float(
                        macro["b_shuffle"]
                    )
                    / max(float(macro["b_core"]), 1e-12)
                    <= float(guard["shuffle_to_matched_ratio_max"]),
                }
        passed_guard = not guard_checks or all(guard_checks.values())
        pilot_status = {
            "status": (
                "PASSED_LOSS_SCALE_PILOT"
                if passed_guard
                else "FAILED_LOSS_SCALE_PILOT_SENSITIVITY_GUARD"
            ),
            "seed": args.seed,
            "variant": args.variant,
            "update": args.updates,
            "target_updates": args.updates,
            "last_losses": last,
            "sensitivity_guard": {
                "thresholds": guard,
                "checks": guard_checks,
                "passed": passed_guard,
            },
            "completed_at_utc": utc_now(),
        }
        if metrics:
            pilot_status["selected_validation"] = selected_validation
        atomic_json(args.output / "status.json", pilot_status)
        return

    sufficiency_status, sufficiency = training_sufficiency(metrics)
    selection_updates = set(
        contract["training"]["selection_window_updates"]
    )
    candidates = [
        row for row in metrics if int(row["update"]) in selection_updates
    ]
    selected = min(
        candidates, key=lambda row: row["validation"]["macro"]["b_core"]
    )
    sufficiency_path = args.output / "training_sufficiency.json"
    atomic_json(
        sufficiency_path,
        {
            "format_version": "before-we-act.a4-training-sufficiency/1",
            "status": sufficiency_status,
            "variant": args.variant,
            "minimum_exposure_met": True,
            "u_b0h": MAX_UPDATES,
            "learning_rate_drop_completed": True,
            "selected_update": int(selected["update"]),
            "all_evaluation_points": [int(row["update"]) for row in metrics],
            "receipt": sufficiency,
        },
    )
    selected_checkpoint = args.output / (
        f"checkpoint_{int(selected['update']):06d}.pt"
    )
    deployment = args.output / "deployment_checkpoint.pt"
    safety_calibration = selected["validation"][
        "deployment_safety_calibration"
    ]
    safety_config = calibrated_residual_safety(safety_calibration).to_dict()
    safety_config["calibration"] = safety_calibration
    export_deployment(
        selected_checkpoint,
        deployment,
        contract,
        config,
        residual_safety=safety_config,
    )
    customize_deployment(deployment, args.variant, args.contract)
    atomic_json(
        args.output / "status.json",
        {
            "status": sufficiency_status,
            "seed": args.seed,
            "variant": args.variant,
            "update": MAX_UPDATES,
            "selected_update": int(selected["update"]),
            "selected_validation": selected["validation"],
            "training_sufficiency_sha256": sha256_file(sufficiency_path),
            "deployment_checkpoint_sha256": sha256_file(deployment),
            "elapsed_hours": (time.time() - started) / 3600,
            "completed_at_utc": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
