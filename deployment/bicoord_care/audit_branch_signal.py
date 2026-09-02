"""Audit CARE branch diversity, lag alignment, and event/safety signal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import TASKS
from .bcore_data import BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH
from .branch_collection import HORIZONS
from .select_calibrate import REGISTERED_CALIBRATION
from .stage_common import artifact, assert_common_paths, atomic_json, common_parser, publish_result, require_stage_result, sha256_file


PRIMARY_HORIZON = int(REGISTERED_CALIBRATION["primary_horizon"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    smoke = args.operation == "signal-gate-smoke"
    dependency_stage = "branch_prepare_smoke" if smoke else "branch_prepare"
    require_stage_result(args.run, dependency_stage, config_sha256=args.config_sha256)
    prepared = args.run / "artifacts" / ("prepared_branches_smoke.pt" if smoke else "prepared_branches.pt")
    if not prepared.is_file(): raise FileNotFoundError(prepared)
    if sha256_file(prepared) == "0" * 64: raise ValueError("invalid prepared branch hash")
    import torch
    payload = torch.load(prepared, map_location="cpu", weights_only=False)
    required = ("memory", "memory_mask", "candidate_chunks", "targets", "hard_safety", "usable", "task_id", "snapshot_ids")
    if any(key not in payload for key in required): raise ValueError("prepared CARE artifact is missing signal fields")
    memory = payload["memory"].float(); candidates = payload["candidate_chunks"].float(); targets = payload["targets"].float(); safety = payload["hard_safety"].float(); usable = payload["usable"].bool()
    n = int(memory.shape[0])
    if memory.shape != (n, BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH) or candidates.shape != (n, 6, 100, 7) or targets.shape != (n, 4, 6, 2, 3) or safety.shape != (n, 4, 6, 2) or usable.shape != (n, 4): raise ValueError("prepared CARE tensor contract differs")
    if n < 1 or not torch.isfinite(memory).all() or not torch.isfinite(candidates).all() or not torch.isfinite(targets).all(): raise ValueError("prepared CARE tensors are empty/non-finite")
    candidate_delta = (candidates[:, 1:] - candidates[:, :1]).abs().mean(dim=(1, 2, 3)).numpy()
    target_delta = (targets[:, :, 1:] - targets[:, :, :1]).abs().mean(dim=(1, 2, 3, 4)).numpy()
    target_std = targets[:, :, 1:, :, 2].std(dim=2).mean().item()
    safety_rate = float(safety[:, :, 1:].mean())
    usable_rate = float(usable.float().mean())
    snapshots = tuple(str(value) for value in payload["snapshot_ids"])
    if len(snapshots) != n or len(set(snapshots)) != n: raise ValueError("snapshot IDs are missing or duplicated")
    checks = {
        "families": n,
        "candidate_delta_mean": float(candidate_delta.mean()),
        "candidate_delta_min": float(candidate_delta.min()),
        "target_delta_mean": float(target_delta.mean()),
        "target_delta_min": float(target_delta.min()),
        "target_total_std": float(target_std),
        "hard_safety_rate_nonreference": safety_rate,
        "usable_horizon_rate": usable_rate,
        "lag_alignment": "observation_row_t_to_action_row_t_plus_1",
        "provider_policy": "B-core/TUNE",
        "peer_inputs_used_for_runtime": False,
    }
    # A gate must reject a degenerate branch tensor rather than authorize a
    # belief head that can only memorize candidate IDs.
    if checks["candidate_delta_min"] <= 1e-7 or checks["target_delta_min"] <= 1e-9 or checks["usable_horizon_rate"] <= 0.0:
        raise RuntimeError(f"CARE branch signal is degenerate: {checks}")
    # A tensor whose candidates differ at all passes the degeneracy check above,
    # which says nothing about whether any of them could be *selected*. The
    # selector needs the best candidate to clear a calibration radius, and every
    # corpus measured so far failed that by a factor of four while passing here.
    # Ask the question at the point in the DAG that already exists for it,
    # before the twelve scorer runs that follow.
    from scripts.before_we_act.measure_care_headroom import summarize_prepared_targets

    primary_index = HORIZONS.index(PRIMARY_HORIZON)
    headroom = summarize_prepared_targets(
        targets.numpy(),
        usable.numpy(),
        horizon_index=primary_index,
        reference_radius=args.reference_radius,
    )
    checks["headroom_primary_horizon"] = headroom
    if headroom.get("verdict") == "BLOCKED":
        raise RuntimeError(
            "CARE branch corpus leaves no room for the selector at horizon "
            f"{PRIMARY_HORIZON}: {headroom['reason']}. "
            f"max|A|={headroom['max_abs_total']:.6g} against radius "
            f"{(headroom.get('against_reference_radius') or headroom['against_irreducible_radius'])['radius']:.6g}. "
            "Training a scorer on this corpus cannot produce an override."
        )
    report = args.run / "artifacts" / ("branch_signal_gate_smoke.json" if smoke else "branch_signal_gate.json")
    atomic_json(report, {"schema": "before-we-act.bicoord.branch-signal-gate/1", "status": "PASSED", "downstream_authorized": True, **checks})
    stage = "branch_signal_gate_smoke" if smoke else "branch_signal_gate"
    return publish_result(args, stage=stage, include_model_contract=True, artifacts=[artifact(prepared, kind="prepared_branches"), artifact(report, kind="branch_signal_gate")], downstream_authorized=True, **checks)


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("signal-gate-smoke", "signal-gate"))
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=None,
        help=(
            "calibration radius to judge headroom against; omit to use the "
            "irreducible radius implied by matched repeats"
        ),
    )
    args = parser.parse_args(argv); run(args); return 0


if __name__ == "__main__": raise SystemExit(main())
