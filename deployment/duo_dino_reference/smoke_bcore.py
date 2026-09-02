"""Duo B-core contract smoke: train four updates and test real deployment wiring."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from before_we_act.temporal_history_data import task_text_tensor
from .bcore_data import (
    BCORE_UPDATES,
    DUO_BELIEF_CONFIG,
    DuoBeliefRequest,
    DuoTeamBeliefDataset,
    sha256_file,
    validate_b0h_payload,
)
from .data import ACTION_HORIZON, ACTION_LAG_ROWS, TASKS, TASK_TEXT, load_duo_episodes, resize_rgb_batch
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _instantiate_policy(
    b0h_checkpoint: Path,
    *,
    dino_model: str | None,
    device: torch.device,
) -> tuple[TemporalHistoryPolicy, PredictiveTeamBeliefPolicy, dict[str, Any]]:
    payload = torch.load(b0h_checkpoint, map_location="cpu", weights_only=False)
    config = dict(validate_b0h_payload(payload))
    model_name = str(dino_model or config.get("dino_model") or "")
    if not model_name:
        raise ValueError("B-core smoke has no DINO model")
    b0h = TemporalHistoryPolicy(
        state_dim=8,
        action_dim=8,
        variant="hidden_residual",
        horizon=ACTION_HORIZON,
        d_model=int(config.get("d_model", 384)),
        enc_layers=int(config.get("enc_layers", 4)),
        dec_layers=int(config.get("dec_layers", 7)),
        roles=int(config.get("roles", 4)),
        role_rank=int(config.get("role_rank", 32)),
        history_layers=int(config.get("history_layers", 2)),
        dino_model=model_name,
        image_height=int(config.get("image_height", 224)),
        image_width=int(config.get("image_width", 224)),
        strict_dino_contract=True,
    ).to(device)
    b0h.load_state_dict(payload["model"], strict=True)
    b0h.eval()
    policy = PredictiveTeamBeliefPolicy(
        DUO_BELIEF_CONFIG,
        state_dim=8,
        action_dim=8,
        horizon=ACTION_HORIZON,
        d_model=384,
        enc_layers=int(config.get("enc_layers", 4)),
        dec_layers=int(config.get("dec_layers", 7)),
        roles=int(config.get("roles", 4)),
        role_rank=int(config.get("role_rank", 32)),
        history_layers=int(config.get("history_layers", 2)),
        dino_model=model_name,
        image_height=int(config.get("image_height", 224)),
        image_width=int(config.get("image_width", 224)),
        strict_dino_contract=True,
        include_teacher=False,
        residual_safety={"enabled": False},
    ).to(device)
    incompatible = policy.load_state_dict(payload["model"], strict=False)
    expected_missing = {
        key
        for key in policy.state_dict()
        if key.startswith(("belief_core.", "direct_belief_residual."))
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"B-core policy/B0-H attachment differs: {incompatible}"
        )
    policy.eval()
    return b0h, policy, config


def _check_b_off(
    prepared_data: Path,
    visual_cache: Path,
    bcore_cache: Path,
    b0h_checkpoint: Path,
    *,
    dino_model: str | None,
    device: torch.device,
) -> dict[str, Any]:
    b0h, policy, config = _instantiate_policy(
        b0h_checkpoint, dino_model=dino_model, device=device
    )
    episodes = load_duo_episodes(prepared_data, require_formal=True)
    dataset = DuoTeamBeliefDataset(
        prepared_data, episodes, visual_cache, bcore_cache, cache_limit=2
    )
    episode_index = next(i for i, episode in enumerate(episodes) if episode.task == TASKS[0])
    episode = episodes[episode_index]
    request = DuoBeliefRequest(
        episode_index,
        0,
        min(1, episode.length - ACTION_LAG_ROWS - 1),
        "bcore-smoke",
        TASKS[0],
    )
    raw = default_collate([dataset[request]])
    batch = {
        key: value.to(device)
        for key, value in raw.items()
        if isinstance(value, torch.Tensor)
    }
    task_root = prepared_data / episode.task
    absolute = episode.start + request.time_index
    global_rgb = resize_rgb_batch(
        np.array(np.load(task_root / "head.npy", mmap_mode="r")[absolute], copy=True),
        int(config.get("image_height", 224)),
        int(config.get("image_width", 224)),
    ).unsqueeze(0).to(device)
    local_rgb = resize_rgb_batch(
        np.array(np.load(task_root / "left.npy", mmap_mode="r")[absolute], copy=True),
        int(config.get("image_height", 224)),
        int(config.get("image_width", 224)),
    ).unsqueeze(0).to(device)
    task_bytes, task_text_mask = task_text_tensor(TASK_TEXT[episode.task])
    base_inputs = {
        "global_rgb": global_rgb,
        "local_rgb": local_rgb,
        "history_visual_raw": batch["runtime_visual_tokens"][:, :, :, 0],
        "history_qpos": batch["history_qpos"],
        "history_action": batch["history_action"],
        "history_mask": batch["history_mask"],
        "action_history_mask": batch["action_history_mask"],
        "task_bytes": task_bytes.unsqueeze(0).to(device),
        "task_text_mask": task_text_mask.unsqueeze(0).to(device),
        "episode_reset_mask": batch["episode_reset_mask"],
    }
    # The temporal backbone uses the historical ``episode_reset`` name.  The
    # B-core policy additionally requires the explicit reset mask and receives
    # it separately; this call deliberately tests both paths on identical data.
    base_inputs["episode_reset"] = batch["episode_reset_mask"][:, -1]
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        bcore_output = policy(
            global_rgb=base_inputs["global_rgb"].float().div_(255),
            local_rgb=base_inputs["local_rgb"].float().div_(255),
            history_visual_raw=batch["runtime_visual_tokens"][:, :, :, 0],
            history_qpos=batch["history_qpos"],
            history_action=batch["history_action"],
            history_mask=batch["history_mask"],
            action_history_mask=batch["action_history_mask"],
            task_bytes=base_inputs["task_bytes"],
            task_text_mask=base_inputs["task_text_mask"],
            episode_reset=batch["episode_reset_mask"][:, -1],
            episode_reset_mask=batch["episode_reset_mask"],
            actions=None,
            belief_enabled=False,
        )
    # The cache's RGB tensors are uint8; b0h above needs [0,1].  Re-run with
    # the normalized tensors to keep the comparison meaningful.
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        b0h_prediction = b0h(
            global_rgb=base_inputs["global_rgb"].float().div_(255),
            local_rgb=base_inputs["local_rgb"].float().div_(255),
            history_visual_raw=base_inputs["history_visual_raw"],
            history_qpos=base_inputs["history_qpos"],
            history_action=base_inputs["history_action"],
            history_mask=base_inputs["history_mask"],
            action_history_mask=base_inputs["action_history_mask"],
            task_bytes=base_inputs["task_bytes"],
            task_text_mask=base_inputs["task_text_mask"],
            episode_reset=base_inputs["episode_reset"],
        )[0]
    torch.testing.assert_close(
        bcore_output.base_prediction,
        b0h_prediction,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        bcore_output.prediction,
        bcore_output.base_prediction,
        rtol=0.0,
        atol=0.0,
    )
    teacher_keys = [
        key
        for key in policy.state_dict()
        if key.startswith("belief_core.teacher_branch.")
    ]
    return {
        "policy_instantiated": True,
        "b_off_matches_b0h": True,
        "teacher_absent": not teacher_keys,
        "b0h_prediction_shape": list(b0h_prediction.shape),
        "bcore_belief_shape": list(bcore_output.belief.mu.shape),
        "source_frequency_hz": DUO_BELIEF_CONFIG.source_frequency_hz,
        "future_offsets_steps": list(DUO_BELIEF_CONFIG.future_offsets_steps),
        "future_offsets_seconds": list(DUO_BELIEF_CONFIG.future_offsets_seconds),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.updates != 4:
        raise ValueError("Duo B-core smoke is frozen to four updates")
    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    report: dict[str, Any] = {
        "schema": "before-we-act.duobench.bcore-smoke/1",
        "status": "FAILED",
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "benchmark_adapter": "DuoBench",
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "strict_dino_contract": True,
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "all_550_demonstrations": True,
        "checks": {},
    }
    # Four updates are executed by the same formal trainer, not a synthetic
    # no-op.  This catches data/collation/loss and checkpoint wiring failures.
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "deployment.duo_dino_reference.train_bcore",
        "--prepared-data",
        str(args.prepared_data),
        "--visual-cache",
        str(args.visual_cache),
        "--bcore-cache",
        str(args.bcore_cache),
        "--b0h-checkpoint",
        str(args.b0h_checkpoint),
        "--output",
        str(output),
        "--seed",
        str(args.seed),
        "--updates",
        "4",
        "--stage",
        "smoke",
        "--workers",
        str(args.workers),
        "--batch-size",
        "48",
        "--save-every",
        "1",
        "--eval-every",
        "4",
    ]
    if args.dino_model:
        command.extend(("--dino-model", str(args.dino_model)))
    subprocess.run(command, check=True)
    report["checks"]["four_update_training"] = (
        (output / "checkpoint_latest.pt").is_file()
        and json.loads((output / "status.json").read_text()).get("status")
        == "PASSED_SMOKE"
    )
    try:
        report["checks"].update(
            {
                "real_predictive_policy": True,
                **{
                    key: value
                    for key, value in _check_b_off(
                        args.prepared_data,
                        args.visual_cache,
                        args.bcore_cache,
                        args.b0h_checkpoint,
                        dino_model=args.dino_model,
                        device=device,
                    ).items()
                    if key
                    in (
                        "b_off_matches_b0h",
                        "teacher_absent",
                        "policy_instantiated",
                    )
                },
                "future_time_contract_30hz": (
                    DUO_BELIEF_CONFIG.source_frequency_hz == 30
                    and DUO_BELIEF_CONFIG.future_offsets_steps == (6, 12, 24, 48)
                ),
            }
        )
    except Exception as error:
        report["checks"]["real_predictive_policy"] = False
        report["error"] = repr(error)
    report["updates"] = 4
    report["checkpoint_sha256"] = (
        sha256_file(output / "checkpoint_latest.pt")
        if (output / "checkpoint_latest.pt").is_file()
        else None
    )
    report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    _atomic_json(output / "smoke_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "PASSED":
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--bcore-cache", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke",), default="smoke")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="")
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()


__all__ = ["run_smoke"]
