#!/usr/bin/env python3
"""Evaluate pre-defined B-core token interventions on frozen validation data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from before_we_act.action_grounded_belief import load_split, split_by_episode_key
from before_we_act.deployment_safety import ResidualSafetyConfig
from before_we_act.predictive_team_belief_data import PredictiveTeamBeliefDataset
from before_we_act.predictive_team_belief_policy import DirectBeliefResidual
from before_we_act.team_belief.predictive_core import (
    PredictiveTeamBeliefCore,
    TeamBeliefConfig,
)
from before_we_act.train_predictive_team_belief import (
    device_batch,
    fixed_loader,
    row_action_mse,
    shuffle_permutation,
)


CONDITIONS = (
    "full",
    "teammate_anchor_shuffle",
    "interaction_slots_shuffle",
    "ego_anchor_shuffle",
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    selected = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"checkpoint has no state under {prefix!r}")
    return selected


def intervene_mu(
    mu: torch.Tensor, permutation: torch.Tensor, condition: str, n_agent_anchors: int
) -> torch.Tensor:
    if mu.ndim != 3 or n_agent_anchors != 2 or mu.shape[1] <= n_agent_anchors:
        raise ValueError("token intervention requires two agent anchors and free slots")
    if permutation.shape != (mu.shape[0],):
        raise ValueError("intervention permutation shape differs")
    result = mu.clone()
    if condition == "full":
        return result
    if condition == "teammate_anchor_shuffle":
        result[:, 1] = mu[permutation, 1]
    elif condition == "interaction_slots_shuffle":
        result[:, n_agent_anchors:] = mu[permutation, n_agent_anchors:]
    elif condition == "ego_anchor_shuffle":
        result[:, 0] = mu[permutation, 0]
    else:
        raise ValueError(f"unknown intervention: {condition}")
    return result


def cluster_bootstrap_ci(
    differences: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    samples: int = 20_000,
) -> tuple[float, float]:
    clusters = np.unique(cluster_ids)
    cluster_means = np.asarray(
        [differences[cluster_ids == cluster].mean() for cluster in clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1_000):
        count = min(1_000, samples - start)
        draw = rng.integers(0, len(cluster_means), size=(count, len(cluster_means)))
        estimates[start : start + count] = cluster_means[draw].mean(1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def mechanism_gate(row: Mapping[str, object]) -> bool:
    return (
        float(row["mse_increase"]) > 0.0
        and float(row["episode_bootstrap_ci95"][0]) > 0.0
        and int(row["positive_tasks"]) >= 4
    )


def load_runtime(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "before-we-act.b3-n2-deployment-checkpoint/1":
        raise ValueError("wrong B-core deployment checkpoint format")
    values = dict(saved["config"]["n2_config"])
    for key in ("future_offsets_steps", "future_offsets_seconds"):
        if key in values:
            values[key] = tuple(values[key])
    config = TeamBeliefConfig(**values)
    core = PredictiveTeamBeliefCore(config, include_teacher=False).to(device)
    core.load_state_dict(prefixed_state(saved["model"], "belief_core."), strict=True)
    residual = DirectBeliefResidual(
        config.d_model,
        config.action_dim,
        safety=ResidualSafetyConfig.from_mapping(saved["config"].get("residual_safety")),
    ).to(device)
    residual.load_state_dict(
        prefixed_state(saved["model"], "direct_belief_residual."), strict=True
    )
    core.eval()
    residual.eval()
    return core, residual, config, saved


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    loader,
    device: torch.device,
    *,
    expected_full_mse: float,
    bootstrap_seed: int,
) -> dict:
    core, residual, config, saved = load_runtime(checkpoint, device)
    losses = {condition: [] for condition in CONDITIONS}
    tasks: list[int] = []
    episodes: list[int] = []
    shuffled_rows = 0
    total_rows = 0
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            belief = core(
                batch["runtime_visual_tokens"],
                batch["runtime_visual_mask"],
                batch["history_qpos"],
                batch["history_action"],
                batch["history_mask"],
                batch["action_history_mask"],
                batch["task_token"],
                batch["episode_reset_mask"],
            )
            permutation = shuffle_permutation(
                batch["task_index"], batch["phase_bin"]
            )
            shuffled_rows += int((permutation != torch.arange(len(permutation), device=device)).sum())
            total_rows += len(permutation)
            for condition in CONDITIONS:
                mu = intervene_mu(
                    belief.mu,
                    permutation,
                    condition,
                    config.n_agent_anchors,
                )
                candidate_residual, _ = residual(
                    batch["decoded_action_hidden"],
                    mu,
                    belief.sigma,
                    belief.reliability,
                )
                prediction = batch["base_action"] + candidate_residual
                row = row_action_mse(
                    prediction, batch["action"], batch["action_mask"]
                )
                losses[condition].extend(row.float().cpu().tolist())
        tasks.extend(batch["task_index"].cpu().tolist())
        episodes.extend((batch["pair_id"] // 1_000_000).cpu().tolist())

    arrays = {
        condition: np.asarray(values, dtype=np.float64)
        for condition, values in losses.items()
    }
    task_array = np.asarray(tasks, dtype=np.int64)
    episode_array = np.asarray(episodes, dtype=np.int64)
    full_mse = float(arrays["full"].mean())
    if abs(full_mse - expected_full_mse) > 5e-8:
        raise RuntimeError(
            f"deployment validation MSE {full_mse} does not reproduce {expected_full_mse}"
        )
    interventions = {}
    for condition in CONDITIONS[1:]:
        difference = arrays[condition] - arrays["full"]
        per_task = {
            str(task): {
                "full_mse": float(arrays["full"][task_array == task].mean()),
                "intervention_mse": float(arrays[condition][task_array == task].mean()),
                "mse_increase": float(difference[task_array == task].mean()),
            }
            for task in range(6)
        }
        ci = cluster_bootstrap_ci(
            difference,
            episode_array,
            seed=bootstrap_seed + CONDITIONS.index(condition),
        )
        row = {
            "intervention_mse": float(arrays[condition].mean()),
            "mse_increase": float(difference.mean()),
            "relative_mse_increase_percent": float(
                100.0 * difference.mean() / max(full_mse, 1e-12)
            ),
            "episode_bootstrap_ci95": list(ci),
            "positive_tasks": sum(
                value["mse_increase"] > 0.0 for value in per_task.values()
            ),
            "per_task": per_task,
        }
        row["passed"] = mechanism_gate(row)
        interventions[condition] = row
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_training_update": int(saved["update"]),
        "rows": len(task_array),
        "episodes": len(np.unique(episode_array)),
        "shuffled_rows": shuffled_rows,
        "shuffled_fraction": shuffled_rows / max(total_rows, 1),
        "full_mse": full_mse,
        "expected_full_mse": expected_full_mse,
        "interventions": interventions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--action-context-cache", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_MECHANISM_RESULT":
        raise RuntimeError("mechanism contract is not frozen")
    for key, path in (
        ("scenario_split", args.scenario_split),
        ("signal_cache_metadata", args.cache / "metadata.json"),
        ("action_context_cache_receipt", args.action_context_cache / "cache_receipt.json"),
    ):
        if sha256_file(path) != contract["immutable_inputs"][key]["sha256"]:
            raise RuntimeError(f"immutable input drifted: {key}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(min(12, os.cpu_count() or 12))
    dataset = PredictiveTeamBeliefDataset(args.cache, args.action_context_cache)
    split = split_by_episode_key(load_split(args.scenario_split))
    seed_results = {}
    for index, (seed, specification) in enumerate(
        sorted(contract["immutable_inputs"]["deployments"].items())
    ):
        checkpoint = Path(specification["path"])
        if sha256_file(checkpoint) != specification["sha256"]:
            raise RuntimeError(f"deployment checkpoint drifted: {seed}")
        validation = fixed_loader(dataset, split, "validation")
        seed_results[seed] = evaluate_checkpoint(
            checkpoint,
            validation,
            device,
            expected_full_mse=float(specification["expected_full_mse"]),
            bootstrap_seed=int(contract["bootstrap_seed"]) + index * 100,
        )

    primary_pass = all(
        row["interventions"]["teammate_anchor_shuffle"]["passed"]
        for row in seed_results.values()
    )
    interaction_pass = all(
        row["interventions"]["interaction_slots_shuffle"]["passed"]
        for row in seed_results.values()
    )
    if primary_pass:
        status = "TEAMMATE_ANCHOR_SUPPORTED"
    elif interaction_pass:
        status = "INTERACTION_STATE_SUPPORTED_TEAMMATE_ANCHOR_NOT_SUPPORTED"
    else:
        status = "TARGETED_MECHANISM_NOT_SUPPORTED"
    result = {
        "format_version": "before-we-act.b3-n3-targeted-token-intervention/1",
        "stage": contract["stage"],
        "status": status,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "primary_teammate_anchor_passed_all_seeds": primary_pass,
        "secondary_interaction_slots_passed_all_seeds": interaction_pass,
        "seeds": seed_results,
        "interpretation_boundary_zh": [
            "相对队友锚点是架构事前定义的位置，不是看结果后选择的潜在因子。",
            "同任务同阶段置换保留边际数值分布，只删除当前局面的对应信息。",
            "这是冻结验证集上的动作机制检查，不是新的闭环成功率或最终统计确认。",
            "若只有自由交互位置通过，只能声称一般交互预测状态，不能升级成明确队友锚点贡献。"
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
