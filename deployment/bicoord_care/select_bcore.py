"""Select the formal BiCoord B-core/TUNE seed using offline diagnostics only.

Selection is intentionally completed before the Validation20 stage.  It reads
only deterministic cached-label diagnostics emitted by ``train_bcore`` and
uses a stable ``(MSE, seed)`` tie-break.  A deployment checkpoint is copied
byte-for-byte after being validated as a genuine teacher-free
``PredictiveTeamBeliefPolicy`` state; no closed-loop result is consulted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import torch

from .bcore_data import (
    BCORE_DEPLOYMENT_FORMAT,
    BCORE_SEEDS,
    BCORE_TRAINING_FORMAT,
    BICOORD_BELIEF_CONFIG,
    BICOORD_CARE_MEMORY_SEMANTICS,
    BICOORD_CARE_MEMORY_TOKENS,
    BICOORD_CARE_MEMORY_WIDTH,
    BICOORD_FUTURE_OFFSETS_STEPS,
    BICOORD_SOURCE_FREQUENCY_HZ,
    validate_b0h_payload,
)
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    DATASET_REVISION,
    D_MODEL,
    DECODER_LAYERS,
    ENCODER_LAYERS,
    HISTORY_LAYERS,
    HISTORY_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ROLES,
    ROLE_RANK,
    STATE_DIM,
    TASKS,
    TOTAL_EPISODES,
    VISION_BACKBONE,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID
from .stage_common import (
    artifact,
    atomic_json,
    common_parser,
    publish_result,
    read_json,
    sha256_file,
)
from .train_bcore import FINAL_SUFFICIENCY_WINDOW, validate_deployment_payload


SELECTION_SCHEMA = "before-we-act.bicoord.bcore-selection/1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _offline_score(payload: Mapping[str, Any]) -> tuple[float, int]:
    """Return the best score/update from the fixed final sufficiency window."""

    evaluations = payload.get("evaluations")
    rows: list[tuple[int, float]] = []
    if isinstance(evaluations, list):
        for item in evaluations:
            if not isinstance(item, Mapping):
                continue
            try:
                update = int(item["update"])
                score = float(item["validation"]["macro"]["b_core"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if update >= 120_000 - FINAL_SUFFICIENCY_WINDOW and score == score and score < float("inf"):
                rows.append((update, score))
    if not rows:
        raise ValueError(
            "B-core seed has no finite b_core offline diagnostic in the final 20k updates"
        )
    update, score = min(rows, key=lambda row: (row[1], row[0]))
    return float(score), int(update)


def _load_mapping(path: Path, context: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} is not a checkpoint mapping: {path}")
    return payload


def _validate_training_checkpoint(
    root: Path,
    *,
    seed: int,
    b0h_sha: str,
    expected_cache_sha: str | None,
    expected_norm_sha: str | None,
) -> tuple[Mapping[str, Any], Path, float, int]:
    status_path = root / "status.json"
    status = read_json(status_path)
    if status.get("status") not in {"COMPLETED", "PASSED", "PASSED_SMOKE"}:
        raise ValueError(f"B-core seed {seed} has non-terminal status {status.get('status')!r}")
    checkpoint = root / "checkpoint_latest.pt"
    payload = _load_mapping(checkpoint, f"B-core seed {seed} checkpoint")
    if (payload.get("format") or payload.get("format_version")) != BCORE_TRAINING_FORMAT:
        raise ValueError(f"B-core seed {seed} has wrong training format")
    if int(payload.get("update", -1)) != 120_000:
        raise ValueError(f"B-core seed {seed} did not complete 120000 updates")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"B-core seed {seed} has no provenance")
    expected_provenance = {
        "seed": seed,
        "policy_episode_count": TOTAL_EPISODES,
        "policy_training_split": "all_1800_demonstrations_no_holdout",
        "b0h_checkpoint_sha256": b0h_sha,
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"B-core seed {seed} provenance differs at {key}: "
                f"{provenance.get(key)!r} != {value!r}"
            )
    if expected_cache_sha is not None and provenance.get("bcore_cache_receipt_sha256") != expected_cache_sha:
        raise ValueError(f"B-core seed {seed} cache provenance differs")
    if expected_norm_sha is not None and provenance.get("normalization_receipt_sha256") != expected_norm_sha:
        raise ValueError(f"B-core seed {seed} normalization provenance differs")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"B-core seed {seed} has no config")
    required = {
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "BiCoord",
        "vision_backbone": VISION_BACKBONE,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "action_encoding": ACTION_ENCODING,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
        "d_model": D_MODEL,
        "enc_layers": ENCODER_LAYERS,
        "dec_layers": DECODER_LAYERS,
        "roles": ROLES,
        "role_rank": ROLE_RANK,
        "history_layers": HISTORY_LAYERS,
        "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
        "strictly_decentralized": True,
        "strict_local": True,
        "shared_weights": True,
        "shared_checkpoint_for_both_arms": True,
        "arm_id_input": False,
        "peer_runtime_input": False,
        "act_provider_allowed": False,
        "strict_dino_contract": True,
        "teacher_training_only": True,
        "all_1800_demonstrations": True,
        "held_out_demonstrations": 0,
        "state_clipping": False,
        "action_clipping": False,
        "gripper_reparameterization": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"B-core seed {seed} config differs at {key}")
    if tuple(config.get("future_offsets_steps", ())) != BICOORD_FUTURE_OFFSETS_STEPS:
        raise ValueError(f"B-core seed {seed} future offsets differ")
    state = payload.get("model")
    if not isinstance(state, Mapping) or not any(str(key).startswith("belief_core.") for key in state):
        raise ValueError(f"B-core seed {seed} has no belief-core training state")
    score, score_update = _offline_score(payload)
    deployment = root / "deployment_checkpoint.pt"
    deployed = _load_mapping(deployment, f"B-core seed {seed} deployment")
    validate_deployment_payload(deployed)
    if deployed.get("source_b0h_checkpoint_sha256") != b0h_sha:
        raise ValueError(f"B-core seed {seed} deployment B0-H provenance differs")
    if int(deployed.get("update", -1)) != score_update:
        raise ValueError(
            f"B-core seed {seed} deployment update differs from its fixed "
            f"offline diagnostic choice: {deployed.get('update')!r} != {score_update}"
        )
    return payload, deployment, score, score_update


def _resolve_paths(args: argparse.Namespace) -> None:
    run = args.run.expanduser().resolve() if args.run is not None else None
    if run is not None:
        args.training_root = args.training_root or run / "artifacts" / "bcore_train_3seeds"
        if args.output is None:
            args.output = run / "artifacts" / "bcore_select"
        if args.b0h_checkpoint is None:
            result_path = run / "stage_results" / "b0h_formal.json"
            candidates: list[Path] = []
            if result_path.is_file():
                result = read_json(result_path)
                for row in result.get("artifacts", []):
                    if isinstance(row, Mapping) and row.get("kind") in {"checkpoint", "b0h_checkpoint", "training_checkpoint"}:
                        path = Path(str(row.get("path", "")))
                        if path.is_file() and sha256_file(path) == row.get("sha256"):
                            candidates.append(path.resolve())
            candidates.extend(
                path.resolve()
                for path in (
                    run / "artifacts" / "b0h_formal" / "checkpoint_latest.pt",
                    run / "artifacts" / "b0h_formal" / "final.pt",
                )
                if path.is_file()
            )
            unique = list(dict.fromkeys(candidates))
            if len(unique) != 1:
                raise ValueError(f"could not resolve one B0-H checkpoint: {unique}")
            args.b0h_checkpoint = unique[0]
        # These receipts are optional at the CLI boundary but used when found
        # to prove all three seeds saw the same immutable source artifacts.
        args.bcore_cache = args.bcore_cache or next(
            (
                root
                for root in (
                    run / "artifacts" / "bcore_cache",
                    run / "bcore_cache",
                )
                if (root / "cache_receipt.json").is_file()
            ),
            None,
        )
        args.normalization = args.normalization or next(
            (
                path
                for path in (
                    run / "artifacts" / "dataset_audit" / "normalization.json",
                    run / "normalization" / "normalization.json",
                )
                if path.is_file()
            ),
            None,
        )
    required = {
        "training_root": args.training_root,
        "b0h_checkpoint": args.b0h_checkpoint,
        "output": args.output,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"B-core selection paths are missing: {missing}")
    args.training_root = Path(args.training_root).expanduser().resolve()
    args.b0h_checkpoint = Path(args.b0h_checkpoint).expanduser().resolve()
    args.output = Path(args.output).expanduser().resolve()
    if args.bcore_cache is not None:
        args.bcore_cache = Path(args.bcore_cache).expanduser().resolve()
    if args.normalization is not None:
        args.normalization = Path(args.normalization).expanduser().resolve()


def select_bcore(
    training_root: Path,
    b0h_checkpoint: Path,
    output: Path,
    *,
    config_sha256: str = "",
    result: Path | None = None,
    bcore_cache: Path | None = None,
    normalization: Path | None = None,
) -> dict[str, Any]:
    """Select and copy one seed using no closed-loop metrics."""

    b0h = _load_mapping(b0h_checkpoint, "B0-H checkpoint")
    validate_b0h_payload(b0h)
    b0h_sha = sha256_file(b0h_checkpoint)
    cache_sha = (
        sha256_file(bcore_cache / "cache_receipt.json")
        if bcore_cache is not None and (bcore_cache / "cache_receipt.json").is_file()
        else None
    )
    norm_sha = sha256_file(normalization) if normalization is not None and normalization.is_file() else None
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, Path, int]] = []
    for seed in BCORE_SEEDS:
        root = training_root / f"seed_{seed}"
        payload, deployment, score, score_update = _validate_training_checkpoint(
            root,
            seed=seed,
            b0h_sha=b0h_sha,
            expected_cache_sha=cache_sha,
            expected_norm_sha=norm_sha,
        )
        rows.append(
            {
                "seed": seed,
                "offline_score_b_core_mse": score,
                "offline_score_update": score_update,
                "training_checkpoint": str((root / "checkpoint_latest.pt").resolve()),
                "training_checkpoint_sha256": sha256_file(root / "checkpoint_latest.pt"),
                "deployment_checkpoint": str(deployment.resolve()),
                "deployment_checkpoint_sha256": sha256_file(deployment),
                "update": int(payload.get("update", -1)),
                "closed_loop_results_used": False,
            }
        )
        candidates.append((score, seed, deployment, score_update))
    score, selected_seed, selected_path, selected_update = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "deployment_checkpoint.pt"
    temporary = output / f".deployment_checkpoint.{os.getpid()}.tmp"
    shutil.copyfile(selected_path, temporary)
    os.replace(temporary, destination)
    # Validate bytes after the copy; this catches partial/NFS writes before the
    # result receipt is published.
    selected_payload = _load_mapping(destination, "selected deployment")
    validate_deployment_payload(selected_payload)
    receipt: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "status": "PASSED",
        "selected_seed": int(selected_seed),
        "selected_offline_score_b_core_mse": float(score),
        "selected_offline_score_update": int(selected_update),
        "candidates": rows,
        "seeds": list(BCORE_SEEDS),
        "updates_per_seed": 120_000,
        "final_sufficiency_window_updates": FINAL_SUFFICIENCY_WINDOW,
        "closed_loop_results_used_for_selection": False,
        "selection_stage": "pre_closed_loop_offline_only",
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "BiCoord",
        "vision": VISION_BACKBONE,
        "vision_backbone": VISION_BACKBONE,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_encoding": ACTION_ENCODING,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
        "d_model": D_MODEL,
        "enc_layers": ENCODER_LAYERS,
        "dec_layers": DECODER_LAYERS,
        "roles": ROLES,
        "role_rank": ROLE_RANK,
        "history_layers": HISTORY_LAYERS,
        "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
        "future_offsets_steps": list(BICOORD_FUTURE_OFFSETS_STEPS),
        "future_offsets_seconds": list(BICOORD_BELIEF_CONFIG.future_offsets_seconds),
        "strictly_decentralized": True,
        "strict_local": True,
        "shared_weights": True,
        "shared_checkpoint_for_both_arms": True,
        "arm_id_input": False,
        "peer_runtime_input": False,
        "act_provider_allowed": False,
        "teacher_present": False,
        "memory_semantics": BICOORD_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": BICOORD_CARE_MEMORY_TOKENS,
        "care_memory_width": BICOORD_CARE_MEMORY_WIDTH,
        "all_1800_demonstrations": True,
        "held_out_demonstrations": 0,
        "source_b0h_checkpoint": str(b0h_checkpoint.resolve()),
        "source_b0h_checkpoint_sha256": b0h_sha,
        "source_checkpoint": str(selected_path.resolve()),
        "source_checkpoint_sha256": sha256_file(selected_path),
        "deployment_checkpoint": str(destination.resolve()),
        "deployment_checkpoint_sha256": sha256_file(destination),
        "created_at_utc": _now(),
    }
    atomic_json(output / "selection_receipt.json", receipt)
    atomic_json(
        output / "status.json",
        {
            "status": "PASSED",
            "selected_seed": int(selected_seed),
            "selected_offline_score_b_core_mse": float(score),
            "deployment_checkpoint_sha256": sha256_file(destination),
            "closed_loop_results_used_for_selection": False,
            "completed_at_utc": _now(),
        },
    )
    if result is not None:
        namespace = argparse.Namespace(result=result, config_sha256=config_sha256)
        publish_result(
            namespace,
            stage="bcore_select",
            include_model_contract=True,
            artifacts=(
                artifact(destination, kind="deployment_checkpoint"),
                artifact(output / "selection_receipt.json", kind="selection_receipt"),
                artifact(output / "status.json", kind="status"),
            ),
            selected_seed=int(selected_seed),
            selected_offline_score_b_core_mse=float(score),
            selected_offline_score_update=int(selected_update),
            candidates=rows,
            seeds=list(BCORE_SEEDS),
            updates_per_seed=120_000,
            closed_loop_results_used_for_selection=False,
            checkpoint=str(destination.resolve()),
            checkpoint_sha256=sha256_file(destination),
            teacher_present=False,
            all_1800_demonstrations=True,
        )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = common_parser(__doc__, ("offline-select", "select"))
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--b0h-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bcore-cache", type=Path)
    parser.add_argument("--normalization", type=Path)
    args = parser.parse_args(argv)
    _resolve_paths(args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    select_bcore(
        args.training_root,
        args.b0h_checkpoint,
        args.output,
        config_sha256=args.config_sha256,
        result=args.result,
        bcore_cache=args.bcore_cache,
        normalization=args.normalization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SELECTION_SCHEMA", "select_bcore"]
