"""Build the frozen B0-H action-context cache used by BiCoord B-core.

The cache contains only deterministic outputs of the upstream B0-H temporal
backbone (decoded action hidden states and complete hidden-residual action
predictions).  It does not contain a teacher or a second policy.  One cache
row corresponds to one legal BiCoord decision ``observation[t] -> action[t+1]``
and is stored for both arm-local streams.

The command supports both the standalone interface and the generic BiCoord
supervisor interface.  In supervisor mode each process receives ``--rank`` /
``--world-size`` and writes a rank receipt; the last rank atomically publishes
the aggregate ``cache_receipt.json``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data._utils.collate import default_collate

from before_we_act.temporal_history_policy import TemporalHistoryPolicy

from .bcore_data import (
    BCORE_CACHE_SCHEMA,
    BICOORD_SOURCE_FREQUENCY_HZ,
    sha256_file,
    validate_b0h_payload,
)
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    DATASET_REVISION,
    EFFECTIVE_BATCH,
    HISTORY_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    STATE_DIM,
    TASKS,
    TOTAL_EPISODES,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
)
from .data import (
    BiCoordEpisode,
    BiCoordTemporalDataset,
    BiCoordTemporalRequest,
    BiCoordVisualCache,
    discover_bicoord_episodes,
    load_normalization_receipt,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID
from .stage_common import (
    require_stage_result,
)


BCORE_CACHE_SHARD_SCHEMA = "before-we-act.bicoord.bcore-cache-shard/1"
SMOKE_EPISODE_SELECTION = "minimum_episode_id_then_path_per_task"
_B0H_ARTIFACT_KINDS = frozenset(
    {"checkpoint", "b0h_checkpoint", "training_checkpoint"}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _verified_artifact_paths(
    run: Path,
    stage: str,
    *,
    config_sha256: str | None,
    kinds: frozenset[str],
) -> tuple[Path, ...]:
    """Return only immutable artifacts published by a completed stage."""

    result = require_stage_result(
        run,
        stage,
        config_sha256=config_sha256 or None,
    )
    paths: list[Path] = []
    for row in result.get("artifacts", []):
        if not isinstance(row, Mapping) or row.get("kind") not in kinds:
            continue
        source = Path(str(row.get("path", "")))
        if not source.is_absolute():
            source = run / source
        try:
            source = source.expanduser().resolve(strict=True)
        except FileNotFoundError:
            continue
        if source.is_file() and sha256_file(source) == row.get("sha256"):
            paths.append(source)
    return tuple(dict.fromkeys(paths))


def _require_layout_artifact(
    run: Path,
    stage: str,
    expected: Path,
    *,
    config_sha256: str | None,
    kinds: frozenset[str],
) -> Path:
    expected = expected.expanduser().resolve(strict=True)
    verified = _verified_artifact_paths(
        run,
        stage,
        config_sha256=config_sha256,
        kinds=kinds,
    )
    if verified.count(expected) != 1:
        raise ValueError(
            f"{stage} did not publish the expected immutable artifact "
            f"{expected}; verified={list(verified)}"
        )
    return expected


def _b0h_checkpoint_from_stage(
    run: Path,
    stage: str,
    *,
    config_sha256: str | None,
) -> Path:
    """Resolve exactly one hashed B0-H checkpoint below its artifact root."""

    result = require_stage_result(
        run,
        stage,
        config_sha256=config_sha256 or None,
    )
    verified = _verified_artifact_paths(
        run,
        stage,
        config_sha256=config_sha256,
        kinds=_B0H_ARTIFACT_KINDS,
    )
    root = (run / "artifacts" / stage).resolve()
    verified = tuple(path for path in verified if root in path.parents)

    # The trainer publishes one canonical checkpoint field.  Prefer that
    # explicit identity when it is also present in the hashed artifact list.
    declared: list[Path] = []
    for key in ("checkpoint", "final_checkpoint"):
        raw = result.get(key)
        if not raw:
            continue
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = run / candidate
        try:
            candidate = candidate.expanduser().resolve(strict=True)
        except FileNotFoundError:
            continue
        if candidate in verified:
            declared.append(candidate)
    declared = list(dict.fromkeys(declared))
    candidates = tuple(declared) if declared else verified
    if len(candidates) != 1:
        raise ValueError(
            f"{stage} must publish exactly one hash-verified B0-H checkpoint "
            f"below {root}; found={list(candidates)}"
        )
    return candidates[0]


def _read_receipt(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} receipt: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} receipt must be a JSON object: {path}")
    return value


def _episode_source_row(episode: BiCoordEpisode) -> dict[str, Any]:
    """Return the immutable identity used in cache receipts.

    ``source_identity`` is the content hash and therefore the primary key used
    by the cache files.  Keeping the task, episode number, and resolved path
    alongside it makes a smoke selection auditable even when two source files
    happen to have the same episode number.
    """

    return {
        "task": str(episode.task),
        "episode_id": int(episode.episode_id),
        "path": str(Path(episode.path).expanduser().resolve()),
        "source_identity": str(episode.source_identity),
    }


def _canonical_episode_order(
    episodes: Sequence[BiCoordEpisode],
) -> list[BiCoordEpisode]:
    """Sort episodes independently of filesystem traversal order."""

    return sorted(
        list(episodes),
        key=lambda item: (
            TASKS.index(item.task) if item.task in TASKS else len(TASKS),
            int(item.episode_id),
            str(Path(item.path).expanduser().resolve()),
        ),
    )


def _select_cache_episodes(
    episodes: Sequence[BiCoordEpisode], *, formal: bool
) -> list[BiCoordEpisode]:
    """Select the exact source set used by every B-core cache process.

    Formal runs consume the complete discovered corpus.  Smoke runs still
    discover the complete corpus (so malformed/missing task data is visible),
    then choose exactly one deterministic source per task: the minimum
    ``(episode_id, resolved_path)`` pair.  This function is intentionally pure
    and is called both by the parent and by each worker; no worker trusts a
    list passed from another process.
    """

    ordered = _canonical_episode_order(episodes)
    if formal:
        return ordered
    by_task: dict[str, list[BiCoordEpisode]] = {task: [] for task in TASKS}
    for episode in ordered:
        if episode.task in by_task:
            by_task[episode.task].append(episode)
    missing = [task for task in TASKS if not by_task[task]]
    if missing:
        raise ValueError(
            "BiCoord B-core smoke requires at least one source episode per task; "
            f"missing={missing}"
        )
    selected = [
        min(
            by_task[task],
            key=lambda item: (
                int(item.episode_id),
                str(Path(item.path).expanduser().resolve()),
            ),
        )
        for task in TASKS
    ]
    if len(selected) != len(TASKS) or {
        item.task for item in selected
    } != set(TASKS):
        raise RuntimeError("BiCoord smoke episode selection is not one-per-task")
    return selected


def _normalization_file(path: Path) -> Path:
    """Resolve a normalization directory to its canonical JSON receipt."""

    path = path.expanduser().resolve()
    if path.is_dir():
        for candidate in (
            path / "normalization.json",
            path / "normalization_receipt.json",
            path / "manifest.json",
        ):
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"no normalization receipt under {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_dependency_receipts(
    *,
    episodes: Sequence[BiCoordEpisode],
    normalization: Path,
    visual_cache: Path,
    formal: bool,
) -> None:
    """Validate immutable upstream receipts before any expensive inference."""

    load_normalization_receipt(normalization, require_formal=formal)
    visual_receipt_path = visual_cache / "cache_receipt.json"
    receipt = _read_receipt(visual_receipt_path, label="DINO cache")
    expected_status = "PASSED" if formal else receipt.get("status")
    if receipt.get("schema") != "before-we-act.bicoord.dino-cache/1":
        raise ValueError("DINO visual cache has an unexpected schema")
    if expected_status not in {"PASSED", "SMOKE"}:
        raise ValueError("DINO visual cache is not complete")
    if formal and receipt.get("status") != "PASSED":
        raise ValueError("formal B-core cache requires a PASSED DINO visual cache")
    required = {
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "patch_size": 16,
        "feature_width": 768,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "dataset_revision": DATASET_REVISION,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"DINO visual cache differs at {key}: "
                f"{receipt.get(key)!r} != {expected!r}"
            )
    cached_episode_count = int(receipt.get("episodes", -1))
    if formal and cached_episode_count != len(episodes):
        raise ValueError(
            "DINO visual cache episode coverage differs from formal B-core input "
            f"({cached_episode_count} != {len(episodes)})"
        )
    if not formal and cached_episode_count < len(episodes):
        raise ValueError(
            "DINO visual cache does not cover the complete B-core smoke subset "
            f"({cached_episode_count} < {len(episodes)})"
        )
    counts = receipt.get("episodes_per_task")
    expected_counts = {
        task: sum(episode.task == task for episode in episodes) for task in TASKS
    }
    if not isinstance(counts, Mapping):
        raise ValueError("DINO visual cache per-task coverage is missing")
    observed_counts = {task: int(counts.get(task, -1)) for task in TASKS}
    if formal:
        if observed_counts != expected_counts:
            raise ValueError("DINO visual cache per-task coverage differs")
    elif any(observed_counts[task] < expected_counts[task] for task in TASKS):
        raise ValueError("DINO visual cache does not cover smoke tasks")
    # A formal DINO cache is normally reused by the B-core smoke stage.  Make
    # that reuse fail closed if any of the selected source files is absent or
    # mapped to a different content hash in the visual receipt.
    if not formal:
        rows = receipt.get("files")
        if not isinstance(rows, list):
            raise ValueError("DINO visual cache lacks source identity rows")
        by_pair = {
            (str(row.get("task")), int(row.get("episode_id", -1))): str(
                row.get("source_identity", "")
            )
            for row in rows
            if isinstance(row, Mapping)
        }
        for episode in episodes:
            key = (episode.task, int(episode.episode_id))
            if by_pair.get(key) != episode.source_identity:
                raise ValueError(
                    "DINO visual cache does not contain selected smoke source "
                    f"{episode.source_identity}"
                )


def _load_model(
    checkpoint: Path, dino_model: str | None, device: torch.device
) -> tuple[TemporalHistoryPolicy, Mapping[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = validate_b0h_payload(payload)
    model_name = str(dino_model or config.get("dino_model") or "")
    if not model_name:
        raise ValueError("BiCoord B-core cache requires a DINO model path")
    model = TemporalHistoryPolicy(
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        variant="hidden_residual",
        horizon=int(config["horizon"]),
        d_model=int(config.get("d_model", 384)),
        enc_layers=int(config.get("enc_layers", 4)),
        dec_layers=int(config.get("dec_layers", 7)),
        roles=int(config.get("roles", 4)),
        role_rank=int(config.get("role_rank", 32)),
        history_layers=int(config.get("history_layers", 2)),
        dino_model=model_name,
        image_height=int(config.get("image_height", IMAGE_HEIGHT)),
        image_width=int(config.get("image_width", IMAGE_WIDTH)),
        strict_dino_contract=True,
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    if model.hidden_residual is None:
        raise RuntimeError("BiCoord B-core cache requires hidden-residual B0-H")
    return model, config


def _cached_paths(output: Path, episode: BiCoordEpisode) -> tuple[Path, Path, Path]:
    root = output / episode.task
    key = episode.source_identity
    return (
        root / f"{key}.decoded.npy",
        root / f"{key}.base_action.npy",
        root / f"{key}.complete.json",
    )


def _valid_episode_cache(output: Path, episode: BiCoordEpisode, b0h_sha: str) -> bool:
    decoded_path, base_path, marker_path = _cached_paths(output, episode)
    if not decoded_path.is_file() or not base_path.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        decoded = np.load(decoded_path, mmap_mode="r", allow_pickle=False)
        base = np.load(base_path, mmap_mode="r", allow_pickle=False)
        decisions = int(episode.length) - 1
        return bool(
            marker.get("status") == "PASSED"
            and marker.get("b0h_checkpoint_sha256") == b0h_sha
            and marker.get("source_identity") == episode.source_identity
            and int(marker.get("decisions", -1)) == decisions
            and decoded.shape == (decisions, 2, ACTION_HORIZON, 384)
            and base.shape == (decisions, 2, ACTION_HORIZON, ACTION_DIM)
            and decoded.dtype == np.float16
            and base.dtype == np.float16
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _device_for_worker(rank: int, world: int, requested: str) -> torch.device:
    kind = str(requested)
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BiCoord B-core cache requested CUDA but CUDA is unavailable")
        # Under the supervisor each process has one visible GPU, so local
        # rank is always zero.  Direct mp.spawn exposes all devices and uses
        # the rank index when available.
        index = rank if torch.cuda.device_count() > rank else 0
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        return device
    if world != 1:
        raise ValueError("CPU B-core cache supports one worker only")
    return torch.device(kind)


def _worker(values: Mapping[str, Any]) -> None:
    rank = int(values["rank"])
    world = int(values["world_size"])
    data_root = Path(str(values["data_root"])).resolve()
    normalization = Path(str(values["normalization"])).resolve()
    visual_cache = Path(str(values["visual_cache"])).resolve()
    checkpoint = Path(str(values["b0h_checkpoint"])).resolve()
    output = Path(str(values["output"])).resolve()
    batch_size = int(values["batch_size"])
    if batch_size < 1:
        raise ValueError("BiCoord B-core cache batch size must be positive")
    device = _device_for_worker(rank, world, str(values["device"]))
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 8) // max(world, 1))))
    formal = bool(values.get("formal", True))
    discovered = discover_bicoord_episodes(
        data_root, require_formal=formal, verify_schema=formal
    )
    # Recompute the selection inside every worker.  Passing episode indices
    # from the parent alone is unsafe: a worker could otherwise silently use
    # a different filesystem ordering or stale dataset snapshot.
    episodes = _select_cache_episodes(discovered, formal=formal)
    expected_identities = [
        str(value) for value in values.get("episode_source_identities", ())
    ]
    actual_identities = [episode.source_identity for episode in episodes]
    if expected_identities and expected_identities != actual_identities:
        raise ValueError(
            "B-core cache worker episode selection differs from parent: "
            f"expected={expected_identities[:3]}... actual={actual_identities[:3]}..."
        )
    # Validate the immutable upstream receipts in each worker as well as in
    # the parent.  This matters for independently launched supervisor ranks,
    # where a worker can otherwise observe a changed/mounted cache after the
    # parent-side preflight has completed.
    _validate_dependency_receipts(
        episodes=episodes,
        normalization=normalization,
        visual_cache=visual_cache,
        formal=formal,
    )
    model, model_config = _load_model(checkpoint, str(values.get("dino_model") or "") or None, device)
    image_height = int(model_config.get("image_height", IMAGE_HEIGHT))
    image_width = int(model_config.get("image_width", IMAGE_WIDTH))
    dataset = BiCoordTemporalDataset(
        episodes,
        normalization,
        visual_cache,
        image_height=image_height,
        image_width=image_width,
        cache_limit=8,
        require_visual_cache=True,
    )
    b0h_sha = sha256_file(checkpoint)
    normalization_sha = sha256_file(normalization)
    visual_receipt_sha = sha256_file(visual_cache / "cache_receipt.json")
    assigned = list(range(rank, len(episodes), world))
    assigned_episodes = [episodes[index] for index in assigned]
    assigned_sources = [_episode_source_row(episode) for episode in assigned_episodes]
    completed = 0
    samples = 0
    started = time.time()
    task_tokens: dict[str, list[float]] = {}
    prior_global = _global_receipt_matches(
        output / "cache_receipt.json",
        b0h_sha=b0h_sha,
        formal=formal,
        config_sha256=str(values.get("config_sha256", "")),
        normalization=normalization,
        visual_cache=visual_cache,
        episode_source_identities=actual_identities,
    )
    prior_tokens_path = output / "task_tokens.json"
    if prior_global is not None and prior_tokens_path.is_file():
        prior_tokens = _read_receipt(prior_tokens_path, label="B-core task-token")
        for task, token in prior_tokens.items():
            array = np.asarray(token, dtype=np.float32)
            if task in TASKS and array.shape == (384,) and np.isfinite(array).all():
                task_tokens[task] = array.tolist()
    for ordinal, episode_index in enumerate(assigned, start=1):
        episode = episodes[episode_index]
        decoded_path, base_path, marker_path = _cached_paths(output, episode)
        decisions = int(episode.length) - 1
        if _valid_episode_cache(output, episode, b0h_sha):
            if episode.task not in task_tokens:
                try:
                    marker = _read_receipt(marker_path, label="B-core episode")
                    token = np.asarray(marker.get("task_token"), dtype=np.float32)
                    if token.shape == (384,) and np.isfinite(token).all():
                        task_tokens[episode.task] = token.tolist()
                except (TypeError, ValueError):
                    pass
            if episode.task in task_tokens:
                completed += 1
                samples += decisions * 2
                continue
        decoded = np.empty((decisions, 2, ACTION_HORIZON, 384), dtype=np.float16)
        base = np.empty((decisions, 2, ACTION_HORIZON, ACTION_DIM), dtype=np.float16)
        requests = [
            BiCoordTemporalRequest(
                episode_index,
                arm,
                time_index,
                f"cache:{episode.source_identity}:{arm}:{time_index}",
                episode.task,
            )
            for time_index in range(decisions)
            for arm in (0, 1)
        ]
        for first in range(0, len(requests), batch_size):
            selected = requests[first : first + batch_size]
            batch = default_collate([dataset[request] for request in selected])
            inputs = {
                key: batch[key].to(device, non_blocking=True)
                for key in BiCoordTemporalDataset.MODEL_INPUT_FIELDS
            }
            inputs["global_rgb"] = inputs["global_rgb"].float().div_(255)
            inputs["local_rgb"] = inputs["local_rgb"].float().div_(255)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                context = model._decode_action_context(**inputs, actions=None)
                prediction = model.out(context.decoded)
                history = context.history_summary.unsqueeze(1).expand(
                    -1, ACTION_HORIZON, -1
                )
                prediction = prediction + model.hidden_residual(
                    torch.cat((context.decoded, history), dim=-1)
                )
            if episode.task not in task_tokens:
                task_tokens[episode.task] = context.task_token[0].float().cpu().tolist()
            for local_index, request in enumerate(selected):
                decoded[request.time_index, request.arm] = (
                    context.decoded[local_index].float().cpu().numpy().astype(np.float16)
                )
                base[request.time_index, request.arm] = (
                    prediction[local_index].float().cpu().numpy().astype(np.float16)
                )
        _atomic_npy(decoded_path, decoded)
        _atomic_npy(base_path, base)
        _atomic_json(
            marker_path,
            {
                "status": "PASSED",
                "task": episode.task,
                "episode_id": episode.episode_id,
                "source_identity": episode.source_identity,
                "decisions": decisions,
                "samples": decisions * 2,
                "action_lag_rows": 1,
                "b0h_checkpoint_sha256": b0h_sha,
                "decoded_sha256": sha256_file(decoded_path),
                "base_action_sha256": sha256_file(base_path),
                "task_token": task_tokens[episode.task],
            },
        )
        completed += 1
        samples += decisions * 2
        if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(assigned):
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "episodes": completed,
                        "assigned": len(assigned),
                        "samples": samples,
                        "episodes_per_hour": completed
                        / max(time.time() - started, 1e-6)
                        * 3600,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    rank_receipt = output / f"rank_{rank:02d}_receipt.json"
    _atomic_json(
        rank_receipt,
        {
            "schema": BCORE_CACHE_SHARD_SCHEMA,
            "status": "PASSED",
            "rank": rank,
            "world_size": world,
            "formal": formal,
            "config_sha256": str(values.get("config_sha256", "")),
            "dataset_revision": DATASET_REVISION,
            "episodes": completed,
            "samples": samples,
            "assigned_episode_source_identities": [
                row["source_identity"] for row in assigned_sources
            ],
            "assigned_episode_sources": assigned_sources,
            "episode_selection": (
                SMOKE_EPISODE_SELECTION if not formal else "all_discovered_episodes"
            ),
            # Keep the explicit smoke spelling consumed by train_bcore and
            # preserve the generic field for stage tooling.
            "smoke_episode_selection": (
                SMOKE_EPISODE_SELECTION if not formal else None
            ),
            "task_tokens": task_tokens,
            "b0h_checkpoint_sha256": b0h_sha,
            "normalization_receipt_sha256": normalization_sha,
            "visual_cache_receipt_sha256": visual_receipt_sha,
            "created_at_utc": _now(),
        },
    )


def _model_contract() -> dict[str, Any]:
    # Keep this shape in sync with deployment.bicoord_care.supervisor's
    # frozen contract.  Extra keys are harmless; required keys are exact.
    return {
        "benchmark_adapter": "BiCoord",
        "method_family": "CARE",
        "reference_policy": "B-core/TUNE",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "policy_family": "PredictiveTeamBeliefPolicy",
        "strictly_decentralized": True,
        "shared_checkpoint_for_both_arms": True,
        "global_view": "shared_head_camera",
        "local_view": "own_wrist_camera_only",
        "peer_qpos_action_wrist_allowed": False,
        "state_dim": 7,
        "action_dim": 7,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "state_source": "joint_action_drive_target",
        "temporal_alignment": "observation_row_t_to_action_row_t_plus_1",
        "training_pairs_per_episode": "source_length_minus_1",
        "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
        "history_steps": HISTORY_STEPS,
        "horizon": ACTION_HORIZON,
        "state_clipping": False,
        "action_clipping": False,
        "gripper_reparameterization": False,
        "normalization_population": "all_1800_demos_both_local_arms",
        "vision_backbone": "dinov3_vitb16_frozen",
        "vision_width": 768,
        "d_model": 384,
        "enc_layers": 4,
        "dec_layers": 7,
        "roles": 4,
        "role_rank": 32,
        "history_layers": 2,
    }


def _result_payload(
    *,
    stage: str,
    config_sha256: str,
    artifacts: Sequence[Path],
    **extra: Any,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in artifacts:
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"B-core cache result artifact is missing: {path}")
        rows.append({"path": str(path), "sha256": sha256_file(path)})
    return {
        "schema": "before-we-act.bicoord-care-stage-result/1",
        "stage": stage,
        "status": "PASSED",
        "benchmark_adapter": "BiCoord",
        "config_sha256": config_sha256,
        "model_contract": _model_contract(),
        "artifacts": rows,
        "completed_at": _now(),
        **extra,
    }


def _complete_worker_result(
    *,
    stage: str,
    config_sha256: str,
    output: Path,
    rank: int,
    world: int,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a worker result only after the global cache is complete."""

    expected = {
        "schema": BCORE_CACHE_SCHEMA,
        "status": "PASSED",
        "cache_complete": True,
        "world_size": world,
        "config_sha256": config_sha256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"B-core cache worker cannot publish before aggregate "
                f"completion: {key}={receipt.get(key)!r} != {value!r}"
            )
    shard = _read_receipt(
        output / f"rank_{rank:02d}_receipt.json",
        label="B-core cache shard",
    )
    if (
        shard.get("schema") != BCORE_CACHE_SHARD_SCHEMA
        or shard.get("status") != "PASSED"
        or int(shard.get("rank", -1)) != rank
        or int(shard.get("world_size", -1)) != world
        or shard.get("config_sha256") != config_sha256
    ):
        raise ValueError(f"B-core cache worker {rank} shard receipt is incomplete")
    # Smoke workers must prove that they processed the exact source slice
    # selected by the parent.  Formal receipts also carry this provenance in
    # new runs; accepting an absent field there keeps old formal caches
    # resumable while never weakening the reduced smoke contract.
    if not bool(receipt.get("formal", True)):
        expected_sources = receipt.get("episode_sources")
        actual_sources = shard.get("assigned_episode_sources")
        if not isinstance(expected_sources, list) or not isinstance(actual_sources, list):
            raise ValueError("smoke cache worker is missing source provenance")
        expected_assigned = [
            expected_sources[index]
            for index in range(rank, len(expected_sources), world)
        ]
        if actual_sources != expected_assigned:
            raise ValueError(
                f"smoke cache worker {rank} source slice differs from aggregate"
            )
    return _result_payload(
        stage=stage,
        config_sha256=config_sha256,
        artifacts=(
            output / f"rank_{rank:02d}_receipt.json",
            output / "cache_receipt.json",
            output / "task_tokens.json",
        ),
        episodes=int(receipt["episodes"]),
        samples=int(receipt["samples"]),
        episodes_per_task=receipt["episodes_per_task"],
        cache_receipt=str((output / "cache_receipt.json").resolve()),
        rank=rank,
        world_size=world,
        cache_complete=True,
        episode_source_identities=receipt.get("episode_source_identities", []),
        episode_sources=receipt.get("episode_sources", []),
        episode_selection=receipt.get("episode_selection"),
    )


def _try_finalize(
    *,
    output: Path,
    episodes: Sequence[BiCoordEpisode],
    world: int,
    b0h_sha: str,
    formal: bool,
    visual_cache: Path,
    normalization: Path,
    config_sha256: str,
) -> dict[str, Any] | None:
    # Accept either the complete discovery result or an already-selected
    # subset from callers.  Re-selecting here makes direct/finalizer tests and
    # independently launched workers obey the same canonical smoke rule.
    episodes = _select_cache_episodes(episodes, formal=formal)
    expected_sources = [_episode_source_row(episode) for episode in episodes]
    expected_identities = [row["source_identity"] for row in expected_sources]
    expected_selection = (
        "all_discovered_episodes" if formal else SMOKE_EPISODE_SELECTION
    )
    rank_paths = [output / f"rank_{rank:02d}_receipt.json" for rank in range(world)]
    if not all(path.is_file() for path in rank_paths):
        return None
    rows = [_read_receipt(path, label="B-core cache shard") for path in rank_paths]
    normalization_sha = sha256_file(normalization)
    visual_receipt = visual_cache / "cache_receipt.json"
    visual_receipt_sha = sha256_file(visual_receipt)
    for rank, row in enumerate(rows):
        expected = {
            "schema": BCORE_CACHE_SHARD_SCHEMA,
            "status": "PASSED",
            "rank": rank,
            "world_size": world,
            "formal": formal,
            "config_sha256": config_sha256,
            "dataset_revision": DATASET_REVISION,
            "b0h_checkpoint_sha256": b0h_sha,
            "normalization_receipt_sha256": normalization_sha,
            "visual_cache_receipt_sha256": visual_receipt_sha,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"BiCoord B-core shard {rank} differs at {key}: "
                    f"{row.get(key)!r} != {value!r}"
                )
        assigned = episodes[rank::world]
        expected_samples_for_rank = sum(
            (int(episode.length) - 1) * 2 for episode in assigned
        )
        if int(row.get("episodes", -1)) != len(assigned) or int(
            row.get("samples", -1)
        ) != expected_samples_for_rank:
            raise ValueError(f"BiCoord B-core shard {rank} coverage differs")
        assigned_sources = [_episode_source_row(episode) for episode in assigned]
        recorded_identities = row.get("assigned_episode_source_identities")
        recorded_sources = row.get("assigned_episode_sources")
        if not formal or recorded_identities is not None or recorded_sources is not None:
            if recorded_identities != [item["source_identity"] for item in assigned_sources]:
                raise ValueError(f"BiCoord B-core shard {rank} source identities differ")
            if recorded_sources != assigned_sources:
                raise ValueError(f"BiCoord B-core shard {rank} source records differ")
        if row.get("episode_selection") not in (None, expected_selection):
            raise ValueError(f"BiCoord B-core shard {rank} selection contract differs")
    invalid = [episode.source_identity for episode in episodes if not _valid_episode_cache(output, episode, b0h_sha)]
    if invalid:
        raise RuntimeError(f"BiCoord B-core cache incomplete: {invalid[:4]}")
    task_tokens: dict[str, list[float]] = {}
    for row in rows:
        task_tokens.update(row.get("task_tokens", {}))
    if set(task_tokens) != set(TASKS):
        raise RuntimeError(f"BiCoord B-core task-token coverage differs: {sorted(task_tokens)}")
    for task in TASKS:
        token = np.asarray(task_tokens[task], dtype=np.float32)
        if token.shape != (384,) or not np.isfinite(token).all():
            raise RuntimeError(f"BiCoord B-core task token is invalid: {task}")
        task_tokens[task] = token.tolist()
    _atomic_json(output / "task_tokens.json", task_tokens)
    expected_samples = sum((int(episode.length) - 1) * 2 for episode in episodes)
    observed_samples = sum(int(row.get("samples", 0)) for row in rows)
    if observed_samples != expected_samples:
        raise RuntimeError(f"BiCoord B-core sample coverage differs: {observed_samples} != {expected_samples}")
    receipt = {
        "schema": BCORE_CACHE_SCHEMA,
        "status": "PASSED",
        "cache_complete": True,
        "formal": formal,
        "episodes": len(episodes),
        "episodes_per_task": {
            task: sum(1 for episode in episodes if episode.task == task) for task in TASKS
        },
        "samples": expected_samples,
        "tasks": list(TASKS),
        "dtype": "float16",
        "decoded_shape_per_sample": [ACTION_HORIZON, 384],
        "base_action_shape_per_sample": [ACTION_HORIZON, ACTION_DIM],
        "base_action_semantics": "complete_TemporalHistoryPolicy_hidden_residual_prediction",
        "policy_family": "TemporalHistoryPolicy",
        "method_family": "CARE",
        "downstream_policy_family": "PredictiveTeamBeliefPolicy",
        "benchmark_adapter": "BiCoord",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "strictly_decentralized": True,
        "act_provider_allowed": False,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
        "action_lag_rows": 1,
        "dataset_revision": DATASET_REVISION,
        "config_sha256": config_sha256,
        "b0h_checkpoint_sha256": b0h_sha,
        "visual_cache_receipt_sha256": visual_receipt_sha,
        "normalization_receipt_sha256": normalization_sha,
        "world_size": world,
        "episode_source_identities": expected_identities,
        "episode_sources": expected_sources,
        "episode_selection": expected_selection,
        "smoke_episode_selection": (
            SMOKE_EPISODE_SELECTION if not formal else None
        ),
        "rank_receipts": [path.name for path in rank_paths],
        "rank_receipt_sha256": {
            path.name: sha256_file(path) for path in rank_paths
        },
        "created_at_utc": _now(),
    }
    _atomic_json(output / "cache_receipt.json", receipt)
    return receipt


def _global_receipt_matches(
    path: Path,
    *,
    b0h_sha: str,
    formal: bool,
    config_sha256: str,
    normalization: Path,
    visual_cache: Path,
    episode_source_identities: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Read a globally finalized receipt and reject stale-run artifacts."""

    if not path.is_file():
        return None
    try:
        value = _read_receipt(path, label="B-core cache")
        expected = {
            "schema": BCORE_CACHE_SCHEMA,
            "status": "PASSED",
            "cache_complete": True,
            "formal": formal,
            "config_sha256": config_sha256,
            "dataset_revision": DATASET_REVISION,
            "b0h_checkpoint_sha256": b0h_sha,
            "normalization_receipt_sha256": sha256_file(normalization),
            "visual_cache_receipt_sha256": sha256_file(
                visual_cache / "cache_receipt.json"
            ),
        }
        # ``formal`` is included in new receipts; tolerate old receipts only
        # when no supervisor config was supplied (standalone compatibility).
        for key, item in expected.items():
            if key == "formal" and key not in value and not config_sha256:
                continue
            if value.get(key) != item:
                return None
        if int(value.get("world_size", 0)) < 1:
            return None
        if episode_source_identities is not None and value.get(
            "episode_source_identities"
        ) != list(episode_source_identities):
            return None
        return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()
    normalization = _normalization_file(Path(args.normalization))
    visual_cache = Path(args.visual_cache).expanduser().resolve()
    checkpoint = Path(args.b0h_checkpoint).expanduser().resolve()
    formal = not bool(args.smoke)
    discovered = discover_bicoord_episodes(
        data_root, require_formal=formal, verify_schema=formal
    )
    episodes = _select_cache_episodes(discovered, formal=formal)
    episode_source_identities = [
        episode.source_identity for episode in episodes
    ]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    b0h_config = validate_b0h_payload(payload)
    stage = str(b0h_config.get("stage", ""))
    expected_stage = "formal" if formal else "smoke"
    if stage != expected_stage:
        raise ValueError(
            f"BiCoord B-core cache requires a {expected_stage} B0-H checkpoint, "
            f"observed stage={stage!r}"
        )
    _validate_dependency_receipts(
        episodes=episodes,
        normalization=normalization,
        visual_cache=visual_cache,
        formal=formal,
    )
    b0h_sha = sha256_file(checkpoint)
    if args.rank is not None:
        rank = int(args.rank)
        world = int(args.world_size or 1)
        if not 0 <= rank < world:
            raise ValueError("invalid cache rank/world-size")
        _worker(
            {
                "rank": rank,
                "world_size": world,
                "data_root": data_root,
                "normalization": normalization,
                "visual_cache": visual_cache,
                "b0h_checkpoint": checkpoint,
                "output": output,
                "batch_size": int(args.batch_size),
                "device": args.device,
                "dino_model": args.dino_model or "",
                "formal": formal,
                "config_sha256": str(args.config_sha256 or ""),
                "episode_source_identities": episode_source_identities,
            }
        )
        receipt: dict[str, Any] | None = None
        deadline = time.time() + float(
            os.environ.get("BICOORD_CACHE_WAIT_SECONDS", "3600")
        )
        if rank == 0:
            while time.time() < deadline:
                try:
                    receipt = _try_finalize(
                        output=output,
                        episodes=episodes,
                        world=world,
                        b0h_sha=b0h_sha,
                        formal=formal,
                        visual_cache=visual_cache,
                        normalization=normalization,
                        config_sha256=str(args.config_sha256 or ""),
                    )
                except (OSError, ValueError, RuntimeError, TypeError, json.JSONDecodeError):
                    # A stale shard or a peer still writing an episode marker
                    # is retried until the bounded lease expires.
                    receipt = None
                if receipt is not None:
                    break
                time.sleep(0.25 if all(
                    (output / f"rank_{index:02d}_receipt.json").is_file()
                    for index in range(world)
                ) else 1.0)
        else:
            # Nonzero workers must not publish a PASSED stage result before
            # rank zero has atomically finalized the global receipt.  This
            # keeps the supervisor's worker aggregation fail-closed.
            while time.time() < deadline:
                receipt = _global_receipt_matches(
                    output / "cache_receipt.json",
                    b0h_sha=b0h_sha,
                    formal=formal,
                    config_sha256=str(args.config_sha256 or ""),
                    normalization=normalization,
                    visual_cache=visual_cache,
                    episode_source_identities=episode_source_identities,
                )
                if receipt is not None:
                    break
                time.sleep(0.25)
        if receipt is None:
            raise TimeoutError("timed out waiting for B-core cache finalization")
        if args.result:
            evidence = [
                output / f"rank_{rank:02d}_receipt.json",
                output / "cache_receipt.json",
                output / "task_tokens.json",
            ]
            result = _complete_worker_result(
                stage=args.stage_name or "bcore_cache",
                config_sha256=args.config_sha256 or "",
                output=output,
                rank=rank,
                world=world,
                receipt=receipt,
            )
            _atomic_json(Path(args.result), result)
        return receipt

    requested_world = int(args.gpus)
    if args.device == "cuda":
        visible = torch.cuda.device_count()
        world = min(requested_world if requested_world > 0 else visible, visible)
        if world < 1:
            raise RuntimeError("BiCoord B-core cache requires at least one visible CUDA GPU")
    else:
        world = 1
    values = {
        "rank": 0,
        "world_size": world,
        "data_root": data_root,
        "normalization": normalization,
        "visual_cache": visual_cache,
        "b0h_checkpoint": checkpoint,
        "output": output,
        "batch_size": int(args.batch_size),
        "device": args.device,
        "dino_model": args.dino_model or "",
        "formal": formal,
        "config_sha256": str(args.config_sha256 or ""),
        "episode_source_identities": episode_source_identities,
    }
    if world == 1:
        _worker(values)
    else:
        mp.spawn(_spawn_worker, args=(world, values), nprocs=world, join=True)
    receipt = _try_finalize(
        output=output,
        episodes=episodes,
        world=world,
        b0h_sha=b0h_sha,
        formal=formal,
        visual_cache=visual_cache,
        normalization=normalization,
        config_sha256=str(args.config_sha256 or ""),
    )
    if receipt is None:
        raise RuntimeError("BiCoord B-core cache did not produce aggregate receipt")
    if args.result:
        result = _result_payload(
            stage=args.stage_name or "bcore_cache",
            config_sha256=args.config_sha256 or "",
            artifacts=[output / "cache_receipt.json", output / "task_tokens.json"],
            episodes=receipt["episodes"],
            samples=receipt["samples"],
            episodes_per_task=receipt["episodes_per_task"],
            cache_receipt=str((output / "cache_receipt.json").resolve()),
            episode_source_identities=receipt["episode_source_identities"],
            episode_sources=receipt["episode_sources"],
            episode_selection=receipt["episode_selection"],
        )
        _atomic_json(Path(args.result), result)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _spawn_worker(rank: int, world: int, values: Mapping[str, Any]) -> None:
    copied = dict(values)
    copied["rank"] = rank
    copied["world_size"] = world
    _worker(copied)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", choices=("cache-all", "prepare"))
    parser.add_argument("--data-root", "--prepared-data", "--dataset", dest="data_root", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--visual-cache", type=Path)
    parser.add_argument("--b0h-checkpoint", type=Path)
    parser.add_argument("--dino-model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-name")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--benchmark-repo", type=Path)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args(argv)
    # The stage flag/name, never an incidental path substring, determines the
    # reduced-data mode.  This prevents a misplaced formal output/result path
    # from silently turning a full cache job into smoke (or vice versa).
    if args.stage_name == "bcore_smoke_cache":
        args.smoke = True
    smoke_path_marker = (
        (args.output is not None and "bcore_smoke_cache" in args.output.parts)
        or (
            args.result is not None
            and any("bcore_smoke_cache" in part for part in args.result.parts)
        )
    )
    if smoke_path_marker and not args.smoke:
        raise ValueError(
            "B-core smoke cache paths require explicit --smoke or "
            "--stage-name bcore_smoke_cache"
        )
    if args.smoke and args.stage_name not in (None, "bcore_smoke_cache"):
        raise ValueError(
            "smoke B-core cache must use the bcore_smoke_cache stage namespace"
        )
    args.stage_name = args.stage_name or (
        "bcore_smoke_cache" if args.smoke else "bcore_cache"
    )
    if args.run is not None:
        run = args.run.expanduser().resolve()
        config_sha256 = str(args.config_sha256 or "") or None
        if config_sha256 is None or len(config_sha256) != 64:
            raise ValueError(
                "generic B-core cache invocation requires a 64-character config SHA-256"
            )
        try:
            int(config_sha256, 16)
        except ValueError as error:
            raise ValueError("generic B-core cache config SHA-256 is not hexadecimal") from error

        expected_output = run / "artifacts" / args.stage_name
        if args.output is not None and Path(args.output).expanduser().resolve() != expected_output.resolve():
            raise ValueError(
                "B-core cache output differs from the frozen stage namespace: "
                f"{args.output} != {expected_output}"
            )
        args.output = expected_output

        expected_normalization = run / "artifacts" / "dataset_audit" / "normalization.json"
        if args.normalization is not None and Path(args.normalization).expanduser().resolve() != expected_normalization.resolve():
            raise ValueError(
                "B-core cache normalization differs from the frozen dataset-audit artifact"
            )
        args.normalization = _require_layout_artifact(
                run,
                "dataset_audit",
                expected_normalization,
                config_sha256=config_sha256,
                kinds=frozenset({"normalization"}),
            )
        expected_visual = run / "artifacts" / "dino_cache"
        if args.visual_cache is not None and Path(args.visual_cache).expanduser().resolve() != expected_visual.resolve():
            raise ValueError(
                "B-core cache visual cache differs from the frozen dino-cache namespace"
            )
        _require_layout_artifact(
                run,
                "dino_cache",
                expected_visual / "cache_receipt.json",
                config_sha256=config_sha256,
                kinds=frozenset({"dino_cache"}),
            )
        args.visual_cache = expected_visual
        stage = "b0h_smoke_train" if args.smoke else "b0h_formal"
        expected_checkpoint = _b0h_checkpoint_from_stage(
            run,
            stage,
            config_sha256=config_sha256,
        )
        if args.b0h_checkpoint is not None and Path(args.b0h_checkpoint).expanduser().resolve() != expected_checkpoint:
            raise ValueError(
                "B-core cache B0-H checkpoint differs from the frozen stage result: "
                f"{args.b0h_checkpoint} != {expected_checkpoint}"
            )
        args.b0h_checkpoint = expected_checkpoint
        args.data_root = args.data_root or run / "dataset"
    required = {
        "data root": args.data_root,
        "normalization": args.normalization,
        "visual cache": args.visual_cache,
        "B0-H checkpoint": args.b0h_checkpoint,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"{', '.join(missing)} are required")
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.normalization = Path(args.normalization).expanduser().resolve()
    args.visual_cache = Path(args.visual_cache).expanduser().resolve()
    args.b0h_checkpoint = Path(args.b0h_checkpoint).expanduser().resolve()
    args.output = Path(args.output).expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    build_cache(_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_cache", "main"]
