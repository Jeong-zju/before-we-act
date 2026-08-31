"""Train CARE's upstream TemporalHistoryPolicy B0-H on all BiCoord data.

The policy weights are shared by both arms.  Every row contains only the
shared scene camera, the focal arm's wrist camera, local qpos/action history,
and task text; peer wrist/proprioception/action and arm identity are excluded.

Run with ``python -m deployment.bicoord_care.train_b0h``.  Formal training is
frozen at 120k global updates with effective batch 48 and, by default, four
DDP workers.  Smoke and formal runs use the same model and optimizer path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    EFFECTIVE_BATCH,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
    HISTORY_STEPS,
    STATE_DIM,
    TASKS,
)
from .data import (
    BiCoordBalancedDistributedBatchSampler,
    BiCoordEpisode,
    BiCoordTemporalDataset,
    discover_bicoord_episodes,
    load_normalization_receipt,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID
from .stage_common import publish_result, artifact


FORMAL_UPDATES = 120_000
FORMAL_EPISODES = 1_800
DEFAULT_SEED = 20260901
CACHE_SCHEMA = "before-we-act.bicoord.dino-cache/1"
NORMALIZATION_SCHEMA = "before-we-act.bicoord.normalization/1"
CHECKPOINT_FORMAT = "before-we-act.bicoord.dino-b0h/1"
CONFIG_FORMAT = "before-we-act.bicoord.dino-b0h-config/1"
SMOKE_EPISODE_SELECTION = "minimum_episode_id_then_path_per_task"
# The source stores the seventh channel as a continuous [0, 1] gripper
# command.  Keep the canonical spelling in config.py so checkpoints and
# runtime receipts cannot drift back to the historical binary label.
POLICY_CONTRACT = (
    "shared_weights_strictly_decentralized_shared_head_rgb_own_wrist_rgb_"
    "local_qpos7_local_executed_action7_no_arm_id_to_local_absolute_action7"
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


def _atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON receipt is not an object: {path}")
    return value


def _distributed_sum(value: torch.Tensor, world: int) -> torch.Tensor:
    result = value.detach().clone()
    if world > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _episode_identity(episode: BiCoordEpisode) -> tuple[str, int]:
    return str(episode.task), int(episode.episode_id)


def _canonical_episode_order(
    episodes: Sequence[BiCoordEpisode],
) -> list[BiCoordEpisode]:
    """Return the stable source ordering used by every B0-H stage."""

    return sorted(
        list(episodes),
        key=lambda item: (
            TASKS.index(item.task) if item.task in TASKS else len(TASKS),
            int(item.episode_id),
            str(Path(item.path).expanduser().resolve()),
        ),
    )


def _episode_source_row(episode: BiCoordEpisode) -> dict[str, Any]:
    return {
        "task": str(episode.task),
        "episode_id": int(episode.episode_id),
        "path": str(Path(episode.path).expanduser().resolve()),
        "source_identity": str(episode.source_identity),
    }


def _smoke_episode_subset(
    episodes: Sequence[BiCoordEpisode], receipt: Mapping[str, Any]
) -> list[BiCoordEpisode]:
    """Select exactly one canonical source episode for each smoke task.

    The formal DINO cache intentionally contains all 1,800 demonstrations and
    is also the visual dependency for the smoke stage.  Consequently its
    ``episodes`` count must *not* be interpreted as the B0-H smoke training
    population.  We derive the reduced population from the immutable source
    discovery and require every selected identity to be present in the DINO
    receipt.  This keeps smoke and formal data contracts distinct while
    retaining the same frozen image encoder/cache.
    """

    ordered = _canonical_episode_order(episodes)
    by_task: dict[str, list[BiCoordEpisode]] = {task: [] for task in TASKS}
    for episode in ordered:
        if episode.task in by_task:
            by_task[episode.task].append(episode)
    missing = [task for task in TASKS if not by_task[task]]
    if missing:
        raise ValueError(f"B0-H smoke dataset is missing tasks: {missing}")
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
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(
            "B0-H smoke requires DINO receipt.files to identify source episodes"
        )
    # A DINO row's path points to the encoded feature ``.npz`` rather than
    # the source HDF5.  Pair and content hash are therefore the authoritative
    # identity checks here; source paths come from ``episodes`` above.
    rows_by_pair: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    rows_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("cache receipt file row is not an object")
        try:
            pair = (str(row["task"]), int(row["episode_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"cache receipt row lacks task/episode identity: {row}") from error
        rows_by_pair.setdefault(pair, []).append(row)
        source = str(row.get("source_identity", ""))
        if source:
            rows_by_source.setdefault(source, []).append(row)
    for episode in selected:
        pair = _episode_identity(episode)
        matches = [
            row
            for row in rows_by_pair.get(pair, [])
            if str(row.get("source_identity", "")) == episode.source_identity
        ]
        if len(matches) != 1:
            raise ValueError(
                f"DINO receipt must identify smoke source exactly once: {pair}"
            )
        source_matches = rows_by_source.get(episode.source_identity, [])
        if len(source_matches) != 1:
            raise ValueError(
                f"DINO receipt source identity is duplicated: {episode.source_identity}"
            )
    if receipt.get("smoke_episode_selection") not in (None, SMOKE_EPISODE_SELECTION):
        raise ValueError("DINO receipt has an incompatible smoke selection contract")
    if len(selected) != len(TASKS) or [item.task for item in selected] != list(TASKS):
        raise AssertionError("B0-H smoke selection lost task coverage")
    return selected


def _validate_smoke_cache_artifacts(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    episodes: Sequence[BiCoordEpisode],
) -> None:
    """Hash-check the 18 DINO files selected for smoke training."""

    cache_root = receipt_path.parent.expanduser().resolve()
    rows = receipt.get("files")
    if not isinstance(rows, list):
        raise ValueError("DINO smoke receipt lacks file rows")
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            by_source.setdefault(str(row.get("source_identity", "")), []).append(row)
    for episode in episodes:
        matches = by_source.get(episode.source_identity, [])
        if len(matches) != 1:
            raise ValueError(
                f"DINO smoke source must have one cache artifact: {episode.source_identity}"
            )
        row = matches[0]
        cache_path = Path(str(row.get("path", "")))
        if not cache_path.is_absolute():
            cache_path = cache_root / cache_path
        try:
            cache_path = cache_path.expanduser().resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"selected DINO smoke cache artifact is missing: {cache_path}"
            ) from error
        expected = (
            cache_root / episode.task / f"{episode.source_identity}.npz"
        ).resolve()
        if cache_path != expected or cache_root not in cache_path.parents:
            raise ValueError(
                f"selected DINO smoke cache path differs: {cache_path} != {expected}"
            )
        digest = str(row.get("sha256", ""))
        if len(digest) != 64 or _sha256(cache_path) != digest:
            raise ValueError(f"selected DINO smoke cache hash differs: {cache_path}")


def _validate_cache_receipt(
    path: Path,
    *,
    stage: str,
    image_height: int,
    image_width: int,
    episodes: Sequence[BiCoordEpisode],
) -> tuple[dict[str, Any], list[BiCoordEpisode]]:
    receipt = _read_json(path)
    if receipt.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"unexpected BiCoord DINO cache schema: {receipt.get('schema')}")
    accepted = {"PASSED", "SMOKE"} if stage == "smoke" else {"PASSED"}
    if receipt.get("status") not in accepted:
        raise ValueError("BiCoord DINO cache is incomplete for this stage")
    required = {
        "image_height": image_height,
        "image_width": image_width,
        "feature_width": 768,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"BiCoord DINO cache differs at {key}: "
                f"{receipt.get(key)!r} != {expected!r}"
            )
    cached_count = int(receipt.get("episodes", 0))
    files = receipt.get("files")
    if not isinstance(files, list) or len(files) != cached_count:
        raise ValueError(
            "BiCoord DINO cache receipt file coverage differs from its episode count"
        )
    if stage == "formal":
        if cached_count != FORMAL_EPISODES:
            raise ValueError("formal BiCoord DINO cache must cover 1800 episodes")
        counts = receipt.get("episodes_per_task")
        if not isinstance(counts, Mapping) or any(
            int(counts.get(task, 0)) != 100 for task in TASKS
        ):
            raise ValueError("formal BiCoord DINO cache is incomplete by task")
        if receipt.get("strict_dino_contract") is not True:
            raise ValueError("formal BiCoord cache lacks strict DINO provenance")
        return receipt, list(episodes)
    if cached_count < len(TASKS):
        raise ValueError(
            "BiCoord smoke requires a visual cache covering all 18 tasks; "
            f"observed {cached_count} episodes"
        )
    counts = receipt.get("episodes_per_task")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(task, 0)) < 1 for task in TASKS
    ):
        raise ValueError("BiCoord smoke visual cache is incomplete by task")
    selected = _smoke_episode_subset(episodes, receipt)
    _validate_smoke_cache_artifacts(path, receipt, selected)
    return receipt, selected


def _validate_normalization(receipt: Mapping[str, Any], *, formal: bool) -> None:
    if receipt.get("schema") != NORMALIZATION_SCHEMA:
        raise ValueError("unexpected BiCoord normalization schema")
    if receipt.get("status") != "PASSED":
        raise ValueError("BiCoord normalization receipt has not passed")
    expected = {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_encoding": ACTION_ENCODING,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"normalization differs at {key}: {receipt.get(key)!r} != {value!r}"
            )
    alignment = receipt.get("recording_alignment")
    if not isinstance(alignment, Mapping) or {
        "observation_row_offset": int(alignment.get("observation_row_offset", -1)),
        "action_row_offset": int(alignment.get("action_row_offset", -1)),
        "action_lag_rows": int(alignment.get("action_lag_rows", -1)),
    } != {
        "observation_row_offset": 0,
        "action_row_offset": 1,
        "action_lag_rows": 1,
    }:
        raise ValueError("BiCoord normalization has wrong observation/action alignment")
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        value = np.asarray(receipt.get(key), dtype=np.float32)
        if value.shape != (STATE_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"invalid normalization vector {key}: {value.shape}")
        if key.endswith("std") and np.any(value <= 0):
            raise ValueError(f"normalization vector {key} is non-positive")
    ranges: dict[str, np.ndarray] = {}
    for key in ("qpos_min", "qpos_max", "action_min", "action_max"):
        value = np.asarray(receipt.get(key), dtype=np.float32)
        if value.shape != (STATE_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"invalid normalization source range {key}")
        ranges[key] = value
    if np.any(ranges["qpos_min"] > ranges["qpos_max"]) or np.any(
        ranges["action_min"] > ranges["action_max"]
    ):
        raise ValueError("normalization source ranges are inverted")
    low, high = GRIPPER_NATIVE_RANGE
    if not (
        float(ranges["qpos_min"][-1]) == low
        and float(ranges["action_min"][-1]) == low
        and float(ranges["qpos_max"][-1]) == high
        and float(ranges["action_max"][-1]) == high
    ):
        raise ValueError("normalization gripper population range is not [0,1]")
    if formal:
        if int(receipt.get("episodes", 0)) != FORMAL_EPISODES:
            raise ValueError("formal normalization does not cover all 1800 episodes")
        counts = receipt.get("episodes_per_task")
        if not isinstance(counts, Mapping) or any(
            int(counts.get(task, 0)) != 100 for task in TASKS
        ):
            raise ValueError("formal normalization is incomplete by task")


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the historical trainer CLI and the supervisor's generic CLI.

    The supervisor invokes every stage with ``operation --dataset --run
    --result``.  The original B0-H trainer predates that interface and uses
    ``--data-root --output --stage``.  Both map to the same immutable training
    graph; this parser is only an adapter and does not alter model settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", choices=("smoke-train", "formal-train", "train"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--benchmark-repo", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--global-batch", type=int, default=EFFECTIVE_BATCH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--visual-cache", type=Path)
    parser.add_argument("--dino-model", default="/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("smoke", "formal"))
    parser.add_argument("--updates", type=int)
    parser.add_argument("--protocol-updates", type=int, default=FORMAL_UPDATES)
    parser.add_argument("--image-height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--capability-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-every", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-world-size", type=int, default=4,
                        help="set to zero to accept any divisor of global batch 48")
    args = parser.parse_args(argv)
    generic = args.operation is not None
    if generic:
        if args.global_batch != EFFECTIVE_BATCH:
            raise ValueError("BiCoord B0-H effective global batch is frozen at 48")
        if any(value is None for value in (args.repo, args.benchmark_repo, args.dataset, args.run, args.result)):
            parser.error("supervisor invocation requires --repo, --benchmark-repo, --dataset, --run and --result")
        if len(str(args.config_sha256)) != 64:
            raise ValueError("supervisor config SHA-256 must contain 64 characters")
        try:
            int(str(args.config_sha256), 16)
        except ValueError as error:
            raise ValueError("supervisor config SHA-256 is not hexadecimal") from error
        args.repo = args.repo.expanduser().resolve()
        args.benchmark_repo = args.benchmark_repo.expanduser().resolve()
        args.dataset = args.dataset.expanduser().resolve()
        args.run = args.run.expanduser().resolve()
        args.result = args.result.expanduser().resolve()
        for path in (args.repo, args.benchmark_repo, args.dataset):
            if not path.is_dir():
                raise FileNotFoundError(path)
        args.data_root = args.data_root or args.dataset
        args.normalization = args.normalization or args.run / "artifacts" / "dataset_audit" / "normalization.json"
        args.visual_cache = args.visual_cache or args.run / "artifacts" / "dino_cache"
        if args.operation == "formal-train" and args.smoke:
            raise ValueError("formal B0-H operation cannot be marked smoke")
        derived_stage = "smoke" if args.operation == "smoke-train" or args.smoke else "formal"
        if args.stage is not None and args.stage != derived_stage:
            raise ValueError("B0-H operation and explicit stage disagree")
        args.output = args.output or args.run / "artifacts" / (
            "b0h_smoke_train" if derived_stage == "smoke" else "b0h_formal"
        )
        args.stage = derived_stage
        args.updates = args.updates if args.updates is not None else (5 if args.stage == "smoke" else FORMAL_UPDATES)
        # The supervisor's --auto-resume is intentionally propagated.  A
        # manual generic launch with no flag starts from a clean directory.
        if args.auto_resume is None:
            args.auto_resume = False
    else:
        required = {
            "--data-root": args.data_root,
            "--normalization": args.normalization,
            "--visual-cache": args.visual_cache,
            "--output": args.output,
            "--stage": args.stage,
            "--updates": args.updates,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("legacy invocation missing " + ", ".join(missing))
        # Historical command resumed by default; preserve that behavior only
        # when no generic operation is supplied.
        if args.auto_resume is None:
            args.auto_resume = True
    for name in ("data_root", "normalization", "visual_cache", "output", "resume"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    args.dino_model = str(Path(args.dino_model).expanduser().resolve())
    return args


def _completion_receipt(
    *,
    args: argparse.Namespace,
    update: int,
    checkpoint: Path,
    normalization_sha256: str,
    cache_sha256: str,
    world: int,
    episodes: Sequence[BiCoordEpisode] | None = None,
) -> dict[str, Any]:
    selected_episodes = list(episodes or ())
    source_identities = [
        str(episode.source_identity) for episode in selected_episodes
    ]
    selection = (
        SMOKE_EPISODE_SELECTION
        if args.stage == "smoke"
        else "all_discovered_episodes"
    )
    value = {
        "schema": "before-we-act.bicoord.dino-b0h-checkpoint/1",
        "status": "PASSED_SMOKE" if args.stage == "smoke" else "PASSED",
        "format": CHECKPOINT_FORMAT,
        "stage": args.stage,
        "update": update,
        "policy_family": "TemporalHistoryPolicy",
        "method_family": "CARE",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "benchmark_adapter": "BiCoord",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "history_steps": HISTORY_STEPS,
        "action_horizon": ACTION_HORIZON,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "effective_batch": EFFECTIVE_BATCH,
        "world_size": world,
        "local_batch": EFFECTIVE_BATCH // world,
        "sampling": "uniform_task_then_uniform_episode_time_and_arm",
        "strictly_decentralized": True,
        "shared_weights": True,
        "arm_id_input": False,
        "peer_proprioception_input": False,
        "peer_action_input": False,
        "all_1800_demonstrations": args.stage == "formal",
        "policy_episode_count": len(selected_episodes),
        "episode_source_identities": source_identities,
        "episode_selection": selection,
        "smoke_episode_selection": (
            SMOKE_EPISODE_SELECTION if args.stage == "smoke" else None
        ),
        "normalization_receipt_sha256": normalization_sha256,
        "dino_cache_receipt_sha256": cache_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "created_at_utc": _now(),
    }
    if selected_episodes and args.stage == "smoke":
        value["episode_sources"] = [_episode_source_row(episode) for episode in selected_episodes]
    return value


def _publish_supervisor_result(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    receipt: Path,
    rank: int,
    episodes: Sequence[BiCoordEpisode] | None = None,
) -> None:
    """Publish the generic stage result expected by ``Supervisor``.

    The upstream trainer writes its own checkpoint/status receipts.  This
    bridge only hashes those existing files and records their paths; it never
    changes model construction, optimizer settings, or checkpoint bytes.
    """
    if args.result is None or rank != 0:
        return
    if len(str(args.config_sha256)) != 64:
        raise ValueError("generic B0-H publication requires config SHA-256")
    stage = "b0h_smoke_train" if args.stage == "smoke" else "b0h_formal"
    artifacts = [
        artifact(checkpoint, kind="checkpoint"),
        artifact(receipt, kind="checkpoint_receipt"),
        artifact(args.output / "config.json", kind="training_config"),
        artifact(args.output / "status.json", kind="status"),
    ]
    publish_result(
        args,
        stage=stage,
        artifacts=artifacts,
        include_model_contract=True,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=_sha256(checkpoint),
        update=int(args.updates),
        target_updates=int(args.updates),
        effective_batch=EFFECTIVE_BATCH,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_batch=EFFECTIVE_BATCH // int(os.environ.get("WORLD_SIZE", "1")),
        all_1800_demonstrations=args.stage == "formal",
        policy_episode_count=len(episodes or ()),
        episode_source_identities=[
            str(episode.source_identity) for episode in (episodes or ())
        ],
        episode_selection=(
            SMOKE_EPISODE_SELECTION
            if args.stage == "smoke"
            else "all_discovered_episodes"
        ),
        smoke_episode_selection=(
            SMOKE_EPISODE_SELECTION if args.stage == "smoke" else None
        ),
        held_out_demonstrations=0,
        closed_loop_results_used_for_selection=False,
        teacher_present=False,
        strictly_decentralized=True,
        shared_weights=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.protocol_updates != FORMAL_UPDATES:
        raise ValueError("BiCoord B0-H protocol is fixed at 120000 updates")
    if not 1 <= args.updates <= args.protocol_updates:
        raise ValueError("invalid BiCoord B0-H update target")
    if args.stage == "formal" and args.updates != FORMAL_UPDATES:
        raise ValueError("formal BiCoord B0-H training requires 120000 updates")
    if args.stage == "smoke" and args.updates > 10:
        raise ValueError("BiCoord smoke training is capped at ten updates")
    if args.image_height % 16 or args.image_width % 16:
        raise ValueError("DINO dimensions must be divisible by 16")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.save_every < 1 or args.log_every < 1:
        raise ValueError("save/log intervals must be positive")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("BiCoord B0-H requires CUDA")
    if args.expected_world_size and world != args.expected_world_size:
        raise ValueError(
            f"expected {args.expected_world_size} DDP ranks, observed {world}"
        )
    if EFFECTIVE_BATCH % world:
        raise ValueError(f"world size must divide global batch {EFFECTIVE_BATCH}")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError(f"invalid LOCAL_RANK={local_rank}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(max(1, min(12, (os.cpu_count() or 12) // world)))

    normalization = load_normalization_receipt(
        args.normalization, require_formal=args.stage == "formal"
    )
    _validate_normalization(normalization, formal=args.stage == "formal")
    normalization_path = args.normalization
    if normalization_path.is_dir():
        candidates = (
            normalization_path / "normalization.json",
            normalization_path / "normalization_receipt.json",
            normalization_path / "manifest.json",
        )
        normalization_path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    if not normalization_path.is_file():
        raise FileNotFoundError(normalization_path)
    normalization_sha256 = _sha256(normalization_path)

    episodes = discover_bicoord_episodes(
        args.data_root,
        require_formal=args.stage == "formal",
        verify_schema=args.stage == "formal",
    )
    cache_path = args.visual_cache / "cache_receipt.json"
    cache_receipt, episodes = _validate_cache_receipt(
        cache_path,
        stage=args.stage,
        image_height=args.image_height,
        image_width=args.image_width,
        episodes=episodes,
    )
    cache_sha256 = _sha256(cache_path)
    if args.stage == "formal" and len(episodes) != FORMAL_EPISODES:
        raise ValueError("formal BiCoord corpus must contain exactly 1800 episodes")

    args.output.mkdir(parents=True, exist_ok=True)
    latest_path = args.output / "checkpoint_latest.pt"
    resume_path = args.resume
    if resume_path is None and args.auto_resume and latest_path.is_file():
        resume_path = latest_path
    saved = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("resume checkpoint has wrong BiCoord B0-H format")
        expected_resume = {
            "stage": args.stage,
            "seed": args.seed,
            "protocol_updates": args.protocol_updates,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_encoding": ACTION_ENCODING,
            "normalization_receipt_sha256": normalization_sha256,
            "dino_cache_receipt_sha256": cache_sha256,
            "policy_episode_count": len(episodes),
            "episode_source_identities": [
                episode.source_identity for episode in episodes
            ],
            "episode_selection": (
                SMOKE_EPISODE_SELECTION
                if args.stage == "smoke"
                else "all_discovered_episodes"
            ),
        }
        saved_config = saved.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("resume checkpoint lacks config")
        for key, expected in expected_resume.items():
            if saved_config.get(key) != expected:
                raise ValueError(
                    f"resume checkpoint differs at {key}: "
                    f"{saved_config.get(key)!r} != {expected!r}"
                )
    start = int(saved.get("update", 0)) if saved else 0
    if not 0 <= start <= args.updates:
        raise ValueError("resume update is outside the requested budget")

    sampler = BiCoordBalancedDistributedBatchSampler(
        episodes,
        updates=args.protocol_updates,
        seed=args.seed,
        rank=rank,
        world_size=world,
        start_update=start,
    )
    if saved:
        sampler.validate_cursor(saved["sample_cursor"])
    dataset = BiCoordTemporalDataset(
        episodes,
        normalization,
        args.visual_cache,
        image_height=args.image_height,
        image_width=args.image_width,
        cache_limit=max(16, args.workers * 4),
        require_visual_cache=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = TemporalHistoryPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        variant="hidden_residual",
        horizon=ACTION_HORIZON,
        d_model=384,
        enc_layers=4,
        dec_layers=7,
        roles=4,
        role_rank=32,
        history_layers=2,
        dino_model=args.dino_model,
        image_height=args.image_height,
        image_width=args.image_width,
        strict_dino_contract=True,
    ).to(device)
    router_prefix = (
        "compatibility",
        "role_prototypes",
        "route_state",
        "route_observation",
        "route_mlp",
    )
    body: list[torch.nn.Parameter] = []
    router: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (router if name.startswith(router_prefix) else body).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": args.lr}, {"params": router, "lr": args.router_lr}],
        weight_decay=1e-4,
    )

    def schedule(step: int) -> float:
        warmup = min(1.0, (step + 1) / max(1, args.warmup))
        progress = min(1.0, (step + 1) / args.protocol_updates)
        return warmup * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
    wrapped = (
        DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world > 1
        else model
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config = {
        "format_version": CONFIG_FORMAT,
        "policy_family": "TemporalHistoryPolicy",
        "method_family": "CARE",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "benchmark_adapter": "BiCoord",
        "stage": args.stage,
        "seed": args.seed,
        "protocol_updates": args.protocol_updates,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
        "variant": "hidden_residual",
        "d_model": 384,
        "enc_layers": 4,
        "dec_layers": 7,
        "roles": 4,
        "role_rank": 32,
        "history_layers": 2,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "dino_model": args.dino_model,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "recording_alignment": {
            "observation_row_offset": 0,
            "action_row_offset": 1,
            "action_lag_rows": 1,
        },
        "policy_contract": POLICY_CONTRACT,
        "tasks": list(TASKS),
        "effective_batch": EFFECTIVE_BATCH,
        "world_size": world,
        "local_batch": EFFECTIVE_BATCH // world,
        "sampling": "uniform_task_then_uniform_episode_time_and_arm",
        "shared_weights": True,
        "arm_id_input": False,
        "peer_proprioception_input": False,
        "peer_action_input": False,
        "all_1800_demonstrations": args.stage == "formal",
        "policy_episode_count": len(episodes),
        "episode_source_identities": [
            episode.source_identity for episode in episodes
        ],
        "episode_sources": (
            [_episode_source_row(episode) for episode in episodes]
            if args.stage == "smoke"
            else None
        ),
        "episode_selection": (
            SMOKE_EPISODE_SELECTION
            if args.stage == "smoke"
            else "all_discovered_episodes"
        ),
        "smoke_episode_selection": (
            SMOKE_EPISODE_SELECTION if args.stage == "smoke" else None
        ),
        "normalization_receipt_sha256": normalization_sha256,
        "dino_cache_receipt_sha256": cache_sha256,
        "dataset_repo_id": normalization.get("dataset_repo_id"),
        "dataset_revision": normalization.get("dataset_revision"),
    }
    if rank == 0:
        _atomic_json(args.output / "config.json", config)
        _atomic_json(
            args.output / "status.json",
            {
                "status": "TRAINING",
                "stage": args.stage,
                "update": start,
                "target_updates": args.updates,
                "world_size": world,
                "started_at_utc": _now(),
            },
        )
    if start == args.updates:
        if rank == 0:
            receipt = _completion_receipt(
                args=args,
                update=start,
                checkpoint=latest_path,
                normalization_sha256=normalization_sha256,
                cache_sha256=cache_sha256,
                world=world,
                episodes=episodes,
            )
            _atomic_json(args.output / "checkpoint_receipt.json", receipt)
            _atomic_json(
                args.output / "status.json",
                {
                    "status": receipt["status"],
                    "stage": args.stage,
                    "update": start,
                    "target_updates": args.updates,
                    "checkpoint": str(latest_path.resolve()),
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "completed_at_utc": _now(),
                },
            )
            _publish_supervisor_result(
                args,
                checkpoint=latest_path,
                receipt=args.output / "checkpoint_receipt.json",
                rank=rank,
                episodes=episodes,
            )
        if world > 1:
            dist.destroy_process_group()
        return 0

    started = time.time()
    last_metrics: dict[str, Any] = {}
    for update, batch in enumerate(loader, start=start + 1):
        if update > args.updates:
            break
        step_seed = args.seed + 10_000_019 * update + 100_003 * rank
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        inputs = {
            key: batch[key].to(device, non_blocking=True)
            for key in BiCoordTemporalDataset.MODEL_INPUT_FIELDS
        }
        inputs["global_rgb"] = inputs["global_rgb"].float().div_(255)
        inputs["local_rgb"] = inputs["local_rgb"].float().div_(255)
        actions = batch["action"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        do_counterfactual = update % args.counterfactual_every == 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (
                prediction,
                mu,
                logvar,
                routes,
                counterfactual,
                counterfactual_target,
                _base,
                _residual,
                _visual,
            ) = wrapped(
                **inputs,
                actions=actions,
                return_routing=True,
                counterfactual=do_counterfactual,
            )
            numerator = ((prediction - actions).square().mean(-1) * mask).sum()
            denominator = _distributed_sum(mask.sum().float(), world).clamp_min(1)
            action_loss = numerator * world / denominator
            local_kl = -0.5 * (
                1 + logvar - mu.square() - logvar.exp()
            ).sum(-1).sum()
            kl_loss = local_kl * world / EFFECTIVE_BATCH
            coupling = prediction.new_zeros(())
            if do_counterfactual:
                errors = (
                    counterfactual - counterfactual_target.unsqueeze(2)
                ).square().mean(-1)
                target = (
                    -errors.detach()
                    / errors.detach().std(-1, keepdim=True).clamp_min(1e-3)
                ).softmax(-1)
                coupling = F.kl_div(
                    routes[:1].clamp_min(1e-8).log(),
                    target,
                    reduction="batchmean",
                )
            loss = action_loss + args.beta * kl_loss + args.capability_weight * coupling
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite BiCoord B0-H loss at {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite BiCoord B0-H gradient at {update}")
        optimizer.step()
        scheduler.step()
        last_metrics = {
            "status": "TRAINING",
            "stage": args.stage,
            "update": update,
            "target_updates": args.updates,
            "loss": float(loss.detach()),
            "action_loss": float(action_loss.detach()),
            "kl_loss": float(kl_loss.detach()),
            "coupling_loss": float(coupling.detach()),
            "gradient_norm": float(gradient.detach()),
            "body_lr": scheduler.get_last_lr()[0],
            "router_lr": scheduler.get_last_lr()[1],
            "elapsed_seconds": time.time() - started,
            "strictly_decentralized": True,
            "world_size": world,
        }
        should_log = (
            update == start + 1
            or update % args.log_every == 0
            or update == args.updates
        )
        if rank == 0 and should_log:
            print(json.dumps(last_metrics, sort_keys=True), flush=True)
            with (args.output / "progress.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            _atomic_json(args.output / "status.json", last_metrics)
        if rank == 0 and (
            update % args.save_every == 0 or update == args.updates
        ):
            payload = {
                "format": CHECKPOINT_FORMAT,
                "format_version": CHECKPOINT_FORMAT,
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "sample_cursor": sampler.cursor_receipt(update),
                "stats": {
                    "q_mean": normalization["qpos_mean"],
                    "q_std": normalization["qpos_std"],
                    "a_mean": normalization["action_mean"],
                    "a_std": normalization["action_std"],
                    "q_min": normalization["qpos_min"],
                    "q_max": normalization["qpos_max"],
                    "a_min": normalization["action_min"],
                    "a_max": normalization["action_max"],
                    "action_encoding": ACTION_ENCODING,
                    "gripper_encoding": GRIPPER_ENCODING,
                    "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
                    "gripper_thresholding": False,
                    "gripper_reparameterization": False,
                },
                "config": config,
                "last_metrics": last_metrics,
            }
            numbered = args.output / f"checkpoint_{update:06d}.pt"
            _atomic_save(payload, numbered)
            _atomic_save(payload, latest_path)
            if update == args.updates:
                _atomic_save(payload, args.output / "final.pt")

    if world > 1:
        dist.barrier()
    if rank == 0:
        if not latest_path.is_file():
            raise RuntimeError("BiCoord B0-H completed without a checkpoint")
        receipt = _completion_receipt(
            args=args,
            update=args.updates,
            checkpoint=latest_path,
            normalization_sha256=normalization_sha256,
            cache_sha256=cache_sha256,
            world=world,
            episodes=episodes,
        )
        _atomic_json(args.output / "checkpoint_receipt.json", receipt)
        _atomic_json(
            args.output / "status.json",
            {
                "status": receipt["status"],
                "stage": args.stage,
                "update": args.updates,
                "target_updates": args.updates,
                "checkpoint": str(latest_path.resolve()),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "completed_at_utc": _now(),
            },
        )
        _publish_supervisor_result(
            args,
            checkpoint=latest_path,
            receipt=args.output / "checkpoint_receipt.json",
            rank=rank,
            episodes=episodes,
        )
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    main()
