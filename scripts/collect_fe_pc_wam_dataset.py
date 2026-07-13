"""One-command dataset collection with deterministic train/val/test splits.

The collector deliberately mixes scripted, disturbed, and recovery behavior
across nominal and communication-relevant scenarios.  Every written episode
uses the strict local-observation schema; legacy files are never
mixed into these directories.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collect import collect_one_episode  # noqa: E402
from data.local_observation import SensorSimulationConfig  # noqa: E402
from data.policies import ScriptedPolicy  # noqa: E402
from data.schema import (  # noqa: E402
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
    save_episode,
)
from envs.two_robot_carry_env import (  # noqa: E402
    CarryEnvConfig,
    TwoRobotCarryNarrowPassageEnv,
)


SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}

# Repeated names are intentional categorical weights.  Selection is derived
# from each episode seed, so interrupted collection can resume reproducibly.
BALANCED_SCENARIOS = (
    "nominal",
    "nominal",
    "nominal",
    "nominal",
    "narrow",
    "narrow",
    "occlusion",
    "occlusion",
    "asymmetric_obstacle",
    "asymmetric_obstacle",
    "blocked_passage",
    "blocked_passage",
    "false_belief",
    "hard_comm",
)
BALANCED_MODES = (
    "scripted",
    "scripted",
    "scripted",
    "scripted",
    "noisy",
    "noisy",
    "noisy",
    "noisy",
    "recovery",
    "recovery",
)
NOISE_BY_MODE = {"scripted": 0.0, "noisy": 0.45, "recovery": 0.0}
OBJECT_DROPOUT_BY_SCENARIO = {
    "nominal": 0.05,
    "narrow": 0.08,
    "occlusion": 0.40,
    "asymmetric_obstacle": 0.12,
    "blocked_passage": 0.15,
    "false_belief": 0.20,
    "hard_comm": 0.50,
}


@dataclass(frozen=True)
class CollectionConfig:
    out_dir: str
    train_episodes: int = 2400
    val_episodes: int = 400
    test_episodes: int = 400
    pilot_episodes: int = 100
    pilot_only: bool = False
    seed: int = 1000
    episode_len: int = 500
    randomize: bool = True
    object_position_std: float = 0.025
    object_yaw_std: float = 0.035
    base_object_dropout: float = 0.05
    rgb_camera_names: tuple[str, ...] = ()
    rgb_calibration_reference: str = ""
    resume: bool = False
    profile: str = "private_gates"

    def __post_init__(self) -> None:
        for name in ("train_episodes", "val_episodes", "test_episodes"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.train_episodes <= 0:
            raise ValueError("train_episodes must be positive")
        if self.episode_len <= 0:
            raise ValueError("episode_len must be positive")
        if self.object_position_std < 0 or self.object_yaw_std < 0:
            raise ValueError("object measurement standard deviations cannot be negative")
        if not 0.0 <= self.base_object_dropout <= 1.0:
            raise ValueError("base_object_dropout must be in [0, 1]")
        if self.pilot_episodes < 0:
            raise ValueError("pilot_episodes cannot be negative")
        if self.profile not in {"balanced", "private_gates"}:
            raise ValueError(f"unsupported collection profile {self.profile!r}")


def episode_recipe(seed: int, config: CollectionConfig) -> dict[str, Any]:
    """Return a deterministic behavior/sensor recipe for one episode seed."""

    rng = np.random.default_rng(seed)
    if config.profile == "private_gates":
        scenario = "private_gates"
        # The formal dataset is expert data. Counterfactual/off-policy
        # behavior is generated separately from saved decision snapshots.
        mode = "scripted"
    else:
        scenario = str(rng.choice(BALANCED_SCENARIOS))
        mode = str(rng.choice(BALANCED_MODES))
    dropout = max(
        float(config.base_object_dropout),
        float(OBJECT_DROPOUT_BY_SCENARIO.get(scenario, 0.10)),
    )
    return {
        "scenario": scenario,
        "mode": mode,
        "noise_std": float(NOISE_BY_MODE[mode]),
        "object_dropout_prob": dropout,
    }


def collect_dataset(config: CollectionConfig) -> dict[str, Any]:
    root = Path(config.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pilot_report = None
    if config.pilot_episodes > 0:
        pilot_report = _collect_split(root, "pilot", config.pilot_episodes, config)
        if pilot_report["success_rate"] < 0.95:
            raise RuntimeError(
                "pilot success rate is below 95%; fix the task/expert before formal collection"
            )
    target_counts = {
        "train": config.train_episodes,
        "val": config.val_episodes,
        "test": config.test_episodes,
    }
    split_reports = {}
    for split, target_count in target_counts.items():
        if config.pilot_only:
            break
        if target_count == 0:
            continue
        split_reports[split] = _collect_split(root, split, target_count, config)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_code_sha256": _collection_code_sha256(),
        "profile": config.profile,
        "root": str(root),
        "config": asdict(config),
        "split_seed_offsets": dict(SPLIT_SEED_OFFSETS),
        "seed_manifest": {
            split: {
                "start": int(config.seed + SPLIT_SEED_OFFSETS[split]),
                "stop_exclusive": int(
                    config.seed + SPLIT_SEED_OFFSETS[split] + target_counts[split]
                ),
            }
            for split in target_counts
            if target_counts[split] > 0
        },
        "splits": split_reports,
        "pilot": pilot_report,
        "split_seeds_frozen_before_collection": True,
        "old_data_or_checkpoint_compatible": False,
        "legacy_data_compatible": False,
        "deployable_input_contract": (
            "ego-local sensors/task/previous action + optional object estimate; "
            "no teammate state or object truth"
        ),
        "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
        "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
        "local_force_units": LOCAL_FORCE_UNITS,
        "local_force_scale_newtons": float(
            CarryEnvConfig().local_force_scale_newtons
        ),
        "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
    }
    _atomic_json(root / "dataset_manifest.json", manifest)
    return manifest


def _collect_split(
    root: Path,
    split: str,
    target_count: int,
    config: CollectionConfig,
) -> dict[str, Any]:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(split_dir.glob("episode_*.hdf5"))
    _validate_existing_indices(existing)
    if existing and not config.resume:
        raise FileExistsError(
            f"{split_dir} already contains {len(existing)}  episodes; "
            "pass --resume to continue without overwriting"
        )
    if existing and config.resume:
        # Preflight every existing artifact before appending anything.  This
        # prevents a partial legacy directory from becoming a mixed
        # legacy/strict dataset when collection resumes.
        _summarize_split(split_dir)
    if len(existing) > target_count:
        raise ValueError(
            f"{split_dir} contains {len(existing)} episodes, above requested target {target_count}"
        )

    environments: dict[str, TwoRobotCarryNarrowPassageEnv] = {}
    start = len(existing)
    seed_base = config.seed + ({**SPLIT_SEED_OFFSETS, "pilot": 3_000_000})[split]
    progress = tqdm(range(start, target_count), desc=f"collect {split}")
    for episode_index in progress:
        seed = seed_base + episode_index
        recipe = episode_recipe(seed, config)
        scenario = recipe["scenario"]
        env = environments.get(scenario)
        if env is None:
            env = TwoRobotCarryNarrowPassageEnv(
                CarryEnvConfig(
                    scenario=scenario,
                    episode_len=config.episode_len,
                    seed=seed,
                )
            )
            environments[scenario] = env
        policy = ScriptedPolicy(
            noise_std=recipe["noise_std"],
            seed=seed,
            mode=recipe["mode"],
        )
        sensor_config = SensorSimulationConfig(
            control_dt=env.cfg.control_dt,
            object_dropout_prob=recipe["object_dropout_prob"],
            object_position_std=config.object_position_std,
            object_yaw_std=config.object_yaw_std,
        )
        episode, _, spec = collect_one_episode(
            env,
            policy,
            seed,
            randomize=config.randomize,
            sensor_config=sensor_config,
        )
        episode.metadata.update(
            {
                "split": split,
                "episode_index": episode_index,
                "collection_profile": config.profile,
                "mode": recipe["mode"],
                "policy_noise_std": recipe["noise_std"],
                "object_dropout_prob": recipe["object_dropout_prob"],
                "rgb_camera_names": list(config.rgb_camera_names),
                "rgb_calibration_reference": config.rgb_calibration_reference,
            }
        )
        path = split_dir / f"episode_{episode_index:06d}.hdf5"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing episode {path}")
        save_episode(path, episode, spec)
        progress.set_postfix(
            scenario=scenario,
            mode=recipe["mode"],
            success=int(bool(episode.metadata["success"])),
        )

    report = _summarize_split(split_dir)
    _atomic_json(split_dir / "summary.json", report)
    return report


def _summarize_split(split_dir: Path) -> dict[str, Any]:
    paths = sorted(split_dir.glob("episode_*.hdf5"))
    scenarios: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    informed_agents: Counter[str] = Counter()
    maneuvers: Counter[str] = Counter()
    successes = 0
    transitions = 0
    for path in paths:
        with h5py.File(path, "r") as file:
            if str(file.attrs.get("schema_version", "")) != SCHEMA_VERSION:
                raise ValueError(f"incompatible episode found in split: {path}")
            if (
                str(file.attrs.get("local_contact_semantics", ""))
                != STRICT_LOCAL_CONTACT_SEMANTICS
                or str(file.attrs.get("local_force_semantics", ""))
                != STRICT_LOCAL_FORCE_SEMANTICS
                or str(file.attrs.get("local_force_units", ""))
                != LOCAL_FORCE_UNITS
                or str(file.attrs.get("local_sensor_provenance", ""))
                != STRICT_LOCAL_SENSOR_PROVENANCE
            ):
                raise ValueError(
                    f"incompatible local sensor semantics found in split: {path}; "
                    "recollect into a new dataset root"
                )
            scale = float(file.attrs.get("local_force_scale_newtons", np.nan))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"invalid local force scale in {path}")
            metadata = file["metadata"].attrs
            scenarios[str(metadata.get("scenario", "unknown"))] += 1
            modes[str(metadata.get("mode", "unknown"))] += 1
            success = bool(metadata.get("success", False))
            successes += int(success)
            if not success:
                failures[str(metadata.get("failure_reason", "unknown"))] += 1
            transitions += int(file.attrs["num_transitions"])
            event_truth = np.asarray(file["privileged/observations/private_event_truth"][:])
            if event_truth.size:
                # Count each event once at its first appearance.
                seen: set[tuple[int, int, int, int]] = set()
                for row in event_truth:
                    key = tuple(int(v) for v in row.tolist())
                    if key[0] < 0 or key[1] < 0 or key in seen:
                        continue
                    seen.add(key)
                    event_types[str(key[1])] += 1
                    informed_agents[str(key[2])] += 1
                    maneuvers[str(key[3])] += 1
    count = len(paths)
    return {
        "path": str(split_dir.resolve()),
        "episodes": count,
        "transitions": transitions,
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
        "scenario_counts": dict(sorted(scenarios.items())),
        "mode_counts": dict(sorted(modes.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
        "event_type_counts": dict(sorted(event_types.items())),
        "informed_agent_counts": dict(sorted(informed_agents.items())),
        "maneuver_counts": dict(sorted(maneuvers.items())),
    }


def _validate_existing_indices(paths: list[Path]) -> None:
    indices = []
    for path in paths:
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError as exc:
            raise ValueError(f"invalid episode filename: {path.name}") from exc
    if indices != list(range(len(indices))):
        raise ValueError(
            "existing episode indices must be contiguous from zero before --resume"
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _collection_code_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "envs/two_robot_carry_env.py",
        "data/collect.py",
        "data/local_observation.py",
        "data/schema.py",
        "data/policies.py",
    ):
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a complete decentralized FE-PC-WAM dataset"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-episodes", type=int, default=2400)
    parser.add_argument("--val-episodes", type=int, default=400)
    parser.add_argument("--test-episodes", type=int, default=400)
    parser.add_argument("--pilot-episodes", type=int, default=100)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--randomize", type=int, choices=(0, 1), default=1)
    parser.add_argument("--object-position-std", type=float, default=0.025)
    parser.add_argument("--object-yaw-std", type=float, default=0.035)
    parser.add_argument("--base-object-dropout", type=float, default=0.05)
    parser.add_argument("--rgb-camera-names", nargs="*", default=[])
    parser.add_argument("--rgb-calibration-reference", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--profile", choices=("private_gates", "balanced"), default="private_gates"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect 2 train, 1 val, 1 test episodes with at most 64 steps",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> CollectionConfig:
    train_episodes = 2 if args.smoke else args.train_episodes
    val_episodes = 1 if args.smoke else args.val_episodes
    test_episodes = 1 if args.smoke else args.test_episodes
    episode_len = min(args.episode_len, 64) if args.smoke else args.episode_len
    return CollectionConfig(
        out_dir=args.out_dir,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        test_episodes=test_episodes,
        pilot_episodes=0 if args.smoke else args.pilot_episodes,
        pilot_only=args.pilot_only,
        seed=args.seed,
        episode_len=episode_len,
        randomize=bool(args.randomize),
        object_position_std=args.object_position_std,
        object_yaw_std=args.object_yaw_std,
        base_object_dropout=args.base_object_dropout,
        rgb_camera_names=tuple(args.rgb_camera_names),
        rgb_calibration_reference=args.rgb_calibration_reference,
        resume=args.resume,
        profile=args.profile,
    )


def main() -> None:
    config = config_from_args(build_parser().parse_args())
    manifest = collect_dataset(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
