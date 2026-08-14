#!/usr/bin/env python3
"""Audit SSC-V7 M1 HDF5 schema, timing, sample keys, and simulator replay."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


BASE_COMMIT = "945d1b49247612f6e67d79104726b67915cf86bf"
ROBOFACTORY_COMMIT = "5868242322414a91454e22f1dd9641f613ba1bcf"
QPOS_TOLERANCE = 1e-4
TASKS: dict[str, dict[str, Any]] = {
    "lift_barrier": {
        "env_id": "LiftBarrier-rf",
        "config": "lift_barrier.yaml",
        "agents": 2,
        "cameras": ("global", "agent_0", "agent_1"),
    },
    "camera_alignment": {
        "env_id": "CameraAlignment-rf",
        "config": "camera_alignment.yaml",
        "agents": 3,
        "cameras": ("global", "agent_0", "agent_1", "agent_2"),
    },
    "long_pipeline_delivery": {
        "env_id": "LongPipelineDelivery-rf",
        "config": "long_pipeline_delivery.yaml",
        "agents": 4,
        "cameras": ("global", "agent_0", "agent_1", "agent_2", "agent_3"),
    },
    "take_photo": {
        "env_id": "TakePhoto-rf",
        "config": "take_photo.yaml",
        "agents": 4,
        "cameras": ("global", "agent_0", "agent_1", "agent_2", "agent_3"),
    },
    "pass_shoe": {
        "env_id": "PassShoe-rf",
        "config": "pass_shoe.yaml",
        "agents": 2,
        "cameras": ("global", "agent_0", "agent_1"),
    },
    "place_food": {
        "env_id": "PlaceFood-rf",
        "config": "place_food.yaml",
        "agents": 2,
        "cameras": ("global",),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/datasets/robofactory_multitask"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2"),
    )
    parser.add_argument(
        "--robofactory-root", type=Path, default=Path("/workspace/RoboFactory")
    )
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=None,
        help="Defaults to RUN_ROOT/pre_registration/contracts/seed_contract.json.",
    )
    parser.add_argument(
        "--w10-seed-root",
        type=Path,
        default=Path("/workspace/bwa_runs/w10-six-task-v1/seeds/validation"),
    )
    parser.add_argument("--hash-workers", type=int, default=6)
    parser.add_argument("--replay-qpos-tolerance", type=float, default=QPOS_TOLERANCE)
    parser.add_argument(
        "--replay-gate-mode",
        choices=("strict", "benchmark_diagnostic"),
        default="strict",
        help=(
            "strict gates on qpos and terminal success; benchmark_diagnostic "
            "reports both but gates only on replay completion and exact state forks"
        ),
    )
    parser.add_argument(
        "--gate-revision",
        type=Path,
        default=None,
        help="Required frozen gate revision for benchmark_diagnostic mode.",
    )
    parser.add_argument(
        "--mode", choices=("all", "schema", "replay"), default="all"
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), text=True
    ).strip()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def field_roles() -> dict[str, Any]:
    return {
        "format_version": "ssc-v7.m1.field_roles/1",
        "deployment_current": [
            "data/observation/images/global",
            "data/observation/images/agent_N (only the matching ego agent)",
            "data/observation/agents/panda_N/qpos (only ego qpos)",
            "data/task/text",
        ],
        "legal_history_16_control_steps": [
            "past global RGB",
            "past matching-agent RGB (Place Food reuses global RGB)",
            "past ego qpos",
            "past ego commanded action",
        ],
        "measurement_label_or_audit_only": [
            "peer qpos/qvel",
            "peer commanded action",
            "own qvel",
            "success/done/terminated/truncated",
            "timestamps and frame indices (alignment and primary-key audit only)",
            "camera calibration",
        ],
        "excluded": {
            "data/action/executed": (
                "command echo, not independent actuator feedback; do not present it as "
                "an executed-action sensor"
            ),
            "data/task/id": "task ID is not an approved model input",
            "frame_index_or_timestamp_as_progress": "forbidden shortcut",
        },
        "not_present_in_hdf5": [
            "simulator object pose",
            "contact/grasp truth",
            "role or subtask boundary",
            "communication",
            "failure reason",
        ],
        "place_food_camera_rule": (
            "only global RGB is stored; the frozen input contract reuses it as the "
            "matching-agent view"
        ),
    }


def all_dataset_paths(group: h5py.Group) -> tuple[str, ...]:
    result: list[str] = []

    def visitor(name: str, value: h5py.Dataset | h5py.Group) -> None:
        if isinstance(value, h5py.Dataset) and not name.startswith("schema/"):
            result.append(name)

    group.visititems(visitor)
    return tuple(sorted(result))


def require_dataset(stream: h5py.File, path: str) -> h5py.Dataset:
    value = stream.get(path)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"missing dataset {path}")
    return value


def exact_equal(first: h5py.Dataset, second: h5py.Dataset) -> bool:
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    rows = int(first.shape[0])
    chunk = 64
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        if not np.array_equal(first[start:stop], second[start:stop]):
            return False
    return True


def concatenate_equal(
    centralized: h5py.Dataset, components: Sequence[h5py.Dataset]
) -> bool:
    rows = int(centralized.shape[0])
    chunk = 256
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        expected = np.concatenate(
            [np.asarray(value[start:stop]) for value in components], axis=-1
        )
        if not np.array_equal(np.asarray(centralized[start:stop]), expected):
            return False
    return True


def all_finite(dataset: h5py.Dataset) -> bool:
    rows = int(dataset.shape[0])
    for start in range(0, rows, 512):
        if not np.isfinite(np.asarray(dataset[start : start + 512])).all():
            return False
    return True


def validate_episode(
    *,
    path: Path,
    task: str,
    entry: Mapping[str, Any],
    actual_sha256: str,
    expected_signature: tuple[str, ...] | None,
    key_stream: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    issues: list[str] = []
    spec = TASKS[task]
    with h5py.File(path, "r") as stream:
        steps = int(stream.attrs.get("num_steps", -1))
        seed = int(stream.attrs.get("seed", -1))
        episode_index = int(stream.attrs.get("episode_index", -1))
        fps = float(stream.attrs.get("fps", float("nan")))
        signature = all_dataset_paths(stream)
        if expected_signature is not None and signature != expected_signature:
            issues.append("dataset path signature changed within task")
        expected_attrs = {
            "format_version": "wam.trajectory.hdf5/1",
            "schema_profile": "robofactory_m1",
            "schema_version": "wam.robofactory.multimodal/1.0",
            "transition_semantics": "observation[t], action[t], observation[t+1]",
            "control_mode": "pd_joint_pos",
            "executed_action_source": "command_echo",
            "independent_actuator_feedback_available": False,
            "task_id": task,
        }
        for name, expected in expected_attrs.items():
            actual = stream.attrs.get(name)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                issues.append(f"attribute {name}={actual!r}, expected {expected!r}")
        if steps <= 0:
            issues.append(f"invalid num_steps={steps}")
        if steps != int(entry["recorded_steps"]):
            issues.append("num_steps differs from training manifest")
        if seed != int(entry["seed"]):
            issues.append("seed differs from training manifest")
        if episode_index != int(entry["episode_index"]):
            issues.append("episode_index differs from training manifest")
        if fps != 20.0 or float(stream.attrs.get("control_frequency_hz", -1)) != 20.0:
            issues.append("control frequency is not exactly 20 Hz")
        if actual_sha256 != str(entry["hdf5_sha256"]):
            issues.append("SHA256 differs from training manifest")
        if path.stat().st_size != int(entry["hdf5_size_bytes"]):
            issues.append("size differs from training manifest")

        for name in signature:
            dataset = require_dataset(stream, name)
            if not dataset.shape or int(dataset.shape[0]) != steps:
                issues.append(f"{name} does not have leading length T={steps}")

        agents = tuple(f"panda_{index}" for index in range(int(spec["agents"])))
        cameras = tuple(spec["cameras"])
        commanded = require_dataset(stream, "data/action/commanded")
        executed = require_dataset(stream, "data/action/executed")
        if commanded.shape != (steps, 8 * len(agents)):
            issues.append("central commanded-action shape is invalid")
        if not all_finite(commanded):
            issues.append("central commanded action contains non-finite values")
        if not exact_equal(commanded, executed):
            issues.append("executed action is not an exact commanded-action echo")
        commanded_agents = [
            require_dataset(stream, f"data/action/agents/{agent}/commanded")
            for agent in agents
        ]
        executed_agents = [
            require_dataset(stream, f"data/action/agents/{agent}/executed")
            for agent in agents
        ]
        if not concatenate_equal(commanded, commanded_agents):
            issues.append("central commanded action differs from per-agent concat")
        if not concatenate_equal(executed, executed_agents):
            issues.append("central executed action differs from per-agent concat")

        observation_components: list[h5py.Dataset] = []
        next_components: list[h5py.Dataset] = []
        for agent in agents:
            for component in ("qpos", "qvel"):
                current = require_dataset(
                    stream, f"data/observation/agents/{agent}/{component}"
                )
                following = require_dataset(
                    stream, f"data/next_observation/agents/{agent}/{component}"
                )
                if current.shape != (steps, 9) or following.shape != (steps, 9):
                    issues.append(f"{agent} {component} shape is not (T,9)")
                if not all_finite(current) or not all_finite(following):
                    issues.append(f"{agent} {component} contains non-finite values")
                if steps > 1 and not np.array_equal(current[1:], following[:-1]):
                    issues.append(f"{agent} {component} breaks obs[t+1] continuity")
                observation_components.append(current)
                next_components.append(following)
        if not concatenate_equal(
            require_dataset(stream, "data/observation/state"), observation_components
        ):
            issues.append("central observation state differs from per-agent concat")
        if not concatenate_equal(
            require_dataset(stream, "data/next_observation/state"), next_components
        ):
            issues.append("central next-observation state differs from per-agent concat")

        frame = np.asarray(require_dataset(stream, "data/frame_index")[:])
        timestamp = np.asarray(require_dataset(stream, "data/timestamp")[:])
        expected_frame = np.arange(steps, dtype=frame.dtype)
        expected_timestamp = np.arange(steps, dtype=np.float64) / 20.0
        expected_next_timestamp = (
            np.arange(1, steps + 1, dtype=np.float64) / 20.0
        )
        if not np.array_equal(frame, expected_frame):
            issues.append("frame_index is not exactly 0..T-1")
        if not np.array_equal(timestamp, expected_timestamp):
            issues.append("timestamp is not exactly frame_index / 20 Hz")
        if not np.all(np.asarray(require_dataset(stream, "data/seed")[:]) == seed):
            issues.append("per-frame seed is not constant")
        if not np.all(
            np.asarray(require_dataset(stream, "data/episode_index")[:])
            == episode_index
        ):
            issues.append("per-frame episode_index is not constant")
        task_ids = require_dataset(stream, "data/task/id").asstr()[:]
        if not np.all(task_ids == task):
            issues.append("per-frame task ID is inconsistent")

        for camera in cameras:
            image = require_dataset(stream, f"data/observation/images/{camera}")
            next_image = require_dataset(
                stream, f"data/next_observation/images/{camera}"
            )
            if (
                image.shape != (steps, 480, 640, 3)
                or next_image.shape != (steps, 480, 640, 3)
                or image.dtype != np.uint8
                or next_image.dtype != np.uint8
            ):
                issues.append(f"camera {camera} is not lossless uint8 480x640 RGB")
            current_frame = np.asarray(
                require_dataset(
                    stream, f"data/observation/image_frame_index/{camera}"
                )[:]
            )
            next_frame = np.asarray(
                require_dataset(
                    stream, f"data/next_observation/image_frame_index/{camera}"
                )[:]
            )
            if not np.array_equal(current_frame, expected_frame):
                issues.append(f"camera {camera} current frame index is misaligned")
            if not np.array_equal(next_frame, expected_frame + 1):
                issues.append(f"camera {camera} next frame index is misaligned")
            for name in ("image_timestamp", "image_state_timestamp"):
                current_time = np.asarray(
                    require_dataset(stream, f"data/observation/{name}/{camera}")[:]
                )
                next_time = np.asarray(
                    require_dataset(
                        stream, f"data/next_observation/{name}/{camera}"
                    )[:]
                )
                if not np.array_equal(current_time, expected_timestamp):
                    issues.append(f"camera {camera} current {name} is misaligned")
                if not np.array_equal(next_time, expected_next_timestamp):
                    issues.append(f"camera {camera} next {name} is misaligned")

        done = np.asarray(require_dataset(stream, "data/done")[:], dtype=bool)
        terminated = np.asarray(
            require_dataset(stream, "data/terminated")[:], dtype=bool
        )
        truncated = np.asarray(
            require_dataset(stream, "data/truncated")[:], dtype=bool
        )
        success = np.asarray(require_dataset(stream, "data/success")[:], dtype=bool)
        if not np.array_equal(done, terminated | truncated):
            issues.append("done differs from terminated OR truncated")
        if bool(success[-1]) != bool(entry["success"]):
            issues.append("terminal success differs from training manifest")
        if bool(terminated[-1]) != bool(entry["terminated"]):
            issues.append("terminal terminated differs from training manifest")
        if bool(truncated[-1]) != bool(entry["truncated"]):
            issues.append("terminal truncated differs from training manifest")

        for index in range(steps):
            key_stream.write(
                json.dumps(
                    {
                        "task": task,
                        "episode_sha256": actual_sha256,
                        "frame_index": index,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return (
        {
            "episode_index": int(entry["episode_index"]),
            "seed": int(entry["seed"]),
            "steps": steps,
            "sha256": actual_sha256,
            "issues": issues,
        },
        signature,
    )


def hash_manifest_entries(
    dataset_root: Path, entries: Sequence[tuple[str, Mapping[str, Any]]], workers: int
) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for task, entry in entries:
            path = dataset_root / task / str(entry["hdf5_path"])
            future = pool.submit(sha256_file, path)
            futures[future] = (task, int(entry["episode_index"]), path)
        completed = 0
        for future in as_completed(futures):
            task, episode_index, path = futures[future]
            result[(task, episode_index)] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"[M1 hash] {completed}/{len(futures)} {path}",
                    flush=True,
                )
    return result


def run_schema(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    flat_entries: list[tuple[str, Mapping[str, Any]]] = []
    for task in TASKS:
        manifest_path = args.dataset_root / task / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[task] = manifest
        entries = sorted(manifest["episodes"], key=lambda value: value["episode_index"])
        if len(entries) != 150:
            raise ValueError(f"{task}: expected 150 episodes, found {len(entries)}")
        flat_entries.extend((task, value) for value in entries)
    hashes = hash_manifest_entries(args.dataset_root, flat_entries, args.hash_workers)

    key_path = output_root / "sample_keys.jsonl.gz"
    task_results: dict[str, Any] = {}
    total_steps = 0
    total_issues = 0
    seen_hashes: set[str] = set()
    with gzip.open(key_path, "wt", encoding="utf-8", compresslevel=6) as key_stream:
        for task in TASKS:
            entries = sorted(
                manifests[task]["episodes"], key=lambda value: value["episode_index"]
            )
            signature: tuple[str, ...] | None = None
            episodes: list[dict[str, Any]] = []
            for position, entry in enumerate(entries, start=1):
                episode_index = int(entry["episode_index"])
                actual_hash = hashes[(task, episode_index)]
                if actual_hash in seen_hashes:
                    raise ValueError(f"duplicate episode SHA256: {actual_hash}")
                seen_hashes.add(actual_hash)
                path = args.dataset_root / task / str(entry["hdf5_path"])
                result, current_signature = validate_episode(
                    path=path,
                    task=task,
                    entry=entry,
                    actual_sha256=actual_hash,
                    expected_signature=signature,
                    key_stream=key_stream,
                )
                if signature is None:
                    signature = current_signature
                total_steps += int(result["steps"])
                total_issues += len(result["issues"])
                episodes.append(result)
                if position % 25 == 0 or position == len(entries):
                    print(f"[M1 schema] {task} {position}/{len(entries)}", flush=True)
            task_results[task] = {
                "episodes_checked": len(episodes),
                "frames_checked": sum(int(value["steps"]) for value in episodes),
                "issue_count": sum(len(value["issues"]) for value in episodes),
                "schema_signature_sha256": sha256_json(signature),
                "episodes_with_issues": [
                    value for value in episodes if value["issues"]
                ],
            }
    key_hash = sha256_file(key_path)
    return {
        "status": "PASSED" if total_issues == 0 else "FAILED",
        "files_checked": len(flat_entries),
        "files_with_hash_mismatch": sum(
            hashes[(task, int(entry["episode_index"]))] != entry["hdf5_sha256"]
            for task, entry in flat_entries
        ),
        "unique_episode_sha256": len(seen_hashes),
        "frames_checked": total_steps,
        "sample_keys": {
            "path": str(key_path),
            "count": total_steps,
            "sha256": key_hash,
            "primary_key": ["task", "episode_sha256", "frame_index"],
        },
        "issue_count": total_issues,
        "tasks": task_results,
    }


def derive_seed(
    namespace: str, purpose: str, task: str, index: int, retry: int
) -> int:
    message = f"{namespace}|{purpose}|{task}|{index}|{retry}".encode("utf-8")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_646 + 1


def expanded_replay_audit_seeds(
    contract_path: Path, w10_seed_root: Path
) -> dict[str, list[int]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    historical: set[int] = set()
    for task in contract["tasks"]:
        payload = json.loads(
            (w10_seed_root / f"{task}.json").read_text(encoding="utf-8")
        )
        historical.update(int(seed) for seed in payload["seeds"])
    used = set(historical)
    replay: dict[str, list[int]] = {}
    for task in contract["tasks"]:
        for purpose, count in (
            list(contract["measurement_purposes_per_task"].items())
            + list(contract["shared_candidate_evaluation_purposes_per_task"].items())
        ):
            values: list[int] = []
            for index in range(int(count)):
                retry = 0
                while True:
                    value = derive_seed(
                        contract["namespace"], purpose, task, index, retry
                    )
                    if value not in used:
                        break
                    retry += 1
                used.add(value)
                values.append(value)
            if purpose == "replay_audit":
                replay[task] = values
    return replay


def replay_selection(
    entries: Sequence[Mapping[str, Any]], audit_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    """Map frozen audit seeds to formal episodes without reading episode outcomes."""
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    count = len(entries)
    by_index = {int(entry["episode_index"]): dict(entry) for entry in entries}
    if set(by_index) != set(range(count)):
        raise ValueError("episode indices must be contiguous before modulo selection")
    for audit_seed in audit_seeds:
        index = int(audit_seed) % count
        while index in used:
            index = (index + 1) % count
        used.add(index)
        selected.append({"audit_seed": int(audit_seed), "entry": by_index[index]})
    return selected


def numpy_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value).reshape(-1)


def clone_state_tree(value: Any) -> Any:
    """Clone a simulator state without moving its tensors between devices."""
    if isinstance(value, Mapping):
        return {key: clone_state_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_state_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_state_tree(item) for item in value]
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def compare_state_trees(first: Any, second: Any, path: str = "state") -> dict[str, Any]:
    """Require exact equality across every leaf of two saved simulator states."""
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return {"exact": False, "max_abs_error": None, "first_mismatch": path}
        if set(first) != set(second):
            return {
                "exact": False,
                "max_abs_error": None,
                "first_mismatch": f"{path} keys",
            }
        maximum = 0.0
        first_mismatch: str | None = None
        for key in sorted(first):
            result = compare_state_trees(first[key], second[key], f"{path}/{key}")
            if result["max_abs_error"] is not None:
                maximum = max(maximum, float(result["max_abs_error"]))
            if not result["exact"] and first_mismatch is None:
                first_mismatch = str(result["first_mismatch"])
        return {
            "exact": first_mismatch is None,
            "max_abs_error": maximum,
            "first_mismatch": first_mismatch,
        }
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        if not isinstance(first, (tuple, list)) or not isinstance(
            second, (tuple, list)
        ):
            return {"exact": False, "max_abs_error": None, "first_mismatch": path}
        if len(first) != len(second):
            return {
                "exact": False,
                "max_abs_error": None,
                "first_mismatch": f"{path} length",
            }
        maximum = 0.0
        first_mismatch = None
        for index, (left, right) in enumerate(zip(first, second)):
            result = compare_state_trees(left, right, f"{path}/{index}")
            if result["max_abs_error"] is not None:
                maximum = max(maximum, float(result["max_abs_error"]))
            if not result["exact"] and first_mismatch is None:
                first_mismatch = str(result["first_mismatch"])
        return {
            "exact": first_mismatch is None,
            "max_abs_error": maximum,
            "first_mismatch": first_mismatch,
        }
    left = numpy_vector(first)
    right = numpy_vector(second)
    if left.shape != right.shape:
        return {
            "exact": False,
            "max_abs_error": None,
            "first_mismatch": f"{path} shape",
        }
    exact = bool(np.array_equal(left, right))
    try:
        maximum = float(
            np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
        )
    except (TypeError, ValueError):
        maximum = 0.0 if exact else None
    return {
        "exact": exact,
        "max_abs_error": maximum,
        "first_mismatch": None if exact else path,
    }


def replay_one(
    *,
    task: str,
    entry: Mapping[str, Any],
    audit_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import gymnasium as gym
    import robofactory  # noqa: F401

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from robofactory_rpc import scalar_bool, split_robofactory_action

    spec = TASKS[task]
    path = args.dataset_root / task / str(entry["hdf5_path"])
    with h5py.File(path, "r") as stream:
        metadata = json.loads(str(stream.attrs["episode_metadata_json"]))
        agent_order = tuple(str(value) for value in metadata["agent_order"])
        stored_seed = int(stream.attrs["seed"])
        actions = np.asarray(stream["data/action/commanded"][:], dtype=np.float32)
        current_qpos = {
            agent: np.asarray(
                stream[
                    f"data/observation/agents/{agent.replace('-', '_')}/qpos"
                ][:]
            )
            for agent in agent_order
        }
        next_qpos = {
            agent: np.asarray(
                stream[
                    f"data/next_observation/agents/{agent.replace('-', '_')}/qpos"
                ][:]
            )
            for agent in agent_order
        }
        recorded_success = bool(stream["data/success"][-1])
    config_path = (
        args.robofactory_root
        / "robofactory/configs/table"
        / str(spec["config"])
    )
    env = gym.make(
        str(spec["env_id"]),
        config=str(config_path),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        sim_backend="cpu",
    )
    maximum = 0.0
    first_exceed: dict[str, Any] | None = None
    replay_success = False
    anchor_frame = len(actions) // 2
    fork_repeatability: dict[str, Any] | None = None
    try:
        observation, _ = env.reset(seed=stored_seed)
        for frame_index, action in enumerate(actions):
            for agent in agent_order:
                error = float(
                    np.max(
                        np.abs(
                            numpy_vector(observation["agent"][agent]["qpos"])
                            - current_qpos[agent][frame_index]
                        )
                    )
                )
                maximum = max(maximum, error)
                if error > args.replay_qpos_tolerance and first_exceed is None:
                    first_exceed = {
                        "frame_index": frame_index,
                        "phase": "observation",
                        "agent": agent,
                        "error": error,
                    }
            if frame_index == anchor_frame:
                base_env = getattr(env, "base_env", env.unwrapped)
                anchor_state = clone_state_tree(base_env.get_state_dict())
                elapsed_steps = getattr(env, "_elapsed_steps", None)
                branches: list[dict[str, Any]] = []
                for _ in range(2):
                    base_env.set_state_dict(clone_state_tree(anchor_state))
                    if elapsed_steps is not None:
                        env._elapsed_steps = elapsed_steps
                    (
                        branch_observation,
                        branch_reward,
                        branch_terminated,
                        branch_truncated,
                        branch_info,
                    ) = env.step(
                        split_robofactory_action(action, agent_order=agent_order)
                    )
                    if not isinstance(branch_info, Mapping) or "success" not in branch_info:
                        raise RuntimeError("RoboFactory fork info lacks success")
                    branches.append(
                        {
                            "observation": branch_observation,
                            "reward": numpy_vector(branch_reward).copy(),
                            "terminated": scalar_bool(branch_terminated, name="terminated"),
                            "truncated": scalar_bool(branch_truncated, name="truncated"),
                            "success": scalar_bool(branch_info["success"], name="info.success"),
                            "state": clone_state_tree(base_env.get_state_dict()),
                        }
                    )
                state_comparison = compare_state_trees(
                    branches[0]["state"], branches[1]["state"]
                )
                observation_comparison = compare_state_trees(
                    branches[0]["observation"],
                    branches[1]["observation"],
                    path="observation",
                )
                qpos_exact = all(
                    np.array_equal(
                        numpy_vector(branches[0]["observation"]["agent"][agent]["qpos"]),
                        numpy_vector(branches[1]["observation"]["agent"][agent]["qpos"]),
                    )
                    for agent in agent_order
                )
                terminal_exact = all(
                    branches[0][name] == branches[1][name]
                    for name in ("terminated", "truncated", "success")
                )
                reward_exact = bool(
                    np.array_equal(branches[0]["reward"], branches[1]["reward"])
                )
                fork_repeatability = {
                    "anchor_rule": "frame_index = floor(recorded_steps / 2)",
                    "anchor_frame_index": anchor_frame,
                    "same_joint_action_repeated_after_restore": True,
                    "full_simulator_state_exact": bool(state_comparison["exact"]),
                    "full_simulator_state_max_abs_error": state_comparison[
                        "max_abs_error"
                    ],
                    "first_state_mismatch": state_comparison["first_mismatch"],
                    "full_observation_exact": bool(observation_comparison["exact"]),
                    "first_observation_mismatch": observation_comparison[
                        "first_mismatch"
                    ],
                    "agent_qpos_exact": qpos_exact,
                    "reward_exact": reward_exact,
                    "terminal_flags_and_success_exact": terminal_exact,
                    "passed": bool(
                        state_comparison["exact"]
                        and observation_comparison["exact"]
                        and qpos_exact
                        and reward_exact
                        and terminal_exact
                    ),
                }
                observation = branches[1]["observation"]
                terminated = branches[1]["terminated"]
                truncated = branches[1]["truncated"]
                info = {"success": branches[1]["success"]}
            else:
                observation, _, terminated, truncated, info = env.step(
                    split_robofactory_action(action, agent_order=agent_order)
                )
            for agent in agent_order:
                error = float(
                    np.max(
                        np.abs(
                            numpy_vector(observation["agent"][agent]["qpos"])
                            - next_qpos[agent][frame_index]
                        )
                    )
                )
                maximum = max(maximum, error)
                if error > args.replay_qpos_tolerance and first_exceed is None:
                    first_exceed = {
                        "frame_index": frame_index,
                        "phase": "next_observation",
                        "agent": agent,
                        "error": error,
                    }
            if not isinstance(info, Mapping) or "success" not in info:
                raise RuntimeError("RoboFactory replay info lacks success")
            replay_success = scalar_bool(info["success"], name="info.success")
    finally:
        env.close()
    trajectory_reproduction_passed = (
        maximum <= args.replay_qpos_tolerance
        and replay_success == recorded_success
    )
    fork_passed = (
        fork_repeatability is not None and bool(fork_repeatability["passed"])
    )
    gate_passed = (
        fork_passed
        if args.replay_gate_mode == "benchmark_diagnostic"
        else trajectory_reproduction_passed and fork_passed
    )
    return {
        "task": task,
        "audit_seed": audit_seed,
        "selection_rule": "episode_index = audit_seed modulo 150; linear-probe collisions",
        "episode_index": int(entry["episode_index"]),
        "stored_seed": stored_seed,
        "episode_sha256": str(entry["hdf5_sha256"]),
        "steps": len(actions),
        "max_qpos_abs_error": maximum,
        "qpos_tolerance": args.replay_qpos_tolerance,
        "qpos_passed": maximum <= args.replay_qpos_tolerance,
        "first_tolerance_exceedance": first_exceed,
        "recorded_terminal_success": recorded_success,
        "replay_terminal_success": replay_success,
        "terminal_success_exact_match": replay_success == recorded_success,
        "trajectory_reproduction_passed": trajectory_reproduction_passed,
        "fork_repeatability": fork_repeatability,
        "gate_mode": args.replay_gate_mode,
        "passed": gate_passed,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = args.seed_contract or (
        args.run_root / "pre_registration/contracts/seed_contract.json"
    )
    seeds = expanded_replay_audit_seeds(contract_path, args.w10_seed_root)
    results: list[dict[str, Any]] = []
    for task in TASKS:
        manifest = json.loads(
            (args.dataset_root / task / "training_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        entries = sorted(manifest["episodes"], key=lambda value: value["episode_index"])
        selected = replay_selection(entries, seeds[task])
        for position, item in enumerate(selected, start=1):
            print(
                f"[M1 replay] {task} {position}/{len(selected)} "
                f"episode={item['entry']['episode_index']}",
                flush=True,
            )
            results.append(
                replay_one(
                    task=task,
                    entry=item["entry"],
                    audit_seed=item["audit_seed"],
                    args=args,
                )
            )
    return {
        "gate_mode": args.replay_gate_mode,
        "status": "PASSED" if all(value["passed"] for value in results) else "FAILED",
        "episodes_checked": len(results),
        "episodes_completed": len(results),
        "trajectory_reproduction_passed": sum(
            bool(value["trajectory_reproduction_passed"]) for value in results
        ),
        "qpos_passed": sum(bool(value["qpos_passed"]) for value in results),
        "terminal_success_matched": sum(
            bool(value["terminal_success_exact_match"]) for value in results
        ),
        "fork_repeatability_passed": sum(
            bool(value["fork_repeatability"]["passed"]) for value in results
        ),
        "maximum_qpos_abs_error": max(
            float(value["max_qpos_abs_error"]) for value in results
        ),
        "selection": {
            "source": "frozen replay_audit seeds from seed_contract.json",
            "mapping": "episode_index = audit_seed modulo 150; linear-probe collisions",
            "outcome_blind": True,
        },
        "episodes": results,
    }


def main() -> int:
    args = parse_args()
    if args.hash_workers <= 0:
        raise ValueError("--hash-workers must be positive")
    if args.replay_qpos_tolerance <= 0:
        raise ValueError("--replay-qpos-tolerance must be positive")
    gate_revision: dict[str, Any] | None = None
    if args.replay_gate_mode == "benchmark_diagnostic":
        if args.gate_revision is None:
            raise ValueError(
                "--gate-revision is required for benchmark_diagnostic mode"
            )
        gate_revision = json.loads(args.gate_revision.read_text(encoding="utf-8"))
        if gate_revision.get("stage_id") != "SSC-V7-M1-R1":
            raise RuntimeError("unexpected benchmark-relaxed gate stage_id")
        acceptance = gate_revision.get("acceptance", {})
        if acceptance.get("qpos_reproduction", {}).get("role") != "diagnostic_only":
            raise RuntimeError("gate revision still gates on qpos reproduction")
        if acceptance.get("terminal_success", {}).get("role") != "diagnostic_only":
            raise RuntimeError("gate revision still gates on terminal success")
        if acceptance.get("state_fork_repeatability", {}).get("required") is not True:
            raise RuntimeError("gate revision does not require exact state forks")
    dry_run_receipt_path = args.run_root / "step0_audit/dry_run_receipt.json"
    dry_run_receipt = json.loads(dry_run_receipt_path.read_text(encoding="utf-8"))
    if dry_run_receipt.get("status") != "PASSED":
        raise RuntimeError("SSC-V7-M1 dry-run has not passed")
    output_name = (
        str(gate_revision["output_name"])
        if gate_revision is not None
        else "m1"
    )
    if not output_name or output_name in (".", "..") or "/" in output_name:
        raise ValueError("gate output_name must be one safe path component")
    output_root = args.run_root / "measurement" / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    roles = field_roles()
    write_json(output_root / "field_roles.json", roles)
    started = time.perf_counter()
    provenance = {
        "stage_id": (
            str(gate_revision["stage_id"])
            if gate_revision is not None
            else "SSC-V7-M1/M1"
        ),
        "branch": git_value(Path(__file__).resolve().parents[2], "branch", "--show-current"),
        "commit": git_value(Path(__file__).resolve().parents[2], "rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "robofactory_commit": git_value(args.robofactory_root, "rev-parse", "HEAD"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed_contract_sha256": sha256_file(
            args.seed_contract
            or args.run_root / "pre_registration/contracts/seed_contract.json"
        ),
        "dry_run_receipt_sha256": sha256_file(dry_run_receipt_path),
        "replay_gate_mode": args.replay_gate_mode,
    }
    if gate_revision is not None:
        provenance["gate_revision"] = str(args.gate_revision)
        provenance["gate_revision_sha256"] = sha256_file(args.gate_revision)
        expected_branch = str(gate_revision["implementation"]["branch"])
        expected_commit = str(gate_revision["implementation"]["commit"])
        expected_script_sha256 = str(
            gate_revision["implementation"]["script_sha256"]
        )
        if provenance["branch"] != expected_branch:
            raise RuntimeError("branch differs from the relaxed gate revision")
        if provenance["commit"] != expected_commit:
            raise RuntimeError("commit differs from the relaxed gate revision")
        if provenance["script_sha256"] != expected_script_sha256:
            raise RuntimeError("script differs from the relaxed gate revision")
    elif provenance["branch"] != "feat/ssc-v7-measurement":
        raise RuntimeError("strict M1 must run on feat/ssc-v7-measurement")
    if provenance["robofactory_commit"] != ROBOFACTORY_COMMIT:
        raise RuntimeError("RoboFactory commit differs from the frozen contract")
    schema: dict[str, Any] | None = None
    replay: dict[str, Any] | None = None
    if args.mode in ("all", "schema"):
        schema = run_schema(args, output_root)
        write_json(output_root / "schema_checks.json", schema)
    elif not (output_root / "schema_checks.json").is_file():
        raise FileNotFoundError("--mode replay requires an existing schema_checks.json")
    else:
        schema = json.loads(
            (output_root / "schema_checks.json").read_text(encoding="utf-8")
        )
    if args.mode in ("all", "replay") and schema["status"] == "PASSED":
        replay = run_replay(args)
        write_json(output_root / "replay_checks.json", replay)
    elif (output_root / "replay_checks.json").is_file():
        replay = json.loads(
            (output_root / "replay_checks.json").read_text(encoding="utf-8")
        )

    schema_passed = schema is not None and schema["status"] == "PASSED"
    replay_passed = replay is not None and replay["status"] == "PASSED"
    if not schema_passed:
        decision_code = "FAILED_SCHEMA"
    elif args.mode == "schema" and replay is None:
        decision_code = "SCHEMA_PASSED_REPLAY_PENDING"
    elif not replay_passed:
        decision_code = "FAILED_SCHEMA/REPLAY_NOT_DETERMINISTIC"
    else:
        decision_code = (
            "PASSED_M1_BENCHMARK_RELAXED"
            if args.replay_gate_mode == "benchmark_diagnostic"
            else "PASSED_M1"
        )
    passed_decision = decision_code in (
        "PASSED_M1",
        "PASSED_M1_BENCHMARK_RELAXED",
    )
    receipt = {
        "format_version": "ssc-v7.m1.schema_receipt/1",
        "completed_at_utc": utc_now(),
        "status": (
            "PASSED"
            if passed_decision
            else "PENDING"
            if decision_code == "SCHEMA_PASSED_REPLAY_PENDING"
            else "FAILED"
        ),
        "decision_code": decision_code,
        "provenance": provenance,
        "requirements": {
            "hdf5_files_expected": 900,
            "replay_episodes_per_task": 5,
            "qpos_reproduction": (
                {
                    "role": "diagnostic_only",
                    "pass_threshold": None,
                    "reported_reference_tolerance": args.replay_qpos_tolerance,
                }
                if args.replay_gate_mode == "benchmark_diagnostic"
                else {
                    "role": "required",
                    "max_abs_error_lte": args.replay_qpos_tolerance,
                }
            ),
            "terminal_success": {
                "role": (
                    "diagnostic_only"
                    if args.replay_gate_mode == "benchmark_diagnostic"
                    else "required"
                ),
                "exact_match": (
                    None
                    if args.replay_gate_mode == "benchmark_diagnostic"
                    else True
                ),
            },
            "save_restore_same_action_result_exact_match": True,
        },
        "schema": schema,
        "replay": replay,
        "field_roles": roles,
        "m2_or_oracle_sidecar_unlocked": passed_decision,
        "training_started": False,
        "stop_rule_applied": decision_code.startswith("FAILED"),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    write_json(output_root / "schema_receipt.json", receipt)
    artifact_paths = [
        output_root / name
        for name in (
            "field_roles.json",
            "sample_keys.jsonl.gz",
            "schema_checks.json",
            "replay_checks.json",
            "schema_receipt.json",
        )
        if (output_root / name).is_file()
    ]
    artifact_manifest = {
        "format_version": "ssc-v7.m1.artifact_manifest/1",
        "created_at_utc": utc_now(),
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(output_root / "artifact_manifest.json", artifact_manifest)
    print(decision_code, flush=True)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed_decision or decision_code == "SCHEMA_PASSED_REPLAY_PENDING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
