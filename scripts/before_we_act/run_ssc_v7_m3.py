#!/usr/bin/env python3
"""Run the frozen SSC-V7 M3-R3 convergence-repair measurement."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.before_we_act import audit_ssc_v7_m2 as m2  # noqa: E402


STAGE_ID = "SSC-V7-M3-R3"
M2_DECISION = "PASSED_M2_ORACLE_LABEL_GATE"
TASKS = tuple(m2.TASKS)
TASK_TEXT = {
    "lift_barrier": "Lift the barrier together",
    "camera_alignment": "Align and lift the camera and meat together",
    "long_pipeline_delivery": "Deliver the shoe through the robot pipeline",
    "take_photo": "Support the camera, lift the meat, and take a photo",
    "pass_shoe": "Pass the shoe to the teammate and place it in the goal",
    "place_food": "Pick up the meat and place it into the pot",
}
PUBLIC_TIME_SCALE = {
    "lift_barrier": 120,
    "camera_alignment": 120,
    "long_pipeline_delivery": 900,
    "take_photo": 250,
    "pass_shoe": 450,
    "place_food": 320,
}
SOURCE_SLICES = {
    "P": slice(0, 64),
    "T": slice(64, 128),
    "B": slice(128, 192),
}
CONDITIONS = (
    "E0",
    "HC",
    "label_shuffled",
    "time_phase_only",
    "B",
    "B_hat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("collect", "merge", "pilot-check", "tune", "seal", "test"),
    )
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=Path(
            "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/"
            "pre_registration/contracts/seed_contract.json"
        ),
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
        "--w10-seed-root",
        type=Path,
        default=Path("/workspace/bwa_runs/w10-six-task-v1/seeds/validation"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--purpose", choices=("repair_test",))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *arguments), text=True
    ).strip()


def load_gate(path: Path) -> dict[str, Any]:
    gate = read_json(path)
    if gate.get("stage_id") != STAGE_ID:
        raise RuntimeError("M3 gate has the wrong stage identity")
    if gate.get("status") != "FROZEN_BEFORE_REPAIR_TEST_COLLECTION":
        raise RuntimeError("M3-R3 gate was not frozen before repair-test collection")
    return gate


def run_path(args: argparse.Namespace, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else args.run_root / path


def preflight(args: argparse.Namespace, gate: Mapping[str, Any]) -> dict[str, Any]:
    implementation = gate["implementation"]
    script = REPOSITORY / str(implementation["script"])
    owner_receipt = args.run_root / str(
        gate["prerequisites"]["m2_owner_approval_receipt"]
    )
    owner = read_json(owner_receipt)
    checks = {
        "m2_passed": owner.get("decision_code") == M2_DECISION,
        "m2_receipt_hash": sha256_file(owner_receipt)
        == str(gate["prerequisites"]["m2_owner_approval_sha256"]),
        "repository_clean": git_value(REPOSITORY, "status", "--porcelain") == "",
        "implementation_is_ancestor": subprocess.call(
            (
                "git",
                "-C",
                str(REPOSITORY),
                "merge-base",
                "--is-ancestor",
                str(implementation["implementation_commit"]),
                "HEAD",
            )
        )
        == 0,
        "script_hash": sha256_file(script) == str(implementation["script_sha256"]),
        "robofactory_commit": git_value(
            args.robofactory_root, "rev-parse", "HEAD"
        )
        == m2.ROBOFACTORY_COMMIT,
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "owner_receipt": str(owner_receipt),
        "owner_receipt_sha256": sha256_file(owner_receipt),
        "repository_head": git_value(REPOSITORY, "rev-parse", "HEAD"),
        "gate_sha256": sha256_file(args.gate),
    }


def compact_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (480, 640, 3):
        raise ValueError(f"unexpected frozen RGB shape: {array.shape}")
    value = array.astype(np.float32).reshape(60, 8, 80, 8, 3).mean((1, 3))
    return np.rint(value).clip(0, 255).astype(np.uint8)


class CompactEpisodeWriter(m2.EpisodeWriter):
    """M2-compatible writer that stores a frozen compact transform of legal RGB."""

    def __init__(self, path: Path, task: str, seed: int, agent_count: int):
        super().__init__(path, task, seed, agent_count)
        self.stream.attrs["format_version"] = "ssc-v7.m3.compact_episode/1"
        self.stream.attrs["rgb_source_shape"] = (480, 640, 3)
        self.stream.attrs["rgb_compact_shape"] = (60, 80, 3)
        self.stream.attrs["rgb_transform"] = "8x8_area_mean_round_uint8"

    def append_observation(self, observation: Mapping[str, Any]) -> None:
        sensor_data = observation["sensor_data"]
        images: dict[str, np.ndarray] = {}
        for sensor_name, payload in sensor_data.items():
            if "rgb" not in payload:
                continue
            short = str(sensor_name)
            if short == "head_camera_global":
                key = "global"
            elif short.startswith("head_camera_agent"):
                key = f"agent_{short[len('head_camera_agent') :]}"
            else:
                continue
            raw = m2.tensor_array(payload["rgb"])[0].astype(np.uint8, copy=False)
            images[key] = compact_rgb(raw)
        if "global" not in images:
            raise ValueError("M3 observation lacks the frozen global RGB sensor")
        image_group = self.stream.require_group("data/observation/images")
        for key, image in images.items():
            dataset = self.image_datasets.get(key)
            if dataset is None:
                dataset = image_group.create_dataset(
                    key,
                    shape=(0, 60, 80, 3),
                    maxshape=(None, 60, 80, 3),
                    chunks=(8, 60, 80, 3),
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                self.image_datasets[key] = dataset
            self._append(dataset, image)
        agent_group = self.stream.require_group("data/observation/agents")
        for slot in range(self.agent_count):
            name = f"panda-{slot}"
            qpos = m2.tensor_array(observation["agent"][name]["qpos"])[0].astype(
                np.float32, copy=False
            )
            dataset = self.qpos_datasets.get(name)
            if dataset is None:
                dataset = agent_group.create_dataset(
                    name.replace("-", "_"),
                    shape=(0, qpos.shape[0]),
                    maxshape=(None, qpos.shape[0]),
                    chunks=(256, qpos.shape[0]),
                    dtype=np.float32,
                )
                self.qpos_datasets[name] = dataset
            self._append(dataset, qpos)


def collect_one_task(
    args: argparse.Namespace, gate: Mapping[str, Any], preflight_receipt: Mapping[str, Any]
) -> None:
    if args.task is None or args.purpose is None:
        raise ValueError("collect requires --task and --purpose")
    if args.output_root.exists():
        raise FileExistsError(f"fresh output root required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    seed_contract = read_json(args.seed_contract)
    expanded = m2.expanded_seed_manifest(seed_contract, args.w10_seed_root)
    if args.purpose != "repair_test":
        raise ValueError("M3-R3 collection is restricted to repair_test")
    purpose_name = "expert_candidate_pool"
    required = int(gate["collection"]["repair_test_successes_per_task"])
    first_candidate = int(
        gate["collection"]["first_unused_candidate_index_by_task"][args.task]
    )
    frozen_candidates = expanded["per_task"][args.task][purpose_name]
    candidates = frozen_candidates[first_candidate:]
    m2.EpisodeWriter = CompactEpisodeWriter
    from robofactory.planner import solutions

    spec = m2.TASKS[args.task]
    environment = m2.make_env(args.task, args.robofactory_root)
    wrapper = m2.M2AuditWrapper(environment, args.task, args.output_root)
    solver = getattr(solutions, str(spec["solver"]))
    attempts: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    try:
        for offset, seed in enumerate(candidates):
            candidate_index = first_candidate + offset
            if len(episodes) >= required:
                break
            print(
                f"[M3 {args.purpose}] {args.task} candidate={candidate_index} seed={seed}",
                flush=True,
            )
            wrapper.begin_attempt(int(seed), candidate_index)
            log_path = args.output_root / "logs" / (
                f"{args.task}.candidate_{candidate_index:03d}.seed_{seed}.log"
            )
            try:
                with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
                    result = solver(wrapper, seed=int(seed), debug=False, vis=False)
                success = bool(result != -1 and m2.scalar_bool(result[-1]["success"]))
                item = wrapper.finish_attempt(success, args.output_root)
                attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "seed": int(seed),
                        "success": success,
                        "log_path": str(log_path),
                        "log_sha256": sha256_file(log_path),
                    }
                )
                if item is not None:
                    item["success_rank"] = len(episodes)
                    episodes.append(item)
            except Exception:
                wrapper.abort_attempt()
                raise
        if len(episodes) != required:
            raise RuntimeError(
                f"{args.task} produced {len(episodes)}/{required} successful episodes"
            )
    finally:
        wrapper.abort_attempt()
        environment.close()
    receipt = {
        "format_version": "ssc-v7.m3.task_collection/1",
        "stage_id": STAGE_ID,
        "task": args.task,
        "purpose": args.purpose,
        "required_successes": required,
        "episodes": episodes,
        "attempts": attempts,
        "preflight": preflight_receipt,
        "elapsed_wall_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "completed_at_utc": utc_now(),
        "next_stage_authorized": "merge only",
        "formal_training_authorized": False,
    }
    write_json(args.output_root / "task_collection_receipt.json", receipt)
    print(f"SSC_V7_M3_{args.purpose.upper()}_{args.task.upper()}_COLLECTED")


def merge_collections(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.data_root is None or args.purpose is None:
        raise ValueError("merge requires --data-root and --purpose")
    if args.output_root.exists():
        raise FileExistsError(f"fresh output root required: {args.output_root}")
    if args.purpose != "repair_test":
        raise ValueError("M3-R3 merge is restricted to repair_test")
    base_manifest_path = run_path(
        args, str(gate["repair_training_data"]["r2_formal_manifest"])
    )
    expected_base_hash = str(
        gate["repair_training_data"]["r2_formal_manifest_sha256"]
    )
    if sha256_file(base_manifest_path) != expected_base_hash:
        raise RuntimeError("frozen R2 base manifest hash mismatch")
    base_manifest = read_json(base_manifest_path)
    episodes: list[dict[str, Any]] = []
    for episode in base_manifest["episodes"]:
        if episode["split"] not in {"train", "tune"}:
            continue
        item = deepcopy(episode)
        item["source_stage_id"] = str(base_manifest["stage_id"])
        episodes.append(item)
    for task in TASKS:
        for split, expected in (("train", 36), ("tune", 12)):
            actual = sum(
                1
                for item in episodes
                if item["task"] == task and item["split"] == split
            )
            if actual != expected:
                raise RuntimeError(
                    f"R2 base manifest has {actual} {task}/{split} episodes, expected {expected}"
                )
    receipts: list[dict[str, Any]] = []
    for task in TASKS:
        path = args.data_root / task / "task_collection_receipt.json"
        receipt = read_json(path)
        if receipt.get("task") != task or receipt.get("purpose") != args.purpose:
            raise RuntimeError(f"collection identity mismatch: {path}")
        receipts.append(
            {
                "task": task,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
        task_episodes = list(receipt["episodes"])
        required = int(gate["collection"]["repair_test_successes_per_task"])
        if len(task_episodes) != required:
            raise RuntimeError(f"wrong successful episode count for {task}")
        for rank, episode in enumerate(task_episodes):
            if int(episode["success_rank"]) != rank:
                raise RuntimeError(f"non-canonical success order for {task}")
            item = deepcopy(episode)
            item["split"] = "read_only_test"
            for key in ("hdf5_path", "sidecar_path"):
                path_value = Path(str(item[key]))
                if not path_value.is_file():
                    raise FileNotFoundError(path_value)
            if sha256_file(Path(str(item["hdf5_path"]))) != item["hdf5_sha256"]:
                raise RuntimeError("collected HDF5 hash mismatch")
            if sha256_file(Path(str(item["sidecar_path"]))) != item["sidecar_sha256"]:
                raise RuntimeError("collected sidecar hash mismatch")
            episodes.append(item)
    args.output_root.mkdir(parents=True)
    split_counts: dict[str, dict[str, int]] = {}
    for task in TASKS:
        split_counts[task] = {
            split: sum(
                1
                for item in episodes
                if item["task"] == task and item["split"] == split
            )
            for split in sorted({str(item["split"]) for item in episodes})
        }
    manifest = {
        "format_version": "ssc-v7.m3.dataset_manifest/1",
        "stage_id": STAGE_ID,
        "purpose": args.purpose,
        "created_at_utc": utc_now(),
        "gate": str(args.gate),
        "gate_sha256": sha256_file(args.gate),
        "source_receipts": receipts,
        "repair_training_data": {
            "r2_formal_manifest": str(base_manifest_path),
            "r2_formal_manifest_sha256": expected_base_hash,
            "included_splits": ["train", "tune"],
            "excluded_split": "read_only_test",
        },
        "split_counts": split_counts,
        "episodes": episodes,
        "test_is_sealed": True,
        "test_has_been_read": False,
    }
    write_json(args.output_root / "dataset_manifest.json", manifest)
    receipt = {
        "decision_code": "SSC_V7_M3_DATASET_MERGED",
        "manifest": str(args.output_root / "dataset_manifest.json"),
        "manifest_sha256": sha256_file(args.output_root / "dataset_manifest.json"),
        "split_counts": split_counts,
        "test_is_sealed": True,
        "formal_training_authorized": False,
    }
    write_json(args.output_root / "merge_receipt.json", receipt)
    print(receipt["decision_code"])


def read_sidecar(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def grid_mean(image: np.ndarray, rows: int, columns: int) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    height, width, channels = value.shape
    if height % rows or width % columns or channels != 3:
        raise ValueError("compact RGB is incompatible with the frozen grid")
    return (
        value.reshape(rows, height // rows, columns, width // columns, channels)
        .mean((1, 3))
        .reshape(-1)
        / 255.0
    ).astype(np.float32)


def task_text_feature(task: str) -> np.ndarray:
    raw = TASK_TEXT[task].encode("utf-8")
    result = np.zeros(64, dtype=np.float32)
    for index, byte in enumerate(raw[:64]):
        result[index] = (float(byte) - 127.5) / 127.5
    return result


def hash_feature(vector: np.ndarray, path: str, value: float = 1.0) -> None:
    digest = hashlib.sha256(path.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % vector.shape[0]
    sign = 1.0 if digest[4] & 1 else -1.0
    vector[index] += sign * float(value)


def hash_tree(vector: np.ndarray, path: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            hash_tree(vector, f"{path}.{key}", value[key])
    elif isinstance(value, list):
        for index, item in enumerate(value):
            label = item if isinstance(item, (str, int, bool)) else index
            hash_tree(vector, f"{path}[{label}]", item)
    elif isinstance(value, bool):
        hash_feature(vector, f"{path}={str(value).lower()}")
    elif value is None:
        hash_feature(vector, f"{path}=none")
    elif isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite oracle feature")
        hash_feature(vector, path, float(np.clip(number, -10.0, 10.0)))
    else:
        hash_feature(vector, f"{path}={value}")


def social_features(label: Mapping[str, Any], own_slot: int) -> np.ndarray:
    result = np.zeros(192, dtype=np.float32)
    progress = result[SOURCE_SLICES["P"]]
    for key in ("stage_id", "within_stage_progress", "remaining_goal_mask"):
        hash_tree(progress, key, label[key])
    hash_tree(progress, "factorized_predicates", label.get("factorized_predicates", {}))
    automaton = label["causal_automaton_state"]
    for key in (
        "completed_goal_mask",
        "completed_handoff_mask",
        "custody_transfer_count",
        "previous_stage_id",
    ):
        hash_tree(progress, f"causal.{key}", automaton.get(key))

    teammate = result[SOURCE_SLICES["T"]]
    for item in label["per_agent_contribution"]:
        if int(item["agent_slot"]) != own_slot:
            hash_tree(teammate, "peer.contribution", item)
    for item in label["agent_object_role_slots"]:
        if int(item["agent_slot"]) != own_slot:
            hash_tree(teammate, "peer.role", item)
    for object_name, state in sorted(label["grasp_contact_custody_state"].items()):
        for key in ("contact_agents", "grasp_agents", "controller_agents"):
            peers = [int(value) for value in state[key] if int(value) != own_slot]
            hash_tree(teammate, f"peer.{object_name}.{key}", peers)

    belief = result[SOURCE_SLICES["B"]]
    hash_tree(belief, "remaining_goal_mask", label["remaining_goal_mask"])
    hash_tree(
        belief, "grasp_contact_custody_state", label["grasp_contact_custody_state"]
    )
    hash_tree(
        belief,
        "collision_drop_contention_risk",
        label["collision_drop_contention_risk"],
    )
    hash_tree(
        belief,
        "causal.last_confirmed_custodian",
        automaton.get("last_confirmed_custodian", {}),
    )
    hash_tree(
        belief,
        "causal.custody_transfer_count",
        automaton.get("custody_transfer_count", 0),
    )
    return result


def uniform_indices(first: int, last: int, maximum: int) -> list[int]:
    if last < first:
        return []
    count = min(maximum, last - first + 1)
    if count == 1:
        return [first]
    values = np.rint(np.linspace(first, last, count)).astype(np.int64)
    return sorted({int(value) for value in values})


@dataclass
class ProbeData:
    legal: np.ndarray
    e0: np.ndarray
    social: np.ndarray
    time: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    tasks: np.ndarray
    episode_ids: np.ndarray
    frame_indices: np.ndarray
    agent_slots: np.ndarray

    def subset(self, indices: np.ndarray) -> "ProbeData":
        return ProbeData(
            **{
                name: getattr(self, name)[indices]
                for name in self.__dataclass_fields__
            }
        )

    def __len__(self) -> int:
        return int(self.legal.shape[0])


def stack_probe_rows(rows: list[dict[str, Any]]) -> ProbeData:
    if not rows:
        raise RuntimeError("probe dataset has no eligible rows")
    return ProbeData(
        legal=np.stack([row["legal"] for row in rows]).astype(np.float32),
        e0=np.stack([row["e0"] for row in rows]).astype(np.float32),
        social=np.stack([row["social"] for row in rows]).astype(np.float32),
        time=np.stack([row["time"] for row in rows]).astype(np.float32),
        target=np.stack([row["target"] for row in rows]).astype(np.float32),
        target_mask=np.stack([row["target_mask"] for row in rows]).astype(np.float32),
        tasks=np.asarray([row["task"] for row in rows], dtype="U32"),
        episode_ids=np.asarray([row["episode_id"] for row in rows], dtype="U64"),
        frame_indices=np.asarray([row["frame_index"] for row in rows], dtype=np.int32),
        agent_slots=np.asarray([row["agent_slot"] for row in rows], dtype=np.int16),
    )


def load_probe_data(
    manifest_path: Path, allowed_splits: set[str]
) -> tuple[ProbeData, dict[str, Any]]:
    manifest = read_json(manifest_path)
    if "read_only_test" in allowed_splits:
        raise RuntimeError("sealed test requires the dedicated test command")
    rows: list[dict[str, Any]] = []
    used_episodes: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        if str(episode["split"]) not in allowed_splits:
            continue
        used_episodes.append(episode)
        labels = read_sidecar(Path(str(episode["sidecar_path"])))
        episode_id = str(episode["hdf5_sha256"])
        task = str(episode["task"])
        with h5py.File(str(episode["hdf5_path"]), "r") as stream:
            actions = np.asarray(stream["data/action/commanded"], dtype=np.float32)
            images = stream["data/observation/images"]
            global_rgb = np.asarray(images["global"], dtype=np.uint8)
            agent_count = int(stream.attrs["agent_count"])
            positions = uniform_indices(16, actions.shape[0] - 16, 64)
            for frame_index in positions:
                label = labels[frame_index]["oracle_label"]
                if int(label["ambiguity_code"]) != 0 or not all(
                    bool(value) for value in label["label_validity_mask"].values()
                ):
                    continue
                history = list(range(frame_index - 15, frame_index + 1))
                global_current = grid_mean(global_rgb[frame_index], 6, 8)
                global_history = np.concatenate(
                    [grid_mean(global_rgb[index], 3, 4) for index in history[:-1]]
                )
                for agent_slot in range(agent_count):
                    local_rgb = np.asarray(images[f"agent_{agent_slot}"], dtype=np.uint8)
                    local_current = grid_mean(local_rgb[frame_index], 6, 8)
                    local_history = np.concatenate(
                        [grid_mean(local_rgb[index], 3, 4) for index in history[:-1]]
                    )
                    qpos = np.asarray(
                        stream[f"data/observation/agents/panda_{agent_slot}"],
                        dtype=np.float32,
                    )[history]
                    action_history = np.zeros((16, 8), dtype=np.float32)
                    action_history[:] = actions[
                        frame_index - 16 : frame_index,
                        agent_slot * 8 : (agent_slot + 1) * 8,
                    ]
                    legal = np.concatenate(
                        (
                            global_current,
                            local_current,
                            global_history,
                            local_history,
                            qpos.reshape(-1),
                            action_history.reshape(-1),
                            task_text_feature(task),
                        )
                    ).astype(np.float32)
                    e0_qpos = np.zeros_like(qpos)
                    e0_qpos[-1] = qpos[-1]
                    e0 = np.concatenate(
                        (
                            global_current,
                            local_current,
                            np.zeros_like(global_history),
                            np.zeros_like(local_history),
                            e0_qpos.reshape(-1),
                            np.zeros_like(action_history).reshape(-1),
                            task_text_feature(task),
                        )
                    ).astype(np.float32)
                    target = np.zeros((100, 8), dtype=np.float32)
                    target_mask = np.zeros(100, dtype=np.float32)
                    available = min(100, actions.shape[0] - frame_index)
                    target[:available] = actions[
                        frame_index : frame_index + available,
                        agent_slot * 8 : (agent_slot + 1) * 8,
                    ]
                    target_mask[:available] = 1.0
                    ratio = min(
                        1.0, frame_index / float(PUBLIC_TIME_SCALE[task])
                    )
                    time_vector = np.zeros(192, dtype=np.float32)
                    time_vector[0] = ratio
                    time_vector[1 + min(3, int(ratio * 4.0))] = 1.0
                    rows.append(
                        {
                            "legal": legal,
                            "e0": e0,
                            "social": social_features(label, agent_slot),
                            "time": time_vector,
                            "target": target,
                            "target_mask": target_mask,
                            "task": task,
                            "episode_id": episode_id,
                            "frame_index": frame_index,
                            "agent_slot": agent_slot,
                        }
                    )
    data = stack_probe_rows(rows)
    audit = {
        "allowed_splits": sorted(allowed_splits),
        "episode_count": len(used_episodes),
        "row_count": len(data),
        "per_task_episodes": {
            task: sum(1 for item in used_episodes if item["task"] == task)
            for task in TASKS
        },
        "per_task_rows": {
            task: int(np.sum(data.tasks == task)) for task in TASKS
        },
        "test_paths_opened": 0,
    }
    return data, audit


def load_sealed_test(manifest_path: Path) -> tuple[ProbeData, dict[str, Any]]:
    manifest = read_json(manifest_path)
    temporary = deepcopy(manifest)
    for item in temporary["episodes"]:
        if item["split"] == "read_only_test":
            item["split"] = "authorized_test"
    temporary_path = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.authorized-test.json"
    )
    write_json(temporary_path, temporary)
    try:
        data, audit = load_probe_data(temporary_path, {"authorized_test"})
    finally:
        temporary_path.unlink(missing_ok=True)
    audit["allowed_splits"] = ["read_only_test"]
    audit["test_paths_opened"] = int(audit["episode_count"])
    return data, audit


def normalizer(train: ProbeData) -> dict[str, Any]:
    action_mean: dict[str, list[float]] = {}
    action_std: dict[str, list[float]] = {}
    for task in TASKS:
        values: list[np.ndarray] = []
        for index in np.flatnonzero(train.tasks == task):
            valid = train.target_mask[index].astype(bool)
            values.append(train.target[index, valid])
        combined = np.concatenate(values, axis=0)
        action_mean[task] = combined.mean(0).astype(float).tolist()
        action_std[task] = np.maximum(combined.std(0), 1e-3).astype(float).tolist()
    social_mean = train.social.mean(0)
    social_std = np.maximum(train.social.std(0), 1e-3)
    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "social_mean": social_mean.astype(float).tolist(),
        "social_std": social_std.astype(float).tolist(),
    }


def normalize_social(data: ProbeData, norms: Mapping[str, Any]) -> None:
    mean = np.asarray(norms["social_mean"], dtype=np.float32)
    std = np.asarray(norms["social_std"], dtype=np.float32)
    data.social = (data.social - mean) / std


def normalized_targets(data: ProbeData, norms: Mapping[str, Any]) -> np.ndarray:
    result = data.target.copy()
    for task in TASKS:
        indices = np.flatnonzero(data.tasks == task)
        mean = np.asarray(norms["action_mean"][task], dtype=np.float32)
        std = np.asarray(norms["action_std"][task], dtype=np.float32)
        result[indices] = (result[indices] - mean) / std
    result *= data.target_mask[:, :, None]
    return result


def episode_permutation(data: ProbeData, seed: int) -> np.ndarray:
    result = np.empty_like(data.social)
    rng = np.random.default_rng(seed)
    for task in TASKS:
        episode_ids = sorted(set(data.episode_ids[data.tasks == task].tolist()))
        sources = episode_ids.copy()
        rng.shuffle(sources)
        if len(sources) > 1 and sources == episode_ids:
            sources = sources[1:] + sources[:1]
        for target_id, source_id in zip(episode_ids, sources, strict=True):
            target_indices = np.flatnonzero(data.episode_ids == target_id)
            source_indices = np.flatnonzero(data.episode_ids == source_id)
            target_order = target_indices[
                np.lexsort(
                    (data.agent_slots[target_indices], data.frame_indices[target_indices])
                )
            ]
            source_order = source_indices[
                np.lexsort(
                    (data.agent_slots[source_indices], data.frame_indices[source_indices])
                )
            ]
            mapped = np.rint(
                np.linspace(0, len(source_order) - 1, len(target_order))
            ).astype(np.int64)
            result[target_order] = data.social[source_order[mapped]]
    return result


def source_mask(name: str) -> np.ndarray:
    result = np.zeros(192, dtype=np.float32)
    for source in ("P", "T", "B"):
        if source in name.replace("_hat", ""):
            result[SOURCE_SLICES[source]] = 1.0
    return result


def condition_input(
    data: ProbeData,
    condition: str,
    noise: np.ndarray,
    shuffled: np.ndarray,
    predicted_social: np.ndarray | None,
) -> np.ndarray:
    if condition == "E0":
        legal = data.e0
        social = np.zeros((len(data), 192), dtype=np.float32)
    elif condition == "HC":
        legal = data.legal
        social = np.broadcast_to(noise, (len(data), 192)).copy()
    elif condition == "label_shuffled":
        legal = data.legal
        social = shuffled
    elif condition == "time_phase_only":
        legal = data.legal
        social = data.time
    elif condition.endswith("_hat"):
        if predicted_social is None:
            raise RuntimeError("deployable condition lacks predicted social features")
        legal = data.legal
        social = predicted_social * source_mask(condition)[None]
    else:
        legal = data.legal
        social = data.social * source_mask(condition)[None]
    return np.concatenate((legal, social), axis=1).astype(np.float32)


def torch_setup(seed: int) -> Any:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def build_action_model(input_width: int, hidden_width: int, seed: int) -> Any:
    torch = torch_setup(seed)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_width),
        torch.nn.Linear(input_width, hidden_width),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_width, hidden_width),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_width, 800),
    )


def build_social_predictor(input_width: int, seed: int) -> Any:
    torch = torch_setup(seed)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_width),
        torch.nn.Linear(input_width, 256),
        torch.nn.SiLU(),
        torch.nn.Linear(256, 192),
    )


def batches(count: int, batch_size: int, seed: int, epoch: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(count)
    for first in range(0, count, batch_size):
        yield order[first : first + batch_size]


def action_loss(model: Any, x: Any, y: Any, mask: Any) -> Any:
    prediction = model(x).reshape(-1, 100, 8)
    squared = (prediction - y).square() * mask[:, :, None]
    return squared.sum() / (mask.sum().clamp_min(1.0) * 8.0)


def train_action_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    tune_x: np.ndarray,
    tune_data: ProbeData,
    tune_targets: np.ndarray,
    device: str,
    learning_rate: float,
    hidden_width: int,
    initialization_seed: int,
    sampler_seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[Any, dict[str, Any]]:
    torch = torch_setup(initialization_seed)
    model = build_action_model(train_x.shape[1], hidden_width, initialization_seed).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    x_train = torch.from_numpy(train_x)
    y_train = torch.from_numpy(train_y)
    mask_train = torch.from_numpy(train_mask)
    x_tune = torch.from_numpy(tune_x).to(device)
    y_tune = torch.from_numpy(tune_targets).to(device)
    mask_tune = torch.from_numpy(tune_data.target_mask).to(device)
    best_primary = float("inf")
    best_full_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        losses: list[float] = []
        for indices in batches(len(train_x), 512, sampler_seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = action_loss(
                model,
                x_train[indices].to(device),
                y_train[indices].to(device),
                mask_train[indices].to(device),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            tune_loss = float(
                action_loss(model, x_tune, y_tune, mask_tune).detach().cpu()
            )
        tune_primary = float(
            evaluate_model(model, tune_x, tune_data, tune_targets, device)[
                "task_macro_primary_16_nrmse"
            ]
        )
        train_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_100_step": train_loss,
                "tune_loss_100_step": tune_loss,
                "tune_task_macro_primary_16_nrmse": tune_primary,
            }
        )
        if tune_primary < best_primary - 1e-7:
            best_primary = tune_primary
            best_full_loss = tune_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("action probe failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_tune_task_macro_primary_16_nrmse": best_primary,
        "diagnostic_100_step_loss_at_best_epoch": best_full_loss,
        "selection_metric": "tune_task_macro_primary_16_nrmse",
        "epochs_run": len(history),
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def train_social_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray | None,
    validation_y: np.ndarray | None,
    device: str,
    initialization_seed: int,
    sampler_seed: int,
    fixed_epochs: int | None = None,
    max_epochs: int = 260,
    patience: int = 25,
) -> tuple[Any, dict[str, Any]]:
    torch = torch_setup(initialization_seed)
    model = build_social_predictor(train_x.shape[1], initialization_seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = torch.from_numpy(train_x)
    y_train = torch.from_numpy(train_y)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    stale = 0
    maximum = fixed_epochs if fixed_epochs is not None else max_epochs
    history: list[dict[str, float]] = []
    for epoch in range(maximum):
        model.train()
        losses: list[float] = []
        for indices in batches(len(train_x), 512, sampler_seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train[indices].to(device))
            loss = (prediction - y_train[indices].to(device)).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if validation_x is None:
            validation_loss = float(np.mean(losses))
        else:
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    (
                        model(torch.from_numpy(validation_x).to(device))
                        - torch.from_numpy(validation_y).to(device)
                    )
                    .square()
                    .mean()
                    .detach()
                    .cpu()
                )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("social predictor failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "max_epochs": maximum,
        "patience": None if fixed_epochs is not None else patience,
    }


def predict_numpy(model: Any, values: np.ndarray, device: str, width: int) -> np.ndarray:
    import torch

    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(values), 2048):
            prediction = model(torch.from_numpy(values[first : first + 2048]).to(device))
            output.append(prediction.detach().cpu().numpy().reshape(-1, width))
    return np.concatenate(output, axis=0).astype(np.float32)


def nested_social_masks(
    train: ProbeData, outer_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return episode-level outer-held, inner-fit, and inner-validation masks."""
    fold_for_episode: dict[str, int] = {}
    for task in TASKS:
        episode_ids = sorted(set(train.episode_ids[train.tasks == task].tolist()))
        for index, episode_id in enumerate(episode_ids):
            fold_for_episode[episode_id] = index % 3
    outer_held = np.asarray(
        [fold_for_episode[value] == outer_fold for value in train.episode_ids],
        dtype=bool,
    )
    inner_validation_ids: set[str] = set()
    for task in TASKS:
        available = sorted(
            episode_id
            for episode_id in set(train.episode_ids[train.tasks == task].tolist())
            if fold_for_episode[episode_id] != outer_fold
        )
        inner_validation_ids.update(
            episode_id
            for index, episode_id in enumerate(available)
            if index % 4 == outer_fold % 4
        )
    inner_validation = np.asarray(
        [value in inner_validation_ids for value in train.episode_ids], dtype=bool
    )
    inner_fit = ~(outer_held | inner_validation)
    return outer_held, inner_fit, inner_validation


def crossfit_social(
    train: ProbeData,
    tune: ProbeData,
    device: str,
    initialization_seed: int,
    sampler_seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
    train_hat = np.zeros_like(train.social)
    fold_receipts: list[dict[str, Any]] = []
    for fold in range(3):
        held, inner_fit, inner_validation = nested_social_masks(train, fold)
        selector, selector_receipt = train_social_model(
            train.legal[inner_fit],
            train.social[inner_fit],
            train.legal[inner_validation],
            train.social[inner_validation],
            device,
            initialization_seed,
            sampler_seed + fold * 1000,
            max_epochs=max_epochs,
            patience=patience,
        )
        selected_epochs = int(selector_receipt["best_epoch"]) + 1
        model, refit_receipt = train_social_model(
            train.legal[~held],
            train.social[~held],
            None,
            None,
            device,
            initialization_seed,
            sampler_seed + fold * 1000,
            fixed_epochs=selected_epochs,
            max_epochs=max_epochs,
            patience=patience,
        )
        train_hat[held] = predict_numpy(model, train.legal[held], device, 192)
        fold_receipts.append(
            {
                "fold": fold,
                "selector": selector_receipt,
                "selected_refit_epochs": selected_epochs,
                "refit": refit_receipt,
                "inner_fit_episode_count": len(
                    set(train.episode_ids[inner_fit].tolist())
                ),
                "inner_validation_episode_count": len(
                    set(train.episode_ids[inner_validation].tolist())
                ),
                "heldout_episode_count": len(set(train.episode_ids[held].tolist())),
                "heldout_row_count": int(held.sum()),
            }
        )
    final_model, final_receipt = train_social_model(
        train.legal,
        train.social,
        tune.legal,
        tune.social,
        device,
        initialization_seed,
        sampler_seed,
        max_epochs=max_epochs,
        patience=patience,
    )
    tune_hat = predict_numpy(final_model, tune.legal, device, 192)
    return train_hat, tune_hat, final_model, {
        "folds": fold_receipts,
        "final": final_receipt,
        "train_oof_mse": float(np.mean((train_hat - train.social) ** 2)),
        "tune_mse": float(np.mean((tune_hat - tune.social) ** 2)),
        "tune_source_mse": {
            source: float(
                np.mean(
                    (
                        tune_hat[:, SOURCE_SLICES[source]]
                        - tune.social[:, SOURCE_SLICES[source]]
                    )
                    ** 2
                )
            )
            for source in ("P", "T", "B")
        },
    }


def episode_errors(
    prediction: np.ndarray,
    data: ProbeData,
    target: np.ndarray,
) -> dict[str, dict[str, Any]]:
    prediction = prediction.reshape(-1, 100, 8)
    result: dict[str, dict[str, Any]] = {}
    for episode_id in sorted(set(data.episode_ids.tolist())):
        indices = np.flatnonzero(data.episode_ids == episode_id)
        task_values = set(data.tasks[indices].tolist())
        if len(task_values) != 1:
            raise RuntimeError("episode spans more than one task")
        mask = data.target_mask[indices]
        squared = (prediction[indices] - target[indices]) ** 2
        primary_mask = mask[:, :16, None]
        primary = math.sqrt(
            float((squared[:, :16] * primary_mask).sum())
            / float(max(1.0, primary_mask.sum() * 8.0))
        )
        full_mask = mask[:, :, None]
        full = math.sqrt(
            float((squared * full_mask).sum())
            / float(max(1.0, full_mask.sum() * 8.0))
        )
        grip_squared = squared[:, :16, 7]
        grip_mask = mask[:, :16]
        grip = math.sqrt(
            float((grip_squared * grip_mask).sum())
            / float(max(1.0, grip_mask.sum()))
        )
        result[episode_id] = {
            "task": next(iter(task_values)),
            "primary_16_nrmse": primary,
            "diagnostic_100_masked_nrmse": full,
            "gripper_16_rmse": grip,
            "rows": int(len(indices)),
        }
    return result


def evaluate_model(
    model: Any,
    values: np.ndarray,
    data: ProbeData,
    targets: np.ndarray,
    device: str,
) -> dict[str, Any]:
    prediction = predict_numpy(model, values, device, 800)
    episodes = episode_errors(prediction, data, targets)
    per_task = {
        task: float(
            np.mean(
                [
                    item["primary_16_nrmse"]
                    for item in episodes.values()
                    if item["task"] == task
                ]
            )
        )
        for task in TASKS
    }
    return {
        "episode_errors": episodes,
        "task_macro_primary_16_nrmse": float(np.mean(list(per_task.values()))),
        "per_task_primary_16_nrmse": per_task,
        "mean_diagnostic_100_masked_nrmse": float(
            np.mean(
                [item["diagnostic_100_masked_nrmse"] for item in episodes.values()]
            )
        ),
        "mean_gripper_16_rmse": float(
            np.mean([item["gripper_16_rmse"] for item in episodes.values()])
        ),
    }


def save_torch_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def load_torch_checkpoint(path: Path, device: str) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location=device, weights_only=False)


def gain_rows(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for episode_id, base in baseline["episode_errors"].items():
        other = candidate["episode_errors"][episode_id]
        value = (base["primary_16_nrmse"] - other["primary_16_nrmse"]) / max(
            base["primary_16_nrmse"], 1e-12
        )
        result[episode_id] = {"task": base["task"], "gain": float(value)}
    return result


def macro_gain(rows: Mapping[str, Mapping[str, Any]]) -> float:
    return float(
        np.mean(
            [
                np.mean([item["gain"] for item in rows.values() if item["task"] == task])
                for task in TASKS
            ]
        )
    )


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def estimated_power(
    rows: Mapping[str, Mapping[str, Any]], effect: float, future_n_per_task: int = 12
) -> dict[str, float]:
    variance_sum = 0.0
    task_sd: dict[str, float] = {}
    for task in TASKS:
        values = np.asarray(
            [item["gain"] for item in rows.values() if item["task"] == task],
            dtype=np.float64,
        )
        sd = float(values.std(ddof=1)) if values.size > 1 else float("inf")
        task_sd[task] = sd
        variance_sum += (sd * sd) / future_n_per_task
    standard_error = math.sqrt(variance_sum / (len(TASKS) ** 2))
    power = (
        normal_cdf(effect / standard_error - 1.6448536269514722)
        if math.isfinite(standard_error) and standard_error > 0
        else 0.0
    )
    return {
        "registered_effect": effect,
        "estimated_standard_error": standard_error,
        "estimated_power": power,
        "task_sd": task_sd,
    }


def pilot_check(args: argparse.Namespace) -> None:
    if args.data_root is None:
        raise ValueError("pilot-check requires --data-root")
    manifest = args.data_root / "dataset_manifest.json"
    data, audit = load_probe_data(manifest, {"pilot"})
    checks = {
        "six_tasks": set(data.tasks.tolist()) == set(TASKS),
        "four_episodes_per_task": all(
            len(set(data.episode_ids[data.tasks == task].tolist())) == 4 for task in TASKS
        ),
        "legal_width_fixed": data.legal.shape[1] == data.e0.shape[1],
        "social_width_192": data.social.shape[1] == 192,
        "target_shape_100x8": data.target.shape[1:] == (100, 8),
        "primary_horizon_available": bool(np.all(data.target_mask[:, :16] == 1.0)),
        "finite": all(
            np.isfinite(value).all()
            for value in (data.legal, data.e0, data.social, data.target)
        ),
        "test_paths_opened_zero": audit["test_paths_opened"] == 0,
    }
    receipt = {
        "decision_code": (
            "SSC_V7_M3_PILOT_PASSED" if all(checks.values()) else "FAILED_SCHEMA"
        ),
        "checks": checks,
        "audit": audit,
        "formal_training_authorized": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(args.output_root / "pilot_check_receipt.json", receipt)
    print(receipt["decision_code"])
    if not all(checks.values()):
        raise RuntimeError("M3 pilot check failed")


def tune(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.data_root is None:
        raise ValueError("tune requires --data-root")
    if args.output_root.exists():
        raise FileExistsError(f"fresh output root required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    manifest_path = args.data_root / "dataset_manifest.json"
    train, train_audit = load_probe_data(manifest_path, {"train"})
    tune_data, tune_audit = load_probe_data(manifest_path, {"tune"})
    norms = normalizer(train)
    normalize_social(train, norms)
    normalize_social(tune_data, norms)
    target_train = normalized_targets(train, norms)
    target_tune = normalized_targets(tune_data, norms)
    seed_contract = read_json(args.seed_contract)
    seeds = m2.expanded_seed_manifest(seed_contract, args.w10_seed_root)
    init_seed = int(seeds["common_training"]["model_initialization"])
    sampler_seed = int(seeds["common_training"]["dataset_sampler"])
    statistics_seed = int(seeds["common_training"]["measurement_statistics"])
    noise = np.random.default_rng(sampler_seed).normal(0, 1, 192).astype(np.float32)
    train_shuffled = episode_permutation(train, sampler_seed)
    tune_shuffled = episode_permutation(tune_data, sampler_seed)

    social_config = gate["social_predictor"]
    action_config = gate["action_probe"]
    train_hat, tune_hat, social_model, social_receipt = crossfit_social(
        train,
        tune_data,
        args.device,
        init_seed,
        sampler_seed,
        max_epochs=int(social_config["max_epochs"]),
        patience=int(social_config["early_stopping_patience"]),
    )
    social_path = args.output_root / "checkpoints/social_predictor.pt"
    save_torch_checkpoint(
        social_path,
        {
            "state_dict": social_model.state_dict(),
            "input_width": int(train.legal.shape[1]),
            "output_width": 192,
            "stage_id": STAGE_ID,
        },
    )

    grid_receipts: list[dict[str, Any]] = []
    for candidate in gate["action_probe"]["hyperparameter_grid"]:
        train_x = condition_input(
            train, "HC", noise, train_shuffled, train_hat
        )
        tune_x = condition_input(
            tune_data, "HC", noise, tune_shuffled, tune_hat
        )
        model, receipt = train_action_model(
            train_x,
            target_train,
            train.target_mask,
            tune_x,
            tune_data,
            target_tune,
            args.device,
            float(candidate["learning_rate"]),
            int(candidate["hidden_width"]),
            init_seed,
            sampler_seed,
            max_epochs=int(action_config["max_epochs"]),
            patience=int(action_config["early_stopping_patience"]),
        )
        metric = evaluate_model(model, tune_x, tune_data, target_tune, args.device)
        grid_receipts.append(
            {
                "candidate": candidate,
                "training": receipt,
                "hc_tune_task_macro_primary_16_nrmse": metric[
                    "task_macro_primary_16_nrmse"
                ],
            }
        )
    selected = min(
        grid_receipts, key=lambda item: item["hc_tune_task_macro_primary_16_nrmse"]
    )["candidate"]

    condition_receipts: dict[str, Any] = {}
    tune_metrics: dict[str, Any] = {}
    parameter_counts: set[int] = set()
    active_conditions = tuple(str(value) for value in gate["conditions"])
    if active_conditions != CONDITIONS:
        raise RuntimeError("gate conditions do not match the M3-R3 implementation")
    for condition in active_conditions:
        print(f"[M3 tune] training {condition}", flush=True)
        train_x = condition_input(
            train, condition, noise, train_shuffled, train_hat
        )
        tune_x = condition_input(
            tune_data, condition, noise, tune_shuffled, tune_hat
        )
        model, receipt = train_action_model(
            train_x,
            target_train,
            train.target_mask,
            tune_x,
            tune_data,
            target_tune,
            args.device,
            float(selected["learning_rate"]),
            int(selected["hidden_width"]),
            init_seed,
            sampler_seed,
            max_epochs=int(action_config["max_epochs"]),
            patience=int(action_config["early_stopping_patience"]),
        )
        checkpoint = args.output_root / f"checkpoints/action_{condition}.pt"
        save_torch_checkpoint(
            checkpoint,
            {
                "state_dict": model.state_dict(),
                "input_width": int(train_x.shape[1]),
                "hidden_width": int(selected["hidden_width"]),
                "condition": condition,
                "stage_id": STAGE_ID,
            },
        )
        metric = evaluate_model(model, tune_x, tune_data, target_tune, args.device)
        receipt["checkpoint"] = str(checkpoint)
        receipt["checkpoint_sha256"] = sha256_file(checkpoint)
        condition_receipts[condition] = receipt
        tune_metrics[condition] = metric
        parameter_counts.add(int(receipt["parameter_count"]))
    if len(parameter_counts) != 1:
        raise RuntimeError("M3 action conditions do not have equal parameter counts")

    baseline = tune_metrics["HC"]
    target_source = str(gate["target_source"])
    if target_source != "B":
        raise RuntimeError("M3-R3 is frozen as a B-only convergence repair")
    oracle_rows = gain_rows(baseline, tune_metrics[target_source])
    power = estimated_power(
        oracle_rows, float(gate["power_audit"]["registered_relative_effect"])
    )
    power["observed_tune_macro_gain"] = macro_gain(oracle_rows)
    power["powered"] = bool(
        power["estimated_power"] >= float(gate["power_audit"]["target_power"])
    )
    oracle_tune = summarize_gain(
        baseline, tune_metrics[target_source], statistics_seed + 1
    )
    deployable_tune = summarize_gain(
        baseline, tune_metrics[f"{target_source}_hat"], statistics_seed + 2
    )
    deployable_vs_shuffled_tune = summarize_gain(
        tune_metrics["label_shuffled"],
        tune_metrics[f"{target_source}_hat"],
        statistics_seed + 3,
    )
    screen_gate = gate["tune_screen"]
    tune_screen_checks = {
        "oracle_powered": bool(power["powered"]),
        "oracle_gain": oracle_tune["macro_gain"]
        >= float(screen_gate["oracle_gain_min"]),
        "oracle_positive_tasks": len(oracle_tune["positive_tasks"])
        >= int(screen_gate["oracle_positive_tasks_min"]),
        "deployable_gain": deployable_tune["macro_gain"]
        > float(screen_gate["deployable_gain_gt"]),
        "deployable_positive_tasks": len(deployable_tune["positive_tasks"])
        >= int(screen_gate["deployable_positive_tasks_min"]),
        "deployable_beats_shuffled": deployable_vs_shuffled_tune["macro_gain"]
        > float(screen_gate["deployable_vs_shuffled_gain_gt"]),
    }
    test_authorized = all(tune_screen_checks.values())
    powered_sources = [target_source] if power["powered"] else []

    normalization_path = args.output_root / "normalization.json"
    write_json(normalization_path, norms)
    metrics_path = args.output_root / "tune_metrics.json"
    write_json(metrics_path, tune_metrics)
    receipt = {
        "format_version": "ssc-v7.m3.tuning_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "train_audit": train_audit,
        "tune_audit": tune_audit,
        "test_paths_opened": 0,
        "grid": grid_receipts,
        "selected_hyperparameters": selected,
        "social_predictor": social_receipt,
        "social_predictor_checkpoint": str(social_path),
        "social_predictor_checkpoint_sha256": sha256_file(social_path),
        "action_conditions": condition_receipts,
        "equal_action_parameter_count": next(iter(parameter_counts)),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "tune_metrics": str(metrics_path),
        "tune_metrics_sha256": sha256_file(metrics_path),
        "power_audit": {target_source: power},
        "tune_screen": {
            "checks": tune_screen_checks,
            "oracle": oracle_tune,
            "deployable": deployable_tune,
            "deployable_vs_label_shuffled": deployable_vs_shuffled_tune,
            "passed": test_authorized,
        },
        "powered_sources": powered_sources,
        "test_authorized": test_authorized,
        "decision_code": (
            "SSC_V7_M3_REPAIR_TEST_COLLECTION_AUTHORIZED"
            if test_authorized
            else "INCONCLUSIVE_MEASUREMENT/INSUFFICIENT_POWER"
        ),
        "formal_training_authorized": False,
    }
    receipt_path = args.output_root / "tuning_receipt.json"
    write_json(receipt_path, receipt)
    frozen_files = [args.gate, manifest_path, normalization_path, metrics_path, social_path]
    frozen_files.extend(
        Path(value["checkpoint"]) for value in condition_receipts.values()
    )
    configuration = {
        "format_version": "ssc-v7.m3.repair_configuration_manifest/1",
        "stage_id": STAGE_ID,
        "created_at_utc": utc_now(),
        "tuning_receipt": str(receipt_path),
        "tuning_receipt_sha256": sha256_file(receipt_path),
        "frozen_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in frozen_files
        ],
        "powered_sources": powered_sources,
        "test_authorized": test_authorized,
        "test_paths_opened_before_configuration_hash": 0,
        "next_action": "collect and seal a new repair test" if test_authorized else "stop",
    }
    write_json(args.output_root / "repair_configuration_manifest.json", configuration)
    print(receipt["decision_code"])


def bootstrap_gain(
    rows: Mapping[str, Mapping[str, Any]], seed: int, resamples: int = 10000
) -> tuple[float, float, float, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    observed = macro_gain(rows)
    draws = np.empty(resamples, dtype=np.float64)
    task_draws: dict[str, np.ndarray] = {}
    for task in TASKS:
        values = np.asarray(
            [item["gain"] for item in rows.values() if item["task"] == task],
            dtype=np.float64,
        )
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        task_draws[task] = values[indices].mean(1)
    draws[:] = np.mean(np.stack([task_draws[task] for task in TASKS]), axis=0)
    task_summary = {
        task: {
            "gain": float(
                np.mean([item["gain"] for item in rows.values() if item["task"] == task])
            ),
            "ci95": [
                float(np.quantile(task_draws[task], 0.025)),
                float(np.quantile(task_draws[task], 0.975)),
            ],
            "episodes": int(len(task_draws[task][0:1]) * sum(1 for item in rows.values() if item["task"] == task)),
        }
        for task in TASKS
    }
    return (
        observed,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        task_summary,
    )


def sign_flip_p(
    rows: Mapping[str, Mapping[str, Any]], seed: int, resamples: int = 10000
) -> float:
    values_by_task = {
        task: np.asarray(
            [item["gain"] for item in rows.values() if item["task"] == task],
            dtype=np.float64,
        )
        for task in TASKS
    }
    observed = macro_gain(rows)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(resamples):
        value = float(
            np.mean(
                [
                    np.mean(
                        values
                        * rng.choice(np.asarray([-1.0, 1.0]), size=values.shape[0])
                    )
                    for values in values_by_task.values()
                ]
            )
        )
        extreme += int(value >= observed)
    return (extreme + 1.0) / (resamples + 1.0)


def holm(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * float(values[name]))
        adjusted[name] = min(1.0, running)
    return adjusted


def summarize_gain(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    rows = gain_rows(baseline, candidate)
    gain, lower, upper, tasks = bootstrap_gain(rows, seed)
    return {
        "macro_gain": gain,
        "ci95": [lower, upper],
        "per_task": tasks,
        "positive_tasks": [task for task, value in tasks.items() if value["gain"] > 0],
        "episode_gains": rows,
    }


def verify_repair_configuration(path: Path) -> dict[str, Any]:
    configuration = read_json(path)
    for item in configuration["frozen_files"]:
        file_path = Path(str(item["path"]))
        if sha256_file(file_path) != str(item["sha256"]):
            raise RuntimeError(f"frozen M3-R3 configuration file changed: {file_path}")
    receipt = Path(str(configuration["tuning_receipt"]))
    if sha256_file(receipt) != str(configuration["tuning_receipt_sha256"]):
        raise RuntimeError("M3-R3 tuning receipt changed after screening")
    if not bool(configuration["test_authorized"]):
        raise RuntimeError("M3-R3 tune screen did not authorize a new test")
    return configuration


def seal_test(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.data_root is None:
        raise ValueError("seal requires --data-root pointing to the repair dataset")
    tuning_root = args.output_root
    configuration_path = tuning_root / "repair_configuration_manifest.json"
    configuration = verify_repair_configuration(configuration_path)
    dataset_manifest = args.data_root / "dataset_manifest.json"
    dataset = read_json(dataset_manifest)
    if dataset.get("stage_id") != STAGE_ID or dataset.get("purpose") != "repair_test":
        raise RuntimeError("wrong M3-R3 repair-test dataset identity")
    if not bool(dataset.get("test_is_sealed")) or bool(dataset.get("test_has_been_read")):
        raise RuntimeError("M3-R3 repair test is not sealed")
    expected_counts = {"train": 36, "tune": 12, "read_only_test": 12}
    for task in TASKS:
        if dataset["split_counts"].get(task) != expected_counts:
            raise RuntimeError(f"wrong M3-R3 split counts for {task}")
    unseal = {
        "format_version": "ssc-v7.m3.test_unseal_manifest/2",
        "stage_id": STAGE_ID,
        "created_at_utc": utc_now(),
        "repair_configuration": str(configuration_path),
        "repair_configuration_sha256": sha256_file(configuration_path),
        "tuning_receipt": str(configuration["tuning_receipt"]),
        "tuning_receipt_sha256": str(configuration["tuning_receipt_sha256"]),
        "test_dataset_manifest": str(dataset_manifest),
        "test_dataset_manifest_sha256": sha256_file(dataset_manifest),
        "frozen_files": [
            {"path": str(configuration_path), "sha256": sha256_file(configuration_path)},
            {"path": str(dataset_manifest), "sha256": sha256_file(dataset_manifest)},
        ],
        "powered_sources": list(configuration["powered_sources"]),
        "test_authorized": True,
        "test_paths_opened_before_manifest_hash": 0,
        "formal_training_authorized": False,
    }
    path = tuning_root / "test_unseal_manifest.json"
    if path.exists():
        raise FileExistsError(f"fresh unseal manifest required: {path}")
    write_json(path, unseal)
    print("SSC_V7_M3_REPAIR_TEST_UNSEAL_AUTHORIZED")


def verify_unseal_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for item in manifest["frozen_files"]:
        file_path = Path(str(item["path"]))
        if sha256_file(file_path) != str(item["sha256"]):
            raise RuntimeError(f"frozen M3 file changed before test: {file_path}")
    configuration_path = Path(str(manifest["repair_configuration"]))
    if sha256_file(configuration_path) != str(
        manifest["repair_configuration_sha256"]
    ):
        raise RuntimeError("M3-R3 repair configuration changed before test")
    verify_repair_configuration(configuration_path)
    if not bool(manifest["test_authorized"]):
        raise RuntimeError("M3 power audit did not authorize test unsealing")
    return manifest


def test(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.data_root is None:
        raise ValueError("test requires --data-root")
    if args.output_root.exists():
        raise FileExistsError(f"fresh output root required: {args.output_root}")
    tuning_root = args.data_root
    unseal_path = tuning_root / "test_unseal_manifest.json"
    unseal = verify_unseal_manifest(unseal_path)
    tuning = read_json(tuning_root / "tuning_receipt.json")
    dataset_manifest = Path(str(unseal["test_dataset_manifest"]))
    data, data_audit = load_sealed_test(dataset_manifest)
    norms = read_json(tuning_root / "normalization.json")
    normalize_social(data, norms)
    target = normalized_targets(data, norms)
    seed_contract = read_json(args.seed_contract)
    seeds = m2.expanded_seed_manifest(seed_contract, args.w10_seed_root)
    sampler_seed = int(seeds["common_training"]["dataset_sampler"])
    statistics_seed = int(seeds["common_training"]["measurement_statistics"])
    init_seed = int(seeds["common_training"]["model_initialization"])
    noise = np.random.default_rng(sampler_seed).normal(0, 1, 192).astype(np.float32)
    shuffled = episode_permutation(data, sampler_seed)

    social_checkpoint = load_torch_checkpoint(
        tuning_root / "checkpoints/social_predictor.pt", args.device
    )
    social_model = build_social_predictor(
        int(social_checkpoint["input_width"]), init_seed
    ).to(args.device)
    social_model.load_state_dict(social_checkpoint["state_dict"])
    social_hat = predict_numpy(social_model, data.legal, args.device, 192)

    metrics: dict[str, Any] = {}
    parameter_counts: set[int] = set()
    for condition in CONDITIONS:
        checkpoint = load_torch_checkpoint(
            tuning_root / f"checkpoints/action_{condition}.pt", args.device
        )
        model = build_action_model(
            int(checkpoint["input_width"]),
            int(checkpoint["hidden_width"]),
            init_seed,
        ).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
        values = condition_input(data, condition, noise, shuffled, social_hat)
        metrics[condition] = evaluate_model(model, values, data, target, args.device)
    if len(parameter_counts) != 1:
        raise RuntimeError("test checkpoints have unequal action-model parameter counts")

    baseline = metrics["HC"]
    effects = {
        condition: summarize_gain(baseline, metrics[condition], statistics_seed + index)
        for index, condition in enumerate(CONDITIONS)
        if condition != "HC"
    }
    source = "B"
    source_p = sign_flip_p(
        gain_rows(baseline, metrics[source]), statistics_seed + 100
    )
    powered_sources = set(str(value) for value in unseal["powered_sources"])
    source_gate = gate["acceptance"]["per_source"]
    oracle = effects[source]
    deployable = effects[f"{source}_hat"]
    retention = (
        deployable["macro_gain"] / oracle["macro_gain"]
        if oracle["macro_gain"] > 0
        else float("-inf")
    )
    oracle_harms = [
        task
        for task, value in oracle["per_task"].items()
        if value["gain"] <= float(source_gate["stable_harm_threshold"])
        and value["ci95"][1] < 0.0
    ]
    deployable_harms = [
        task
        for task, value in deployable["per_task"].items()
        if value["gain"] <= float(source_gate["stable_harm_threshold"])
        and value["ci95"][1] < 0.0
    ]
    oracle_vs_shuffled = summarize_gain(
        metrics["label_shuffled"], metrics[source], statistics_seed + 200
    )
    deployable_vs_shuffled = summarize_gain(
        metrics["label_shuffled"], metrics[f"{source}_hat"], statistics_seed + 201
    )
    checks = {
        "powered_before_test": source in powered_sources,
        "oracle_gain": oracle["macro_gain"] >= float(source_gate["oracle_gain_min"]),
        "oracle_ci": oracle["ci95"][0] > 0.0,
        "source_p": source_p <= float(source_gate["source_p_lte"]),
        "oracle_positive_tasks": len(oracle["positive_tasks"])
        >= int(source_gate["positive_tasks_min"]),
        "no_oracle_stable_harm": not oracle_harms,
        "deployable_retention": retention
        >= float(source_gate["deployable_retention_min"]),
        "deployable_ci": deployable["ci95"][0] > 0.0,
        "deployable_positive_tasks": len(deployable["positive_tasks"])
        >= int(source_gate["positive_tasks_min"]),
        "no_deployable_stable_harm": not deployable_harms,
        "oracle_beats_shuffled_ci": oracle_vs_shuffled["ci95"][0]
        > float(source_gate["shortcut_ci_lower_gt"]),
        "deployable_beats_shuffled_ci": deployable_vs_shuffled["ci95"][0]
        > float(source_gate["shortcut_ci_lower_gt"]),
    }
    if source not in powered_sources:
        source_decision = "INCONCLUSIVE_MEASUREMENT/INSUFFICIENT_POWER"
    else:
        source_decision = (
            "PASSED_M3_SOURCE" if all(checks.values()) else "FAILED_M3_SOURCE"
        )
    source_results = {
        source: {
            "decision": source_decision,
            "checks": checks,
            "oracle": oracle,
            "deployable": deployable,
            "deployable_retention": retention,
            "source_p": source_p,
            "oracle_stable_harm_tasks": oracle_harms,
            "deployable_stable_harm_tasks": deployable_harms,
            "oracle_vs_label_shuffled": oracle_vs_shuffled,
            "deployable_vs_label_shuffled": deployable_vs_shuffled,
        }
    }
    passed_sources = [source] if source_decision == "PASSED_M3_SOURCE" else []
    if passed_sources:
        decision = "PASSED_M3_SOCIAL_SIGNAL_GATE"
    elif source_decision == "INCONCLUSIVE_MEASUREMENT/INSUFFICIENT_POWER":
        decision = source_decision
    else:
        decision = "FAILED_MEASUREMENT/NO_DEPLOYABLE_B_HEADROOM"
    args.output_root.mkdir(parents=True)
    metrics_path = args.output_root / "test_metrics.json"
    write_json(metrics_path, metrics)
    result = {
        "format_version": "ssc-v7.m3.result/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "decision_code": decision,
        "plain_language_conclusion": "filled by the roadmap handoff after numerical adjudication",
        "data_audit": data_audit,
        "unseal_manifest": str(unseal_path),
        "unseal_manifest_sha256": sha256_file(unseal_path),
        "equal_action_parameter_count": next(iter(parameter_counts)),
        "social_predictor_test_mse": float(np.mean((social_hat - data.social) ** 2)),
        "social_predictor_test_source_mse": {
            source: float(
                np.mean(
                    (
                        social_hat[:, SOURCE_SLICES[source]]
                        - data.social[:, SOURCE_SLICES[source]]
                    )
                    ** 2
                )
            )
            for source in ("P", "T", "B")
        },
        "negative_controls": {
            "E0": effects["E0"],
            "label_shuffled": effects["label_shuffled"],
            "time_phase_only": effects["time_phase_only"],
        },
        "source_results": source_results,
        "repair_scope": "B only; P, T, combinations, M4, and M5 were not rerun",
        "passed_sources": passed_sources,
        "test_metrics": str(metrics_path),
        "test_metrics_sha256": sha256_file(metrics_path),
        "m4_authorized_sources": passed_sources,
        "m4_is_not_executed": True,
        "b_core_unlocked": False,
        "b_core_status": (
            "B may proceed to M4; B-core remains locked until M4 and M5 pass."
            if passed_sources
            else "B remains stopped at M3."
        ),
        "formal_training_authorized": False,
        "required_next_action": (
            "Proceed only to M4 for passed sources; do not train B0-H or social models."
            if passed_sources
            else "Stop the affected route(s); do not proceed to formal training."
        ),
    }
    write_json(args.output_root / "m3_result.json", result)
    print(decision)
    print(json.dumps({"passed_sources": passed_sources}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    gate = load_gate(args.gate)
    receipt = preflight(args, gate)
    if receipt["status"] != "PASSED":
        raise RuntimeError(f"M3 preflight failed: {receipt['checks']}")
    if args.command == "collect":
        collect_one_task(args, gate, receipt)
    elif args.command == "merge":
        merge_collections(args, gate)
    elif args.command == "pilot-check":
        pilot_check(args)
    elif args.command == "tune":
        tune(args, gate)
    elif args.command == "seal":
        seal_test(args, gate)
    elif args.command == "test":
        test(args, gate)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
