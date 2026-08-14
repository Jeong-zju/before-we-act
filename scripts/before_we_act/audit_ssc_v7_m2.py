#!/usr/bin/env python3
"""Collect and audit SSC-V7 M2 oracle-label sidecars in RoboFactory."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


ROBOFACTORY_COMMIT = "5868242322414a91454e22f1dd9641f613ba1bcf"
M1_DECISION = "PASSED_M1_BENCHMARK_RELAXED"
CONTACT_FORCE_MIN = 0.20
CONTACT_AMBIGUITY_MARGIN = 0.02
ROBOT_COLLISION_FORCE_MIN = 1.0
ROBOT_PROXIMITY_DISTANCE = 0.10

TASKS: dict[str, dict[str, Any]] = {
    "lift_barrier": {
        "env_id": "LiftBarrier-rf",
        "config": "lift_barrier.yaml",
        "agents": 2,
        "objects": ("barrier",),
        "role_agents": {"support_0": 0, "support_1": 1},
        "solver": "solveLiftBarrier",
    },
    "camera_alignment": {
        "env_id": "CameraAlignment-rf",
        "config": "camera_alignment.yaml",
        "agents": 3,
        "objects": ("camera", "meat"),
        "role_agents": {"camera_0": 0, "camera_1": 1, "meat": 2},
        "solver": "solveCameraAlignment",
    },
    "long_pipeline_delivery": {
        "env_id": "LongPipelineDelivery-rf",
        "config": "long_pipeline_delivery.yaml",
        "agents": 4,
        "objects": ("shoe",),
        "role_agents": {
            "pipeline_0": 3,
            "pipeline_1": 2,
            "pipeline_2": 1,
            "pipeline_3": 0,
        },
        "solver": "solveLongPipelineDelivery",
    },
    "take_photo": {
        "env_id": "TakePhoto-rf",
        "config": "take_photo.yaml",
        "agents": 4,
        "objects": ("camera", "meat"),
        "role_agents": {"camera_0": 0, "camera_1": 1, "meat": 2, "button": 3},
        "solver": "solveTakePhoto",
    },
    "pass_shoe": {
        "env_id": "PassShoe-rf",
        "config": "pass_shoe.yaml",
        "agents": 2,
        "objects": ("shoe",),
        "role_agents": {"source": 0, "receiver": 1},
        "solver": "solvePassShoe",
    },
    "place_food": {
        "env_id": "PlaceFood-rf",
        "config": "place_food.yaml",
        "agents": 2,
        "objects": ("meat", "pot"),
        "role_agents": {"meat": 0, "pot": 1},
        "solver": "solvePlaceFood",
    },
}

REQUIRED_FIELDS = (
    "stage_id",
    "within_stage_progress",
    "per_agent_contribution",
    "remaining_goal_mask",
    "agent_object_role_slots",
    "grasp_contact_custody_state",
    "collision_drop_contention_risk",
    "causal_automaton_state",
    "label_validity_mask",
    "ambiguity_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed-contract", type=Path, required=True)
    parser.add_argument("--oracle-spec", type=Path, required=True)
    parser.add_argument("--gate-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "execute", "finalize"), required=True)
    parser.add_argument("--manual-review", type=Path)
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def tensor_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_bool(value: Any) -> bool:
    array = tensor_array(value).reshape(-1)
    if array.size != 1:
        raise ValueError("expected one environment boolean")
    return bool(array[0])


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def derive_seed(namespace: str, purpose: str, task: str, index: int, retry: int) -> int:
    message = f"{namespace}|{purpose}|{task}|{index}|{retry}".encode("utf-8")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_646 + 1


def expanded_seed_manifest(contract: Mapping[str, Any], w10_root: Path) -> dict[str, Any]:
    historical: set[int] = set()
    for task in contract["tasks"]:
        payload = json.loads((w10_root / f"{task}.json").read_text(encoding="utf-8"))
        historical.update(int(seed) for seed in payload["seeds"])
    used = set(historical)
    expanded: dict[str, Any] = {"per_task": {}, "common_training": {}}
    for task in contract["tasks"]:
        task_payload: dict[str, list[int]] = {}
        purposes = list(contract["measurement_purposes_per_task"].items())
        purposes += list(contract["shared_candidate_evaluation_purposes_per_task"].items())
        for purpose, count in purposes:
            values: list[int] = []
            for index in range(int(count)):
                retry = 0
                while True:
                    value = derive_seed(
                        str(contract["namespace"]), purpose, task, index, retry
                    )
                    if value not in used:
                        break
                    retry += 1
                used.add(value)
                values.append(value)
            task_payload[purpose] = values
        expanded["per_task"][task] = task_payload
    for purpose in contract["common_training_purposes"]:
        retry = 0
        while True:
            value = derive_seed(
                str(contract["namespace"]), purpose, "common", 0, retry
            )
            if value not in used:
                break
            retry += 1
        used.add(value)
        expanded["common_training"][purpose] = value
    expanded["historical_w10_seed_count"] = len(historical)
    expanded["generated_seed_count"] = len(used) - len(historical)
    return expanded


def contact_targets(actor: Any) -> list[Any]:
    if hasattr(actor, "_bodies"):
        return [actor]
    if hasattr(actor, "get_links"):
        links = list(actor.get_links())
        if links:
            return links
    raise TypeError(f"unsupported M2 contact object type {type(actor)!r}")


def finger_contact_vectors(
    base_env: Any, agent: Any, actor: Any
) -> tuple[np.ndarray, np.ndarray]:
    scene = base_env.scene
    left = np.zeros((1, 3), dtype=np.float64)
    right = np.zeros((1, 3), dtype=np.float64)
    for target in contact_targets(actor):
        left += tensor_array(
            scene.get_pairwise_contact_forces(agent.finger1_link, target)
        )
        right += tensor_array(
            scene.get_pairwise_contact_forces(agent.finger2_link, target)
        )
    return left, right


def vector_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 180.0
    cosine = float(np.dot(first, second) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def grasp_from_contacts(agent: Any, left: np.ndarray, right: np.ndarray) -> bool:
    left_vector = np.asarray(left[0], dtype=np.float64)
    right_vector = np.asarray(right[0], dtype=np.float64)
    left_force = float(np.linalg.norm(left_vector))
    right_force = float(np.linalg.norm(right_vector))
    left_direction = tensor_array(agent.finger1_link.pose.to_transformation_matrix())[
        0, :3, 1
    ]
    right_direction = -tensor_array(
        agent.finger2_link.pose.to_transformation_matrix()
    )[0, :3, 1]
    return bool(
        left_force >= 0.5
        and right_force >= 0.5
        and vector_angle_degrees(left_direction, left_vector) <= 85.0
        and vector_angle_degrees(right_direction, right_vector) <= 85.0
    )


def robot_contact_force(base_env: Any, first: Any, second: Any) -> float:
    maximum = 0.0
    for first_link in (first.finger1_link, first.finger2_link):
        for second_link in (second.finger1_link, second.finger2_link):
            force = tensor_array(
                base_env.scene.get_pairwise_contact_forces(first_link, second_link)
            )
            maximum = max(maximum, float(np.linalg.norm(force, axis=-1).max()))
    return maximum


def extract_snapshot(
    base_env: Any,
    task: str,
    environment_success: bool,
) -> dict[str, Any]:
    spec = TASKS[task]
    agents = list(base_env.agent.agents)
    if len(agents) != int(spec["agents"]):
        raise ValueError("RoboFactory agent count differs from the M2 contract")
    object_positions: dict[str, list[float]] = {}
    object_velocities: dict[str, list[float]] = {}
    grasp: dict[str, list[bool]] = {}
    contact: dict[str, list[bool]] = {}
    contact_force: dict[str, list[float]] = {}
    threshold_ambiguous = False
    base_z = float(tensor_array(agents[0].robot.pose.p)[0, 2])
    for object_name in spec["objects"]:
        actor = getattr(base_env, object_name)
        object_positions[object_name] = [
            float(value) for value in tensor_array(actor.pose.p)[0]
        ]
        velocity = (
            actor.linear_velocity
            if hasattr(actor, "linear_velocity")
            else actor.root_linear_velocity
        )
        object_velocities[object_name] = [
            float(value) for value in tensor_array(velocity)[0]
        ]
        grasp_values: list[bool] = []
        contact_values: list[bool] = []
        force_values: list[float] = []
        for agent in agents:
            left_vector, right_vector = finger_contact_vectors(base_env, agent, actor)
            left = float(np.linalg.norm(left_vector, axis=-1).max())
            right = float(np.linalg.norm(right_vector, axis=-1).max())
            force = max(left, right)
            force_values.append(force)
            contact_values.append(force >= CONTACT_FORCE_MIN)
            grasp_values.append(grasp_from_contacts(agent, left_vector, right_vector))
            if abs(force - CONTACT_FORCE_MIN) <= CONTACT_AMBIGUITY_MARGIN:
                threshold_ambiguous = True
        grasp[object_name] = grasp_values
        contact[object_name] = contact_values
        contact_force[object_name] = force_values
    tcp_positions = [
        [float(value) for value in tensor_array(agent.tcp.pose.p)[0]] for agent in agents
    ]
    base_positions = [
        [float(value) for value in tensor_array(agent.robot.pose.p)[0]] for agent in agents
    ]
    maximum_robot_force = 0.0
    minimum_tcp_distance = float("inf")
    for first_index, first in enumerate(agents):
        for second_index in range(first_index + 1, len(agents)):
            maximum_robot_force = max(
                maximum_robot_force,
                robot_contact_force(base_env, first, agents[second_index]),
            )
            minimum_tcp_distance = min(
                minimum_tcp_distance,
                float(
                    np.linalg.norm(
                        np.asarray(tcp_positions[first_index])
                        - np.asarray(tcp_positions[second_index])
                    )
                ),
            )
    goal_positions: dict[str, list[float]] = {"goal": [0.0, 0.0, 0.0]}
    if hasattr(base_env, "goal_region"):
        goal_positions["goal"] = [
            float(value) for value in tensor_array(base_env.goal_region.pose.p)[0]
        ]
    task_predicates = {
        "button_aligned": False,
        "planar_meat_to_pot_distance": 1.0,
    }
    if task == "take_photo":
        camera = np.asarray(object_positions["camera"][:2], dtype=np.float64)
        button = camera + np.asarray([0.035, -0.09], dtype=np.float64)
        operator = int(spec["role_agents"]["button"])
        delta = np.asarray(tcp_positions[operator][:2], dtype=np.float64) - button
        task_predicates["button_aligned"] = bool(np.all(np.abs(delta) < 0.035))
    if task == "place_food":
        task_predicates["planar_meat_to_pot_distance"] = float(
            np.linalg.norm(
                np.asarray(object_positions["meat"][:2])
                - np.asarray(object_positions["pot"][:2])
            )
        )
    drop_risk = {
        object_name: bool(
            not any(grasp[object_name])
            and object_velocities[object_name][2] < -0.25
            and object_positions[object_name][2] > base_z + 0.05
        )
        for object_name in spec["objects"]
    }
    return {
        "task": task,
        "agent_count": len(agents),
        "role_agents": deepcopy(spec["role_agents"]),
        "tcp_positions": tcp_positions,
        "base_positions": base_positions,
        "object_positions": object_positions,
        "object_velocities": object_velocities,
        "goal_positions": goal_positions,
        "reference_heights": {"robot_base": base_z},
        "grasp": grasp,
        "contact": contact,
        "contact_force": contact_force,
        "task_predicates": task_predicates,
        "robot_collision": maximum_robot_force >= ROBOT_COLLISION_FORCE_MIN,
        "robot_proximity_risk": minimum_tcp_distance < ROBOT_PROXIMITY_DISTANCE,
        "drop_risk": drop_risk,
        "contact_threshold_ambiguous": threshold_ambiguous,
        "environment_success": bool(environment_success),
    }


class EpisodeWriter:
    def __init__(self, path: Path, task: str, seed: int, agent_count: int):
        self.path = path
        self.task = task
        self.seed = int(seed)
        self.agent_count = int(agent_count)
        self.stream = h5py.File(path, "w")
        self.stream.attrs["format_version"] = "ssc-v7.m2.audit_episode/1"
        self.stream.attrs["task"] = task
        self.stream.attrs["seed"] = int(seed)
        self.stream.attrs["agent_count"] = int(agent_count)
        self.image_datasets: dict[str, h5py.Dataset] = {}
        self.qpos_datasets: dict[str, h5py.Dataset] = {}
        self.action_dataset: h5py.Dataset | None = None
        self.labels: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.automatic_checks = {
            "deterministic": True,
            "agent_slot_equivariant": True,
            "required_fields": True,
            "finite_and_domain_valid": True,
            "terminal_success_exact": True,
        }

    @staticmethod
    def _append(dataset: h5py.Dataset, value: np.ndarray) -> None:
        next_index = int(dataset.shape[0])
        dataset.resize(next_index + 1, axis=0)
        dataset[next_index] = value

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
            images[key] = tensor_array(payload["rgb"])[0].astype(np.uint8, copy=False)
        if "global" not in images:
            raise ValueError("M2 observation lacks the frozen global RGB sensor")
        image_group = self.stream.require_group("data/observation/images")
        for key, image in images.items():
            dataset = self.image_datasets.get(key)
            if dataset is None:
                dataset = image_group.create_dataset(
                    key,
                    shape=(0, *image.shape),
                    maxshape=(None, *image.shape),
                    chunks=(1, *image.shape),
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=4,
                )
                self.image_datasets[key] = dataset
            self._append(dataset, image)
        agent_group = self.stream.require_group("data/observation/agents")
        for slot in range(self.agent_count):
            name = f"panda-{slot}"
            qpos = tensor_array(observation["agent"][name]["qpos"])[0].astype(
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

    def append_action(self, action: Mapping[str, Any]) -> None:
        expected = [f"panda-{slot}" for slot in range(self.agent_count)]
        if sorted(action) != expected:
            raise ValueError("M2 action dictionary differs from frozen agent slots")
        flattened = np.concatenate(
            [np.asarray(action[name], dtype=np.float32).reshape(-1) for name in expected]
        )
        if self.action_dataset is None:
            self.action_dataset = self.stream.require_group("data/action").create_dataset(
                "commanded",
                shape=(0, flattened.shape[0]),
                maxshape=(None, flattened.shape[0]),
                chunks=(256, flattened.shape[0]),
                dtype=np.float32,
            )
        self._append(self.action_dataset, flattened)

    def close(self) -> None:
        if self.stream:
            image_group = self.stream.get("data/observation/images")
            if isinstance(image_group, h5py.Group) and "global" in image_group:
                for slot in range(self.agent_count):
                    key = f"agent_{slot}"
                    if key not in image_group:
                        image_group[key] = image_group["global"]
            self.stream.attrs["observation_frames"] = len(self.labels)
            self.stream.attrs["action_frames"] = (
                0 if self.action_dataset is None else int(self.action_dataset.shape[0])
            )
            self.stream.flush()
            self.stream.close()


class M2AuditWrapper:
    def __init__(self, env: Any, task: str, scratch_root: Path):
        self.env = env
        self.task = task
        self.scratch_root = scratch_root
        self.writer: EpisodeWriter | None = None
        self.seed: int | None = None
        self.candidate_index: int | None = None
        self.memory: dict[str, Any] | None = None

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def begin_attempt(self, seed: int, candidate_index: int) -> None:
        if self.writer is not None:
            raise RuntimeError("previous M2 attempt was not finalized")
        self.seed = int(seed)
        self.candidate_index = int(candidate_index)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        from before_we_act.ssc_v7_oracle_labels import initial_automaton_state

        result = self.env.reset(*args, **kwargs)
        observation, info = result
        if self.seed is None or self.candidate_index is None:
            raise RuntimeError("M2 attempt identity was not set before reset")
        path = self.scratch_root / (
            f".{self.task}.candidate_{self.candidate_index:03d}.seed_{self.seed}.hdf5.tmp"
        )
        self.writer = EpisodeWriter(path, self.task, self.seed, TASKS[self.task]["agents"])
        self.memory = initial_automaton_state(self.task)
        success = False
        if isinstance(info, Mapping) and "success" in info:
            success = scalar_bool(info["success"])
        else:
            success = scalar_bool(self.unwrapped.evaluate()["success"])
        self._record(observation, success)
        return result

    def step(self, action: Mapping[str, Any]) -> Any:
        if self.writer is None:
            raise RuntimeError("M2 step arrived before reset")
        self.writer.append_action(action)
        result = self.env.step(action)
        observation, _, _, _, info = result
        if not isinstance(info, Mapping) or "success" not in info:
            raise RuntimeError("RoboFactory M2 step lacks environment success")
        self._record(observation, scalar_bool(info["success"]))
        return result

    def _record(self, observation: Mapping[str, Any], success: bool) -> None:
        from before_we_act.ssc_v7_oracle_labels import (
            build_oracle_label,
            permute_agent_slots,
            permute_automaton_state,
            permute_label_slots,
        )

        if self.writer is None or self.memory is None:
            raise RuntimeError("M2 record state is not initialized")
        snapshot = extract_snapshot(self.unwrapped, self.task, success)
        prior_memory = deepcopy(self.memory)
        first = build_oracle_label(snapshot, prior_memory)
        repeated = build_oracle_label(deepcopy(snapshot), deepcopy(prior_memory))
        if first != repeated:
            self.writer.automatic_checks["deterministic"] = False
        count = int(snapshot["agent_count"])
        permutation = list(reversed(range(count)))
        renamed = build_oracle_label(
            permute_agent_slots(snapshot, permutation),
            permute_automaton_state(prior_memory, permutation),
        )
        expected = permute_label_slots(first, permutation)
        if renamed != expected:
            self.writer.automatic_checks["agent_slot_equivariant"] = False
        if not all(field in first for field in REQUIRED_FIELDS):
            self.writer.automatic_checks["required_fields"] = False
        if not finite_tree(first) or not 0.0 <= float(first["within_stage_progress"]) <= 1.0:
            self.writer.automatic_checks["finite_and_domain_valid"] = False
        if bool(first["task_complete"]) != bool(success):
            self.writer.automatic_checks["terminal_success_exact"] = False
        self.memory = deepcopy(first["causal_automaton_state"])
        self.writer.append_observation(observation)
        self.writer.snapshots.append(snapshot)
        self.writer.labels.append(first)

    def finish_attempt(self, success: bool, output_root: Path) -> dict[str, Any] | None:
        if self.writer is None or self.seed is None or self.candidate_index is None:
            raise RuntimeError("no active M2 attempt")
        writer = self.writer
        writer.close()
        self.writer = None
        if not success:
            writer.path.unlink()
            return None
        episode_dir = output_root / "episodes" / self.task
        episode_dir.mkdir(parents=True, exist_ok=True)
        hdf5_path = episode_dir / (
            f"candidate_{self.candidate_index:03d}_seed_{self.seed}.hdf5"
        )
        os.replace(writer.path, hdf5_path)
        episode_sha = sha256_file(hdf5_path)
        sidecar_path = hdf5_path.with_suffix(".oracle.jsonl.gz")
        with gzip.open(sidecar_path, "wt", encoding="utf-8", compresslevel=6) as stream:
            for frame_index, (snapshot, label) in enumerate(
                zip(writer.snapshots, writer.labels, strict=True)
            ):
                payload = {
                    "primary_key": {
                        "task": self.task,
                        "episode_sha256": episode_sha,
                        "frame_index": frame_index,
                    },
                    "privileged_snapshot": snapshot,
                    "oracle_label": label,
                }
                stream.write(canonical_bytes(payload).decode("utf-8") + "\n")
        return {
            "task": self.task,
            "candidate_index": self.candidate_index,
            "seed": self.seed,
            "success": True,
            "frames": len(writer.labels),
            "actions": len(writer.labels) - 1,
            "hdf5_path": str(hdf5_path),
            "hdf5_sha256": episode_sha,
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "automatic_checks": writer.automatic_checks,
        }

    def abort_attempt(self) -> None:
        if self.writer is not None:
            self.writer.close()
            if self.writer.path.exists():
                self.writer.path.unlink()
        self.writer = None


def validate_contracts(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seed = json.loads(args.seed_contract.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle_spec.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_contract.read_text(encoding="utf-8"))
    if seed.get("stage_id") != "SSC-V7-M1":
        raise RuntimeError("M2 seed contract is not the frozen M1 seed revision")
    if not str(gate.get("stage_id", "")).startswith("SSC-V7-M2-R"):
        raise RuntimeError("M2 gate contract has the wrong stage identity")
    if tuple(seed["tasks"]) != tuple(TASKS):
        raise RuntimeError("M2 task order differs from the frozen seed contract")
    if tuple(oracle["required_fields"]) != REQUIRED_FIELDS:
        raise RuntimeError("M2 required label fields differ from oracle_label_spec")
    return seed, oracle, gate


def static_dependency_audit() -> dict[str, Any]:
    from before_we_act.ssc_v7_oracle_labels import build_oracle_label

    forbidden = (
        "frame_index",
        "wall_clock_time",
        "episode_length",
        "future_done",
        "future_success",
        "future_observation",
        "future_action",
    )
    source = inspect.getsource(build_oracle_label)
    hits = [value for value in forbidden if value in source]
    return {"forbidden_symbols": list(forbidden), "hits": hits, "passed": not hits}


def preflight(args: argparse.Namespace, gate: Mapping[str, Any]) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    m1_receipt = args.run_root / "measurement/m1_relaxed_rerun_v2/schema_receipt.json"
    payload = json.loads(m1_receipt.read_text(encoding="utf-8"))
    checks = {
        "m1_unlocked": payload.get("decision_code") == M1_DECISION,
        "repository_clean_before_collection": git_value(repository, "status", "--porcelain") == "",
        "robofactory_commit": git_value(args.robofactory_root, "rev-parse", "HEAD")
        == ROBOFACTORY_COMMIT,
        "implementation_commit_is_ancestor": subprocess.call(
            (
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                str(gate["implementation"]["commit"]),
                "HEAD",
            )
        )
        == 0,
        "implementation_module_sha256": sha256_file(
            repository / str(gate["implementation"]["module"])
        )
        == str(gate["implementation"]["module_sha256"]),
        "implementation_script_sha256": sha256_file(Path(__file__))
        == str(gate["implementation"]["script_sha256"]),
        "static_forbidden_dependency_scan": static_dependency_audit()["passed"],
        "output_root_fresh": not args.output_root.exists(),
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "m1_receipt": str(m1_receipt),
        "m1_receipt_sha256": sha256_file(m1_receipt),
        "static_dependency_audit": static_dependency_audit(),
    }


def make_env(task: str, robofactory_root: Path) -> Any:
    import gymnasium as gym
    import robofactory  # noqa: F401

    spec = TASKS[task]
    return gym.make(
        str(spec["env_id"]),
        config=str(robofactory_root / "robofactory/configs/table" / spec["config"]),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sensor_configs={"shader_pack": "default", "width": 640, "height": 480},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        sim_backend="cpu",
    )


def collect(args: argparse.Namespace, seeds: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    from robofactory.planner import solutions

    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "logs").mkdir()
    mode = str(args.mode)
    purpose = "pilot" if mode == "dry-run" else "expert_candidate_pool"
    required_successes = int(gate["collection"][mode]["successful_episodes_per_task"])
    candidate_start = int(gate["collection"][mode].get("candidate_index_start", 0))
    attempts_limit = int(gate["collection"][mode]["candidate_prefix_max"])
    episodes: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for task, spec in TASKS.items():
        environment = make_env(task, args.robofactory_root)
        wrapper = M2AuditWrapper(environment, task, args.output_root)
        solver = getattr(solutions, str(spec["solver"]))
        task_successes = 0
        try:
            candidates = seeds["per_task"][task][purpose][
                candidate_start:attempts_limit
            ]
            for candidate_index, seed in enumerate(candidates, start=candidate_start):
                if task_successes >= required_successes:
                    break
                print(
                    f"[M2 {mode}] {task} candidate={candidate_index} seed={seed}",
                    flush=True,
                )
                wrapper.begin_attempt(int(seed), candidate_index)
                log_path = args.output_root / "logs" / (
                    f"{task}.candidate_{candidate_index:03d}.seed_{seed}.log"
                )
                try:
                    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
                        result = solver(wrapper, seed=int(seed), debug=False, vis=False)
                    success = bool(result != -1 and scalar_bool(result[-1]["success"]))
                    item = wrapper.finish_attempt(success, args.output_root)
                    attempts.append(
                        {
                            "task": task,
                            "candidate_index": candidate_index,
                            "seed": int(seed),
                            "success": success,
                            "log_path": str(log_path),
                            "log_sha256": sha256_file(log_path),
                        }
                    )
                    if item is not None:
                        task_successes += 1
                        episodes.append(item)
                except Exception:
                    wrapper.abort_attempt()
                    raise
            if task_successes < required_successes:
                raise RuntimeError(
                    f"{task} produced {task_successes}/{required_successes} successful M2 episodes"
                )
        finally:
            wrapper.abort_attempt()
            environment.close()
    return {
        "format_version": "ssc-v7.m2.collection/1",
        "mode": mode,
        "purpose": purpose,
        "candidate_index_start": candidate_start,
        "candidate_index_stop_exclusive": attempts_limit,
        "required_successes_per_task": required_successes,
        "episodes": episodes,
        "attempts": attempts,
    }


def read_sidecar(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def transition_allowed(
    task: str, previous: str, current: str, gate: Mapping[str, Any]
) -> bool:
    if previous == current:
        return True
    allowed = gate["legal_transitions"][task].get(previous, [])
    return current in allowed


def support_categories(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        label = row["oracle_label"]
        result.add(f"stage:{label['stage_id']}")
        for key, value in label["factorized_predicates"].items():
            if value is True:
                result.add(f"predicate:{key}")
        for object_name, value in label["grasp_contact_custody_state"].items():
            if value["current_custodian"] is not None:
                result.add(f"custody:{object_name}")
        for value in label["agent_object_role_slots"]:
            for role in value["roles"]:
                if role != "none":
                    result.add(f"role:{value['object']}:{role}")
    return result


def event_signature(label: Mapping[str, Any]) -> str:
    payload = {
        "stage": label["stage_id"],
        "predicates": label["factorized_predicates"],
        "roles": label["agent_object_role_slots"],
        "custody": label["grasp_contact_custody_state"],
        "complete": label["task_complete"],
    }
    return sha256_json(payload)


def select_manual_events(
    collection: Mapping[str, Any],
    event_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    gate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Select frozen hash-ranked windows and add only missing episode coverage.

    The gate requires at least 20 transition windows *and* five independent
    episodes per task.  A global lowest-20 ranking can accidentally omit an
    episode.  Keep those first 20 identities unchanged, then append the
    lowest-ranked boundary from each omitted collected episode.  This makes
    the coverage correction deterministic and prevents choosing an easy-looking
    supplement after inspecting the rendered video.
    """

    selected_events: list[dict[str, Any]] = []
    supplements: list[dict[str, Any]] = []
    missing_without_event = 0
    required_events = int(gate["manual_audit"]["transition_windows_per_task_min"])
    for task in TASKS:
        candidates = sorted(
            (dict(value) for value in event_candidates[task]),
            key=lambda value: str(value["rank"]),
        )
        primary = candidates[:required_events]
        selected_events.extend(primary)
        selected_episodes = {str(value["episode_sha256"]) for value in primary}
        collected_episodes = sorted(
            {
                str(episode["hdf5_sha256"])
                for episode in collection["episodes"]
                if str(episode["task"]) == task
            }
        )
        for episode_sha in collected_episodes:
            if episode_sha in selected_episodes:
                continue
            choices = [
                value
                for value in candidates
                if str(value["episode_sha256"]) == episode_sha
            ]
            if not choices:
                missing_without_event += 1
                continue
            supplement = dict(choices[0])
            supplement["coverage_supplement"] = True
            supplements.append(supplement)
    selected_events.extend(supplements)
    return selected_events, missing_without_event


def automatic_audit(
    collection: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    episode_rows: dict[str, list[dict[str, Any]]] = {}
    issue_counts = {
        "required_field": 0,
        "finite_or_domain": 0,
        "terminal_success": 0,
        "determinism": 0,
        "agent_slot_equivariance": 0,
        "illegal_transition": 0,
    }
    ambiguous_frames = 0
    total_frames = 0
    support: dict[str, dict[str, int]] = {task: {} for task in TASKS}
    event_candidates: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for episode in collection["episodes"]:
        task = str(episode["task"])
        rows = read_sidecar(Path(str(episode["sidecar_path"])))
        episode_rows[str(episode["hdf5_sha256"])] = rows
        total_frames += len(rows)
        ambiguous_frames += sum(
            int(row["oracle_label"]["ambiguity_code"] != 0) for row in rows
        )
        for check_name, passed in episode["automatic_checks"].items():
            if not passed:
                key = {
                    "deterministic": "determinism",
                    "agent_slot_equivariant": "agent_slot_equivariance",
                    "required_fields": "required_field",
                    "finite_and_domain_valid": "finite_or_domain",
                    "terminal_success_exact": "terminal_success",
                }[check_name]
                issue_counts[key] += 1
        previous = rows[0]["oracle_label"]["stage_id"]
        previous_signature = event_signature(rows[0]["oracle_label"])
        for frame_index, row in enumerate(rows[1:], start=1):
            current = str(row["oracle_label"]["stage_id"])
            if not transition_allowed(task, str(previous), current, gate):
                issue_counts["illegal_transition"] += 1
            signature = event_signature(row["oracle_label"])
            if signature != previous_signature:
                rank = hashlib.sha256(
                    f"{task}|{episode['hdf5_sha256']}|{frame_index}".encode("utf-8")
                ).hexdigest()
                event_candidates[task].append(
                    {
                        "rank": rank,
                        "task": task,
                        "episode_sha256": episode["hdf5_sha256"],
                        "hdf5_path": episode["hdf5_path"],
                        "sidecar_path": episode["sidecar_path"],
                        "frame_index": frame_index,
                    }
                )
            previous = current
            previous_signature = signature
        categories = support_categories(rows)
        for category in categories:
            support[task][category] = support[task].get(category, 0) + 1
    support_requirements: list[dict[str, Any]] = []
    for task, requirements in gate["required_support_categories"].items():
        for item in requirements:
            category = str(item["category"])
            count = int(support[task].get(category, 0))
            support_requirements.append(
                {
                    "task": task,
                    "category": category,
                    "sources": list(item["sources"]),
                    "episode_count": count,
                    "required": int(gate["thresholds"]["independent_episodes_per_category_min"]),
                    "passed": count
                    >= int(gate["thresholds"]["independent_episodes_per_category_min"]),
                }
            )
    ambiguity_fraction = ambiguous_frames / total_frames if total_frames else 1.0
    ambiguity_passed = ambiguity_fraction <= float(
        gate["thresholds"]["ambiguous_frame_fraction_max"]
    )
    required_events = int(gate["manual_audit"]["transition_windows_per_task_min"])
    for task, candidates in event_candidates.items():
        if len(candidates) < required_events and collection["mode"] == "execute":
            issue_counts["insufficient_transition_windows"] = issue_counts.get(
                "insufficient_transition_windows", 0
            ) + (required_events - len(candidates))
    selected_events, missing_manual_episode_coverage = select_manual_events(
        collection, event_candidates, gate
    )
    if missing_manual_episode_coverage and collection["mode"] == "execute":
        issue_counts["insufficient_manual_episode_coverage"] = (
            missing_manual_episode_coverage
        )
    hard_passed = all(value == 0 for value in issue_counts.values())
    return {
        "format_version": "ssc-v7.m2.automatic_audit/1",
        "hard_gate_passed": hard_passed,
        "ambiguity_passed": ambiguity_passed,
        "issue_counts": issue_counts,
        "total_frames": total_frames,
        "ambiguous_frames": ambiguous_frames,
        "ambiguous_frame_fraction": ambiguity_fraction,
        "support_histogram": support,
        "support_requirements": support_requirements,
        "selected_events": selected_events,
        "episode_rows": episode_rows,
    }


def overlay_frame(image: np.ndarray, text_lines: Sequence[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    text = "\n".join(text_lines)
    box = draw.multiline_textbbox((0, 0), text)
    width = min(canvas.width, box[2] + 12)
    height = min(canvas.height, box[3] + 12)
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    draw.multiline_text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(canvas)


def causal_review_indices(center: int, total_frames: int, width: int = 5) -> list[int]:
    """Return a fixed-width past/current window without exposing future frames.

    Events in the first ``width - 1`` frames are left-padded by repeating frame 0.
    This preserves the frozen event identity and five-frame UI shape without letting
    a reviewer use a post-event frame to judge the current label.
    """

    if total_frames <= 0:
        raise ValueError("manual review episode is empty")
    if center < 0 or center >= total_frames:
        raise ValueError("manual review center is outside the episode")
    if width <= 0:
        raise ValueError("manual review width must be positive")
    return [max(0, center - width + 1 + offset) for offset in range(width)]


def build_manual_packets(
    output_root: Path, automatic: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    import imageio.v2 as imageio
    from PIL import Image

    packet_root = output_root / "manual_audit_packets"
    packet_root.mkdir()
    items: list[dict[str, Any]] = []
    montage_images: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    rows_by_episode = automatic["episode_rows"]
    for index, event in enumerate(automatic["selected_events"]):
        task = str(event["task"])
        episode_sha = str(event["episode_sha256"])
        center = int(event["frame_index"])
        rows = rows_by_episode[episode_sha]
        frame_indices = causal_review_indices(center, len(rows))
        frames: list[np.ndarray] = []
        with h5py.File(str(event["hdf5_path"]), "r") as stream:
            dataset = stream["data/observation/images/global"]
            for context_slot, frame_index in enumerate(frame_indices):
                label = rows[frame_index]["oracle_label"]
                custody = {
                    name: value["current_custodian"]
                    for name, value in label["grasp_contact_custody_state"].items()
                }
                frames.append(
                    overlay_frame(
                        np.asarray(dataset[frame_index]),
                        (
                            f"{task} f={frame_index}",
                            f"audit_context={context_slot + 1}/5 past_or_current",
                            f"stage={label['stage_id']}",
                            f"custody={custody}",
                            f"complete={label['task_complete']}",
                        ),
                    )
                )
        packet_id = f"{task}_{index:03d}_{episode_sha[:10]}_f{center:04d}"
        task_root = packet_root / task
        task_root.mkdir(exist_ok=True)
        video_path = task_root / f"{packet_id}.mp4"
        imageio.mimsave(video_path, frames, fps=4, codec="libx264", quality=8)
        contact_sheet = np.concatenate(frames, axis=1)
        sheet_path = task_root / f"{packet_id}.png"
        Image.fromarray(contact_sheet).save(sheet_path)
        montage_images[task].append(frames[len(frames) // 2])
        oracle = rows[center]["oracle_label"]
        items.append(
            {
                "packet_id": packet_id,
                "task": task,
                "episode_sha256": episode_sha,
                "frame_index": center,
                "review_frame_indices": frame_indices,
                "review_window_policy": "past/current only; frame 0 is repeated for left padding",
                "video_path": str(video_path),
                "video_sha256": sha256_file(video_path),
                "contact_sheet_path": str(sheet_path),
                "contact_sheet_sha256": sha256_file(sheet_path),
                "oracle": {
                    "stage_id": oracle["stage_id"],
                    "factorized_predicates": oracle["factorized_predicates"],
                    "agent_object_role_slots": oracle["agent_object_role_slots"],
                    "grasp_contact_custody_state": oracle[
                        "grasp_contact_custody_state"
                    ],
                    "task_complete": oracle["task_complete"],
                },
                "review": None,
            }
        )
    montage_paths: dict[str, str] = {}
    for task, frames in montage_images.items():
        if not frames:
            continue
        target_width = 320
        thumbs = []
        for frame in frames:
            image = Image.fromarray(frame)
            image.thumbnail((target_width, 240))
            thumbs.append(np.asarray(image))
        rows = []
        for start in range(0, len(thumbs), 4):
            row = thumbs[start : start + 4]
            while len(row) < 4:
                row.append(np.zeros_like(thumbs[0]))
            rows.append(np.concatenate(row, axis=1))
        montage = np.concatenate(rows, axis=0)
        path = packet_root / f"{task}_montage.png"
        Image.fromarray(montage).save(path)
        montage_paths[task] = str(path)
    return {
        "format_version": "ssc-v7.m2.manual_review/1",
        "review_instructions": gate["manual_audit"],
        "montage_paths": montage_paths,
        "items": items,
    }


def manifest_for_collection(
    args: argparse.Namespace,
    seed_contract: Mapping[str, Any],
    gate: Mapping[str, Any],
    seeds: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "format_version": "ssc-v7.m2.oracle_sidecar_manifest/1",
        "stage_id": gate["stage_id"],
        "mode": args.mode,
        "status": "COLLECTED_PENDING_AUDIT",
        "created_at_utc": utc_now(),
        "provenance": {
            "repository_branch": git_value(repository, "branch", "--show-current"),
            "repository_commit": git_value(repository, "rev-parse", "HEAD"),
            "implementation_commit": gate["implementation"]["commit"],
            "robofactory_commit": git_value(args.robofactory_root, "rev-parse", "HEAD"),
            "seed_contract": str(args.seed_contract),
            "seed_contract_sha256": sha256_file(args.seed_contract),
            "oracle_spec": str(args.oracle_spec),
            "oracle_spec_sha256": sha256_file(args.oracle_spec),
            "gate_contract": str(args.gate_contract),
            "gate_contract_sha256": sha256_file(args.gate_contract),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "expanded_seed_manifest_sha256": sha256_json(seeds),
        "seed_namespace": seed_contract["namespace"],
        "preflight": preflight_receipt,
        "collection": collection,
    }


def finalize(
    args: argparse.Namespace,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    automatic_path = args.output_root / "automatic_audit.json"
    template_path = args.output_root / "manual_review_template.json"
    if not automatic_path.is_file() or not template_path.is_file():
        raise RuntimeError("M2 finalize requires completed collection and audit packets")
    if args.manual_review is None:
        raise RuntimeError("M2 finalize requires --manual-review")
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    review = json.loads(args.manual_review.read_text(encoding="utf-8"))
    expected = json.loads(template_path.read_text(encoding="utf-8"))
    expected_ids = [item["packet_id"] for item in expected["items"]]
    review_ids = [item["packet_id"] for item in review["items"]]
    if review_ids != expected_ids:
        raise RuntimeError("manual review item identities or order changed")
    required_keys = {
        "predicate_agreement",
        "terminal_agreement",
        "role_agreement",
        "custody_agreement",
        "reviewer",
        "reviewer_type",
        "notes",
    }
    for item in review["items"]:
        if not isinstance(item.get("review"), Mapping) or set(item["review"]) != required_keys:
            raise RuntimeError("manual review row is incomplete")
    predicate_agreement = np.mean(
        [bool(item["review"]["predicate_agreement"]) for item in review["items"]]
    )
    terminal_errors = sum(
        not bool(item["review"]["terminal_agreement"]) for item in review["items"]
    )
    role_errors = sum(
        not bool(item["review"]["role_agreement"]) for item in review["items"]
    )
    custody_errors = sum(
        not bool(item["review"]["custody_agreement"]) for item in review["items"]
    )
    task_counts = {
        task: len({item["episode_sha256"] for item in review["items"] if item["task"] == task})
        for task in TASKS
    }
    task_window_counts = {
        task: sum(item["task"] == task for item in review["items"]) for task in TASKS
    }
    human_review = all(
        item["review"]["reviewer_type"] == "human" for item in review["items"]
    )
    manual_passed = bool(
        predicate_agreement >= float(gate["manual_audit"]["predicate_level_agreement_min"])
        and terminal_errors == 0
        and role_errors == 0
        and custody_errors == 0
        and human_review
        and all(
            value >= int(gate["manual_audit"]["episodes_per_task"])
            for value in task_counts.values()
        )
        and all(
            value >= int(gate["manual_audit"]["transition_windows_per_task_min"])
            for value in task_window_counts.values()
        )
    )
    hard_passed = bool(automatic["hard_gate_passed"] and automatic["ambiguity_passed"])
    insufficient = [
        item for item in automatic["support_requirements"] if not bool(item["passed"])
    ]
    source_status: dict[str, str] = {source: "PASSED" for source in ("P", "T", "B")}
    for item in insufficient:
        for source in item["sources"]:
            source_status[str(source)] = "INCONCLUSIVE_MEASUREMENT/INSUFFICIENT_LABEL_SUPPORT"
    if not hard_passed or not manual_passed:
        decision = "FAILED_SCHEMA/ORACLE_LABEL_MISMATCH"
        source_status = {source: decision for source in source_status}
    elif all(value == "PASSED" for value in source_status.values()):
        decision = "PASSED_M2_ORACLE_LABEL_GATE"
    else:
        decision = "INCONCLUSIVE_MEASUREMENT/INSUFFICIENT_LABEL_SUPPORT"
    result = {
        "format_version": "ssc-v7.m2.oracle_label_audit/1",
        "stage_id": gate["stage_id"],
        "completed_at_utc": utc_now(),
        "decision_code": decision,
        "global_hard_gate_passed": hard_passed,
        "manual_audit_passed": manual_passed,
        "source_status": source_status,
        "automatic_audit": automatic,
        "manual_audit": {
            "review_path": str(args.manual_review),
            "review_sha256": sha256_file(args.manual_review),
            "predicate_agreement": float(predicate_agreement),
            "terminal_errors": terminal_errors,
            "role_errors": role_errors,
            "custody_errors": custody_errors,
            "episodes_covered_per_task": task_counts,
            "transition_windows_per_task": task_window_counts,
            "human_review": human_review,
        },
        "training_authorized": False,
        "next_step_authorized": "M3 only for sources with source_status=PASSED",
    }
    write_json(args.output_root / "oracle_label_audit.json", result)
    return result


def main() -> None:
    started = datetime.now(timezone.utc)
    args = parse_args()
    seed_contract, _, gate = validate_contracts(args)
    if args.mode == "finalize":
        result = finalize(args, gate)
        print(result["decision_code"])
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    repository = Path(__file__).resolve().parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    preflight_receipt = preflight(args, gate)
    if preflight_receipt["status"] != "PASSED":
        raise RuntimeError(f"M2 preflight failed: {preflight_receipt['checks']}")
    seeds = expanded_seed_manifest(seed_contract, args.w10_seed_root)
    collection = collect(args, seeds, gate)
    manifest = manifest_for_collection(
        args, seed_contract, gate, seeds, preflight_receipt, collection
    )
    manifest_path = args.output_root / "oracle_sidecar_manifest.json"
    write_json(manifest_path, manifest)
    automatic = automatic_audit(collection, gate)
    automatic.pop("episode_rows")
    write_json(args.output_root / "automatic_audit.json", automatic)
    automatic_with_rows = automatic_audit(collection, gate)
    template = build_manual_packets(args.output_root, automatic_with_rows, gate)
    write_json(args.output_root / "manual_review_template.json", template)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    automatic_passed = bool(
        automatic["hard_gate_passed"] and automatic["ambiguity_passed"]
    )
    if not automatic_passed:
        decision = "FAILED_SCHEMA/ORACLE_LABEL_MISMATCH"
    elif args.mode == "dry-run":
        decision = "SSC_V7_M2_DRY_RUN_PASSED"
    else:
        decision = "PENDING_M2_MANUAL_AUDIT"
    receipt = {
        "format_version": "ssc-v7.m2.execution_receipt/1",
        "stage_id": gate["stage_id"],
        "mode": args.mode,
        "completed_at_utc": utc_now(),
        "elapsed_wall_seconds": elapsed,
        "decision_code": decision,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "automatic_hard_gate_passed": automatic_passed,
        "manual_review_required": args.mode == "execute" and automatic_passed,
        "training_authorized": False,
    }
    write_json(args.output_root / "execution_receipt.json", receipt)
    print(decision)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
