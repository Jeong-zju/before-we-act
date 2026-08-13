#!/usr/bin/env python3
"""Run the preregistered SSC-V7 M3-R4 action-relevant-belief measurement.

R4 deliberately stops after the oracle gate when action-relevant ground truth does
not beat a frozen, converged history-and-current-observation baseline and matched
residual controls.  The sealed test split is never accepted by this script.
"""

from __future__ import annotations

import argparse
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
import sys
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.before_we_act import run_ssc_v7_m3 as m3  # noqa: E402


STAGE_ID = "SSC-V7-M3-R4"
TASKS = tuple(m3.TASKS)
TOKEN_NAMES = (
    "contact_grasp_custody",
    "handoff_event",
    "teammate_motion_state",
    "blocking_collision",
    "visibility_staleness",
    "uncertainty_missingness",
)
FIELD_NAMES = (
    (
        "self_contact_any",
        "self_grasp_any",
        "self_controller_any",
        "peer_contact_any",
        "peer_grasp_any",
        "peer_controller_any",
        "shared_control_any",
        "peer_current_custodian_any",
    ),
    (
        "handoff_completed_now",
        "custody_changed_now",
        "self_gained_custody_now",
        "self_lost_custody_now",
        "peer_gained_custody_now",
        "peer_lost_custody_now",
        "shared_control_started_now",
        "shared_control_ended_now",
    ),
    (
        "peer_active_any",
        "peer_inactive_any",
        "multiple_peers_active",
        "peer_contacting_any",
        "peer_grasping_any",
        "peer_support_role_any",
        "peer_receiver_role_any",
        "peer_handler_or_button_role_any",
    ),
    (
        "robot_collision",
        "robot_proximity_risk",
        "contested_object_any",
        "dropped_object_any",
        "multi_agent_contact_any",
        "multi_agent_grasp_any",
        "shared_control_risk_any",
        "custody_conflict_any",
    ),
    (
        "relation_changed_within_1",
        "relation_changed_within_4",
        "relation_changed_within_8",
        "relation_changed_within_16",
        "relation_unchanged_for_16",
        "peer_relation_present_now",
        "relevant_event_present_now",
        "source_is_current",
    ),
    (
        "contact_label_valid",
        "handoff_label_valid",
        "motion_label_valid",
        "risk_label_valid",
        "ambiguity_free",
        "source_is_fresh",
        "source_not_missing",
        "oracle_truth_available",
    ),
)
TOKEN_COUNT = len(TOKEN_NAMES)
TOKEN_WIDTH = len(FIELD_NAMES[0])
ARB_WIDTH = TOKEN_COUNT * TOKEN_WIDTH
HC_INPUT_WIDTH = 1896
LEGACY_WIDTH = 192
PRIMARY_STEPS = 16
ACTION_WIDTH = 8
OUTPUT_WIDTH = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("schema-audit", "pilot", "train-hc", "train-branch", "aggregate-oracle"),
    )
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/"
            "measurement/m3_r2/formal_dataset/dataset_manifest.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--condition", choices=("oracle_arb", "zero_residual", "noise_residual", "legacy_concat")
    )
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2))
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
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_gate(path: Path) -> dict[str, Any]:
    gate = read_json(path)
    if gate.get("stage_id") != STAGE_ID:
        raise RuntimeError("wrong M3-R4 gate identity")
    if gate.get("status") != "FROZEN_BEFORE_R4_TUNE_ACCESS":
        raise RuntimeError("M3-R4 gate is not frozen before tune access")
    schema = gate["arb_schema"]
    if tuple(schema["token_names"]) != TOKEN_NAMES:
        raise RuntimeError("gate token names do not match implementation")
    if tuple(tuple(row) for row in schema["field_names"]) != FIELD_NAMES:
        raise RuntimeError("gate ARB fields do not match implementation")
    if int(schema["token_count"]) != TOKEN_COUNT or int(schema["token_width"]) != TOKEN_WIDTH:
        raise RuntimeError("gate ARB dimensions do not match implementation")
    unsigned = {key: value for key, value in gate.items() if key != "integrity"}
    payload_hash = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if payload_hash != str(gate["integrity"]["payload_sha256"]):
        raise RuntimeError("gate payload hash mismatch")
    gate["_runtime_gate_sha256"] = sha256_file(path)
    return gate


def label_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line)["oracle_label"] for line in stream if line.strip()]


def int_agents(values: Iterable[Any]) -> set[int]:
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def custody_map(label: Mapping[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name, state in label["grasp_contact_custody_state"].items():
        value = state.get("current_custodian")
        try:
            result[str(name)] = None if value is None else int(value)
        except (TypeError, ValueError):
            result[str(name)] = None
    return result


def shared_map(label: Mapping[str, Any]) -> dict[str, bool]:
    return {
        str(name): bool(state.get("shared_control", False))
        for name, state in label["grasp_contact_custody_state"].items()
    }


def relation_signature(label: Mapping[str, Any], own_slot: int) -> tuple[Any, ...]:
    parts: list[Any] = []
    for name, state in sorted(label["grasp_contact_custody_state"].items()):
        contacts = int_agents(state.get("contact_agents", []))
        grasps = int_agents(state.get("grasp_agents", []))
        controllers = int_agents(state.get("controller_agents", []))
        custodian = custody_map(label)[str(name)]
        parts.append(
            (
                str(name),
                own_slot in contacts,
                bool(contacts - {own_slot}),
                own_slot in grasps,
                bool(grasps - {own_slot}),
                own_slot in controllers,
                bool(controllers - {own_slot}),
                custodian == own_slot,
                custodian is not None and custodian != own_slot,
                bool(state.get("shared_control", False)),
            )
        )
    peer_states = []
    for item in label["per_agent_contribution"]:
        if int(item["agent_slot"]) == own_slot:
            continue
        peer_states.append(
            (
                bool(item.get("active", False)),
                bool(item.get("contact_objects", [])),
                bool(item.get("grasp_objects", [])),
                tuple(sorted(str(value) for value in item.get("roles", []))),
            )
        )
    risk = label["collision_drop_contention_risk"]
    parts.append(tuple(sorted(peer_states)))
    parts.append(
        (
            bool(risk.get("robot_collision", False)),
            bool(risk.get("robot_proximity_risk", False)),
            bool(risk.get("contested_objects", [])),
            bool(risk.get("dropped_objects", [])),
        )
    )
    return tuple(parts)


def arb_tokens(
    labels: Sequence[Mapping[str, Any]],
    frame: int,
    own_slot: int,
    signatures: Sequence[tuple[Any, ...]] | None = None,
) -> np.ndarray:
    current = labels[frame]
    previous = labels[max(0, frame - 1)]
    current_states = current["grasp_contact_custody_state"]
    previous_states = previous["grasp_contact_custody_state"]
    own_contact = own_grasp = own_controller = False
    peer_contact = peer_grasp = peer_controller = False
    shared = peer_custodian = False
    multi_contact = multi_grasp = custody_conflict = False
    for name, state in current_states.items():
        contacts = int_agents(state.get("contact_agents", []))
        grasps = int_agents(state.get("grasp_agents", []))
        controllers = int_agents(state.get("controller_agents", []))
        own_contact |= own_slot in contacts
        own_grasp |= own_slot in grasps
        own_controller |= own_slot in controllers
        peer_contact |= bool(contacts - {own_slot})
        peer_grasp |= bool(grasps - {own_slot})
        peer_controller |= bool(controllers - {own_slot})
        shared |= bool(state.get("shared_control", False))
        custodian = custody_map(current).get(str(name))
        peer_custodian |= custodian is not None and custodian != own_slot
        multi_contact |= len(contacts) >= 2
        multi_grasp |= len(grasps) >= 2
        custody_conflict |= (custodian is None and bool(controllers)) or len(controllers) >= 2

    current_custody = custody_map(current)
    previous_custody = custody_map(previous)
    custody_changed = current_custody != previous_custody
    self_gained = any(
        current_custody.get(name) == own_slot and previous_custody.get(name) != own_slot
        for name in current_custody
    )
    self_lost = any(
        previous_custody.get(name) == own_slot and current_custody.get(name) != own_slot
        for name in current_custody
    )
    peer_gained = any(
        current_custody.get(name) not in (None, own_slot)
        and previous_custody.get(name) != current_custody.get(name)
        for name in current_custody
    )
    peer_lost = any(
        previous_custody.get(name) not in (None, own_slot)
        and current_custody.get(name) != previous_custody.get(name)
        for name in current_custody
    )
    current_completed = set(current["causal_automaton_state"].get("completed_handoff_mask", []))
    previous_completed = set(previous["causal_automaton_state"].get("completed_handoff_mask", []))
    current_shared = shared_map(current)
    previous_shared = shared_map(previous)

    peers = [
        item
        for item in current["per_agent_contribution"]
        if int(item["agent_slot"]) != own_slot
    ]
    active_count = sum(bool(item.get("active", False)) for item in peers)
    roles = {
        str(role)
        for item in peers
        for role in item.get("roles", [])
        if str(role) != "none"
    }
    peer_contacting = any(bool(item.get("contact_objects", [])) for item in peers)
    peer_grasping = any(bool(item.get("grasp_objects", [])) for item in peers)

    risk = current["collision_drop_contention_risk"]
    if signatures is None:
        signatures = [relation_signature(label, own_slot) for label in labels]
    changed: dict[int, bool] = {}
    for horizon in (1, 4, 8, 16):
        first = max(1, frame - horizon + 1)
        changed[horizon] = any(
            signatures[index] != signatures[index - 1]
            for index in range(first, frame + 1)
        )
    relation_present = any(
        (own_contact, own_grasp, own_controller, peer_contact, peer_grasp, peer_controller, shared)
    )
    relevant_event = any(
        (
            custody_changed,
            bool(current_completed - previous_completed),
            bool(risk.get("robot_collision", False)),
            bool(risk.get("robot_proximity_risk", False)),
            bool(risk.get("contested_objects", [])),
            bool(risk.get("dropped_objects", [])),
        )
    )
    valid = current["label_validity_mask"]
    values = np.asarray(
        [
            [own_contact, own_grasp, own_controller, peer_contact, peer_grasp, peer_controller, shared, peer_custodian],
            [
                bool(current_completed - previous_completed),
                custody_changed,
                self_gained,
                self_lost,
                peer_gained,
                peer_lost,
                any(current_shared.get(name, False) and not previous_shared.get(name, False) for name in current_shared),
                any(previous_shared.get(name, False) and not current_shared.get(name, False) for name in current_shared),
            ],
            [
                active_count > 0,
                active_count < len(peers),
                active_count >= 2,
                peer_contacting,
                peer_grasping,
                "support" in roles,
                "receiver" in roles,
                bool(roles & {"handler", "button_operator", "custodian"}),
            ],
            [
                bool(risk.get("robot_collision", False)),
                bool(risk.get("robot_proximity_risk", False)),
                bool(risk.get("contested_objects", [])),
                bool(risk.get("dropped_objects", [])),
                multi_contact,
                multi_grasp,
                shared,
                custody_conflict,
            ],
            [changed[1], changed[4], changed[8], changed[16], not changed[16], relation_present, relevant_event, True],
            [
                bool(valid.get("grasp_contact_custody_state", False)),
                bool(valid.get("causal_automaton_state", False)),
                bool(valid.get("per_agent_contribution", False)),
                bool(valid.get("collision_drop_contention_risk", False)),
                int(current.get("ambiguity_code", 1)) == 0,
                True,
                True,
                True,
            ],
        ],
        dtype=np.float32,
    )
    if values.shape != (TOKEN_COUNT, TOKEN_WIDTH):
        raise AssertionError(values.shape)
    return values


@dataclass
class ArbBundle:
    base: m3.ProbeData
    arb: np.ndarray

    def subset(self, indices: np.ndarray) -> "ArbBundle":
        return ArbBundle(self.base.subset(indices), self.arb[indices])

    def __len__(self) -> int:
        return len(self.base)


def load_bundle(manifest_path: Path, splits: set[str]) -> tuple[ArbBundle, dict[str, Any]]:
    if "read_only_test" in splits:
        raise RuntimeError("M3-R4 implementation is forbidden from opening a test split")
    base, audit = m3.load_probe_data(manifest_path, splits)
    manifest = read_json(manifest_path)
    features: list[np.ndarray] = []
    identities: list[tuple[str, int, int]] = []
    for episode in manifest["episodes"]:
        if str(episode["split"]) not in splits:
            continue
        labels = label_rows(Path(str(episode["sidecar_path"])))
        episode_id = str(episode["hdf5_sha256"])
        with h5py.File(str(episode["hdf5_path"]), "r") as stream:
            action_count = int(stream["data/action/commanded"].shape[0])
            agent_count = int(stream.attrs["agent_count"])
        episode_signatures = {
            own_slot: [relation_signature(label, own_slot) for label in labels]
            for own_slot in range(agent_count)
        }
        for frame in m3.uniform_indices(16, action_count - 16, 64):
            label = labels[frame]
            if int(label["ambiguity_code"]) != 0 or not all(
                bool(value) for value in label["label_validity_mask"].values()
            ):
                continue
            for own_slot in range(agent_count):
                features.append(
                    arb_tokens(labels, frame, own_slot, episode_signatures[own_slot])
                )
                identities.append((episode_id, frame, own_slot))
    expected = list(
        zip(
            base.episode_ids.astype(str).tolist(),
            base.frame_indices.astype(int).tolist(),
            base.agent_slots.astype(int).tolist(),
            strict=True,
        )
    )
    if identities != expected:
        raise RuntimeError("ARB extraction order does not match the frozen probe rows")
    values = np.stack(features).astype(np.float32)
    audit = dict(audit)
    audit["arb_shape"] = list(values.shape)
    audit["test_paths_opened"] = 0
    return ArbBundle(base, values), audit


def fixed_hc_noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, LEGACY_WIDTH).astype(np.float32)


def hc_input(bundle: ArbBundle, seed: int) -> np.ndarray:
    noise = np.broadcast_to(fixed_hc_noise(seed), (len(bundle), LEGACY_WIDTH))
    return np.concatenate((bundle.base.legal, noise), axis=1).astype(np.float32)


def arb_input(bundle: ArbBundle, hc_seed: int, arb: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    if arb.shape != (len(bundle), TOKEN_COUNT, TOKEN_WIDTH):
        raise ValueError(f"wrong ARB shape: {arb.shape}")
    if reliability.shape != (len(bundle), 1):
        raise ValueError(f"wrong reliability shape: {reliability.shape}")
    return np.concatenate((hc_input(bundle, hc_seed), arb.reshape(len(bundle), -1), reliability), axis=1).astype(np.float32)


def legacy_input(bundle: ArbBundle, hc_seed: int, reliability: np.ndarray) -> np.ndarray:
    legacy = bundle.base.social.copy()
    legacy[:, : m3.SOURCE_SLICES["B"].start] = 0.0
    legacy[:, m3.SOURCE_SLICES["B"].stop :] = 0.0
    return np.concatenate((hc_input(bundle, hc_seed), legacy, reliability), axis=1).astype(np.float32)


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


def build_frozen_hc(payload: Mapping[str, Any]) -> Any:
    import torch

    input_width = int(payload["input_width"])
    hidden_width = int(payload["hidden_width"])
    net = torch.nn.Sequential(
        torch.nn.LayerNorm(input_width),
        torch.nn.Linear(input_width, hidden_width),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_width, hidden_width),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_width, OUTPUT_WIDTH),
    )
    net.load_state_dict(payload["state_dict"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


class HCWrapper:
    """Factory namespace; the returned class remains a normal torch Module."""

    @staticmethod
    def create(payload: Mapping[str, Any]) -> Any:
        import torch

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hc = build_frozen_hc(payload)

            def forward(self, values: Any) -> Any:
                return self.hc(values[:, :HC_INPUT_WIDTH])

        return Model()


def hc_forward_hidden(hc: Any, values: Any) -> tuple[Any, Any]:
    x = values[:, :HC_INPUT_WIDTH]
    hidden = hc[0](x)
    hidden = hc[2](hc[1](hidden))
    hidden = hc[4](hc[3](hidden))
    return hc[5](hidden), hidden


class ArbResidualFactory:
    @staticmethod
    def create(payload: Mapping[str, Any], seed: int) -> Any:
        torch = torch_setup(seed)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hc = build_frozen_hc(payload)
                self.context_projection = torch.nn.Linear(256, 64)
                self.token_projection = torch.nn.Linear(TOKEN_WIDTH, 64, bias=False)
                self.token_type = torch.nn.Parameter(torch.empty(TOKEN_COUNT, 64))
                self.query_type = torch.nn.Parameter(torch.empty(4, 64))
                self.attention = torch.nn.MultiheadAttention(64, 4, batch_first=True)
                self.norm = torch.nn.LayerNorm(64)
                self.immediate = torch.nn.Linear(64, 4 * ACTION_WIDTH)
                self.short = torch.nn.Linear(64, 4 * ACTION_WIDTH)
                self.late = torch.nn.Linear(64, 8 * ACTION_WIDTH)
                self.gate = torch.nn.Linear(64, 1)
                torch.nn.init.normal_(self.token_type, std=0.02)
                torch.nn.init.normal_(self.query_type, std=0.02)
                for layer in (self.immediate, self.short, self.late):
                    torch.nn.init.zeros_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

            def forward(self, values: Any) -> Any:
                base, hidden = hc_forward_hidden(self.hc, values)
                first = HC_INPUT_WIDTH
                tokens = values[:, first : first + ARB_WIDTH].reshape(-1, TOKEN_COUNT, TOKEN_WIDTH)
                reliability = values[:, first + ARB_WIDTH : first + ARB_WIDTH + 1]
                keys = self.token_projection(tokens) + self.token_type[None]
                queries = self.context_projection(hidden)[:, None] + self.query_type[None]
                attended, _ = self.attention(queries, keys, keys, need_weights=False)
                states = self.norm(queries + attended)
                residual_16 = torch.cat(
                    (self.immediate(states[:, 0]), self.short(states[:, 1]), self.late(states[:, 2])),
                    dim=1,
                )
                gate = reliability * torch.sigmoid(self.gate(states[:, 3]))
                tail = torch.zeros((values.shape[0], OUTPUT_WIDTH - residual_16.shape[1]), device=values.device, dtype=values.dtype)
                return base + torch.cat((gate * residual_16, tail), dim=1)

        return Model()


class LegacyResidualFactory:
    @staticmethod
    def create(payload: Mapping[str, Any], seed: int) -> Any:
        torch = torch_setup(seed)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hc = build_frozen_hc(payload)
                self.fusion = torch.nn.Sequential(
                    torch.nn.LayerNorm(256 + LEGACY_WIDTH),
                    torch.nn.Linear(256 + LEGACY_WIDTH, 128),
                    torch.nn.SiLU(),
                )
                self.residual = torch.nn.Linear(128, PRIMARY_STEPS * ACTION_WIDTH)
                self.gate = torch.nn.Linear(128, 1)
                torch.nn.init.zeros_(self.residual.weight)
                torch.nn.init.zeros_(self.residual.bias)

            def forward(self, values: Any) -> Any:
                base, hidden = hc_forward_hidden(self.hc, values)
                first = HC_INPUT_WIDTH
                legacy = values[:, first : first + LEGACY_WIDTH]
                reliability = values[:, first + LEGACY_WIDTH : first + LEGACY_WIDTH + 1]
                state = self.fusion(torch.cat((hidden, legacy), dim=1))
                delta = reliability * torch.sigmoid(self.gate(state)) * self.residual(state)
                tail = torch.zeros((values.shape[0], OUTPUT_WIDTH - delta.shape[1]), device=values.device, dtype=values.dtype)
                return base + torch.cat((delta, tail), dim=1)

        return Model()


def primary_action_loss(model: Any, x: Any, y: Any, mask: Any) -> Any:
    prediction = model(x).reshape(-1, 100, ACTION_WIDTH)[:, :PRIMARY_STEPS]
    squared = (prediction - y[:, :PRIMARY_STEPS]).square() * mask[:, :PRIMARY_STEPS, None]
    return squared.sum() / (mask[:, :PRIMARY_STEPS].sum().clamp_min(1.0) * ACTION_WIDTH)


def train_residual(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    validation_x: np.ndarray,
    validation_data: m3.ProbeData,
    validation_y: np.ndarray,
    device: str,
    learning_rate: float,
    sampler_seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[Any, dict[str, Any]]:
    torch = torch_setup(sampler_seed)
    model = model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    x_train = torch.from_numpy(train_x)
    y_train = torch.from_numpy(train_y)
    mask_train = torch.from_numpy(train_mask)
    best_primary = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        losses: list[float] = []
        for indices in m3.batches(len(train_x), 512, sampler_seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = primary_action_loss(
                model,
                x_train[indices].to(device),
                y_train[indices].to(device),
                mask_train[indices].to(device),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        primary = float(
            m3.evaluate_model(model, validation_x, validation_data, validation_y, device)[
                "task_macro_primary_16_nrmse"
            ]
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_16_step": float(np.mean(losses)),
                "validation_task_macro_primary_16_nrmse": primary,
            }
        )
        if primary < best_primary - 1e-7:
            best_primary = primary
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
        raise RuntimeError("residual training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    converged = len(history) - 1 - best_epoch >= patience
    return model, {
        "best_epoch": best_epoch,
        "best_validation_task_macro_primary_16_nrmse": best_primary,
        "epochs_run": len(history),
        "max_epochs": max_epochs,
        "patience": patience,
        "converged_by_patience": converged,
        "cap_reached_while_improving": not converged,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "history": history,
    }


def save_checkpoint(path: Path, model: Any, metadata: Mapping[str, Any]) -> None:
    m3.save_torch_checkpoint(
        path,
        {
            "state_dict": model.state_dict(),
            "stage_id": STAGE_ID,
            **dict(metadata),
        },
    )


def deterministic_episode_split(data: m3.ProbeData, holdout_per_task: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    fit_ids: set[str] = set()
    holdout_ids: set[str] = set()
    for task in TASKS:
        episode_ids = sorted(set(data.episode_ids[data.tasks == task].astype(str).tolist()))
        ranked = sorted(
            episode_ids,
            key=lambda value: hashlib.sha256(f"{seed}|{task}|{value}".encode()).hexdigest(),
        )
        holdout_ids.update(ranked[:holdout_per_task])
        fit_ids.update(ranked[holdout_per_task:])
    fit = np.asarray([str(value) in fit_ids for value in data.episode_ids], dtype=bool)
    holdout = np.asarray([str(value) in holdout_ids for value in data.episode_ids], dtype=bool)
    if np.any(fit & holdout) or not np.all(fit | holdout):
        raise RuntimeError("invalid pilot episode split")
    return fit, holdout


def schema_audit(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    bundle, audit = load_bundle(args.manifest, {"train"})
    prevalence = bundle.arb.mean(axis=0)
    episode_support: dict[str, dict[str, int]] = {}
    for token_index, token_name in enumerate(TOKEN_NAMES):
        episode_support[token_name] = {}
        for field_index, field_name in enumerate(FIELD_NAMES[token_index]):
            positive = set(
                bundle.base.episode_ids[bundle.arb[:, token_index, field_index] > 0.5].astype(str).tolist()
            )
            negative = set(
                bundle.base.episode_ids[bundle.arb[:, token_index, field_index] <= 0.5].astype(str).tolist()
            )
            episode_support[token_name][field_name] = min(len(positive), len(negative))
    receipt = {
        "format_version": "ssc-v7.m3_r4.schema_audit/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": sha256_file(args.gate),
        "manifest_sha256": sha256_file(args.manifest),
        "train_only": True,
        "audit": audit,
        "prevalence": {
            token: {
                field: float(prevalence[token_index, field_index])
                for field_index, field in enumerate(FIELD_NAMES[token_index])
            }
            for token_index, token in enumerate(TOKEN_NAMES)
        },
        "minority_episode_support": episode_support,
        "test_paths_opened": 0,
    }
    write_json(args.output_root / "schema_audit.json", receipt)
    print("SSC_V7_M3_R4_SCHEMA_AUDITED_TRAIN_ONLY")


def pilot(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.output_root.exists():
        raise FileExistsError(f"fresh pilot output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    bundle, audit = load_bundle(args.manifest, {"train"})
    pilot_config = gate["r4_0_pilot"]
    fit_mask, holdout_mask = deterministic_episode_split(
        bundle.base,
        int(pilot_config["holdout_episodes_per_task"]),
        int(pilot_config["split_seed"]),
    )
    fit = bundle.subset(fit_mask)
    holdout = bundle.subset(holdout_mask)
    norms = m3.normalizer(fit.base)
    target_fit = m3.normalized_targets(fit.base, norms)
    target_holdout = m3.normalized_targets(holdout.base, norms)
    hc_seed = int(gate["seeds"]["hc_noise_seed"])
    max_epochs = int(pilot_config["max_epochs"])
    patience = int(pilot_config["patience"])
    hc_model, hc_training = m3.train_action_model(
        hc_input(fit, hc_seed),
        target_fit,
        fit.base.target_mask,
        hc_input(holdout, hc_seed),
        holdout.base,
        target_holdout,
        args.device,
        float(gate["training"]["hc_learning_rate"]),
        256,
        int(gate["seeds"]["hc_initialization_seed"]),
        int(gate["seeds"]["hc_sampler_seed"]),
        max_epochs,
        patience,
    )
    hc_best = int(hc_training["best_epoch"])
    hc_converged = int(hc_training["epochs_run"]) - 1 - hc_best >= patience
    hc_payload = {
        "state_dict": hc_model.state_dict(),
        "input_width": HC_INPUT_WIDTH,
        "hidden_width": 256,
        "condition": "pilot_hc",
    }
    reliability_fit = np.ones((len(fit), 1), dtype=np.float32)
    reliability_holdout = np.ones((len(holdout), 1), dtype=np.float32)
    residual = ArbResidualFactory.create(hc_payload, int(gate["seeds"]["residual_initialization"][0]))
    residual, residual_training = train_residual(
        residual,
        arb_input(fit, hc_seed, fit.arb, reliability_fit),
        target_fit,
        fit.base.target_mask,
        arb_input(holdout, hc_seed, holdout.arb, reliability_holdout),
        holdout.base,
        target_holdout,
        args.device,
        float(gate["training"]["residual_learning_rate"]),
        int(gate["seeds"]["residual_sampler"][0]),
        max_epochs,
        patience,
    )
    converged = hc_converged and bool(residual_training["converged_by_patience"])
    margin = int(pilot_config["formal_budget_margin_epochs"])
    floor = int(pilot_config["formal_max_epochs_floor"])
    ceiling = int(pilot_config["formal_max_epochs_ceiling"])
    formal_max = min(ceiling, max(floor, hc_best + patience + margin, int(residual_training["best_epoch"]) + patience + margin))
    resolved = {
        "format_version": "ssc-v7.m3_r4.resolved_budget/1",
        "stage_id": STAGE_ID,
        "gate_sha256": sha256_file(args.gate),
        "manifest_sha256": sha256_file(args.manifest),
        "source": "train-only episode-disjoint R4-0 pilot",
        "max_epochs": formal_max,
        "patience": patience,
        "pilot_converged": converged,
        "test_paths_opened": 0,
    }
    budget_path = args.output_root / "resolved_budget.json"
    write_json(budget_path, resolved)
    receipt = {
        "format_version": "ssc-v7.m3_r4.pilot_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "train_only": True,
        "audit": audit,
        "fit_rows": len(fit),
        "holdout_rows": len(holdout),
        "fit_episode_count": len(set(fit.base.episode_ids.tolist())),
        "holdout_episode_count": len(set(holdout.base.episode_ids.tolist())),
        "hc_training": hc_training,
        "hc_converged_by_patience": hc_converged,
        "oracle_residual_training": residual_training,
        "resolved_budget": str(budget_path),
        "resolved_budget_sha256": sha256_file(budget_path),
        "decision_code": "SSC_V7_M3_R4_PILOT_CONVERGED" if converged else "INCONCLUSIVE_TRAINING/CAP_REACHED",
        "tune_paths_opened": 0,
        "test_paths_opened": 0,
    }
    write_json(args.output_root / "pilot_receipt.json", receipt)
    print(receipt["decision_code"])


def load_resolved_budget(root: Path, gate: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_receipt = read_json(root / "r4_0_pilot" / "pilot_receipt.json")
    if pilot_receipt.get("decision_code") != "SSC_V7_M3_R4_PILOT_CONVERGED":
        raise RuntimeError("R4-0 pilot did not converge")
    if pilot_receipt.get("test_paths_opened") != 0 or pilot_receipt.get("tune_paths_opened") != 0:
        raise RuntimeError("R4-0 pilot accessed a forbidden split")
    budget_path = Path(str(pilot_receipt["resolved_budget"]))
    if sha256_file(budget_path) != str(pilot_receipt["resolved_budget_sha256"]):
        raise RuntimeError("resolved budget hash mismatch")
    budget = read_json(budget_path)
    if budget.get("gate_sha256") != gate["_runtime_gate_sha256"]:
        raise RuntimeError("resolved budget belongs to another gate")
    return budget, pilot_receipt


def train_hc(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    output = args.output_root / "r4_a" / "hc"
    if output.exists():
        raise FileExistsError(f"fresh HC output required: {output}")
    output.mkdir(parents=True)
    budget, pilot_receipt = load_resolved_budget(args.output_root, gate)
    train_bundle, train_audit = load_bundle(args.manifest, {"train"})
    tune_bundle, tune_audit = load_bundle(args.manifest, {"tune"})
    norms = m3.normalizer(train_bundle.base)
    target_train = m3.normalized_targets(train_bundle.base, norms)
    target_tune = m3.normalized_targets(tune_bundle.base, norms)
    hc_seed = int(gate["seeds"]["hc_noise_seed"])
    model, training = m3.train_action_model(
        hc_input(train_bundle, hc_seed),
        target_train,
        train_bundle.base.target_mask,
        hc_input(tune_bundle, hc_seed),
        tune_bundle.base,
        target_tune,
        args.device,
        float(gate["training"]["hc_learning_rate"]),
        256,
        int(gate["seeds"]["hc_initialization_seed"]),
        int(gate["seeds"]["hc_sampler_seed"]),
        int(budget["max_epochs"]),
        int(budget["patience"]),
    )
    converged = int(training["epochs_run"]) - 1 - int(training["best_epoch"]) >= int(budget["patience"])
    checkpoint = output / "action_HC.pt"
    save_checkpoint(
        checkpoint,
        model,
        {"input_width": HC_INPUT_WIDTH, "hidden_width": 256, "condition": "HC"},
    )
    metric = m3.evaluate_model(model, hc_input(tune_bundle, hc_seed), tune_bundle.base, target_tune, args.device)
    normalization_path = output / "normalization.json"
    metrics_path = output / "tune_metrics.json"
    write_json(normalization_path, norms)
    write_json(metrics_path, metric)
    receipt = {
        "format_version": "ssc-v7.m3_r4.hc_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": sha256_file(args.gate),
        "manifest_sha256": sha256_file(args.manifest),
        "resolved_budget_sha256": str(pilot_receipt["resolved_budget_sha256"]),
        "train_audit": train_audit,
        "tune_audit": tune_audit,
        "training": training,
        "converged_by_patience": converged,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "decision_code": "SSC_V7_M3_R4_HC_FROZEN" if converged else "INCONCLUSIVE_TRAINING/CAP_REACHED",
        "test_paths_opened": 0,
    }
    write_json(output / "hc_receipt.json", receipt)
    print(receipt["decision_code"])


def load_hc_artifacts(root: Path, gate: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = read_json(root / "r4_a" / "hc" / "hc_receipt.json")
    if receipt.get("decision_code") != "SSC_V7_M3_R4_HC_FROZEN":
        raise RuntimeError("formal HC did not converge")
    if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
        raise RuntimeError("HC belongs to another gate")
    checkpoint_path = Path(str(receipt["checkpoint"]))
    normalization_path = Path(str(receipt["normalization"]))
    if sha256_file(checkpoint_path) != str(receipt["checkpoint_sha256"]):
        raise RuntimeError("HC checkpoint hash mismatch")
    if sha256_file(normalization_path) != str(receipt["normalization_sha256"]):
        raise RuntimeError("normalization hash mismatch")
    payload = m3.load_torch_checkpoint(checkpoint_path, "cpu")
    return receipt, payload, read_json(normalization_path)


def branch_values(
    condition: str,
    bundle: ArbBundle,
    gate: Mapping[str, Any],
    noise: np.ndarray,
) -> np.ndarray:
    hc_seed = int(gate["seeds"]["hc_noise_seed"])
    reliability = np.ones((len(bundle), 1), dtype=np.float32)
    if condition == "oracle_arb":
        return arb_input(bundle, hc_seed, bundle.arb, reliability)
    if condition == "zero_residual":
        return arb_input(bundle, hc_seed, np.zeros_like(bundle.arb), reliability)
    if condition == "noise_residual":
        values = np.broadcast_to(noise, bundle.arb.shape).copy()
        return arb_input(bundle, hc_seed, values, reliability)
    if condition == "legacy_concat":
        return legacy_input(bundle, hc_seed, reliability)
    raise ValueError(condition)


def train_branch(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.condition is None or args.seed_index is None:
        raise ValueError("train-branch requires --condition and --seed-index")
    output = args.output_root / "r4_a" / "branches" / args.condition / f"seed_{args.seed_index}"
    if output.exists():
        raise FileExistsError(f"fresh branch output required: {output}")
    output.mkdir(parents=True)
    budget, _ = load_resolved_budget(args.output_root, gate)
    hc_receipt, hc_payload, norms = load_hc_artifacts(args.output_root, gate)
    train_bundle, train_audit = load_bundle(args.manifest, {"train"})
    tune_bundle, tune_audit = load_bundle(args.manifest, {"tune"})
    m3.normalize_social(train_bundle.base, norms)
    m3.normalize_social(tune_bundle.base, norms)
    target_train = m3.normalized_targets(train_bundle.base, norms)
    target_tune = m3.normalized_targets(tune_bundle.base, norms)
    noise_seed = int(gate["seeds"]["input_independent_noise_seed"])
    noise = np.random.default_rng(noise_seed).uniform(0.0, 1.0, size=(TOKEN_COUNT, TOKEN_WIDTH)).astype(np.float32)
    init_seed = int(gate["seeds"]["residual_initialization"][args.seed_index])
    sampler_seed = int(gate["seeds"]["residual_sampler"][args.seed_index])
    if args.condition == "legacy_concat":
        model = LegacyResidualFactory.create(hc_payload, init_seed)
    else:
        model = ArbResidualFactory.create(hc_payload, init_seed)
    train_x = branch_values(args.condition, train_bundle, gate, noise)
    tune_x = branch_values(args.condition, tune_bundle, gate, noise)
    model, training = train_residual(
        model,
        train_x,
        target_train,
        train_bundle.base.target_mask,
        tune_x,
        tune_bundle.base,
        target_tune,
        args.device,
        float(gate["training"]["residual_learning_rate"]),
        sampler_seed,
        int(budget["max_epochs"]),
        int(budget["patience"]),
    )
    checkpoint = output / "action_residual.pt"
    save_checkpoint(
        checkpoint,
        model,
        {
            "condition": args.condition,
            "seed_index": args.seed_index,
            "input_width": int(train_x.shape[1]),
        },
    )
    metric = m3.evaluate_model(model, tune_x, tune_bundle.base, target_tune, args.device)
    metrics_path = output / "tune_metrics.json"
    write_json(metrics_path, metric)
    receipt = {
        "format_version": "ssc-v7.m3_r4.branch_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "condition": args.condition,
        "seed_index": args.seed_index,
        "initialization_seed": init_seed,
        "sampler_seed": sampler_seed,
        "gate_sha256": sha256_file(args.gate),
        "manifest_sha256": sha256_file(args.manifest),
        "hc_checkpoint_sha256": hc_receipt["checkpoint_sha256"],
        "train_audit": train_audit,
        "tune_audit": tune_audit,
        "training": training,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "decision_code": "SSC_V7_M3_R4_BRANCH_CONVERGED" if training["converged_by_patience"] else "INCONCLUSIVE_TRAINING/CAP_REACHED",
        "test_paths_opened": 0,
    }
    write_json(output / "branch_receipt.json", receipt)
    print(f"{receipt['decision_code']} {args.condition} seed={args.seed_index}")


def median_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("empty metric collection")
    episode_ids = tuple(metrics[0]["episode_errors"])
    episodes: dict[str, dict[str, Any]] = {}
    for episode_id in episode_ids:
        values = [item["episode_errors"][episode_id] for item in metrics]
        episodes[episode_id] = {
            "task": values[0]["task"],
            "primary_16_nrmse": float(np.median([value["primary_16_nrmse"] for value in values])),
            "diagnostic_100_masked_nrmse": float(np.median([value["diagnostic_100_masked_nrmse"] for value in values])),
            "gripper_16_rmse": float(np.median([value["gripper_16_rmse"] for value in values])),
            "rows": int(values[0]["rows"]),
        }
    per_task = {
        task: float(np.mean([value["primary_16_nrmse"] for value in episodes.values() if value["task"] == task]))
        for task in TASKS
    }
    return {
        "episode_errors": episodes,
        "task_macro_primary_16_nrmse": float(np.mean(list(per_task.values()))),
        "per_task_primary_16_nrmse": per_task,
        "aggregation": "per-episode median over the three frozen residual seeds",
    }


def stable_task_harms(summary: Mapping[str, Any], threshold: float) -> list[str]:
    return [
        task
        for task, value in summary["per_task"].items()
        if float(value["ci95"][1]) <= -threshold
    ]


def aggregate_oracle(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    output = args.output_root / "r4_a" / "oracle_gate_receipt.json"
    if output.exists():
        raise FileExistsError(f"fresh aggregate output required: {output}")
    hc_receipt, _, _ = load_hc_artifacts(args.output_root, gate)
    hc_metrics = read_json(Path(str(hc_receipt["metrics"])))
    conditions = ("oracle_arb", "zero_residual", "noise_residual", "legacy_concat")
    loaded: dict[str, list[dict[str, Any]]] = {}
    branch_receipts: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        loaded[condition] = []
        branch_receipts[condition] = []
        for seed_index in range(3):
            root = args.output_root / "r4_a" / "branches" / condition / f"seed_{seed_index}"
            receipt = read_json(root / "branch_receipt.json")
            if receipt.get("decision_code") != "SSC_V7_M3_R4_BRANCH_CONVERGED":
                raise RuntimeError(f"branch did not converge: {condition}/{seed_index}")
            if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
                raise RuntimeError("branch gate hash mismatch")
            metric_path = Path(str(receipt["metrics"]))
            if sha256_file(metric_path) != str(receipt["metrics_sha256"]):
                raise RuntimeError("branch metric hash mismatch")
            loaded[condition].append(read_json(metric_path))
            branch_receipts[condition].append(receipt)
    aggregate = {condition: median_metrics(values) for condition, values in loaded.items()}
    statistics_seed = int(gate["seeds"]["statistics_seed"])
    oracle = m3.summarize_gain(hc_metrics, aggregate["oracle_arb"], statistics_seed)
    comparisons = {
        condition: m3.summarize_gain(
            aggregate[condition], aggregate["oracle_arb"], statistics_seed + offset
        )
        for offset, condition in enumerate(("zero_residual", "noise_residual", "legacy_concat"), start=1)
    }
    seed_summaries = [
        m3.summarize_gain(hc_metrics, metric, statistics_seed + 10 + seed_index)
        for seed_index, metric in enumerate(loaded["oracle_arb"])
    ]
    threshold = float(gate["r4_a_acceptance"]["oracle_gain_min"])
    stable_harms = stable_task_harms(oracle, float(gate["r4_a_acceptance"]["stable_harm_threshold_abs"]))
    checks = {
        "oracle_gain_at_least_3pct": float(oracle["macro_gain"]) >= threshold,
        "oracle_ci_lower_positive": float(oracle["ci95"][0]) > 0.0,
        "at_least_two_positive_tasks": len(oracle["positive_tasks"]) >= int(gate["r4_a_acceptance"]["positive_tasks_min"]),
        "no_stable_task_harm_at_3pct": not stable_harms,
        "at_least_two_of_three_seeds_positive": sum(float(item["macro_gain"]) > 0.0 for item in seed_summaries) >= 2,
        "no_seed_stably_harmed_at_3pct": all(float(item["ci95"][1]) > -float(gate["r4_a_acceptance"]["stable_harm_threshold_abs"]) for item in seed_summaries),
        "beats_zero_residual_ci": float(comparisons["zero_residual"]["ci95"][0]) > 0.0,
        "beats_noise_residual_ci": float(comparisons["noise_residual"]["ci95"][0]) > 0.0,
        "beats_legacy_concat_ci": float(comparisons["legacy_concat"]["ci95"][0]) > 0.0,
    }
    passed = all(checks.values())
    receipt = {
        "format_version": "ssc-v7.m3_r4.oracle_gate_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": sha256_file(args.gate),
        "manifest_sha256": sha256_file(args.manifest),
        "hc_receipt_sha256": sha256_file(args.output_root / "r4_a" / "hc" / "hc_receipt.json"),
        "aggregation": "per-episode median over three frozen seeds; episode-level task-stratified paired bootstrap",
        "oracle_vs_hc": oracle,
        "oracle_vs_controls": comparisons,
        "oracle_per_seed_vs_hc": seed_summaries,
        "stable_task_harms": stable_harms,
        "checks": checks,
        "passed": passed,
        "decision_code": "PASSED_M3_R4_A_ORACLE_UTILITY" if passed else "FAILED_MEASUREMENT/NO_ACTION_RELEVANT_ORACLE_UTILITY",
        "r4_b_authorized": passed,
        "new_test_authorized": False,
        "m4_authorized": False,
        "b_core_authorized": False,
        "test_paths_opened": 0,
    }
    write_json(output, receipt)
    print(receipt["decision_code"])


def main() -> None:
    args = parse_args()
    gate = load_gate(args.gate)
    if sha256_file(args.manifest) != str(gate["data"]["manifest_sha256"]):
        raise RuntimeError("frozen data manifest hash mismatch")
    if args.command == "schema-audit":
        schema_audit(args, gate)
    elif args.command == "pilot":
        pilot(args, gate)
    elif args.command == "train-hc":
        train_hc(args, gate)
    elif args.command == "train-branch":
        train_branch(args, gate)
    elif args.command == "aggregate-oracle":
        aggregate_oracle(args, gate)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
