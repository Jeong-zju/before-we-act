"""Causal batch assembly shared by offline training and online replay tests."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Mapping

import torch

from models.slot_encoder import LocalBeliefSlotEncoder


def build_future_belief_histories(
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Roll the decision-time history forward without using future truth early.

    At post-transition step ``h``, the appended row is exactly
    ``(observation[t+h+1], action[t+h])``.  This mirrors the online ring buffer
    and produces belief targets for WAM horizons without repeating the current
    frame as fake history.
    """

    local = batch["local_history"]
    history_mask = batch["history_mask"]
    future_observation = batch["future_model_observation"]
    future_action = batch["ego_future_action"]
    object_observation = batch["object_observation_history"]
    object_valid = batch["object_valid_history"]
    object_confidence = batch["object_confidence_history"]
    object_age = batch["object_age_history"]

    if local.ndim != 3:
        raise ValueError("local_history must have shape [B, L, D]")
    B, L, local_dim = local.shape
    if history_mask.shape != (B, L) or history_mask.dtype != torch.bool:
        raise TypeError("history_mask must be boolean with shape [B, L]")
    if future_observation.ndim != 3 or future_action.ndim != 3:
        raise ValueError("future observations/actions must have shape [B, H, D]")
    if future_observation.shape[:2] != future_action.shape[:2] or future_action.shape[0] != B:
        raise ValueError("future observation/action batch and horizon must agree")
    H = future_action.shape[1]
    expected_local_dim = future_observation.shape[-1] + future_action.shape[-1]
    if local_dim != expected_local_dim:
        raise ValueError(
            "local_history width must equal future model observation + ego action widths; "
            f"got {local_dim} vs {expected_local_dim}"
        )

    if object_observation.shape[:2] != (B, L):
        raise ValueError("object_observation_history must have shape [B, L, object_dim]")
    for name, value in (
        ("object_valid_history", object_valid),
        ("object_confidence_history", object_confidence),
        ("object_age_history", object_age),
    ):
        if value.shape != (B, L):
            raise ValueError(f"{name} must have shape [B, L]")
    if object_valid.dtype != torch.bool:
        raise TypeError("object_valid_history must be boolean")

    future_object = batch["future_object_observation"]
    future_valid = batch["future_object_valid"]
    future_confidence = batch["future_object_confidence"]
    future_age = batch["future_object_age"]
    if future_object.shape[:2] != (B, H):
        raise ValueError("future_object_observation must have shape [B, H, object_dim]")
    for name, value in (
        ("future_object_valid", future_valid),
        ("future_object_confidence", future_confidence),
        ("future_object_age", future_age),
    ):
        if value.shape != (B, H):
            raise ValueError(f"{name} must have shape [B, H]")
    if future_valid.dtype != torch.bool:
        raise TypeError("future_object_valid must be boolean")

    running_local = local
    running_mask = history_mask
    running_object = object_observation
    running_valid = object_valid
    running_confidence = object_confidence
    running_age = object_age
    histories = []
    masks = []
    object_histories = []
    valid_histories = []
    confidence_histories = []
    age_histories = []

    for step in range(H):
        appended = torch.cat(
            [future_observation[:, step], future_action[:, step]], dim=-1
        ).unsqueeze(1)
        running_local = torch.cat([running_local[:, 1:], appended], dim=1)
        running_mask = torch.cat(
            [
                running_mask[:, 1:],
                torch.ones(B, 1, device=running_mask.device, dtype=torch.bool),
            ],
            dim=1,
        )
        running_object = torch.cat(
            [running_object[:, 1:], future_object[:, step : step + 1]], dim=1
        )
        running_valid = torch.cat(
            [running_valid[:, 1:], future_valid[:, step : step + 1]], dim=1
        )
        running_confidence = torch.cat(
            [running_confidence[:, 1:], future_confidence[:, step : step + 1]], dim=1
        )
        running_age = torch.cat(
            [running_age[:, 1:], future_age[:, step : step + 1]], dim=1
        )
        histories.append(running_local)
        masks.append(running_mask)
        object_histories.append(running_object)
        valid_histories.append(running_valid)
        confidence_histories.append(running_confidence)
        age_histories.append(running_age)

    return {
        "local_history": torch.stack(histories, dim=1),
        "history_mask": torch.stack(masks, dim=1),
        "object_observation": torch.stack(object_histories, dim=1),
        "object_valid": torch.stack(valid_histories, dim=1),
        "object_confidence": torch.stack(confidence_histories, dim=1),
        "object_age": torch.stack(age_histories, dim=1),
    }


def encode_current_and_future_beliefs(
    encoder: LocalBeliefSlotEncoder,
    batch: Mapping[str, torch.Tensor],
    *,
    detach_future: bool = True,
) -> Dict[str, torch.Tensor]:
    """Encode current ego belief and causal post-transition belief targets."""

    current = encoder(
        batch["local_history"],
        batch["history_mask"],
        batch["ego_id"],
        object_observation=batch["object_observation_history"],
        object_valid=batch["object_valid_history"],
        object_age=batch["object_age_history"],
        object_confidence=batch["object_confidence_history"],
    )["slots"]

    rolled = build_future_belief_histories(batch)
    B, H, L, D = rolled["local_history"].shape
    agent_id = batch["ego_id"].unsqueeze(1).expand(B, H).reshape(B * H)
    context = torch.no_grad() if detach_future else nullcontext()
    with context:
        future = encoder(
            rolled["local_history"].reshape(B * H, L, D),
            rolled["history_mask"].reshape(B * H, L),
            agent_id,
            object_observation=rolled["object_observation"].reshape(
                B * H, L, rolled["object_observation"].shape[-1]
            ),
            object_valid=rolled["object_valid"].reshape(B * H, L),
            object_age=rolled["object_age"].reshape(B * H, L),
            object_confidence=rolled["object_confidence"].reshape(B * H, L),
        )["slots"].reshape(B, H, encoder.NUM_ROLES, encoder.cfg.slot_dim)
    if detach_future:
        future = future.detach()
    return {"ego_slots": current, "target_ego_slots": future, **rolled}
