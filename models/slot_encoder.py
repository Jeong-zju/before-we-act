from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Local-belief encoder
# ---------------------------------------------------------------------------


@dataclass
class LocalBeliefSlotEncoderConfig:
    """Configuration for the strictly local  belief encoder.

    ``local_dim`` is the width of a fused deployable vector with the explicit
    layout ``[onboard_sensors, ego_task_context, previous_ego_action]``.  Thus
    callers compute ``local_dim = sensor_dim + task_dim + action_dim`` and the
    final history row contains :math:`a_{t-1}`, never the candidate
    :math:`a_t`.  It must not contain object truth, simulator phase labels, or
    another robot's private state.  Object estimates use the separate inputs
    below so their missingness cannot be hidden inside a zero-filled vector.
    ``object_dim`` is reserved for an optional *local perception estimate*
    supplied by a future RGB perception bridge; setting it to zero is valid.
    """

    history: int = 8
    local_dim: int = 21
    object_dim: int = 0
    slot_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_history_layers: int = 2
    num_slot_layers: int = 2
    num_agents: int = 2
    dropout: float = 0.1
    max_object_age: float = 100.0
    privileged_aux_dims: Dict[str, int] = field(default_factory=dict)
    privileged_aux_roles: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.history <= 0 or self.local_dim <= 0 or self.object_dim < 0:
            raise ValueError("history/local_dim must be positive and object_dim cannot be negative")
        if self.slot_dim <= 0 or self.hidden_dim <= 0 or self.num_heads <= 0:
            raise ValueError("slot_dim, hidden_dim and num_heads must be positive")
        if self.slot_dim % self.num_heads != 0:
            raise ValueError("slot_dim must be divisible by num_heads")
        if self.num_history_layers <= 0 or self.num_slot_layers <= 0:
            raise ValueError("transformer layer counts must be positive")
        if self.num_agents <= 0 or self.max_object_age <= 0:
            raise ValueError("num_agents and max_object_age must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name, out_dim in self.privileged_aux_dims.items():
            if not name or "." in name:
                raise ValueError("privileged auxiliary names must be non-empty and cannot contain '.'")
            if int(out_dim) <= 0:
                raise ValueError("privileged auxiliary output dimensions must be positive")
        if set(self.privileged_aux_roles) != set(self.privileged_aux_dims):
            raise ValueError("every privileged auxiliary head must have exactly one configured role")
        valid_roles = set(LocalBeliefSlotEncoder.ROLE_NAMES)
        for name, role in self.privileged_aux_roles.items():
            if role not in valid_roles:
                raise ValueError(f"privileged auxiliary head '{name}' has unknown role '{role}'")


class LocalBeliefSlotEncoder(nn.Module):
    """Encode one robot's deployable history into four fixed-role beliefs.

    Slot order is an API invariant::

        0 self, 1 object-belief, 2 teammate-belief, 3 task-context

    No teammate pose/state or plan message is accepted here.  The teammate
    belief is inferred only from the robot's own history and object-coupling
    evidence; selective plan messages are fused by downstream intention/WAM
    modules.  Privileged targets are never forward inputs.  Optional auxiliary
    heads predict them from already-constructed slots for simulation training.

    Object missingness has explicit semantics:

    * ``object_valid=True``: a fresh local-perception measurement;
    * ``object_valid=False, confidence>0``: a stale retained estimate, whose
      value is attenuated by confidence but remains usable;
    * ``object_valid=False, confidence=0``: no estimate; its value is ignored;
    * ``object_age>=0``: age of the most recent local estimate;
    * ``object_confidence in [0,1]``: confidence of that estimate;
    * ``history_mask=False``: a padded timestep, ignored by all attention.
    """

    ROLE_NAMES = ("self", "object-belief", "teammate-belief", "task-context")
    SELF_INDEX = 0
    OBJECT_INDEX = 1
    TEAMMATE_INDEX = 2
    TASK_INDEX = 3
    NUM_ROLES = 4

    def __init__(self, cfg: LocalBeliefSlotEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.local_projection = nn.Linear(cfg.local_dim, cfg.slot_dim)
        self.object_projection = nn.Linear(cfg.object_dim, cfg.slot_dim) if cfg.object_dim > 0 else None
        self.object_status_projection = nn.Linear(3, cfg.slot_dim)
        self.temporal_embedding = nn.Parameter(torch.randn(1, cfg.history, cfg.slot_dim) * 0.02)
        self.agent_id_embed = nn.Embedding(cfg.num_agents, cfg.slot_dim)

        history_layer = nn.TransformerEncoderLayer(
            d_model=cfg.slot_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer,
            num_layers=cfg.num_history_layers,
            enable_nested_tensor=False,
        )

        self.role_queries = nn.Parameter(torch.randn(1, self.NUM_ROLES, cfg.slot_dim) * 0.02)
        slot_layer = nn.TransformerDecoderLayer(
            d_model=cfg.slot_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(slot_layer, num_layers=cfg.num_slot_layers)
        self.out_norm = nn.LayerNorm(cfg.slot_dim)

        self.privileged_aux_role_indices = {
            name: self.ROLE_NAMES.index(cfg.privileged_aux_roles[name]) for name in cfg.privileged_aux_dims
        }
        self.privileged_aux_heads = nn.ModuleDict(
            {
                name: MLP(cfg.slot_dim, cfg.hidden_dim, int(out_dim), depth=3, dropout=cfg.dropout)
                for name, out_dim in cfg.privileged_aux_dims.items()
            }
        )

    def _validate_inputs(
        self,
        local_history: torch.Tensor,
        history_mask: torch.Tensor,
        agent_id: torch.Tensor,
    ) -> None:
        cfg = self.cfg
        if local_history.ndim != 3 or tuple(local_history.shape[1:]) != (cfg.history, cfg.local_dim):
            raise ValueError(f"local_history must have shape [B, {cfg.history}, {cfg.local_dim}]")
        if history_mask.shape != local_history.shape[:2]:
            raise ValueError(f"history_mask must have shape [B, {cfg.history}]")
        if agent_id.shape != (local_history.shape[0],):
            raise ValueError("agent_id must have shape [B]")
        if agent_id.dtype not in (torch.int32, torch.int64):
            raise TypeError("agent_id must use an integer dtype")
        if history_mask.dtype != torch.bool:
            raise TypeError("history_mask must be boolean (True=observed, False=padding)")
        if not history_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one observed history timestep")
        if (agent_id < 0).any() or (agent_id >= cfg.num_agents).any():
            raise ValueError("agent_id lies outside the configured agent embedding range")
        if not torch.isfinite(local_history[history_mask]).all():
            raise ValueError("observed local_history values must be finite")

    def _prepare_object_inputs(
        self,
        local_history: torch.Tensor,
        history_mask: torch.Tensor,
        object_observation: torch.Tensor | None,
        object_valid: torch.Tensor | None,
        object_age: torch.Tensor | None,
        object_confidence: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        cfg = self.cfg
        B = local_history.shape[0]
        device = local_history.device
        dtype = local_history.dtype

        provided = (object_observation, object_valid, object_age, object_confidence)
        if object_observation is None:
            if any(value is not None for value in provided[1:]):
                raise ValueError("object metadata cannot be supplied without object_observation")
            if cfg.object_dim > 0:
                object_values: torch.Tensor | None = torch.zeros(
                    B, cfg.history, cfg.object_dim, device=device, dtype=dtype
                )
            else:
                object_values = None
            valid = torch.zeros(B, cfg.history, device=device, dtype=torch.bool)
            age = torch.full((B, cfg.history), cfg.max_object_age, device=device, dtype=dtype)
            confidence = torch.zeros(B, cfg.history, device=device, dtype=dtype)
        else:
            if cfg.object_dim == 0:
                raise ValueError("object_observation requires config.object_dim > 0")
            if object_valid is None or object_age is None or object_confidence is None:
                raise ValueError("object_observation requires valid, age, and confidence metadata")
            if object_observation.shape != (B, cfg.history, cfg.object_dim):
                raise ValueError(f"object_observation must have shape [B, {cfg.history}, {cfg.object_dim}]")
            if object_valid.shape != (B, cfg.history) or object_valid.dtype != torch.bool:
                raise TypeError(f"object_valid must be boolean with shape [B, {cfg.history}]")
            if object_age.shape != (B, cfg.history) or object_confidence.shape != (B, cfg.history):
                raise ValueError(f"object_age/confidence must have shape [B, {cfg.history}]")
            age = object_age.to(device=device, dtype=dtype)
            confidence = object_confidence.to(device=device, dtype=dtype)
            valid = object_valid.to(device=device) & history_mask
            if not torch.isfinite(age).all() or (age < 0).any():
                raise ValueError("object_age must be finite and non-negative")
            if not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
                raise ValueError("object_confidence must be finite and in [0, 1]")
            available = (confidence > 0) & history_mask
            if available.any() and not torch.isfinite(object_observation[available]).all():
                raise ValueError("available object observations must be finite")
            object_values = torch.where(
                available.unsqueeze(-1),
                object_observation.to(device=device, dtype=dtype) * confidence.unsqueeze(-1),
                torch.zeros((), device=device, dtype=dtype),
            )

        # Padded timesteps convey no missingness information.  Invalid object
        # measurements retain age/confidence and may carry an attenuated stale
        # estimate; confidence=0 is the unambiguous fully-missing case.
        valid = valid & history_mask
        age = torch.where(history_mask, age.clamp_max(cfg.max_object_age), torch.zeros_like(age))
        confidence = torch.where(history_mask, confidence, torch.zeros_like(confidence))
        age_scale = torch.log1p(age) / torch.log1p(age.new_tensor(cfg.max_object_age))
        status = torch.stack([valid.to(dtype), age_scale, confidence], dim=-1)
        return object_values, status

    def forward(
        self,
        local_history: torch.Tensor,
        history_mask: torch.Tensor,
        agent_id: torch.Tensor,
        *,
        object_observation: torch.Tensor | None = None,
        object_valid: torch.Tensor | None = None,
        object_age: torch.Tensor | None = None,
        object_confidence: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        self._validate_inputs(local_history, history_mask, agent_id)
        object_values, object_status = self._prepare_object_inputs(
            local_history,
            history_mask,
            object_observation,
            object_valid,
            object_age,
            object_confidence,
        )

        masked_local = torch.where(
            history_mask.unsqueeze(-1), local_history, torch.zeros((), device=local_history.device, dtype=local_history.dtype)
        )
        agent_context = self.agent_id_embed(agent_id)
        memory = self.local_projection(masked_local)
        memory = memory + self.object_status_projection(object_status)
        if self.object_projection is not None and object_values is not None:
            memory = memory + self.object_projection(object_values)
        memory = memory + self.temporal_embedding + agent_context.unsqueeze(1)
        memory = self.history_encoder(memory, src_key_padding_mask=~history_mask)

        role_queries = self.role_queries.expand(local_history.shape[0], -1, -1) + agent_context.unsqueeze(1)
        slots = self.out_norm(
            self.slot_decoder(
                role_queries,
                memory,
                memory_key_padding_mask=~history_mask,
            )
        )
        privileged_predictions = {
            name: head(slots[:, self.privileged_aux_role_indices[name]])
            for name, head in self.privileged_aux_heads.items()
        }

        return {
            "slots": slots,
            "self_slot": slots[:, self.SELF_INDEX],
            "object_belief_slot": slots[:, self.OBJECT_INDEX],
            "teammate_belief_slot": slots[:, self.TEAMMATE_INDEX],
            "task_context_slot": slots[:, self.TASK_INDEX],
            "privileged_predictions": privileged_predictions,
        }

    @torch.no_grad()
    def encode_slots(
        self,
        local_history: torch.Tensor,
        history_mask: torch.Tensor,
        agent_id: torch.Tensor,
        **object_inputs: torch.Tensor,
    ) -> Dict[str, object]:
        self.eval()
        return self.forward(local_history, history_mask, agent_id, **object_inputs)


def compute_local_belief_auxiliary_losses(
    model: LocalBeliefSlotEncoder,
    batch: Mapping[str, torch.Tensor],
    privileged_targets: Mapping[str, torch.Tensor],
    *,
    privileged_masks: Mapping[str, torch.Tensor] | None = None,
    privileged_weights: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Train optional privileged heads without feeding targets into slots.

    Targets are expected to be normalized continuous simulation labels.  They
    are consumed only after ``model.forward`` has completed.
    """

    out = model(
        batch["local_history"],
        batch["history_mask"],
        batch["agent_id"],
        object_observation=batch.get("object_observation"),
        object_valid=batch.get("object_valid"),
        object_age=batch.get("object_age"),
        object_confidence=batch.get("object_confidence"),
    )
    predictions = out["privileged_predictions"]
    assert isinstance(predictions, dict)
    total = out["slots"].sum() * 0.0
    losses: Dict[str, torch.Tensor] = {}
    masks = privileged_masks or {}
    weights = privileged_weights or {}

    unknown = set(privileged_targets) - set(predictions)
    if unknown:
        raise KeyError(f"no configured privileged auxiliary head(s): {sorted(unknown)}")
    for name, target in privileged_targets.items():
        prediction = predictions[name]
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        if target.shape != prediction.shape:
            raise ValueError(f"privileged target '{name}' must have shape {tuple(prediction.shape)}")
        element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
        if name in masks:
            mask = masks[name].to(device=prediction.device, dtype=prediction.dtype)
            while mask.ndim < element_loss.ndim:
                mask = mask.unsqueeze(-1)
            try:
                mask = mask.expand_as(element_loss)
            except RuntimeError as exc:
                raise ValueError(f"privileged mask '{name}' is not broadcastable to its target") from exc
            denominator = mask.sum().clamp_min(1.0)
            head_loss = (element_loss * mask).sum() / denominator
        else:
            head_loss = element_loss.mean()
        weight = float(weights.get(name, 1.0))
        if weight < 0:
            raise ValueError("privileged auxiliary weights cannot be negative")
        total = total + weight * head_loss
        losses[f"loss_aux_{name}"] = head_loss.detach()

    return {"loss": total, "slots": out["slots"].detach(), **losses}
