"""Registry and shared utilities for isolated R10 perception extensions.

This parent module intentionally registers no concrete candidate.  Each R10
branch adds exactly one implementation and registers one ``bridge.kind``.
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

try:
    from .bwa_contracts import CorePerceptionExtension
except ImportError:
    from bwa_contracts import CorePerceptionExtension


BRIDGE_REGISTRY: dict[str, type["TrainablePerceptionExtension"]] = {}
EXPECTED_PARENT_SHA256 = "061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d"
FORMAL_BATCH = 40
FORMAL_SEED = 20260803


def load_r10_config(path) -> dict[str, Any]:
    """Validate shared identity, budget, seed and precision without vision imports."""
    from pathlib import Path

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("R10 config must be a mapping")
    required = {
        "schema_version", "candidate_id", "parent_commit", "checkpoint_sha256",
        "bridge", "training", "loss_weights", "calibration", "intervention",
    }
    if set(config) != required:
        raise ValueError(
            f"R10 config keys differ: missing={sorted(required - set(config))}, "
            f"extra={sorted(set(config) - required)}"
        )
    if config["schema_version"] != 1 or config["candidate_id"] not in {"p0", "p1", "p2", "p3"}:
        raise ValueError("unsupported R10 config identity")
    if config["checkpoint_sha256"] != EXPECTED_PARENT_SHA256:
        raise ValueError("R10 parent checkpoint hash drift")
    training = config["training"]
    locked = {
        "batch_size": FORMAL_BATCH,
        "seed": FORMAL_SEED,
        "screen_updates": 10_000,
        "selection_updates": 30_000,
        "precision": "bfloat16",
    }
    failed = [key for key, value in locked.items() if training.get(key) != value]
    if failed:
        raise ValueError(f"locked training protocol drift: {failed}")
    return config


def register_bridge(kind: str) -> Callable[[type["TrainablePerceptionExtension"]], type["TrainablePerceptionExtension"]]:
    """Register one fail-closed R10 bridge kind."""
    if not kind or any(character.isspace() for character in kind):
        raise ValueError(f"invalid bridge kind: {kind!r}")

    def decorate(cls: type["TrainablePerceptionExtension"]):
        if kind in BRIDGE_REGISTRY:
            raise ValueError(f"duplicate bridge kind: {kind}")
        if cls.kind != kind:
            raise ValueError(f"class kind {cls.kind!r} differs from registry key {kind!r}")
        BRIDGE_REGISTRY[kind] = cls
        return cls

    return decorate


class TrainablePerceptionExtension(CorePerceptionExtension):
    """Common zero-gate behavior and generic auxiliary-loss protocol."""

    kind = "unregistered"

    def __init__(self, *, d_model: int, gate_max: float) -> None:
        super().__init__()
        if not 0.0 < float(gate_max) <= 1.0:
            raise ValueError("gate_max must be in (0, 1]")
        self.d_model = int(d_model)
        self.gate_max = float(gate_max)
        self.raw_perception_gate = nn.Parameter(torch.zeros(()))

    @property
    def perception_gate(self) -> torch.Tensor:
        return self.raw_perception_gate

    @property
    def requires_history_views(self) -> bool:
        return False

    @property
    def future_feature_horizons(self) -> tuple[int, ...]:
        return ()

    def clamp_gate_(self) -> None:
        bound = float(torch.atanh(torch.tensor(self.gate_max)).item())
        with torch.no_grad():
            self.raw_perception_gate.clamp_(-bound, bound)

    def training_losses(
        self,
        auxiliary: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        """Return named scalar losses without exposing targets to ``forward``."""
        del targets
        return {
            name.removeprefix("loss_"): value
            for name, value in auxiliary.items()
            if name.startswith("loss_") and isinstance(value, torch.Tensor)
        }

    def apply_diagnostic_intervention(
        self, name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Candidate-specific interventions override this fail-closed hook."""
        if name in ("", "normal"):
            return tensor
        raise ValueError(f"unsupported {self.kind} intervention: {name}")

    @abstractmethod
    def forward(self, views, state_vec, deployment_context):
        raise NotImplementedError


def build_perception_extension(config: Mapping[str, Any]) -> TrainablePerceptionExtension:
    """Instantiate the one bridge kind present in the current branch."""
    kind = str(config.get("kind", ""))
    if kind not in BRIDGE_REGISTRY:
        raise ValueError(
            f"bridge.kind {kind!r} is not registered on this branch; "
            f"available={sorted(BRIDGE_REGISTRY)}"
        )
    return BRIDGE_REGISTRY[kind](config=dict(config))


def zero_linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def normalized_uv(tokens: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return deterministic 2-D coordinates without assuming aligned views."""
    height = max(1, int(round(tokens ** 0.5)))
    while height > 1 and tokens % height:
        height -= 1
    width = tokens // height
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(tokens, 2)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=-2)
    if mask.shape != tokens.shape[:-1]:
        raise ValueError(f"mask shape {tuple(mask.shape)} != {tuple(tokens.shape[:-1])}")
    weights = mask.to(dtype=tokens.dtype)
    denominator = weights.sum(dim=-1, keepdim=True)
    if bool((denominator == 0).any()):
        raise ValueError("all-masked perception input is forbidden")
    return (tokens * weights.unsqueeze(-1)).sum(dim=-2) / denominator


def cosine_regression(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean()
