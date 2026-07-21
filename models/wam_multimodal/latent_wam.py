"""Phase M1 visual-conditioned latent World-Action Model wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from models.wam import RWMARRolloutPredictions, RWMARWorldModel
from models.wam.api import WorldModelSequenceInputs
from models.wam_multimodal.latent_world_head import (
    ActionConditionedFutureLatentHead,
    CANONICAL_FUTURE_HORIZONS,
    FutureLatentHeadConfig,
)
from models.wam_multimodal.token_resampler import (
    PerceiverResampler,
    PerceiverResamplerConfig,
)
from models.wam_multimodal.vision_encoder import VisionEncoderOutput


CapacityControl = Literal["none", "future_head", "action_mlp"]


@dataclass(frozen=True)
class LatentWAMConfig:
    """Serializable modality, fusion, and equal-capacity ablation contract."""

    task_vocabulary: tuple[str, ...]
    use_state: bool = True
    use_vision: bool = True
    capacity_control: CapacityControl = "future_head"
    action_dim: int = 8
    task_embedding_dim: int = 128
    fusion_hidden_dim: int = 2048
    action_mlp_hidden_dim: int = 1024
    future_hidden_dim: int = 2048
    future_action_hidden_dim: int = 512
    future_latent_dim: int = 512
    visual_skip_initial_scale: float = 0.1
    future_horizons: tuple[int, ...] = CANONICAL_FUTURE_HORIZONS
    resampler: PerceiverResamplerConfig = field(
        default_factory=PerceiverResamplerConfig
    )

    def __post_init__(self) -> None:
        vocabulary = tuple(str(value) for value in self.task_vocabulary)
        if not vocabulary or any(not value for value in vocabulary):
            raise ValueError("task_vocabulary must contain non-empty task ids")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("task_vocabulary must be unique")
        if not self.use_state and not self.use_vision:
            raise ValueError("at least one of use_state/use_vision must be enabled")
        if self.capacity_control not in {"none", "future_head", "action_mlp"}:
            raise ValueError(
                "capacity_control must be 'none', 'future_head', or 'action_mlp'"
            )
        for name in (
            "action_dim",
            "task_embedding_dim",
            "fusion_hidden_dim",
            "action_mlp_hidden_dim",
            "future_hidden_dim",
            "future_action_hidden_dim",
            "future_latent_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.visual_skip_initial_scale) <= 1.0:
            raise ValueError("visual_skip_initial_scale must be in (0,1]")
        horizons = tuple(int(value) for value in self.future_horizons)
        if horizons != CANONICAL_FUTURE_HORIZONS:
            raise ValueError(
                f"Phase M1 future_horizons must be {CANONICAL_FUTURE_HORIZONS}"
            )
        if (
            self.resampler.width != 512
            or self.resampler.num_latents != 16
            or self.resampler.num_layers != 3
        ):
            raise ValueError(
                "Phase M1 requires a 3-layer, width-512, 16-token resampler"
            )
        if self.resampler.input_dim != self.future_latent_dim:
            raise ValueError(
                "resampler input_dim and future_latent_dim must match the "
                "frozen visual teacher width"
            )
        object.__setattr__(self, "task_vocabulary", vocabulary)
        object.__setattr__(self, "future_horizons", horizons)

    @property
    def use_future_head(self) -> bool:
        return self.capacity_control == "future_head"

    @property
    def variant(self) -> str:
        modality = (
            "state_vision"
            if self.use_state and self.use_vision
            else "state_only"
            if self.use_state
            else "vision_only"
        )
        return f"{modality}_{self.capacity_control}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_vocabulary"] = list(self.task_vocabulary)
        payload["future_horizons"] = list(self.future_horizons)
        payload["variant"] = self.variant
        payload["use_future_head"] = self.use_future_head
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentWAMConfig":
        values = dict(payload)
        values.pop("variant", None)
        values.pop("use_future_head", None)
        raw_resampler = values.get("resampler")
        if isinstance(raw_resampler, Mapping):
            values["resampler"] = PerceiverResamplerConfig(**dict(raw_resampler))
        if "task_vocabulary" in values:
            values["task_vocabulary"] = tuple(values["task_vocabulary"])
        if "future_horizons" in values:
            values["future_horizons"] = tuple(values["future_horizons"])
        return cls(**values)


@dataclass(frozen=True)
class LatentWAMEncoding:
    """Deployable state/vision representation consumed by the action flow."""

    planning_features: Tensor
    state_planning_features: Tensor
    visual_tokens: Tensor
    visual_summary: Tensor
    teacher_current_pooled_latent: Tensor
    task_embedding: Tensor
    hidden: Tensor | None
    current_state: Tensor | None


@dataclass(frozen=True)
class LatentWAMOutput:
    """Trainer-facing joint output with no future-image input surface."""

    encoding: LatentWAMEncoding
    world_predictions: RWMARRolloutPredictions | None
    future_visual_latents: Tensor | None
    future_horizons: tuple[int, ...]


class _PlanningFeatureFusion(nn.Module):
    def __init__(
        self,
        *,
        planning_dim: int,
        visual_dim: int,
        task_dim: int,
        hidden_dim: int,
        visual_skip_initial_scale: float,
    ) -> None:
        super().__init__()
        input_dim = planning_dim + visual_dim + task_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, planning_dim),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, std=1e-3)
        nn.init.zeros_(final.bias)
        # The deep residual remains the expressive fusion route, while this
        # shallow projection prevents spatial RGB from being attenuated by
        # another three-layer bottleneck before reaching the action flow.
        self.visual_skip_norm = nn.LayerNorm(visual_dim)
        self.visual_skip = nn.Linear(visual_dim, planning_dim, bias=False)
        nn.init.xavier_uniform_(self.visual_skip.weight)
        self.visual_skip_gain = nn.Parameter(
            torch.tensor(float(visual_skip_initial_scale))
        )

    def forward(
        self,
        state_features: Tensor,
        visual_summary: Tensor,
        task_embedding: Tensor,
    ) -> Tensor:
        residual = self.network(
            torch.cat((state_features, visual_summary, task_embedding), dim=-1)
        )
        visual_skip = self.visual_skip(self.visual_skip_norm(visual_summary))
        return state_features + residual + self.visual_skip_gain * visual_skip


class _CapacityMatchedActionMLP(nn.Module):
    """Fully active residual MLP with exactly the future-head capacity.

    Widths are solved deterministically from the required parameter count.
    Every matched parameter therefore participates in ``forward``; there is no
    inert padding tensor that could make the nominal and active capacities
    disagree.
    """

    def __init__(
        self,
        *,
        planning_dim: int,
        visual_dim: int,
        task_dim: int,
        hidden_dim: int,
        target_parameter_count: int,
    ) -> None:
        super().__init__()
        input_dim = planning_dim + visual_dim + task_dim
        widths = _exact_capacity_widths(
            input_dim,
            planning_dim,
            int(target_parameter_count),
            preferred_width=int(hidden_dim),
        )
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous = input_dim
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.GELU()))
            previous = width
        layers.append(nn.Linear(previous, planning_dim))
        self.network = nn.Sequential(*layers)
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, std=1e-3)
        nn.init.zeros_(final.bias)
        functional_count = sum(
            parameter.numel() for parameter in self.network.parameters()
        )
        if functional_count != int(target_parameter_count):
            raise RuntimeError("exact action-MLP capacity solver returned a mismatch")
        self.hidden_widths = widths
        self.functional_parameter_count = functional_count
        self.target_parameter_count = int(target_parameter_count)

    @property
    def padding_parameter_count(self) -> int:
        return 0

    def forward(
        self,
        planning_features: Tensor,
        visual_summary: Tensor,
        task_embedding: Tensor,
    ) -> Tensor:
        residual = self.network(
            torch.cat((planning_features, visual_summary, task_embedding), dim=-1)
        )
        return planning_features + residual


@lru_cache(maxsize=32)
def _exact_capacity_widths(
    input_dim: int,
    output_dim: int,
    target_parameter_count: int,
    *,
    preferred_width: int,
) -> tuple[int, ...]:
    """Find a compact two/three-hidden-layer MLP with an exact parameter sum."""

    input_dim = int(input_dim)
    output_dim = int(output_dim)
    target = int(target_parameter_count)
    preferred = int(preferred_width)
    if min(input_dim, output_dim, target, preferred) <= 0:
        raise ValueError("capacity dimensions and target must be positive")
    # LayerNorm contributes 2*input_dim.  For two hidden widths the remaining
    # expression is linear in h2 once h1 is fixed, so the complete exact search
    # is inexpensive and lets us prefer a balanced functional network.
    constant = 2 * input_dim + output_dim
    two_layer: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    maximum_h1 = max(0, (target - constant - 1) // (input_dim + 1))
    for h1 in range(1, maximum_h1 + 1):
        remaining = target - constant - (input_dim + 1) * h1
        denominator = h1 + output_dim + 1
        if remaining > 0 and remaining % denominator == 0:
            h2 = remaining // denominator
            if h2 > 0:
                score = (
                    max(h1, h2),
                    abs(h1 - h2),
                    abs(h1 - preferred) + abs(h2 - preferred),
                    h1,
                    h2,
                )
                two_layer.append((score, (h1, h2)))
    if two_layer:
        return min(two_layer, key=lambda value: value[0])[1]

    # Equal-width approximation for three hidden layers:
    # target ~= constant + 2*h^2 + (input+output+3)*h.
    coefficient = input_dim + output_dim + 3
    discriminant = coefficient * coefficient + 8 * max(target - constant, 0)
    base = max(
        1,
        round((-coefficient + math.sqrt(discriminant)) / 4.0),
    )
    for radius in (512, 1_024, 2_048):
        lower = max(1, base - radius)
        upper = base + radius
        best: tuple[tuple[int, ...], tuple[int, int, int]] | None = None
        for h1 in range(lower, upper + 1):
            after_h1 = target - constant - (input_dim + 1) * h1
            for h2 in range(lower, upper + 1):
                remaining = after_h1 - (h1 + 1) * h2
                denominator = h2 + output_dim + 1
                if remaining <= 0 or remaining % denominator:
                    continue
                h3 = remaining // denominator
                if h3 <= 0:
                    continue
                score = (
                    max(abs(h1 - base), abs(h2 - base), abs(h3 - base)),
                    abs(h1 - h2) + abs(h2 - h3),
                    abs(h1 - preferred) + abs(h2 - preferred) + abs(h3 - preferred),
                    h1,
                    h2,
                    h3,
                )
                candidate = (score, (h1, h2, h3))
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is not None:
            return best[1]
    raise ValueError(
        "cannot construct a fully active action MLP with the requested exact capacity"
    )


class LatentWAM(nn.Module):
    """Wrap the accepted recurrent world model with visual latent conditioning."""

    def __init__(
        self,
        config: LatentWAMConfig,
        world_model: RWMARWorldModel,
        vision_encoder: nn.Module | None,
    ) -> None:
        super().__init__()
        if world_model.config.action_dim != config.action_dim:
            raise ValueError("Latent WAM action dimension differs from world model")
        if config.use_vision and vision_encoder is None:
            raise ValueError("use_vision=True requires a frozen vision encoder")
        if vision_encoder is not None and int(
            getattr(vision_encoder, "output_dim", -1)
        ) != int(config.resampler.input_dim):
            raise ValueError("visual teacher output_dim must match resampler input_dim")
        self.config = config
        self.world_model = world_model
        self.vision_encoder = vision_encoder
        self.resampler = PerceiverResampler(config.resampler)
        planning_dim = int(world_model.planning_feature_dim)
        self.task_embedding = nn.Embedding(
            len(config.task_vocabulary), config.task_embedding_dim
        )
        self.fusion = _PlanningFeatureFusion(
            planning_dim=planning_dim,
            visual_dim=config.resampler.width,
            task_dim=config.task_embedding_dim,
            hidden_dim=config.fusion_hidden_dim,
            visual_skip_initial_scale=config.visual_skip_initial_scale,
        )
        future_config = FutureLatentHeadConfig(
            planning_feature_dim=planning_dim,
            action_dim=config.action_dim,
            visual_dim=config.resampler.width,
            latent_dim=config.future_latent_dim,
            action_hidden_dim=config.future_action_hidden_dim,
            hidden_dim=config.future_hidden_dim,
            horizons=config.future_horizons,
        )
        self.future_head: ActionConditionedFutureLatentHead | None = None
        self.action_capacity_mlp: _CapacityMatchedActionMLP | None = None
        self.capacity_target_parameter_count = 0
        if config.capacity_control == "future_head":
            self.future_head = ActionConditionedFutureLatentHead(future_config)
            self.capacity_target_parameter_count = sum(
                parameter.numel() for parameter in self.future_head.parameters()
            )
        elif config.capacity_control == "action_mlp":
            reference = ActionConditionedFutureLatentHead(future_config)
            target = sum(parameter.numel() for parameter in reference.parameters())
            del reference
            self.action_capacity_mlp = _CapacityMatchedActionMLP(
                planning_dim=planning_dim,
                visual_dim=config.resampler.width,
                task_dim=config.task_embedding_dim,
                hidden_dim=config.action_mlp_hidden_dim,
                target_parameter_count=target,
            )
            self.capacity_target_parameter_count = target
        self._task_to_index = {
            task_id: index for index, task_id in enumerate(config.task_vocabulary)
        }

    @property
    def planning_feature_dim(self) -> int:
        return int(self.world_model.planning_feature_dim)

    @property
    def future_horizons(self) -> tuple[int, ...]:
        return self.config.future_horizons

    def task_indices(
        self,
        task_ids: str | Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        values = (task_ids,) if isinstance(task_ids, str) else tuple(task_ids)
        if not values:
            raise ValueError("task_ids cannot be empty")
        unknown = sorted(set(values) - set(self._task_to_index))
        if unknown:
            raise ValueError(f"unknown task ids: {unknown}")
        return torch.tensor(
            [self._task_to_index[value] for value in values],
            dtype=torch.int64,
            device=device,
        )

    def encode(
        self,
        states: Tensor | None,
        past_actions: Tensor | None,
        valid_mask: Tensor | None,
        images: Tensor | None,
        task_index: Tensor,
        *,
        image_valid_mask: Tensor | None = None,
        vision_features: VisionEncoderOutput | None = None,
    ) -> LatentWAMEncoding:
        batch_size = self._validate_task_index(task_index)
        task_embedding = self.task_embedding(task_index)
        hidden: Tensor | None = None
        current_state: Tensor | None = None
        if self.config.use_state:
            if states is None or past_actions is None or valid_mask is None:
                raise ValueError("state-enabled Latent WAM requires complete history")
            history = WorldModelSequenceInputs(
                states=states,
                past_actions=past_actions,
                valid_mask=valid_mask,
            )
            hidden, current_state, state_features = (
                self.world_model.encode_planning_history(history)
            )
            if state_features.shape[0] != batch_size:
                raise ValueError("state history and task_index batch sizes differ")
        else:
            # A vision-only ablation must not even invoke the state encoder.
            state_features = torch.zeros(
                batch_size,
                self.planning_feature_dim,
                device=task_embedding.device,
                dtype=task_embedding.dtype,
            )

        if self.config.use_vision:
            if images is None or self.vision_encoder is None:
                raise ValueError("vision-enabled Latent WAM requires RGB images")
            (
                visual_tokens,
                visual_summary,
                teacher_pooled,
            ) = self._encode_visual_context(
                images,
                batch_size=batch_size,
                image_valid_mask=image_valid_mask,
                vision_features=vision_features,
            )
        else:
            visual_tokens = torch.zeros(
                batch_size,
                self.config.resampler.num_latents,
                self.config.resampler.width,
                device=task_embedding.device,
                dtype=task_embedding.dtype,
            )
            visual_summary = visual_tokens.mean(dim=1)
            teacher_pooled = torch.zeros(
                batch_size,
                self.config.future_latent_dim,
                device=task_embedding.device,
                dtype=task_embedding.dtype,
            )

        target_device = task_embedding.device
        if (
            state_features.device != target_device
            or visual_summary.device != target_device
        ):
            raise TypeError("state, vision, and task inputs must share a device")
        fusion_dtype = next(self.fusion.parameters()).dtype
        state_features = state_features.to(dtype=fusion_dtype)
        visual_tokens = visual_tokens.to(dtype=fusion_dtype)
        visual_summary = visual_summary.to(dtype=fusion_dtype)
        task_embedding = task_embedding.to(dtype=fusion_dtype)
        planning_features = self.fusion(
            state_features,
            visual_summary,
            task_embedding,
        )
        if self.action_capacity_mlp is not None:
            planning_features = self.action_capacity_mlp(
                planning_features,
                visual_summary,
                task_embedding,
            )
        return LatentWAMEncoding(
            planning_features=planning_features,
            state_planning_features=state_features,
            visual_tokens=visual_tokens,
            visual_summary=visual_summary,
            teacher_current_pooled_latent=teacher_pooled,
            task_embedding=task_embedding,
            hidden=hidden,
            current_state=current_state,
        )

    def forward(
        self,
        states: Tensor | None,
        past_actions: Tensor | None,
        valid_mask: Tensor | None,
        images: Tensor | None,
        task_index: Tensor,
        candidate_actions: Tensor,
        *,
        image_valid_mask: Tensor | None = None,
        compute_world_predictions: bool = True,
        compute_future_visual_latents: bool = True,
    ) -> LatentWAMOutput:
        encoding = self.encode(
            states,
            past_actions,
            valid_mask,
            images,
            task_index,
            image_valid_mask=image_valid_mask,
        )
        world_predictions: RWMARRolloutPredictions | None = None
        if self.config.use_state and compute_world_predictions:
            assert encoding.hidden is not None and encoding.current_state is not None
            world_predictions = self.world_model.predict_from_encoded_history(
                encoding.hidden,
                encoding.current_state,
                candidate_actions,
            )
        future_visual_latents: Tensor | None = None
        if self.future_head is not None and compute_future_visual_latents:
            future_visual_latents = self.future_head(
                encoding.planning_features,
                encoding.visual_tokens,
                candidate_actions,
            )
        return LatentWAMOutput(
            encoding=encoding,
            world_predictions=world_predictions,
            future_visual_latents=future_visual_latents,
            future_horizons=self.config.future_horizons,
        )

    def _encode_visual_context(
        self,
        images: Tensor,
        *,
        batch_size: int,
        image_valid_mask: Tensor | None,
        vision_features: VisionEncoderOutput | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized = self._normalize_image_shape(images, batch_size=batch_size)
        _, history, cameras, _, _, _ = normalized.shape
        if image_valid_mask is None:
            frame_valid = torch.ones(
                batch_size,
                history,
                cameras,
                dtype=torch.bool,
                device=normalized.device,
            )
        else:
            if image_valid_mask.shape != (batch_size, history, cameras):
                raise ValueError("image_valid_mask must have shape [B,T,Cam]")
            if image_valid_mask.dtype != torch.bool:
                raise TypeError("image_valid_mask must be boolean")
            if image_valid_mask.device != normalized.device:
                raise TypeError("images and image_valid_mask must share a device")
            frame_valid = image_valid_mask
        if not torch.all(frame_valid.flatten(1).any(dim=1)):
            raise ValueError("each sample requires at least one valid RGB frame")
        assert self.vision_encoder is not None
        teacher = (
            self.vision_encoder(normalized)
            if vision_features is None
            else vision_features
        )
        if teacher.spatial_tokens.shape[:3] != (batch_size, history, cameras):
            raise ValueError("cached vision patch tokens have invalid frame axes")
        if teacher.pooled_latent.shape != (
            batch_size,
            history,
            cameras,
            self.config.future_latent_dim,
        ):
            raise ValueError("cached vision pooled latents have invalid shape")
        if (
            teacher.spatial_tokens.device != normalized.device
            or teacher.pooled_latent.device != normalized.device
        ):
            raise TypeError("cached vision features and images must share a device")
        if teacher.spatial_tokens.requires_grad or teacher.pooled_latent.requires_grad:
            raise ValueError("cached frozen-teacher features must be detached")
        adapted = self.resampler.visual_adapter(
            normalized,
            teacher.spatial_tokens,
            frame_valid,
        )
        visual_tokens = self.resampler(
            adapted.context,
            context_valid_mask=adapted.context_valid_mask,
        )
        pooled_frames = teacher.pooled_latent.reshape(
            batch_size, history * cameras, self.config.future_latent_dim
        )
        pooled_mask = frame_valid.reshape(batch_size, history * cameras, 1)
        teacher_pooled = (pooled_frames * pooled_mask).sum(dim=1) / pooled_mask.sum(
            dim=1
        ).clamp_min(1)
        visual_summary = self.resampler.summarize(
            visual_tokens, adapted.spatial_shortcut
        )
        return visual_tokens, visual_summary, teacher_pooled

    @staticmethod
    def _normalize_image_shape(images: Tensor, *, batch_size: int) -> Tensor:
        if images.shape[0] != batch_size:
            raise ValueError("images and task_index batch sizes differ")
        if images.ndim == 4:
            normalized = images[:, None, None]
        elif images.ndim == 5:
            normalized = images[:, None]
        elif images.ndim == 6:
            normalized = images
        else:
            raise ValueError("images must have shape [B,(T),(Cam),3,H,W]")
        if normalized.shape[-3] != 3:
            raise ValueError("RGB channel dimension must be 3")
        return normalized

    def _validate_task_index(self, task_index: Tensor) -> int:
        if task_index.ndim != 1 or task_index.dtype != torch.int64:
            raise TypeError("task_index must be an int64 tensor with shape [B]")
        if task_index.numel() == 0:
            raise ValueError("task_index cannot be empty")
        if int(task_index.min()) < 0 or int(task_index.max()) >= len(
            self.config.task_vocabulary
        ):
            raise ValueError("task_index contains an unknown task")
        return int(task_index.shape[0])

    def parameter_breakdown(
        self, action_flow: nn.Module | None = None
    ) -> dict[str, int]:
        future_head = _trainable_count(self.future_head)
        action_mlp_total = _trainable_count(self.action_capacity_mlp)
        capacity_padding = (
            self.action_capacity_mlp.padding_parameter_count
            if self.action_capacity_mlp is not None
            else 0
        )
        action_mlp_functional = action_mlp_total - capacity_padding
        breakdown = {
            "world_model": _trainable_count(self.world_model),
            "vision_encoder_trainable": _trainable_count(self.vision_encoder),
            "vision_encoder_frozen": _frozen_count(self.vision_encoder),
            "visual_adapter": _trainable_count(self.resampler.visual_adapter),
            "resampler": _trainable_count(self.resampler)
            - _trainable_count(self.resampler.visual_adapter),
            "task_embedding": _trainable_count(self.task_embedding),
            "fusion": _trainable_count(self.fusion),
            "future_head": future_head,
            "action_mlp_functional": action_mlp_functional,
            "capacity_padding": capacity_padding,
            "action_flow": _trainable_count(action_flow),
        }
        breakdown["total_trainable"] = self.trainable_parameter_count(action_flow)
        breakdown["total_active"] = self.active_parameter_count(action_flow)
        return breakdown

    def trainable_parameter_count(self, action_flow: nn.Module | None = None) -> int:
        return _trainable_count(self) + _trainable_count(action_flow)

    def active_parameter_count(self, action_flow: nn.Module | None = None) -> int:
        count = _trainable_count(self.fusion) + _trainable_count(self.task_embedding)
        if self.config.use_state:
            count += _trainable_count(self.world_model)
        if self.config.use_vision:
            count += _trainable_count(self.resampler)
        count += _trainable_count(self.future_head)
        if self.action_capacity_mlp is not None:
            count += self.action_capacity_mlp.functional_parameter_count
        return count + _trainable_count(action_flow)


def _trainable_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _frozen_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not parameter.requires_grad
    )


__all__ = [
    "CapacityControl",
    "LatentWAM",
    "LatentWAMConfig",
    "LatentWAMEncoding",
    "LatentWAMOutput",
]
