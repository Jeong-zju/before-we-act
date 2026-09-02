"""Strict-local DuoBench runtime for a real PredictiveTeamBeliefPolicy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.team_belief.predictive_core import TeamBeliefConfig
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
)
from .bcore_data import (
    BCORE_DEPLOYMENT_FORMAT,
    DUO_CARE_MEMORY_SEMANTICS,
    DUO_CARE_MEMORY_TOKENS,
    DUO_CARE_MEMORY_WIDTH,
)
from .data import (
    ACTION_DIM,
    ACTION_HORIZON,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    STATE_DIM,
    TASKS,
    resize_rgb_batch,
)
from .runtime import (
    DuoB0HRuntime,
    _AbsoluteEnsemble,
    _arm_qpos,
    _controller_equivalent_payload,
    _frames,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


@dataclass
class BcorePrediction:
    prediction: np.ndarray
    base_prediction: np.ndarray
    qpos: np.ndarray
    current_visual_raw: torch.Tensor
    belief_mu: np.ndarray
    belief_event_memory: np.ndarray
    belief_event_mask: np.ndarray
    belief_sigma: np.ndarray
    belief_reliability: np.ndarray
    residual_gate: np.ndarray
    residual: np.ndarray

    @property
    def memory(self) -> np.ndarray:
        """Complete CARE memory (belief tokens followed by event slots)."""

        return np.concatenate((self.belief_mu, self.belief_event_memory), axis=1)

    @property
    def memory_mask(self) -> np.ndarray:
        """Validity mask matching :attr:`memory`, including sparse events."""

        belief_mask = np.ones(self.belief_mu.shape[:2], dtype=bool)
        return np.concatenate((belief_mask, self.belief_event_mask), axis=1)

    def validate_memory_contract(self) -> None:
        """Fail closed if a checkpoint/runtime drops event-memory slots."""

        if self.belief_mu.ndim != 3 or self.belief_mu.shape[-1] != DUO_CARE_MEMORY_WIDTH:
            raise ValueError(f"invalid B-core belief shape: {self.belief_mu.shape}")
        if self.belief_event_memory.shape != (
            self.belief_mu.shape[0],
            DUO_CARE_MEMORY_TOKENS - self.belief_mu.shape[1],
            DUO_CARE_MEMORY_WIDTH,
        ):
            raise ValueError(
                "invalid B-core event-memory shape: "
                f"{self.belief_event_memory.shape}"
            )
        if self.belief_event_mask.shape != self.belief_event_memory.shape[:2]:
            raise ValueError("B-core event-memory validity mask differs")
        if self.belief_event_mask.dtype != np.bool_:
            raise TypeError("B-core event-memory validity mask must be boolean")
        if not np.isfinite(self.belief_event_memory).all():
            raise ValueError("B-core event memory contains a non-finite value")
        if self.memory.shape[1] != DUO_CARE_MEMORY_TOKENS:
            raise ValueError("B-core CARE memory token count differs")
        if self.memory_mask.shape != (
            self.belief_mu.shape[0], DUO_CARE_MEMORY_TOKENS
        ):
            raise ValueError("B-core CARE memory mask shape differs")


def validate_bcore_payload(saved: Mapping[str, Any]) -> Mapping[str, Any]:
    format_value = saved.get("format") or saved.get("format_version")
    if format_value != BCORE_DEPLOYMENT_FORMAT:
        raise ValueError("checkpoint is not a Duo PredictiveTeamBeliefPolicy deployment")
    required = {
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        # ``CARE`` is the method family and must be recorded explicitly; it
        # must never be inferred from (or conflated with) policy_family.
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "DuoBench",
        "vision_backbone": "dinov3_vitb16_frozen",
        "action_encoding": "absolute_joint7_binary_gripper1",
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "teacher_present": False,
        "strict_dino_contract": True,
    }
    config = saved.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Duo B-core deployment has no config mapping")
    for key, expected in required.items():
        if key in {
            "policy_family",
            "reference_policy_family",
            "method_family",
            "strict_dino_contract",
        }:
            if saved.get(key) != expected or config.get(key) != expected:
                raise ValueError(
                    f"Duo B-core checkpoint differs at {key}: "
                    f"top={saved.get(key)!r}, config={config.get(key)!r}, "
                    f"expected={expected!r}"
                )
            continue
        value = saved.get(key, config.get(key))
        if value != expected:
            raise ValueError(
                f"Duo B-core checkpoint differs at {key}: {value!r} != {expected!r}"
            )
    preprocess_id = config.get("image_preprocess_id")
    if preprocess_id != IMAGE_PREPROCESS_ID:
        raise ValueError(
            "Duo B-core checkpoint does not carry the registered image preprocess id"
        )
    if config.get("dino_normalization_id") != DINO_NORMALIZATION_ID:
        raise ValueError(
            "Duo B-core checkpoint does not carry the registered DINO normalization id"
        )
    state = saved.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Duo B-core deployment has no policy state")
    keys = tuple(str(key) for key in state)
    required_prefixes = (
        "vision.",
        "history_encoder.",
        "decoder.",
        "hidden_residual.",
        "belief_core.",
        "direct_belief_residual.",
    )
    missing = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in keys)
    ]
    if missing:
        raise ValueError(f"Duo B-core state is missing architecture keys: {missing}")
    if any(str(key).startswith("belief_core.teacher_branch.") for key in keys):
        raise ValueError("privileged B-core teacher weights are present in deployment")
    if not saved.get("source_b0h_checkpoint_sha256"):
        raise ValueError("Duo B-core deployment has no B0-H provenance hash")
    lowered = " ".join(str(value).lower() for value in config.values())
    if any(marker in lowered for marker in ("convnext", "resnet18", "actpolicy")):
        raise ValueError("legacy ACT/ConvNeXt markers are forbidden in Duo B-core")
    return config


class DuoBcoreRuntime(DuoB0HRuntime):
    """One shared B-core module, two independent local histories."""

    model: PredictiveTeamBeliefPolicy

    def __init__(
        self,
        model: PredictiveTeamBeliefPolicy,
        stats: Mapping[str, torch.Tensor | np.ndarray],
        *,
        device: torch.device,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        ensemble_decay: float = 0.01,
    ) -> None:
        super().__init__(
            model,  # type: ignore[arg-type]
            stats,
            device=device,
            image_height=image_height,
            image_width=image_width,
            ensemble_decay=ensemble_decay,
        )
        self.base_ensemble = _AbsoluteEnsemble(self.arms, ensemble_decay)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dino_model: str | None = None,
        ensemble_decay: float = 0.01,
    ) -> "DuoBcoreRuntime":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping):
            raise ValueError("Duo B-core checkpoint is not a mapping")
        config = validate_bcore_payload(saved)
        values = dict(config.get("n2_config", {}))
        if not values:
            raise ValueError("Duo B-core checkpoint has no N2 config")
        for key in ("future_offsets_steps", "future_offsets_seconds"):
            if key in values:
                values[key] = tuple(values[key])
        belief_config = TeamBeliefConfig(**values)
        model_name = str(dino_model or config.get("dino_model") or "")
        if not model_name:
            raise ValueError("Duo B-core checkpoint has no DINO model path")
        model = PredictiveTeamBeliefPolicy(
            belief_config,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            horizon=int(config.get("horizon", ACTION_HORIZON)),
            d_model=int(config.get("d_model", 384)),
            enc_layers=int(config.get("enc_layers", 4)),
            dec_layers=int(config.get("dec_layers", 7)),
            roles=int(config.get("roles", 4)),
            role_rank=int(config.get("role_rank", 32)),
            history_layers=int(config.get("history_layers", 2)),
            dino_model=model_name,
            image_height=int(config.get("image_height", DEFAULT_IMAGE_HEIGHT)),
            image_width=int(config.get("image_width", DEFAULT_IMAGE_WIDTH)),
            strict_dino_contract=True,
            include_teacher=False,
            residual_safety=config.get("residual_safety", {"enabled": False}),
        )
        model.load_state_dict(saved["model"], strict=True)
        return cls(
            model,
            saved["stats"],
            device=torch.device(device),
            image_height=int(config.get("image_height", DEFAULT_IMAGE_HEIGHT)),
            image_width=int(config.get("image_width", DEFAULT_IMAGE_WIDTH)),
            ensemble_decay=ensemble_decay,
        )

    def reset(self, task: str) -> None:
        super().reset(task)
        self.base_ensemble.reset()

    @torch.inference_mode()
    def predict_chunks(
        self,
        observation: Mapping[str, Any],
        task: str | None = None,
        *,
        belief_enabled: bool = True,
        append_observation: bool = True,
    ) -> BcorePrediction:
        task = task or self.task
        if task is None:
            raise ValueError("runtime.reset(task) is required")
        if task not in TASKS:
            raise ValueError(task)
        if self.task != task:
            self.reset(task)
        heads, wrists, qposes = [], [], []
        for arm in self.arms:
            head, wrist = _frames(observation, arm)
            heads.append(resize_rgb_batch(head, self.image_height, self.image_width))
            wrists.append(resize_rgb_batch(wrist, self.image_height, self.image_width))
            qposes.append(_arm_qpos(observation, arm))
        head_tensor = torch.stack(heads).to(self.device).float().div_(255)
        wrist_tensor = torch.stack(wrists).to(self.device).float().div_(255)
        qraw = torch.as_tensor(np.stack(qposes), device=self.device, dtype=torch.float32)
        qnorm = (qraw - self.q_mean) / self.q_std
        temporal = self._history_batch(qnorm, task)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            output = self.model(
                head_tensor,
                wrist_tensor,
                **temporal,
                belief_enabled=belief_enabled,
            )
        if append_observation:
            self._append_observation(output.current_visual_raw, qnorm)
        prediction = (
            output.prediction.float() * self.a_std + self.a_mean
        ).cpu().numpy()
        base = (
            output.base_prediction.float() * self.a_std + self.a_mean
        ).cpu().numpy()
        result = BcorePrediction(
            prediction=prediction.astype(np.float32),
            base_prediction=base.astype(np.float32),
            qpos=qraw.cpu().numpy().astype(np.float32),
            current_visual_raw=output.current_visual_raw.detach().float().cpu(),
            belief_mu=output.belief.mu.detach().float().cpu().numpy(),
            belief_event_memory=(
                output.belief.event_memory.detach().float().cpu().numpy()
            ),
            belief_event_mask=(
                output.belief.event_mask.detach().bool().cpu().numpy()
            ),
            belief_sigma=output.belief.sigma.detach().float().cpu().numpy(),
            belief_reliability=output.belief.reliability.detach().float().cpu().numpy(),
            residual_gate=output.residual_gate.detach().float().cpu().numpy(),
            residual=(output.belief_residual.float() * self.a_std)
            .detach()
            .cpu()
            .numpy(),
        )
        result.validate_memory_contract()
        return result

    def append_absolute_action(self, absolute: np.ndarray) -> None:
        value = torch.as_tensor(absolute, device=self.device, dtype=torch.float32)
        if tuple(value.shape) != (2, ACTION_DIM) or not torch.isfinite(value).all():
            raise ValueError("Duo B-core executed action must be finite [2,8]")
        normalized = (value - self.a_mean) / self.a_std
        self._append_action({arm: normalized[row] for row, arm in enumerate(self.arms)})
        self.step_index += 1

    @torch.inference_mode()
    def act(
        self,
        observation: Mapping[str, Any],
        task: str | None = None,
        *,
        action_spaces: Mapping[str, Any] | None = None,
        belief_enabled: bool = True,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
        task = task or self.task
        prediction = self.predict_chunks(
            observation, task, belief_enabled=belief_enabled, append_observation=True
        )
        chunks = prediction.prediction if belief_enabled else prediction.base_prediction
        selected = self.ensemble.add_and_select(self.step_index, chunks)
        payload, absolute = _controller_equivalent_payload(
            selected, self.arms, action_spaces=action_spaces
        )
        self.append_absolute_action(absolute)
        diagnostics = {
            "task": task,
            "step": self.step_index - 1,
            "policy_family": "PredictiveTeamBeliefPolicy",
            "method_family": "CARE",
            "belief_enabled": bool(belief_enabled),
            "action_encoding": "absolute_joint7_binary_gripper1",
            "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
            "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
            "rcs_api_limits_used_for_canonicalization": False,
            "strictly_decentralized": True,
            "residual_gate_mean": float(np.mean(prediction.residual_gate)),
            "residual_norm_mean": float(
                np.linalg.norm(prediction.residual, axis=-1).mean()
            ),
            "belief_reliability_mean": float(
                np.mean(prediction.belief_reliability)
            ),
            "belief_sigma_mean": float(np.mean(prediction.belief_sigma)),
            "belief_event_slots_valid": int(
                np.count_nonzero(prediction.belief_event_mask)
            ),
            "care_memory_tokens": int(prediction.memory.shape[1]),
            "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
            "source_frequency_hz": self.model.team_belief_config.source_frequency_hz,
            "future_offsets_steps": list(
                self.model.team_belief_config.future_offsets_steps
            ),
            "future_offsets_seconds": list(
                self.model.team_belief_config.future_offsets_seconds
            ),
        }
        return payload, diagnostics


__all__ = [
    "BcorePrediction",
    "DuoBcoreRuntime",
    "validate_bcore_payload",
]
