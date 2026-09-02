"""Strict checkpoint loader and scorer runtime for isolated MARS CARE-v2.

The training checkpoint is deliberately not directly deployable.  A v2
deployment checkpoint must carry an OOF admission receipt, the exact reference
policy hash, physical utility scales, and an auditable safety mode.  This keeps
same-corpus diagnostics from accidentally becoming a closed-loop policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from before_we_act.care_belief import CARECalibration
from before_we_act.care_belief_v2 import CAREBeliefV2Config, CAREBeliefV2Head
from before_we_act.care_selector_v2 import CARESelectionV2, select_care_candidate_v2
from before_we_act.care_training_data import sha256_file
from deployment.mars_care.common import TASK_BY_NAME


DEPLOYMENT_FORMAT_VERSION = "before-we-act.care-mars-deployment-checkpoint-v2/1"
TRAINING_FORMAT_VERSION = "before-we-act.care-mars-training-checkpoint-v2/1"
OOF_FORMAT_VERSION = "before-we-act.care-mars-oof-gate/1"
TASK_NAMES = tuple(TASK_BY_NAME)


def _finite_nonnegative(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_true(mapping: Mapping[str, Any], key: str, context: str) -> None:
    if mapping.get(key) is not True:
        raise ValueError(f"CARE v2 {context} does not prove {key}")


def validate_v2_deployment_payload(
    saved: Mapping[str, Any], reference_checkpoint: Path
) -> tuple[CAREBeliefV2Config, CARECalibration, torch.Tensor, torch.Tensor, str]:
    """Validate deployment provenance before allocating the scorer model."""

    if saved.get("format_version") != DEPLOYMENT_FORMAT_VERSION:
        raise ValueError("wrong MARS CARE v2 deployment checkpoint")
    if saved.get("source_training_format_version") != TRAINING_FORMAT_VERSION:
        raise ValueError("CARE v2 deployment source is not a v2 training checkpoint")
    if saved.get("reference_checkpoint_sha256") != sha256_file(reference_checkpoint):
        raise ValueError("MARS CARE v2/reference checkpoint hash mismatch")
    task_names = tuple(str(value) for value in saved.get("task_names", ()))
    if task_names != TASK_NAMES:
        raise ValueError("CARE v2 deployment task order differs from MARS contract")

    config = CAREBeliefV2Config.from_mapping(saved["config"])
    if config.action_prefix_steps != 1:
        raise ValueError("current MARS CARE v2 deployment requires one-step branches")
    if saved.get("prepared_intervention_steps") != config.action_prefix_steps:
        raise ValueError("CARE v2 deployment prefix/branch intervention mismatch")
    if config.candidates != 6 or config.action_horizon != 100:
        raise ValueError("CARE v2 fixed candidate/action-horizon contract drifted")

    calibration = CARECalibration.from_mapping(saved["calibration"])
    if calibration.primary_horizon not in config.horizons:
        raise ValueError("CARE v2 primary horizon is absent from scorer horizons")
    _finite_nonnegative(calibration.lower_correction, "CARE v2 correction")
    _finite_nonnegative(calibration.selector_delta, "CARE v2 selector delta")
    if not 0.0 < calibration.hard_safety_probability_max < 1.0:
        raise ValueError("CARE v2 safety threshold is invalid")
    if not 0.0 < calibration.nominal_simultaneous_coverage <= 1.0:
        raise ValueError("CARE v2 nominal coverage is invalid")

    scales = torch.as_tensor(saved.get("task_component_scales"), dtype=torch.float32)
    expected_scales = (len(TASK_NAMES), config.outcome_components)
    if scales.shape != expected_scales:
        raise ValueError(f"CARE v2 utility scales must have shape {expected_scales}")
    if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("CARE v2 utility scales must be finite and positive")

    task_corrections_value = saved.get("task_lower_corrections")
    if task_corrections_value is None:
        task_corrections = torch.full(
            (len(TASK_NAMES),), float(calibration.lower_correction)
        )
    else:
        task_corrections = torch.as_tensor(
            task_corrections_value, dtype=torch.float32
        )
    if task_corrections.shape != (len(TASK_NAMES),):
        raise ValueError("CARE v2 task corrections must be [task]")
    if not torch.isfinite(task_corrections).all() or bool((task_corrections < 0).any()):
        raise ValueError("CARE v2 task corrections must be finite and non-negative")

    safety_mode = str(saved.get("safety_gate_mode", ""))
    safety_support = int(saved.get("safety_positive_label_count", -1))
    if safety_mode == "legality_only":
        if safety_support != 0:
            raise ValueError("legality-only CARE v2 deployment must have zero safety support")
    elif safety_mode == "learned_probability":
        if safety_support <= 0 or saved.get("safety_threshold_calibrated") is not True:
            raise ValueError("learned CARE v2 safety requires positive support and calibration")
    else:
        raise ValueError("CARE v2 deployment safety mode is not auditable")

    provenance = saved.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("CARE v2 deployment provenance is missing")
    if provenance.get("oof_format_version") != OOF_FORMAT_VERSION:
        raise ValueError("CARE v2 deployment lacks the registered OOF gate")
    for key in (
        "admission_passed",
        "family_disjoint",
        "calibration_independent",
        "no_validation20_tuning",
        "physical_unit_runtime_parity",
    ):
        _require_true(provenance, key, "deployment provenance")
    if provenance.get("promotion_scope") not in {"smoke", "formal"}:
        raise ValueError("CARE v2 deployment promotion scope is invalid")
    if provenance.get("promotion_scope") == "formal":
        for key in ("paired_smoke_passed", "decentralized_smoke_passed"):
            _require_true(provenance, key, "formal promotion")
    return config, calibration, scales, task_corrections, safety_mode


@dataclass
class MarsCAREBeliefV2Runtime:
    """Shared scorer weights with task-local physical unit decoding."""

    model: CAREBeliefV2Head
    calibration: CARECalibration
    task_component_scales: torch.Tensor
    task_lower_corrections: torch.Tensor
    safety_gate_mode: str
    saved: Mapping[str, Any]

    def task_indices(
        self, tasks: str | Sequence[str], *, device: torch.device
    ) -> torch.Tensor:
        values = (tasks,) if isinstance(tasks, str) else tuple(tasks)
        try:
            indices = [TASK_NAMES.index(str(value)) for value in values]
        except ValueError as error:
            raise KeyError(f"unknown MARS CARE v2 task in {values}") from error
        return torch.tensor(indices, dtype=torch.long, device=device)

    @torch.inference_mode()
    def score_and_select(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        candidate_chunks: torch.Tensor,
        candidate_legality: torch.Tensor,
        tasks: str | Sequence[str],
        *,
        selector_enabled: bool = True,
    ) -> CARESelectionV2:
        task_id = self.task_indices(tasks, device=memory.device)
        if task_id.shape != (memory.shape[0],):
            if task_id.numel() == 1:
                task_id = task_id.expand(memory.shape[0])
            else:
                raise ValueError("CARE v2 task count must equal scorer batch")
        scale = self.task_component_scales.to(memory.device).index_select(0, task_id)
        correction = self.task_lower_corrections.to(memory.device).index_select(
            0, task_id
        )
        horizon = torch.full(
            (memory.shape[0],),
            self.model.config.horizons.index(self.calibration.primary_horizon),
            dtype=torch.long,
            device=memory.device,
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=memory.device.type == "cuda"
        ):
            output = self.model(
                memory,
                memory_mask,
                candidate_chunks,
                horizon,
                utility_scale=scale,
            )
        return select_care_candidate_v2(
            output,
            self.calibration,
            candidate_legality,
            variant=self.model.config.variant,
            safety_gate_mode=self.safety_gate_mode,
            lower_correction=correction,
            selector_enabled=selector_enabled,
        )


def load_mars_care_v2(
    path: Path, device: torch.device, reference_checkpoint: Path
) -> MarsCAREBeliefV2Runtime:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping):
        raise ValueError("MARS CARE v2 deployment checkpoint must be a mapping")
    config, calibration, scales, corrections, safety_mode = (
        validate_v2_deployment_payload(saved, reference_checkpoint)
    )
    model = CAREBeliefV2Head(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return MarsCAREBeliefV2Runtime(
        model=model,
        calibration=calibration,
        task_component_scales=scales,
        task_lower_corrections=corrections,
        safety_gate_mode=safety_mode,
        saved=saved,
    )


__all__ = [
    "DEPLOYMENT_FORMAT_VERSION",
    "MarsCAREBeliefV2Runtime",
    "OOF_FORMAT_VERSION",
    "TASK_NAMES",
    "TRAINING_FORMAT_VERSION",
    "load_mars_care_v2",
    "validate_v2_deployment_payload",
]
