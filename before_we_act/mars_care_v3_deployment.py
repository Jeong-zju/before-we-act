"""Fail-closed deployment loader for the H8 CARE-v3 scorer.

The V3 checkpoint is intentionally separate from the historical V2 loader.
It keeps the CARE chain unchanged while making candidate-slot/task conditioning,
the executed H8 prefix, and simultaneous all-horizon OOF calibration explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from before_we_act.care_belief import CARECalibration
from before_we_act.care_selector_v2 import CARESelectionV2, select_care_candidate_v2
from before_we_act.care_training_data import sha256_file
from before_we_act.care_belief_v3 import CAREBeliefV3Config, CAREBeliefV3Head
from deployment.mars_care.common import TASK_BY_NAME


DEPLOYMENT_FORMAT_VERSION = "before-we-act.care-mars-deployment-checkpoint-v3/1"
FINAL_TRAINING_FORMAT_VERSION = "before-we-act.care-mars-final-training-v3/1"
OOF_FORMAT_VERSION = "before-we-act.care-mars-oof-training-v3/2-all-horizon"
TASK_NAMES = tuple(TASK_BY_NAME)
HORIZONS = (8, 16, 32, 64)


def _positive_tensor(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if tuple(result.shape) != shape:
        raise ValueError(f"CARE v3 {name} must have shape {shape}")
    if not torch.isfinite(result).all() or bool((result <= 0).any()):
        raise ValueError(f"CARE v3 {name} must be finite and positive")
    return result


def _nonnegative_tensor(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if tuple(result.shape) != shape:
        raise ValueError(f"CARE v3 {name} must have shape {shape}")
    if not torch.isfinite(result).all() or bool((result < 0).any()):
        raise ValueError(f"CARE v3 {name} must be finite and non-negative")
    return result


def validate_v3_deployment_payload(
    saved: Mapping[str, Any], reference_checkpoint: Path
) -> tuple[CAREBeliefV3Config, CARECalibration, torch.Tensor, torch.Tensor, str]:
    if saved.get("format_version") != DEPLOYMENT_FORMAT_VERSION:
        raise ValueError("wrong MARS CARE v3 deployment checkpoint")
    if saved.get("source_training_format_version") != FINAL_TRAINING_FORMAT_VERSION:
        raise ValueError("CARE v3 deployment source is not the final training format")
    if saved.get("reference_checkpoint_sha256") != sha256_file(reference_checkpoint):
        raise ValueError("MARS CARE v3/reference checkpoint hash mismatch")
    if tuple(str(x) for x in saved.get("task_names", ())) != TASK_NAMES:
        raise ValueError("CARE v3 deployment task order differs from MARS contract")
    config = CAREBeliefV3Config.from_mapping(saved["config"])
    if config.action_prefix_steps != 8:
        raise ValueError("H8 CARE v3 deployment requires action_prefix_steps=8")
    if not config.use_candidate_slot_embedding or not config.use_task_embedding:
        raise ValueError("promoted CARE v3 deployment requires slot and task conditioning")
    if config.candidates != 6 or config.action_horizon != 100:
        raise ValueError("CARE v3 candidate/action-horizon contract drifted")
    if tuple(config.horizons) != HORIZONS:
        raise ValueError("CARE v3 horizon contract drifted")
    calibration = CARECalibration.from_mapping(saved["calibration"])
    if calibration.primary_horizon not in HORIZONS:
        raise ValueError("CARE v3 primary horizon is absent")
    values = (
        calibration.lower_correction,
        calibration.selector_delta,
        calibration.hard_safety_probability_max,
        calibration.nominal_simultaneous_coverage,
    )
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("CARE v3 calibration contains NaN/Inf")
    if calibration.lower_correction < 0 or calibration.selector_delta < 0:
        raise ValueError("CARE v3 correction/delta must be non-negative")
    if not 0 < calibration.hard_safety_probability_max < 1:
        raise ValueError("CARE v3 safety threshold is invalid")
    scales = _positive_tensor(
        saved.get("task_horizon_component_scales"),
        (len(TASK_NAMES), len(HORIZONS), config.outcome_components),
        "task_horizon_component_scales",
    )
    corrections = _nonnegative_tensor(
        saved.get("task_horizon_lower_corrections"),
        (len(TASK_NAMES), len(HORIZONS)),
        "task_horizon_lower_corrections",
    )
    if saved.get("intervention_steps") != 8:
        raise ValueError("CARE v3 intervention/prefix contract drifted")
    safety_mode = str(saved.get("safety_gate_mode", ""))
    safety_support = int(saved.get("safety_positive_label_count", -1))
    if safety_mode == "legality_only":
        if safety_support != 0:
            raise ValueError("legality-only CARE v3 deployment has safety support")
    elif safety_mode == "learned_probability":
        if safety_support <= 0 or saved.get("safety_threshold_calibrated") is not True:
            raise ValueError("learned CARE v3 safety lacks calibration proof")
    else:
        raise ValueError("CARE v3 safety mode is not auditable")
    provenance = saved.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("CARE v3 deployment provenance is missing")
    scope = provenance.get("promotion_scope")
    if scope not in {"smoke", "formal", "exploratory"}:
        raise ValueError("CARE v3 promotion scope is invalid")
    if scope == "exploratory":
        if provenance.get("admission_bypassed") is not True:
            raise ValueError("exploratory CARE v3 scope lacks explicit bypass proof")
        if provenance.get("no_validation20_tuning") is not True:
            raise ValueError("exploratory CARE v3 scope lacks no-tuning proof")
    if provenance.get("interface_smoke_only") is True:
        if scope != "smoke" or provenance.get("no_validation20_tuning") is not True:
            raise ValueError("CARE v3 interface smoke provenance is invalid")
        if provenance.get("physical_unit_runtime_parity") is not True:
            raise ValueError("CARE v3 interface smoke lacks runtime parity")
    else:
        for key in (
            "oof_format_version",
            "admission_passed",
            "family_disjoint",
            "calibration_independent",
            "no_validation20_tuning",
            "physical_unit_runtime_parity",
            "horizon_oof_complete",
        ):
            if key == "oof_format_version":
                if provenance.get(key) != OOF_FORMAT_VERSION:
                    raise ValueError("CARE v3 deployment lacks the registered OOF format")
            elif provenance.get(key) is not True:
                raise ValueError(f"CARE v3 provenance does not prove {key}")
    return config, calibration, scales, corrections, safety_mode


@dataclass
class MarsCAREBeliefV3Runtime:
    model: CAREBeliefV3Head
    calibration: CARECalibration
    task_horizon_component_scales: torch.Tensor
    task_horizon_lower_corrections: torch.Tensor
    safety_gate_mode: str
    saved: Mapping[str, Any]

    def task_indices(self, tasks: str | Sequence[str], *, device: torch.device) -> torch.Tensor:
        values = (tasks,) if isinstance(tasks, str) else tuple(tasks)
        try:
            indices = [TASK_NAMES.index(str(value)) for value in values]
        except ValueError as error:
            raise KeyError(f"unknown MARS CARE v3 task in {values}") from error
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
        if task_id.numel() == 1:
            task_id = task_id.expand(memory.shape[0])
        if task_id.shape != (memory.shape[0],):
            raise ValueError("CARE v3 task count must equal scorer batch")
        horizon_index = torch.full(
            (memory.shape[0],),
            self.model.config.horizons.index(self.calibration.primary_horizon),
            dtype=torch.long,
            device=memory.device,
        )
        scale = self.task_horizon_component_scales.to(memory.device)[
            task_id, horizon_index
        ]
        correction = self.task_horizon_lower_corrections.to(memory.device)[
            task_id, horizon_index
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=memory.device.type == "cuda"):
            output = self.model(
                memory,
                memory_mask,
                candidate_chunks,
                horizon_index,
                task_id,
                utility_scale=scale,
            )
        return select_care_candidate_v2(
            output,
            self.calibration,
            candidate_legality,
            variant="care",
            safety_gate_mode=self.safety_gate_mode,
            lower_correction=correction,
            selector_enabled=selector_enabled,
        )


def load_mars_care_v3(
    path: Path, device: torch.device, reference_checkpoint: Path
) -> MarsCAREBeliefV3Runtime:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping):
        raise ValueError("MARS CARE v3 deployment checkpoint must be a mapping")
    config, calibration, scales, corrections, safety_mode = validate_v3_deployment_payload(
        saved, reference_checkpoint
    )
    model = CAREBeliefV3Head(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return MarsCAREBeliefV3Runtime(
        model=model,
        calibration=calibration,
        task_horizon_component_scales=scales,
        task_horizon_lower_corrections=corrections,
        safety_gate_mode=safety_mode,
        saved=saved,
    )


__all__ = [
    "DEPLOYMENT_FORMAT_VERSION",
    "MarsCAREBeliefV3Runtime",
    "load_mars_care_v3",
    "validate_v3_deployment_payload",
]
