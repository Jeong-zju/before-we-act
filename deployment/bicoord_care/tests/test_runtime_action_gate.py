from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.bicoord_care.bcore_data import (
    BICOORD_CARE_MEMORY_TOKENS,
    BICOORD_CARE_MEMORY_WIDTH,
)
from deployment.bicoord_care.bcore_runtime import BiCoordBcoreRuntime
from deployment.bicoord_care.config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
)
from deployment.bicoord_care.runtime import B0HRuntime


def _stats() -> dict[str, object]:
    low = np.r_[np.full(ACTION_DIM - 1, -2.0), 0.0].astype(np.float32)
    high = np.r_[np.full(ACTION_DIM - 1, 2.0), 1.0].astype(np.float32)
    return {
        "q_mean": np.zeros(ACTION_DIM, dtype=np.float32),
        "q_std": np.ones(ACTION_DIM, dtype=np.float32),
        "a_mean": np.zeros(ACTION_DIM, dtype=np.float32),
        "a_std": np.ones(ACTION_DIM, dtype=np.float32),
        "q_min": low.copy(),
        "q_max": high.copy(),
        "a_min": low.copy(),
        "a_max": high.copy(),
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
    }


def _observation() -> dict[str, object]:
    return {
        "observation": {
            "head_camera": {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.full((16, 16, 3), 17, dtype=np.uint8)},
            "right_camera": {"rgb": np.full((16, 16, 3), 29, dtype=np.uint8)},
        },
        "joint_action": {
            "left_arm": np.zeros(6, dtype=np.float32),
            "left_gripper": np.float32(0.25),
            "right_arm": np.zeros(6, dtype=np.float32),
            "right_gripper": np.float32(0.75),
        },
    }


class _FakeVisionPolicy(torch.nn.Module):
    def _raw_vision_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (images.shape[0], 1, 768), dtype=torch.float32, device=images.device
        )


class _FakeB0H(_FakeVisionPolicy):
    def __init__(self, prediction: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("prediction", torch.from_numpy(prediction.copy()))

    def forward(self, **_inputs: torch.Tensor):
        latent = torch.zeros(
            (self.prediction.shape[0], 1), device=self.prediction.device
        )
        return self.prediction.clone(), latent, latent


class _FakeBcore(_FakeVisionPolicy):
    def __init__(self, prediction: np.ndarray, base_prediction: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("prediction", torch.from_numpy(prediction.copy()))
        self.register_buffer(
            "base_prediction", torch.from_numpy(base_prediction.copy())
        )

    def forward(self, **_inputs: torch.Tensor) -> SimpleNamespace:
        batch = self.prediction.shape[0]
        device = self.prediction.device
        belief = SimpleNamespace(
            mu=torch.zeros(
                (batch, 16, BICOORD_CARE_MEMORY_WIDTH), device=device
            ),
            event_memory=torch.zeros(
                (
                    batch,
                    BICOORD_CARE_MEMORY_TOKENS - 16,
                    BICOORD_CARE_MEMORY_WIDTH,
                ),
                device=device,
            ),
            event_mask=torch.zeros(
                (batch, BICOORD_CARE_MEMORY_TOKENS - 16),
                dtype=torch.bool,
                device=device,
            ),
            sigma=torch.zeros(
                (batch, 16, BICOORD_CARE_MEMORY_WIDTH), device=device
            ),
            reliability=torch.ones((batch, 1, 1), device=device),
        )
        return SimpleNamespace(
            prediction=self.prediction.clone(),
            base_prediction=self.base_prediction.clone(),
            belief=belief,
            belief_residual=torch.zeros_like(self.prediction),
            residual_gate=torch.zeros(
                (batch, ACTION_HORIZON, 1), device=device
            ),
        )


def _chunk() -> np.ndarray:
    value = np.zeros(
        (2, ACTION_HORIZON, ACTION_DIM), dtype=np.float32
    )
    value[..., -1] = 0.5
    return value


def test_b0h_future_gripper_oob_is_telemetry_without_output_transform() -> None:
    prediction = _chunk()
    prediction[0, 7, -1] = np.float32(1.25)
    prediction[1, 91, -1] = np.float32(-0.2)
    runtime = B0HRuntime(
        _FakeB0H(prediction), _stats(), device=torch.device("cpu")
    )

    actions = runtime.act(_observation(), "cook")

    np.testing.assert_array_equal(actions[0], prediction[0])
    np.testing.assert_array_equal(actions[1], prediction[1])
    diagnostics = runtime.last_prediction_diagnostics
    assert diagnostics["prediction_gripper_oob_count"] == 2
    assert diagnostics["ensemble_plan_gripper_oob_count"] == 2
    assert diagnostics["prediction_gripper_min"] == pytest.approx(-0.2)
    assert diagnostics["prediction_gripper_max"] == pytest.approx(1.25)
    assert diagnostics["policy_output_clipping"] is False


def test_b0h_current_gripper_oob_fails_at_executed_row_gate() -> None:
    prediction = _chunk()
    prediction[0, 0, -1] = np.float32(1.01)
    runtime = B0HRuntime(
        _FakeB0H(prediction), _stats(), device=torch.device("cpu")
    )

    with pytest.raises(ValueError, match="B0-H executed action.*outside native range"):
        runtime.act(_observation(), "cook")


def test_bcore_future_oob_is_telemetry_but_executed_row_is_fail_closed() -> None:
    prediction = _chunk()
    base_prediction = _chunk()
    prediction[0, 9, -1] = np.float32(1.2)
    prediction[1, 63, -1] = np.float32(-0.3)
    base_prediction[1, 31, -1] = np.float32(1.4)
    runtime = BiCoordBcoreRuntime(
        _FakeBcore(prediction, base_prediction),
        _stats(),
        device=torch.device("cpu"),
    )
    runtime.reset("cook")

    context = runtime.act_with_context(_observation(), "cook", commit=False)

    np.testing.assert_array_equal(context.reference_chunk, prediction)
    np.testing.assert_array_equal(context.base_chunk, base_prediction)
    np.testing.assert_array_equal(context.reference_plan, prediction)
    diagnostics = context.diagnostics
    assert diagnostics["reference_chunk_gripper_oob_count"] == 2
    assert diagnostics["reference_plan_gripper_oob_count"] == 2
    assert diagnostics["base_chunk_gripper_oob_count"] == 1
    assert diagnostics["reference_chunk_gripper_min"] == pytest.approx(-0.3)
    assert diagnostics["reference_chunk_gripper_max"] == pytest.approx(1.2)
    assert diagnostics["policy_output_clipping"] is False

    invalid_current = context.reference_plan[:, 0].copy()
    invalid_current[1, -1] = np.float32(-0.01)
    with pytest.raises(
        ValueError, match="B-core executed actions.*outside native range"
    ):
        runtime.record_executed_actions(invalid_current)
