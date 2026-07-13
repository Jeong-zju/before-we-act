from __future__ import annotations

import numpy as np

from data.local_observation import LocalObservationPacket, LocalObservationSpec, PoseEstimate
from eval.calibration import ParetoPoint, fit_affine_cost_calibration, pareto_frontier, select_pareto_utopia


def _packet(*, cue: np.ndarray, valid: bool) -> LocalObservationPacket:
    return LocalObservationPacket(
        base_twist=np.zeros(3, dtype=np.float32),
        joint_position=np.zeros(0, dtype=np.float32),
        joint_velocity=np.zeros(0, dtype=np.float32),
        joint_torque=np.zeros(0, dtype=np.float32),
        local_force=np.zeros(1, dtype=np.float32),
        contact=np.zeros(1, dtype=np.float32),
        grasp=np.zeros(1, dtype=np.float32),
        object_estimate=PoseEstimate(
            pose=np.zeros(3, dtype=np.float32),
            valid=np.ones(1, dtype=np.float32),
            confidence=np.ones(1, dtype=np.float32),
            age=np.zeros(1, dtype=np.float32),
        ),
        task_goal=np.zeros(3, dtype=np.float32),
        private_event_cue=cue,
        private_event_valid=np.asarray([float(valid)], dtype=np.float32),
    )


def test_private_event_invalid_cue_is_rejected() -> None:
    spec = LocalObservationSpec()
    packet = _packet(cue=np.asarray([1.0, 0.0, 0.0], dtype=np.float32), valid=False)
    try:
        packet.validate(spec)
    except ValueError as exc:
        assert "zeroed" in str(exc)
    else:
        raise AssertionError("invalid cue leaked into deployable packet")


def test_private_event_fields_expand_only_local_task_context() -> None:
    spec = LocalObservationSpec()
    assert spec.model_observation_dim == 17
    assert "task/private_event_cue" in spec.model_field_names()
    assert all("truth" not in name for name in spec.field_shapes())


def test_pareto_selection_removes_dominated_points() -> None:
    points = [
        ParetoPoint(0.0, 0.0, 0.95, 100.0),
        ParetoPoint(0.1, 0.0, 0.95, 80.0),
        ParetoPoint(0.2, 0.0, 0.90, 40.0),
    ]
    frontier = pareto_frontier(points)
    assert points[0] not in frontier
    selected = select_pareto_utopia(points)
    assert selected["selected"] in [point.__dict__ for point in frontier]
    assert selected["test_set_used"] is False


def test_affine_utility_calibration_recovers_scale() -> None:
    result = fit_affine_cost_calibration([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert np.isclose(result["scale"], 2.0)
    assert np.isclose(result["bias"], 1.0)
    assert result["rmse"] < 1e-10
