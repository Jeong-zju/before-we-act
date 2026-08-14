from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts.before_we_act import run_ssc_v7_m3 as m3
from scripts.before_we_act import run_ssc_v7_m3_r4 as r4


def label() -> dict:
    return {
        "ambiguity_code": 0,
        "label_validity_mask": {
            "grasp_contact_custody_state": True,
            "causal_automaton_state": True,
            "per_agent_contribution": True,
            "collision_drop_contention_risk": True,
        },
        "grasp_contact_custody_state": {
            "shoe": {
                "contact_agents": [],
                "grasp_agents": [],
                "controller_agents": [],
                "current_custodian": None,
                "shared_control": False,
            }
        },
        "causal_automaton_state": {"completed_handoff_mask": []},
        "per_agent_contribution": [
            {
                "agent_slot": 0,
                "active": False,
                "contact_objects": [],
                "grasp_objects": [],
                "roles": [],
            },
            {
                "agent_slot": 1,
                "active": False,
                "contact_objects": [],
                "grasp_objects": [],
                "roles": [],
            },
        ],
        "collision_drop_contention_risk": {
            "robot_collision": False,
            "robot_proximity_risk": False,
            "contested_objects": [],
            "dropped_objects": [],
        },
    }


def hc_payload() -> dict:
    model = m3.build_action_model(r4.HC_INPUT_WIDTH, 256, 17)
    return {
        "state_dict": model.state_dict(),
        "input_width": r4.HC_INPUT_WIDTH,
        "hidden_width": 256,
    }


def test_arb_tokens_capture_relative_handoff_without_stage_or_identity() -> None:
    before = label()
    after = deepcopy(before)
    after["grasp_contact_custody_state"]["shoe"].update(
        {
            "contact_agents": [0, 1],
            "grasp_agents": [1],
            "controller_agents": [1],
            "current_custodian": 1,
            "shared_control": True,
        }
    )
    after["causal_automaton_state"]["completed_handoff_mask"] = ["handoff_01"]
    after["per_agent_contribution"][1].update(
        {
            "active": True,
            "contact_objects": ["shoe"],
            "grasp_objects": ["shoe"],
            "roles": ["receiver"],
        }
    )
    value = r4.arb_tokens([before, after], 1, own_slot=0)
    assert value.shape == (6, 8)
    assert value[0, 3] == 1.0  # peer contact
    assert value[0, 4] == 1.0  # peer grasp
    assert value[0, 7] == 1.0  # peer custody
    assert value[1, 0] == 1.0  # handoff completed now
    assert value[1, 1] == 1.0  # custody changed now
    assert value[1, 4] == 1.0  # peer gained custody
    assert value[2, 0] == 1.0  # peer active
    assert value[2, 6] == 1.0  # receiver role


def test_zero_initialized_residual_and_gate_off_exactly_reproduce_hc() -> None:
    torch = pytest.importorskip("torch")
    payload = hc_payload()
    model = r4.ArbResidualFactory.create(payload, seed=23).eval()
    rng = np.random.default_rng(9)
    hc = rng.normal(size=(5, r4.HC_INPUT_WIDTH)).astype(np.float32)
    arb = rng.uniform(size=(5, r4.ARB_WIDTH)).astype(np.float32)
    reliability_on = np.ones((5, 1), dtype=np.float32)
    reliability_off = np.zeros((5, 1), dtype=np.float32)
    with torch.no_grad():
        baseline = r4.HCWrapper.create(payload)(torch.from_numpy(hc))
        initial = model(torch.from_numpy(np.concatenate((hc, arb, reliability_on), axis=1)))
        gate_off = model(torch.from_numpy(np.concatenate((hc, arb, reliability_off), axis=1)))
    assert torch.equal(initial, baseline)
    assert torch.equal(gate_off, baseline)


def test_legacy_zero_initialized_residual_reproduces_hc() -> None:
    torch = pytest.importorskip("torch")
    payload = hc_payload()
    model = r4.LegacyResidualFactory.create(payload, seed=29).eval()
    rng = np.random.default_rng(11)
    hc = rng.normal(size=(3, r4.HC_INPUT_WIDTH)).astype(np.float32)
    legacy = rng.normal(size=(3, r4.LEGACY_WIDTH)).astype(np.float32)
    reliability = np.ones((3, 1), dtype=np.float32)
    with torch.no_grad():
        baseline = r4.HCWrapper.create(payload)(torch.from_numpy(hc))
        prediction = model(torch.from_numpy(np.concatenate((hc, legacy, reliability), axis=1)))
    assert torch.equal(prediction, baseline)


def test_sealed_test_split_is_rejected_before_loading(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        r4.load_bundle(manifest, {"read_only_test"})


def test_median_metrics_uses_episode_median_not_best_seed() -> None:
    def metric(error: float) -> dict:
        return {
            "episode_errors": {
                f"episode_{task}": {
                    "task": task,
                    "primary_16_nrmse": error,
                    "diagnostic_100_masked_nrmse": error,
                    "gripper_16_rmse": error,
                    "rows": 1,
                }
                for task in r4.TASKS
            }
        }

    value = r4.median_metrics([metric(0.1), metric(0.9), metric(0.5)])
    assert value["episode_errors"]["episode_lift_barrier"]["primary_16_nrmse"] == 0.5
