from pathlib import Path
import importlib.util

import pytest
import torch

from before_we_act.direct_reactive_policy import (
    DirectReactiveDeploymentPolicy,
    prefixed_state,
)


def load_summary_module():
    path = Path(__file__).parents[2] / "scripts/before_we_act/summarize_minimal_bcore_attribution.py"
    spec = importlib.util.spec_from_file_location("minimal_attribution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prefixed_state_is_exact_and_nonempty() -> None:
    state = {
        "direct_control.a": torch.tensor([1.0]),
        "direct_control.block.b": torch.tensor([2.0]),
        "belief_core.c": torch.tensor([3.0]),
    }
    selected = prefixed_state(state, "direct_control.")
    assert set(selected) == {"a", "block.b"}
    assert selected["a"].item() == 1.0
    with pytest.raises(ValueError, match="no state"):
        prefixed_state(state, "missing.")


def test_direct_deployment_declares_the_backbone_variants() -> None:
    assert "hidden_residual" in DirectReactiveDeploymentPolicy.VARIANTS


def test_screen_decision_was_frozen_as_a_practical_gate() -> None:
    module = load_summary_module()
    assert module.screen_decision(5) == "ADVANCE_TO_MECHANISM_CHECK"
    assert module.screen_decision(1) == "SMALL_POSITIVE_REQUIRE_MECHANISM_CHECK"
    assert module.screen_decision(0) == "STOP_TEAM_SPECIFIC_MAIN_CLAIM"


def test_exact_mcnemar_is_symmetric() -> None:
    module = load_summary_module()
    assert module.exact_mcnemar_two_sided(7, 2) == module.exact_mcnemar_two_sided(2, 7)
    assert module.exact_mcnemar_two_sided(0, 0) == 1.0
