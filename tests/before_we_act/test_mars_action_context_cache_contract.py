import numpy as np
import pytest

from scripts.before_we_act.build_mars_action_context_cache import require_finite


def test_action_context_finite_guard_accepts_large_float32_hidden_state() -> None:
    value = np.asarray([0.0, 65_505.0, 1_000_000.0], dtype=np.float32)
    require_finite("decoded", value)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_action_context_finite_guard_rejects_nonfinite_values(bad: float) -> None:
    with pytest.raises(FloatingPointError, match="non-finite decoded"):
        require_finite("decoded", np.asarray([0.0, bad], dtype=np.float32))
