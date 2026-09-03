"""Training and evaluation must weight the action chunk on the same scale.

A DuoBench run trained with --action-loss-decay 16 and was scored through the
default ensembler at exp(-0.01*age). Training supervised chunk position t by
exp(-t/16), leaving the tail of the 100-step chunk at 6.8x the head's error,
while the ensembler drew 51% of every executed action from positions weighted
below a tenth of the first step. Closed-loop success was 1.36%, below the ACT
baseline's 7.73%, and the logged action loss -- the same weighted mean -- gave
no sign of it. The mismatch only surfaced after a full sweep had been paid for.
"""
from __future__ import annotations

import pytest

from deployment.duo_dino_reference.evaluate import require_matched_chunk_weighting


def test_uniform_training_constrains_nothing() -> None:
    """Decay zero supervises every position, so any ensembler is consistent."""
    require_matched_chunk_weighting(0.0, 0.01)
    require_matched_chunk_weighting(0.0, 0.25)


def test_the_deployed_mismatch_is_rejected() -> None:
    """The exact configuration that produced 1.36%."""
    with pytest.raises(ValueError, match="different"):
        require_matched_chunk_weighting(16.0, 0.01)


def test_the_rejection_names_both_repairs() -> None:
    with pytest.raises(ValueError) as error:
        require_matched_chunk_weighting(16.0, 0.01)
    message = str(error.value)
    assert "--action-loss-decay 0" in message
    assert "--ensemble-decay 0.0625" in message
    assert "6.2" in message  # the scale ratio, so the size is visible


def test_a_matched_ensembler_is_accepted() -> None:
    """1/16 makes the ensemble weight exactly the training weight."""
    require_matched_chunk_weighting(16.0, 1.0 / 16.0)


@pytest.mark.parametrize("ensemble_decay", [1 / 8.0, 1 / 32.0])
def test_a_factor_of_two_is_tolerated(ensemble_decay: float) -> None:
    """Exact equality is not required; an order-of-magnitude gap is the target."""
    require_matched_chunk_weighting(16.0, ensemble_decay)


@pytest.mark.parametrize("ensemble_decay", [1 / 33.0, 1 / 7.0])
def test_beyond_a_factor_of_two_is_rejected(ensemble_decay: float) -> None:
    with pytest.raises(ValueError):
        require_matched_chunk_weighting(16.0, ensemble_decay)


def test_a_nonpositive_ensemble_decay_is_rejected_when_training_decayed() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        require_matched_chunk_weighting(16.0, 0.0)


def test_the_supervisor_trains_uniformly() -> None:
    """The formal DuoBench B0-H command must not reintroduce the decay."""
    import inspect

    from deployment.duo_dino_reference import supervisor

    source = inspect.getsource(supervisor)
    index = source.index('"--action-loss-decay"')
    following = source[index : index + 200]
    assert '"0"' in following, "formal B0-H must train with uniform action loss"
