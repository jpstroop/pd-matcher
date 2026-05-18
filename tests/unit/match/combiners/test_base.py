"""Tests for :mod:`pd_matcher.match.combiners.base`."""

from pytest import raises

from pd_matcher.match.combiners.base import CombinedScore


def test_combined_score_is_frozen() -> None:
    """The struct must reject attribute mutation."""
    score = CombinedScore(raw=50.0, calibrated=0.5)
    with raises(AttributeError):
        setattr(score, "raw", 99.0)


def test_combined_score_fields_are_stored() -> None:
    """Field assignment from the constructor round-trips."""
    score = CombinedScore(raw=80.0, calibrated=0.9)
    assert score.raw == 80.0
    assert score.calibrated == 0.9
