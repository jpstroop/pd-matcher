"""Tests for :mod:`pd_matcher.match.combiners.learned`."""

from pytest import raises

from pd_matcher.match.combiners.learned import LearnedCombiner


def test_learned_combiner_raises_not_implemented() -> None:
    """The Phase 9 placeholder must raise when ``combine`` is invoked."""
    combiner = LearnedCombiner()
    with raises(NotImplementedError, match="Phase 9"):
        combiner.combine(())
