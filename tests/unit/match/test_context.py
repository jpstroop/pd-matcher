"""Tests for :mod:`pd_matcher.match.scorers.context`."""

from pytest import raises

from pd_matcher.match.scorers.context import ScorerContext


def test_scorer_context_is_frozen(scorer_context: ScorerContext) -> None:
    """ScorerContext must reject attribute mutation."""
    with raises(AttributeError):
        setattr(scorer_context, "language", "fre")
