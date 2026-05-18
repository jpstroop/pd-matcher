"""Tests for :mod:`pd_matcher.match.scorers.year`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.year import score_year


def test_score_year_zero_delta(scorer_context: ScorerContext) -> None:
    """Same year scores 100."""
    ev = score_year(1940, 1940, scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["delta_years"] == 0.0


def test_score_year_one_year_apart(scorer_context: ScorerContext) -> None:
    """One year apart scores 75."""
    ev = score_year(1940, 1941, scorer_context)
    assert ev.score == 75.0


def test_score_year_two_years_apart(scorer_context: ScorerContext) -> None:
    """Two years apart scores 50."""
    ev = score_year(1940, 1942, scorer_context)
    assert ev.score == 50.0


def test_score_year_three_years_apart(scorer_context: ScorerContext) -> None:
    """Three years apart scores 25."""
    ev = score_year(1940, 1943, scorer_context)
    assert ev.score == 25.0


def test_score_year_four_years_apart_floors_at_zero(scorer_context: ScorerContext) -> None:
    """Four years apart floors at 0."""
    ev = score_year(1940, 1944, scorer_context)
    assert ev.score == 0.0


def test_score_year_ten_years_apart_floors_at_zero(scorer_context: ScorerContext) -> None:
    """Large deltas floor at 0, never negative."""
    ev = score_year(1940, 1990, scorer_context)
    assert ev.score == 0.0


def test_score_year_skipped_when_marc_year_none(scorer_context: ScorerContext) -> None:
    """A None MARC year triggers the skipped path."""
    ev = score_year(None, 1940, scorer_context)
    assert ev.skipped is True


def test_score_year_skipped_when_nypl_year_none(scorer_context: ScorerContext) -> None:
    """A None NYPL year triggers the skipped path."""
    ev = score_year(1940, None, scorer_context)
    assert ev.skipped is True
