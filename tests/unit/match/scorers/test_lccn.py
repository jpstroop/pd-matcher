"""Tests for :mod:`pd_matcher.match.scorers.lccn`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.lccn import score_lccn
from pd_matcher.models import IndexedNyplRegRecord


def _record(regnum: str | None) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(uuid="u", title="t", was_renewed=False, regnum=regnum)


def test_score_lccn_exact_match_is_decisive(scorer_context: ScorerContext) -> None:
    """Equal canonical IDs produce a decisive Evidence at max score."""
    ev = score_lccn("A111111", _record("A111111"), scorer_context)
    assert ev.decisive is True
    assert ev.score == ev.max
    assert ev.skipped is False


def test_score_lccn_handles_leading_zeros(scorer_context: ScorerContext) -> None:
    """Leading zeros and whitespace are stripped before comparison."""
    ev = score_lccn("  0040012345  ", _record("40012345"), scorer_context)
    assert ev.decisive is True


def test_score_lccn_unequal_ids_skipped(scorer_context: ScorerContext) -> None:
    """Unequal IDs deliberately skip rather than emit a low score."""
    ev = score_lccn("A111111", _record("B222222"), scorer_context)
    assert ev.skipped is True
    assert ev.decisive is False


def test_score_lccn_skipped_when_marc_lccn_missing(scorer_context: ScorerContext) -> None:
    """A None MARC LCCN triggers the skipped path."""
    ev = score_lccn(None, _record("A111"), scorer_context)
    assert ev.skipped is True


def test_score_lccn_skipped_when_marc_lccn_blank(scorer_context: ScorerContext) -> None:
    """A blank MARC LCCN canonicalises to None and triggers skipped."""
    ev = score_lccn("   ", _record("A111"), scorer_context)
    assert ev.skipped is True


def test_score_lccn_skipped_when_nypl_regnum_missing(scorer_context: ScorerContext) -> None:
    """A None NYPL regnum triggers the skipped path."""
    ev = score_lccn("A111", _record(None), scorer_context)
    assert ev.skipped is True
