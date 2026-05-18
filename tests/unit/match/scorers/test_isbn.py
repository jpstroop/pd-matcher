"""Tests for :mod:`pd_matcher.match.scorers.isbn`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.isbn import score_isbn
from pd_matcher.models import IndexedNyplRegRecord


def _record() -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(uuid="u", title="t", was_renewed=False)


def test_score_isbn_always_skipped(scorer_context: ScorerContext) -> None:
    """The Phase 4 ISBN scorer is a permanent skip until NYPL exposes ISBNs."""
    ev = score_isbn(("9780000000000",), _record(), scorer_context)
    assert ev.skipped is True
    assert ev.decisive is False
    assert dict(ev.features)["marc_isbn_count"] == 1.0


def test_score_isbn_skipped_when_marc_isbns_empty(scorer_context: ScorerContext) -> None:
    """An empty MARC ISBN tuple also lands in the skipped branch."""
    ev = score_isbn((), _record(), scorer_context)
    assert ev.skipped is True
    assert dict(ev.features)["marc_isbn_count"] == 0.0
