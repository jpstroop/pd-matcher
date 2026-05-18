"""Tests for :mod:`pd_matcher.match.scorers.edition`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.edition import score_edition


def test_score_edition_skipped_when_either_empty(scorer_context: ScorerContext) -> None:
    """Empty edition strings on either side skip."""
    assert score_edition(None, "1st", scorer_context).skipped is True
    assert score_edition("1st", "", scorer_context).skipped is True


def test_score_edition_skipped_when_normalisation_yields_empty(
    scorer_context: ScorerContext,
) -> None:
    """Punctuation-only inputs collapse to empty after normalization and skip."""
    ev = score_edition("---", "1st", scorer_context)
    assert ev.skipped is True


def test_score_edition_explicit_match(scorer_context: ScorerContext) -> None:
    """Numeric extraction recognises ``1st`` and ``First`` as equal."""
    ev = score_edition("1st ed.", "First edition", scorer_context)
    assert ev.score == 100.0
    feature_map = dict(ev.features)
    assert feature_map["marc_edition_num"] == 1.0
    assert feature_map["nypl_edition_num"] == 1.0


def test_score_edition_explicit_mismatch(scorer_context: ScorerContext) -> None:
    """Two extractable but different numbers score zero with a mismatch flag."""
    ev = score_edition("1st ed.", "2nd edition", scorer_context)
    assert ev.score == 0.0
    assert dict(ev.features)["explicit_mismatch"] == 1.0


def test_score_edition_fuzzy_fallback_when_no_number(scorer_context: ScorerContext) -> None:
    """When neither side has a number, the scorer falls back to fuzzy comparison."""
    ev = score_edition("revised", "revised", scorer_context)
    assert ev.score == 100.0
    feature_map = dict(ev.features)
    assert feature_map["marc_edition_num"] == -1.0
    assert feature_map["nypl_edition_num"] == -1.0
    assert feature_map["explicit_mismatch"] == 0.0


def test_score_edition_fuzzy_fallback_when_one_side_has_number(
    scorer_context: ScorerContext,
) -> None:
    """An asymmetric ``has-number`` situation also takes the fuzzy fallback."""
    ev = score_edition("revised", "1st edition", scorer_context)
    assert 0.0 <= ev.score < 100.0
    assert dict(ev.features)["explicit_mismatch"] == 0.0
