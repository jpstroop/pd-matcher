"""Tests for :mod:`pd_matcher.match.scorers.name`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher


def test_score_author_identical_inputs(scorer_context: ScorerContext) -> None:
    """Identical author strings yield score == max."""
    ev = score_author("Smith, John", "Smith, John", scorer_context)
    assert ev.score == ev.max == 100.0


def test_score_author_reordered_tokens_still_max(scorer_context: ScorerContext) -> None:
    """Token reordering is the canonical token-set-ratio win."""
    ev = score_author("Smith, John", "John Smith", scorer_context)
    assert ev.score == 100.0


def test_score_author_partial_overlap_between_zero_and_max(
    scorer_context: ScorerContext,
) -> None:
    """A partial overlap should land in ``(0, 100)``."""
    ev = score_author("Smith, John", "Smith, Jane", scorer_context)
    assert 0.0 < ev.score < 100.0


def test_score_author_skipped_when_marc_none(scorer_context: ScorerContext) -> None:
    """A None MARC author triggers the skipped path."""
    ev = score_author(None, "Smith, John", scorer_context)
    assert ev.skipped is True


def test_score_author_skipped_when_inputs_collapse_to_empty(
    scorer_context: ScorerContext,
) -> None:
    """Punctuation-only inputs collapse to empty tokens and are skipped."""
    ev = score_author("...", "Smith", scorer_context)
    assert ev.skipped is True


def test_score_publisher_identical_inputs(scorer_context: ScorerContext) -> None:
    """Identical publisher strings yield score == max."""
    ev = score_publisher("Acme Press", "Acme Press", scorer_context)
    assert ev.score == 100.0


def test_score_publisher_skipped_when_either_empty(scorer_context: ScorerContext) -> None:
    """An empty publisher on either side triggers the skipped path."""
    assert score_publisher("Acme", "", scorer_context).skipped is True
    assert score_publisher(None, "Acme", scorer_context).skipped is True


def test_score_publisher_handles_unicode(scorer_context: ScorerContext) -> None:
    """Diacritics should not throw off scoring after normalization."""
    ev = score_publisher("Éditions Beta", "Editions Beta", scorer_context)
    assert ev.score == 100.0


def test_score_publisher_features_include_lengths_and_overlap(
    scorer_context: ScorerContext,
) -> None:
    """Features expose normalized lengths and token overlap counts."""
    ev = score_publisher("Acme Press", "Acme Press", scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["normalized_marc_len"] > 0.0
    assert feature_map["normalized_nypl_len"] > 0.0
    assert feature_map["token_overlap"] >= 1.0


def test_score_publisher_disjoint_tokens_below_floor_zeroed(
    scorer_context: ScorerContext,
) -> None:
    """Unrelated single-token names collapse to zero via the disjoint floor."""
    ev = score_publisher("Maruzen", "Peter Chiarulli", scorer_context)
    assert ev.score == 0.0
    assert dict(ev.features)["token_overlap"] == 0.0


def test_score_author_disjoint_tokens_below_floor_zeroed(
    scorer_context: ScorerContext,
) -> None:
    """The disjoint floor applies symmetrically to the author scorer."""
    ev = score_author("Maruzen", "Peter Chiarulli", scorer_context)
    assert ev.score == 0.0


def test_score_publisher_disjoint_typo_above_floor_preserved(
    scorer_context: ScorerContext,
) -> None:
    """Single-character typos clear the floor and keep the raw ratio."""
    ev = score_publisher("Wonder", "Woncler", scorer_context)
    assert ev.score > 70.0
    assert dict(ev.features)["token_overlap"] == 0.0


def test_score_publisher_disjoint_in_floor_band_preserved(
    scorer_context: ScorerContext,
) -> None:
    """Disjoint pairs in the (50, 70) band sit above the floor and are kept.

    ``token_set_ratio('smith', 'smyht') == 60``: a two-character permutation
    yields a real-signal score that the chosen floor of 50 must preserve.
    """
    ev = score_publisher("Smith", "Smyht", scorer_context)
    assert 50.0 < ev.score < 70.0
    assert dict(ev.features)["token_overlap"] == 0.0


def test_score_publisher_overlapping_tokens_unaffected(
    scorer_context: ScorerContext,
) -> None:
    """Token intersection bypasses the floor entirely."""
    ev = score_publisher("Macmillan", "Macmillan & Co", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] >= 1.0
