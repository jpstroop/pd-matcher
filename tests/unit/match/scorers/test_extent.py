"""Tests for :mod:`pd_matcher.match.scorers.extent`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.extent import extract_page_count
from pd_matcher.match.scorers.extent import score_extent


def testextract_page_count_handles_marc_300a_plain() -> None:
    """A plain ``"312 p."`` extracts 312."""
    assert extract_page_count("312 p.") == 312


def testextract_page_count_strips_roman_front_matter() -> None:
    """Roman-numeral front matter is stripped before integer extraction."""
    assert extract_page_count("xii, 312 p.") == 312


def testextract_page_count_handles_bracketed_pages() -> None:
    """``"[8], 312 p."`` extracts 312 (the largest count after stripping)."""
    assert extract_page_count("[8], 312 p.") == 312


def testextract_page_count_handles_long_word_form() -> None:
    """``"312 pages"`` extracts 312."""
    assert extract_page_count("312 pages") == 312


def testextract_page_count_takes_largest_for_multivolume_paren() -> None:
    """``"1 v. (312 p.)"`` picks 312 over the leading volume count."""
    assert extract_page_count("1 v. (312 p.)") == 312


def testextract_page_count_skips_loose_leaf_extents() -> None:
    """``"v. (loose-leaf)"`` has no integer and yields ``None``."""
    assert extract_page_count("v. (loose-leaf)") is None


def testextract_page_count_skips_unpaged_extents() -> None:
    """``"unpaged"`` has no integer and yields ``None``."""
    assert extract_page_count("unpaged") is None


def testextract_page_count_handles_cce_prefix_form() -> None:
    """``"p. 312"`` (CCE prefix form) extracts 312."""
    assert extract_page_count("p. 312") == 312


def testextract_page_count_skips_cce_cm_only() -> None:
    """``"p. cm."`` has no integer and yields ``None``."""
    assert extract_page_count("p. cm.") is None


def testextract_page_count_handles_illustrations_suffix() -> None:
    """``"312 p. illus."`` extracts 312."""
    assert extract_page_count("312 p. illus.") == 312


def testextract_page_count_returns_none_for_empty() -> None:
    """An empty string yields ``None``."""
    assert extract_page_count("") is None


def testextract_page_count_returns_none_for_none() -> None:
    """``None`` input yields ``None``."""
    assert extract_page_count(None) is None


def testextract_page_count_returns_none_when_only_zero() -> None:
    """``"0 p."`` has no positive integer and yields ``None``."""
    assert extract_page_count("0 p.") is None


def testextract_page_count_skips_bare_volume_count() -> None:
    """``"3 v"`` is a volume count, not 3 pages — yields ``None`` (skip)."""
    assert extract_page_count("3 v") is None


def testextract_page_count_skips_bare_volume_count_with_period() -> None:
    """``"1 v."`` is a volume count, not 1 page — yields ``None`` (skip)."""
    assert extract_page_count("1 v.") is None


def testextract_page_count_skips_plural_volume_abbreviations() -> None:
    """``"2 vols."`` and ``"3 volumes"`` are volume counts and yield ``None``."""
    assert extract_page_count("2 vols.") is None
    assert extract_page_count("3 volumes") is None


def testextract_page_count_keeps_page_count_past_volume_count() -> None:
    """``"1 v. (312 p.)"`` strips the volume count and keeps the page count."""
    assert extract_page_count("1 v. (312 p.)") == 312


def testextract_page_count_skips_volumes_in_bindings_form() -> None:
    """``"5 v. in 10"`` is volumes-in-bindings, not 10 pages — yields ``None`` (pair 377)."""
    assert extract_page_count("5 v. in 10") is None
    assert extract_page_count("5 v in 10") is None


def test_score_extent_volume_counts_skip_not_match(scorer_context: ScorerContext) -> None:
    """``"3 v"`` vs ``"1 v."`` must skip, not score a false 1.0 (pair 295)."""
    ev = score_extent("3 v", "1 v.", scorer_context)
    assert ev.skipped is True
    assert ev.score == 0.0


def test_score_extent_zero_delta_is_max(scorer_context: ScorerContext) -> None:
    """Identical page counts produce the maximum score."""
    ev = score_extent("312 p.", "312 p.", scorer_context)
    assert ev.skipped is False
    assert ev.score == ev.max


def test_score_extent_small_delta_is_max(scorer_context: ScorerContext) -> None:
    """A delta of 2 still scores at the max (tolerance for foreword counts)."""
    ev = score_extent("312 p.", "310 p.", scorer_context)
    assert ev.score == ev.max


def test_score_extent_mid_delta_is_partial(scorer_context: ScorerContext) -> None:
    """A delta of 5 scores 100 - 5*(5-2) = 85."""
    ev = score_extent("315 p.", "310 p.", scorer_context)
    assert ev.score == 85.0


def test_score_extent_delta_ten_scores_sixty(scorer_context: ScorerContext) -> None:
    """A delta of 10 scores 100 - 5*(10-2) = 60."""
    ev = score_extent("320 p.", "310 p.", scorer_context)
    assert ev.score == 60.0


def test_score_extent_delta_fifteen_scores_thirty_five(scorer_context: ScorerContext) -> None:
    """A delta of 15 scores 100 - 5*(15-2) = 35."""
    ev = score_extent("325 p.", "310 p.", scorer_context)
    assert ev.score == 35.0


def test_score_extent_large_delta_floors_at_zero(scorer_context: ScorerContext) -> None:
    """A delta of 100 floors at 0.0 (catches whole/part mismatches)."""
    ev = score_extent("412 p.", "312 p.", scorer_context)
    assert ev.score == 0.0


def test_score_extent_unparseable_both_skipped(scorer_context: ScorerContext) -> None:
    """When both sides have no integer the evidence is skipped."""
    ev = score_extent("unpaged", "v. (loose-leaf)", scorer_context)
    assert ev.skipped is True


def test_score_extent_unparseable_marc_skipped(scorer_context: ScorerContext) -> None:
    """When MARC side has no integer the evidence is skipped."""
    ev = score_extent("unpaged", "312 p.", scorer_context)
    assert ev.skipped is True


def test_score_extent_unparseable_cce_skipped(scorer_context: ScorerContext) -> None:
    """When CCE side has no integer the evidence is skipped."""
    ev = score_extent("312 p.", "v. cm.", scorer_context)
    assert ev.skipped is True


def test_score_extent_strips_roman_on_both_sides(scorer_context: ScorerContext) -> None:
    """Roman-numeral front matter is excluded from comparison on both sides."""
    ev = score_extent("xii, 312 p.", "viii, 312 p.", scorer_context)
    assert ev.score == ev.max


def test_score_extent_emits_features(scorer_context: ScorerContext) -> None:
    """The emitted Evidence includes parsed page counts and delta as features."""
    ev = score_extent("315 p.", "310 p.", scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_pages"] == 315.0
    assert feature_map["cce_pages"] == 310.0
    assert feature_map["delta"] == 5.0


def test_score_extent_features_when_skipped(scorer_context: ScorerContext) -> None:
    """Skipped evidence still records what was parsed (-1.0 when absent)."""
    ev = score_extent("unpaged", "310 p.", scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_pages"] == -1.0
    assert feature_map["cce_pages"] == 310.0
