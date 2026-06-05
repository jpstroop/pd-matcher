"""Tests for :mod:`pd_matcher.match.scorers.name`."""

from pytest import fixture

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.normalize.publishers import DEFAULT_PUBLISHER_TABLE_PATH
from pd_matcher.normalize.publishers import build_alias_index
from pd_matcher.normalize.publishers import get_default_alias_index
from pd_matcher.normalize.publishers import load_publisher_table
from pd_matcher.normalize.stemming import stemmer_for
from pd_matcher.normalize.stopwords import load_stopwords


@fixture
def alias_index() -> dict[str, str]:
    """Return the bundled alias index for the publisher scorer tests."""
    return build_alias_index(load_publisher_table(DEFAULT_PUBLISHER_TABLE_PATH))


@fixture
def alias_scorer_context(
    matching_config: MatchingConfig,
    idf_table: IdfTable,
    alias_index: dict[str, str],
) -> ScorerContext:
    """Return an English :class:`ScorerContext` with the bundled alias index."""
    return ScorerContext(
        language="eng",
        stopwords=load_stopwords("eng"),
        stemmer=stemmer_for("eng"),
        idf=idf_table,
        config=matching_config,
        publisher_alias_index=alias_index,
    )


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


def test_score_publisher_alias_hit_lifts_imprint_to_parent(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """``Whittlesey House`` / ``McGraw-Hill`` resolve to the same canonical."""
    ev = score_publisher(
        "Whittlesey House",
        "McGraw-Hill Book Company",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.score >= 95.0


def test_score_publisher_alias_hit_does_not_lift_mismatched_canonicals(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """Different canonicals fall through to the fuzzy baseline."""
    with_alias = score_publisher(
        "Whittlesey House",
        "Random House",
        scorer_context,
        alias_index=alias_index,
    )
    without_alias = score_publisher(
        "Whittlesey House",
        "Random House",
        scorer_context,
    )
    assert with_alias.score == without_alias.score
    assert with_alias.score < 95.0


def test_score_publisher_default_path_unchanged_without_alias_index(
    scorer_context: ScorerContext,
) -> None:
    """Omitting ``alias_index`` keeps the legacy fuzzy baseline."""
    ev = score_publisher("Whittlesey House", "McGraw-Hill", scorer_context)
    assert ev.score < 95.0


def test_score_publisher_perfect_match_preserved_under_alias_path(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """``max(fuzzy, floor)`` keeps perfect matches at 100.0."""
    ev = score_publisher(
        "McGraw-Hill Book Company",
        "McGraw-Hill Book Company",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.score == 100.0


def test_score_publisher_alias_path_via_context(
    alias_scorer_context: ScorerContext,
) -> None:
    """A populated ``ctx.publisher_alias_index`` lifts the score without a kwarg."""
    ev = score_publisher(
        "Whittlesey House",
        "McGraw-Hill Book Company",
        alias_scorer_context,
    )
    assert ev.score >= 95.0


def test_score_publisher_kwarg_overrides_context_alias_index(
    alias_scorer_context: ScorerContext,
) -> None:
    """Passing an empty alias index disables the lift even when ctx has one."""
    ev = score_publisher(
        "Whittlesey House",
        "McGraw-Hill Book Company",
        alias_scorer_context,
        alias_index={},
    )
    assert ev.score < 95.0


def test_score_publisher_alias_skipped_when_input_normalizes_empty(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """A stopword-only input never matches a canonical and stays on fuzzy."""
    ev = score_publisher(
        "The Company & Co.",
        "McGraw-Hill",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.score < 95.0


def test_score_publisher_alias_skipped_when_publisher_unknown(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """A publisher not in the table receives no lift."""
    ev = score_publisher(
        "Whittlesey House",
        "Some Unknown House",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.score < 95.0


def test_score_publisher_skipped_evidence_not_lifted(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """Skipped evidence (empty input) is never lifted by the alias path."""
    ev = score_publisher(
        None,
        "McGraw-Hill Book Company",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.skipped is True
    assert ev.score == 0.0


def test_get_default_alias_index_resolves_anchor_pairs(
    scorer_context: ScorerContext,
) -> None:
    """The bundled default index lifts the known anchor pairs."""
    index = get_default_alias_index()
    ev = score_publisher(
        "Aldus Books",
        "Doubleday & Company",
        scorer_context,
        alias_index=index,
    )
    assert ev.score >= 95.0


def test_score_publisher_alias_hit_stamps_canonical_on_note(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """An alias-lifted Evidence carries the human canonical on ``note``."""
    ev = score_publisher(
        "Whittlesey House",
        "McGraw-Hill Book Company",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.note == "McGraw-Hill Book Company"


def test_score_publisher_no_alias_hit_leaves_note_none(
    scorer_context: ScorerContext,
) -> None:
    """The legacy fuzzy path emits ``note=None`` (no breadcrumb to surface)."""
    ev = score_publisher("Whittlesey House", "McGraw-Hill", scorer_context)
    assert ev.note is None


def test_score_publisher_perfect_match_does_not_overwrite_note(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """A perfect literal match short-circuits before the note-stamping path."""
    ev = score_publisher(
        "McGraw-Hill Book Company",
        "McGraw-Hill Book Company",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.note is None
