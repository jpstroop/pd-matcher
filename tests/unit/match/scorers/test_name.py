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
    author_idf_table: IdfTable,
    publisher_idf_table: IdfTable,
    alias_index: dict[str, str],
) -> ScorerContext:
    """Return an English :class:`ScorerContext` with the bundled alias index."""
    return ScorerContext(
        language="eng",
        stopwords=load_stopwords("eng"),
        stemmer=stemmer_for("eng"),
        idf=idf_table,
        author_idf=author_idf_table,
        publisher_idf=publisher_idf_table,
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
    """Token intersection on a distinctive token bypasses the floor entirely.

    ``Macmillan`` (marc) is a full token-subset of ``Macmillan & Co`` (nypl,
    "& co" stopped), so the gate's full-overlap regime keeps the raw 100.0.
    """
    ev = score_publisher("Macmillan", "Macmillan & Co", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] >= 1.0


def test_score_publisher_generic_only_overlap_gated_to_low(
    scorer_context: ScorerContext,
) -> None:
    """Sharing only a low-IDF token ("oxford") collapses a partial overlap.

    ``oxford`` carries a low IDF in the publisher fixture, so when it is the
    only shared token and both sides diverge elsewhere, the gate
    (most-distinctive shared token / default_idf) drives the score near zero
    even though ``token_set_ratio`` alone would inflate it.
    """
    ev = score_publisher("Oxford Brothers", "Oxford Sisters", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score < 20.0


def test_score_publisher_distinctive_token_overlap_stays_high(
    scorer_context: ScorerContext,
) -> None:
    """A distinctive (high-IDF) shared token keeps a real match high.

    ``Knopf`` is rare in the publisher IDF table, so when the marc side is
    the distinctive token plus a moderate extra one the distinctive-hit term
    keeps the gate near ``1.0`` and the high raw ratio survives — Jaccard
    alone would over-penalize the unshared extra token.
    """
    ev = score_publisher("Alfred Knopf", "Knopf", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score > 70.0


def test_score_publisher_distinctive_typo_preserves_fuzzy(
    scorer_context: ScorerContext,
) -> None:
    """An OCR typo on a distinctive token keeps the disjoint-fuzzy score.

    ``Macmillan`` vs ``Macmillian`` share no token after the typo, so the
    gate never applies; the disjoint-fuzzy path keeps the high near-match
    ratio, preserving tolerance for transcription noise.
    """
    ev = score_publisher("Macmillan", "Macmillian", scorer_context)
    assert dict(ev.features)["token_overlap"] == 0.0
    assert ev.score > 70.0


def test_score_publisher_generic_overlap_short_string_still_gated(
    scorer_context: ScorerContext,
) -> None:
    """Generic-only overlap stays low however short either side is.

    The shared token ``oxford`` is the entirety of the marc side here, but
    because the nypl side carries a distinctive token (``knopf``) the shared
    set equals neither full side, so the gate fires and the short-string
    inflation ``token_set_ratio`` would otherwise produce is removed.
    """
    ev = score_publisher("Oxford", "Oxford Knopf", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score < 20.0


def test_score_author_generic_only_overlap_gated_to_low(
    scorer_context: ScorerContext,
) -> None:
    """The author gate discounts overlap on a common surname alone."""
    ev = score_author("John Smith", "Jane Smith", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score < 20.0


def test_score_author_lone_common_token_subset_crushed(
    scorer_context: ScorerContext,
) -> None:
    """A one-word author sharing only a common given name scores near zero.

    Issue #83: ``smith`` is a low-IDF common surname, so a one-word side that
    is a strict subset of a longer side (``token_set_ratio`` inflates to 100)
    is crushed by the single-shared-token floor instead of surviving on
    coverage's short-string credit.
    """
    ev = score_author("Smith", "Smith, John", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score < 20.0


def test_score_author_lone_common_initial_subset_crushed(
    scorer_context: ScorerContext,
) -> None:
    """A bare shared initial does not inflate a one-word-vs-longer match.

    ``john`` stands in for the lone-initial / common-token class; sharing it
    alone against a longer name is driven toward zero (issue #83).
    """
    ev = score_author("John", "John Smith", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score < 20.0


def test_score_author_lone_distinctive_token_subset_stays_high(
    scorer_context: ScorerContext,
) -> None:
    """A one-word side sharing a rare token clears the single-token floor.

    ``albuquerque`` is near ``default_idf``; a lone shared distinctive token
    is strong evidence (a mononym / rare surname), so the single-token floor
    leaves it high rather than crushing it (issue #83 backfire guard).
    """
    ev = score_author("Albuquerque", "Albuquerque, John", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score > 70.0


def test_score_author_identical_single_token_stays_max(
    scorer_context: ScorerContext,
) -> None:
    """An exact one-word match is exempt from the single-token floor.

    The two sets are equal so coverage is ``1.0`` and the floor never fires —
    a mononym matching itself keeps its full 100 even when the token is
    common (issue #83 backfire guard).
    """
    ev = score_author("Smith", "Smith", scorer_context)
    assert ev.score == 100.0


def test_score_publisher_lone_common_token_subset_crushed(
    scorer_context: ScorerContext,
) -> None:
    """The single-token floor applies symmetrically on the publisher side.

    ``oxford`` is low-IDF in the publisher fixture; a one-word side that is a
    subset of a longer side is crushed instead of inflating (issue #83).
    """
    ev = score_publisher("Oxford", "Oxford Macmillan", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score < 20.0


def test_score_publisher_lone_distinctive_token_subset_stays_high(
    scorer_context: ScorerContext,
) -> None:
    """A lone distinctive house token survives the single-token floor.

    ``knopf`` is rare, so a one-word side sharing only it against a longer
    side keeps a high score — distinctive corporate evidence is preserved
    (issue #83 backfire guard).
    """
    ev = score_publisher("Knopf", "Alfred Knopf", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score > 70.0


def test_score_author_multi_token_overlap_unaffected_by_floor(
    scorer_context: ScorerContext,
) -> None:
    """Sharing two tokens bypasses the single-token floor entirely.

    A two-token overlap keeps the ``max(coverage, distinctive_hit)`` path, so
    a real reordered match stays at 100 (issue #83 only narrows the lone-token
    regime).
    """
    ev = score_author("Albuquerque, John", "John Albuquerque", scorer_context)
    assert dict(ev.features)["token_overlap"] == 2.0
    assert ev.score == 100.0


def test_score_author_distinctive_token_overlap_stays_high(
    scorer_context: ScorerContext,
) -> None:
    """A rare shared author token keeps a real match high.

    ``Albuquerque`` is rare in the author IDF table, so the distinctive-hit
    term keeps the gate near ``1.0`` and the subset match ("Albuquerque" is
    the whole nypl side) keeps its raw ratio.
    """
    ev = score_author("Albuquerque, John", "Albuquerque", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score > 70.0


def test_score_publisher_oxford_vs_hawaii_now_low(
    scorer_context: ScorerContext,
) -> None:
    """Separation-test pair #5: a total non-match no longer inflates.

    "Oxford University Press for the Royal Institute of International
    Affairs" vs "University of Hawaii Press" shared only ``university``
    before; with the institutional stopword promotion it now shares nothing
    and the disjoint-fuzzy floor zeroes it.
    """
    ev = score_publisher(
        "Oxford University Press for the Royal Institute of International Affairs",
        "University of Hawaii Press",
        scorer_context,
    )
    assert ev.score < 10.0


def test_score_publisher_alias_lift_survives_generic_gate(
    scorer_context: ScorerContext,
    alias_index: dict[str, str],
) -> None:
    """A curated alias hit overrides a gate-suppressed fuzzy baseline.

    The alias floor is a ``max`` over the (now gated) fuzzy score, so a
    curated imprint hit still lifts to the floor even when the literal
    overlap was generic.
    """
    ev = score_publisher(
        "Whittlesey House",
        "McGraw-Hill Book Company",
        scorer_context,
        alias_index=alias_index,
    )
    assert ev.score >= 95.0


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


def test_score_author_initial_unifies_with_full_name(
    scorer_context: ScorerContext,
) -> None:
    """A bare surname initial matches the spelled-out name at full IDF (#119).

    ``"Faulkner, M."`` prepares to ``(faulkner, m)`` and ``"Faulkner, Morris"``
    to ``(faulkner, morris)``; the initial ``m`` unifies onto ``morris`` so both
    tokens are shared and the pair scores a full match.
    """
    ev = score_author("Faulkner, M.", "Faulkner, Morris", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] == 2.0


def test_score_author_initial_unifies_without_trailing_period(
    scorer_context: ScorerContext,
) -> None:
    """The initial need not carry a period to unify (#119)."""
    ev = score_author("Whitehead, L", "Whitehead, Leslie", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] == 2.0


def test_score_author_double_initial_pseudonym_unifies(
    scorer_context: ScorerContext,
) -> None:
    """Both initials of an all-initials form unify onto the spelled-out name.

    ``"H.D."`` prepares to ``(h, d)`` post-#118 (neither letter is a numbering
    Roman numeral) and ``"Doolittle, Hilda"`` to ``(doolittle, hilda)``; ``h``
    unifies onto ``hilda`` and ``d`` onto ``doolittle`` for a full match (#119).
    """
    ev = score_author("H.D.", "Doolittle, Hilda", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] == 2.0


def test_score_author_initial_unification_is_symmetric(
    scorer_context: ScorerContext,
) -> None:
    """The initial may sit on the CCE side and still unify (#119)."""
    ev = score_author("Faulkner, Morris", "Faulkner, M.", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] == 2.0


def test_score_author_multiple_initials_all_unify(
    scorer_context: ScorerContext,
) -> None:
    """Several initials each unify onto their own compatible full token (#119).

    ``"Eliot, T. S."`` → ``(eliot, t, s)`` and ``"Eliot, Thomas Stearns"`` →
    ``(eliot, thomas, stearns)``; ``t``→``thomas`` and ``s``→``stearns`` give a
    full three-token match.
    """
    ev = score_author("Eliot, T. S.", "Eliot, Thomas Stearns", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] == 3.0


def test_score_author_second_initial_not_double_credited(
    scorer_context: ScorerContext,
) -> None:
    """Two like initials cannot both collapse onto one full token (#119).

    ``"J. J. Smith"`` → ``(j, j, smith)`` and ``"John Smith"`` → ``(john,
    smith)``. The first ``j`` unifies onto ``john``; the second ``j`` finds no
    unconsumed ``j``-token and stays, so the overlap is two (``john``, ``smith``)
    rather than a spurious full match.
    """
    ev = score_author("J. J. Smith", "John Smith", scorer_context)
    assert dict(ev.features)["token_overlap"] == 2.0
    assert ev.score < 100.0


def test_score_author_incompatible_initial_not_unified(
    scorer_context: ScorerContext,
) -> None:
    """An initial with no first-letter-compatible full token stays as-is (#119).

    ``"Rankin, E."`` → ``(rankin, e)`` and ``"Rankin, Morris"`` → ``(rankin,
    morris)``; ``e`` matches neither full token, so only ``rankin`` is shared.
    """
    ev = score_author("Rankin, E.", "Rankin, Morris", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score < 100.0


def test_score_publisher_initial_path_unchanged_by_unification(
    scorer_context: ScorerContext,
) -> None:
    """The publisher scorer never unifies initials — behavior is pinned (#119).

    ``"Smith, J."`` vs ``"Smith, John"`` shares only ``smith`` (the initial
    ``j`` is not unified onto ``john``), so the evidence is byte-identical to the
    pre-#119 single-shared-token path. Any leak of the author-only unification
    into :func:`_evidence` would raise the overlap to 2 and the score above this
    pinned value.
    """
    ev = score_publisher("Smith, J.", "Smith, John", scorer_context)
    assert dict(ev.features)["token_overlap"] == 1.0
    assert ev.score == 83.33333333333333
