"""Tests for :mod:`pd_matcher.match.scorers.title`."""

from collections.abc import Callable

from msgspec import structs

from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.title import TITLE_WINDOW_NOTE
from pd_matcher.match.scorers.title import _align_tokens
from pd_matcher.match.scorers.title import _best_window_score
from pd_matcher.match.scorers.title import _shared_weight
from pd_matcher.match.scorers.title import _strip_cross_field
from pd_matcher.match.scorers.title import prepare_cross_field_stems
from pd_matcher.match.scorers.title import score_title


def _windowless(ctx: ScorerContext) -> ScorerContext:
    """Return a copy of ``ctx`` whose config disables the sliding window (#133)."""
    return ScorerContext(
        language=ctx.language,
        stopwords=ctx.stopwords,
        stemmer=ctx.stemmer,
        idf=ctx.idf,
        author_idf=ctx.author_idf,
        publisher_idf=ctx.publisher_idf,
        config=structs.replace(ctx.config, title_window_trigger_ratio=0.0),
    )


def _with_cross_field(ctx: ScorerContext, stems: frozenset[str]) -> ScorerContext:
    """Return a copy of ``ctx`` carrying ``stems`` as cross-field title stems."""
    return ScorerContext(
        language=ctx.language,
        stopwords=ctx.stopwords,
        stemmer=ctx.stemmer,
        idf=ctx.idf,
        author_idf=ctx.author_idf,
        publisher_idf=ctx.publisher_idf,
        config=ctx.config,
        publisher_alias_index=ctx.publisher_alias_index,
        cross_field_title_stems=stems,
    )


def test_score_title_identical_inputs_max_score(scorer_context: ScorerContext) -> None:
    """Identical inputs produce score == max with no unique tokens."""
    ev = score_title("A study of widgets", "A study of widgets", scorer_context)
    assert ev.score == ev.max == 100.0
    assert ev.skipped is False
    feature_map = dict(ev.features)
    assert feature_map["unique_to_marc"] == 0.0
    assert feature_map["unique_to_nypl"] == 0.0
    assert feature_map["token_overlap"] > 0.0


def test_score_title_emits_token_length_subfeatures(
    scorer_context: ScorerContext,
) -> None:
    """The learned combiner reads per-side token-set lengths off the features."""
    ev = score_title("A study of widgets", "Widgets and machines", scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["marc_token_len"] == 2.0
    assert feature_map["nypl_token_len"] == 2.0


def test_score_title_skipped_when_marc_empty(scorer_context: ScorerContext) -> None:
    """An empty MARC title triggers the skipped branch."""
    ev = score_title("", "A study of widgets", scorer_context)
    assert ev.skipped is True
    assert ev.score == 0.0


def test_score_title_skipped_when_nypl_none(scorer_context: ScorerContext) -> None:
    """A None NYPL title triggers the skipped branch."""
    ev = score_title("A study of widgets", None, scorer_context)
    assert ev.skipped is True


def test_score_title_skipped_when_all_tokens_are_stopwords(
    scorer_context: ScorerContext,
) -> None:
    """A title made entirely of stopwords yields no tokens; the scorer skips."""
    ev = score_title("a the of", "the and of", scorer_context)
    assert ev.skipped is True


def test_score_title_partial_overlap_falls_between_zero_and_max(
    scorer_context: ScorerContext,
) -> None:
    """Partial overlap should fall strictly between zero and max."""
    ev = score_title("A study of widgets", "Widgets and machines", scorer_context)
    assert 0.0 < ev.score < 100.0
    feature_map = dict(ev.features)
    assert feature_map["token_overlap"] == 1.0
    assert feature_map["unique_to_marc"] >= 1.0
    assert feature_map["unique_to_nypl"] >= 1.0
    assert feature_map["avg_token_idf"] > 0.0


def test_score_title_disjoint_inputs_score_zero(scorer_context: ScorerContext) -> None:
    """Disjoint tokens score zero."""
    ev = score_title("Albuquerque", "machines", scorer_context)
    assert ev.score == 0.0
    assert ev.skipped is False


def test_score_title_coverage_one_when_shorter_is_subset(
    scorer_context: ScorerContext,
) -> None:
    """A shorter title fully contained in a longer one yields coverage 1.0 (#85).

    The symmetric ``score`` is deflated by the longer side's extra distinctive
    tokens, but the asymmetric coverage sub-feature stays high because the
    shorter side's whole mass is shared. The lengths (3 vs 2 tokens, ratio
    0.667) stay above the sliding-window trigger (0.5), so the symmetric score
    is measured in isolation without the window (#133) rescuing it.
    """
    ev = score_title("Studies of widgets in Albuquerque", "Widgets Albuquerque", scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["coverage"] == 1.0
    assert ev.score < 100.0


def test_score_title_coverage_bidirectional(scorer_context: ScorerContext) -> None:
    """Coverage is high whether the MARC side or the CCE side is the longer one."""
    longer = "Widgets Albuquerque machines studies"
    shorter = "Widgets Albuquerque"
    long_marc = score_title(longer, shorter, scorer_context)
    long_cce = score_title(shorter, longer, scorer_context)
    assert dict(long_marc.features)["coverage"] == 1.0
    assert dict(long_cce.features)["coverage"] == 1.0


def test_score_title_coverage_low_for_disjoint(scorer_context: ScorerContext) -> None:
    """No shared mass means coverage 0.0."""
    ev = score_title("Albuquerque", "machines", scorer_context)
    assert dict(ev.features)["coverage"] == 0.0


def test_score_title_coverage_does_not_change_score(
    scorer_context: ScorerContext,
) -> None:
    """Coverage is a FEATURE only: the token_set score itself is unchanged (#85).

    The subset shape must produce the same symmetric Jaccard ``score`` it
    produced before coverage existed — the v1 regression was lifting this
    score; coverage must not touch it. The lengths (3 vs 2 tokens, ratio 0.667)
    stay above the sliding-window trigger (0.5) so the symmetric score is
    isolated from the window (#133).
    """
    ev = score_title("Studies of widgets in Albuquerque", "Widgets Albuquerque", scorer_context)
    shared = 3.0 + 5.0
    union = shared + 2.5
    expected_raw = shared / union
    assert ev.score == expected_raw * 100.0


def test_score_title_coverage_partial_between_zero_and_one(
    scorer_context: ScorerContext,
) -> None:
    """A shorter side that is only partly shared yields coverage in (0, 1)."""
    ev = score_title("Widgets machines", "Widgets Albuquerque", scorer_context)
    coverage = dict(ev.features)["coverage"]
    assert 0.0 < coverage < 1.0


def test_score_title_returns_zero_when_unseen_tokens_idf_zero(
    scorer_context: ScorerContext,
) -> None:
    """If every token has zero IDF the union sum is zero and the score is zero."""
    ctx = ScorerContext(
        language=scorer_context.language,
        stopwords=scorer_context.stopwords,
        stemmer=scorer_context.stemmer,
        idf=scorer_context.idf.__class__(
            document_count=0,
            default_idf=0.0,
            source_hash="x",
            language="eng",
            idf={},
        ),
        author_idf=scorer_context.author_idf,
        publisher_idf=scorer_context.publisher_idf,
        config=scorer_context.config,
    )
    ev = score_title("unique tokens here", "different ones entirely", ctx)
    assert ev.score == 0.0
    assert ev.skipped is False


def test_score_title_script_mismatch_emits_non_skipped_zero(
    scorer_context: ScorerContext,
) -> None:
    """A Latin/Hebrew script mismatch emits a non-skipped zero, not a skip.

    The Evidence must count in the combiner's denominator so the pair is
    penalized rather than silently dropped.
    """
    ev = score_title("Bereshit bara Elohim", "בראשית ברא אלהים", scorer_context)
    assert ev.skipped is False
    assert ev.score == 0.0
    assert ev.max == 100.0
    assert ("script_mismatch", 1.0) in ev.features


def test_score_title_script_mismatch_cyrillic_emits_zero(
    scorer_context: ScorerContext,
) -> None:
    """Latin vs. Cyrillic mismatch fires the script-mismatch zero."""
    ev = score_title("Voyna i mir", "Война и мир", scorer_context)
    assert ev.skipped is False
    assert ev.score == 0.0


def test_score_title_same_script_does_not_fire_mismatch(
    scorer_context: ScorerContext,
) -> None:
    """Same-script inputs route through the normal token-set path."""
    ev = score_title("A study of widgets", "A study of widgets", scorer_context)
    assert ev.skipped is False
    feature_map = dict(ev.features)
    assert "script_mismatch" not in feature_map


def test_score_title_precomputed_script_is_byte_identical(
    scorer_context: ScorerContext,
) -> None:
    """Passing the precomputed CCE script yields Evidence identical to deriving it."""
    derived = score_title("A study of widgets", "A study of widgets", scorer_context)
    precomputed = score_title(
        "A study of widgets",
        "A study of widgets",
        scorer_context,
        nypl_title_script="LATIN",
    )
    assert precomputed == derived


def test_score_title_precomputed_script_fires_mismatch(
    scorer_context: ScorerContext,
) -> None:
    """The precomputed CCE script drives the mismatch guard without re-derivation."""
    ev = score_title(
        "Voyna i mir",
        "Война и мир",
        scorer_context,
        nypl_title_script="CYRILLIC",
    )
    assert ev.skipped is False
    assert ev.score == 0.0
    assert ("script_mismatch", 1.0) in ev.features


def test_score_title_precomputed_none_falls_back_to_derivation(
    scorer_context: ScorerContext,
) -> None:
    """An absent precomputed script reproduces the self-derived mismatch zero."""
    ev = score_title("Voyna i mir", "Война и мир", scorer_context, nypl_title_script=None)
    assert ev.skipped is False
    assert ev.score == 0.0
    assert ("script_mismatch", 1.0) in ev.features


def test_align_tokens_exact_only_reduces_to_intersection() -> None:
    """With no near-misses the alignment is the plain set intersection."""
    matched, unique_marc, unique_nypl = _align_tokens({"a", "b"}, {"b", "c"})
    assert matched == (("b", "b"),)
    assert unique_marc == frozenset({"a"})
    assert unique_nypl == frozenset({"c"})


def test_align_tokens_fuzzy_recovers_ocr_corrupted_pair() -> None:
    """A single-character OCR corruption aligns to the clean stem (ratio 93)."""
    matched, unique_marc, unique_nypl = _align_tokens({"immunochemistri"}, {"immunochenistri"})
    assert matched == (("immunochemistri", "immunochenistri"),)
    assert unique_marc == frozenset()
    assert unique_nypl == frozenset()


def test_align_tokens_distinct_words_stay_unmatched() -> None:
    """Genuinely different words (ratio 75 < threshold) do not align."""
    matched, unique_marc, unique_nypl = _align_tokens({"work"}, {"word"})
    assert matched == ()
    assert unique_marc == frozenset({"work"})
    assert unique_nypl == frozenset({"word"})


def test_shared_weight_exact_pair_is_token_idf(idf_table: IdfTable) -> None:
    """An exact pair weighs exactly the token's IDF (Jaccard reduction)."""
    assert _shared_weight("widget", "widget", idf_table) == 3.0


def test_score_title_lone_generic_token_discounted(scorer_context: ScorerContext) -> None:
    """A lone low-IDF shared token is discounted below max (#87).

    "small" stems to idf 1.5; with one shared token IDF cancels in the Jaccard
    (raw == 1.0), so the absolute-mass confidence (1.5 / 2.0 == 0.75) is the only
    thing keeping it off a full 100.
    """
    ev = score_title("small", "small", scorer_context)
    assert ev.score == 75.0
    assert ev.skipped is False


def test_score_title_lone_distinctive_token_keeps_max(scorer_context: ScorerContext) -> None:
    """A lone token at or above default_idf clears the confidence gate (#87).

    "widget" stems to idf 3.0 >= default_idf 2.0, so confidence saturates at 1.0
    and the single-token match keeps its full score.
    """
    ev = score_title("widget", "widget", scorer_context)
    assert ev.score == 100.0
    assert ev.skipped is False


def test_score_title_recovers_ocr_corrupted_token(
    scorer_context: ScorerContext,
) -> None:
    """An OCR'd distinctive token (ratio 90) is recovered; a different one is not."""
    recovered = score_title("albuquerqu widget", "alkuquerqu widget", scorer_context)
    distinct = score_title("albuquerqu widget", "machin widget", scorer_context)
    assert dict(recovered.features)["token_overlap"] == 2.0
    assert dict(distinct.features)["token_overlap"] == 1.0
    assert recovered.score > distinct.score


def test_score_title_distinct_short_words_not_fuzzy_aligned(
    scorer_context: ScorerContext,
) -> None:
    """Different short words (ratio 22) stay unique; only the shared token aligns."""
    ev = score_title("small widget", "part widget", scorer_context)
    feature_map = dict(ev.features)
    assert feature_map["token_overlap"] == 1.0
    assert feature_map["unique_to_marc"] == 1.0
    assert feature_map["unique_to_nypl"] == 1.0


def test_score_title_whole_string_rescues_compound_split(
    scorer_context: ScorerContext,
) -> None:
    """A compound split per-token matching misses is rescued by the whole-string ratio.

    "albuquerqu" vs "albu querqu" shares no aligned token (the parts are too
    short to clear the per-token gate), so the Jaccard alone is zero — but the
    concatenated stems are identical, so the whole-string rescue lifts it to max.
    """
    ev = score_title("albuquerqu", "albu querqu", scorer_context)
    assert ev.score == 100.0
    assert dict(ev.features)["token_overlap"] == 0.0


def test_score_title_whole_string_below_gate_does_not_rescue(
    scorer_context: ScorerContext,
) -> None:
    """A sub-gate whole-string ratio (work/word = 75 < 90) leaves the score at zero."""
    ev = score_title("work", "word", scorer_context)
    assert ev.score == 0.0
    assert ev.skipped is False


def test_score_title_whole_string_length_gate_blocks_short_compound(
    scorer_context: ScorerContext,
) -> None:
    """A short compound (joined "redcoat" = 7 < 10) is below the length floor.

    The whole-string ratio is a perfect 100 here, but the concatenation is too
    short for that to be a trustworthy same-title claim, so the rescue does not
    fire and the score stays at the (zero) Jaccard. The identical-but-longer
    "albuquerqu" / "albu querqu" pair (10 chars) clears the floor and is rescued,
    so this isolates the length gate.
    """
    ev = score_title("redcoat", "red coat", scorer_context)
    assert ev.score == 0.0
    assert ev.skipped is False


def test_prepare_cross_field_stems_normalizes_and_merges(
    english_stemmer: Callable[[str], str],
) -> None:
    """Field values are normalized/stopword-dropped/stemmed and merged to a set."""
    stems = prepare_cross_field_stems(
        ("University of Illinois Press", "Chicago"),
        "eng",
        frozenset({"of"}),
        english_stemmer,
    )
    assert stems == frozenset({"universiti", "illinoi", "press", "chicago"})


def test_prepare_cross_field_stems_empty_input_is_empty() -> None:
    """No field values yields an empty stem set (the no-op default)."""
    assert prepare_cross_field_stems((), "eng", frozenset(), str) == frozenset()


def test_strip_cross_field_removes_explained_token(scorer_context: ScorerContext) -> None:
    """A CCE stem explained only by a non-title MARC field is dropped."""
    kept = _strip_cross_field(
        ("studi", "press"),
        ("studi",),
        _with_cross_field(scorer_context, frozenset({"press"})),
    )
    assert kept == ("studi",)


def test_strip_cross_field_keeps_genuine_title_token(scorer_context: ScorerContext) -> None:
    """A stem in the MARC title is never stripped, even if it recurs in a field."""
    kept = _strip_cross_field(
        ("widget", "press"),
        ("widget", "press"),
        _with_cross_field(scorer_context, frozenset({"press", "widget"})),
    )
    assert kept == ("widget", "press")


def test_strip_cross_field_over_strip_guard_returns_original(
    scorer_context: ScorerContext,
) -> None:
    """When stripping would empty the comparand, the original tokens are kept."""
    kept = _strip_cross_field(
        ("press", "universiti"),
        ("widget",),
        _with_cross_field(scorer_context, frozenset({"press", "universiti"})),
    )
    assert kept == ("press", "universiti")


def test_strip_cross_field_no_stems_is_noop(scorer_context: ScorerContext) -> None:
    """An empty cross-field set returns the CCE tokens unchanged."""
    kept = _strip_cross_field(("studi", "press"), ("studi",), scorer_context)
    assert kept == ("studi", "press")


def test_score_title_cross_field_contamination_recovers(
    scorer_context: ScorerContext,
) -> None:
    """Publisher tokens leaked into the CCE title no longer deflate the score.

    The MARC title is "study widget"; the CCE title repeats it but appends the
    publisher "machines" (a high-IDF token MARC keeps in its publisher field).
    Without the cross-field strip those extra tokens are penalized as a title
    difference; with the strip the comparand reduces to the shared title and
    scores higher.
    """
    contaminated = "study widget machines"
    without = score_title("study widget", contaminated, scorer_context)
    ctx = _with_cross_field(scorer_context, frozenset({"machin"}))
    with_strip = score_title("study widget", contaminated, ctx)
    assert with_strip.score > without.score
    assert with_strip.score == 100.0


def test_score_title_genuine_extra_content_still_penalized(
    scorer_context: ScorerContext,
) -> None:
    """Extra CCE tokens not explained by any MARC field still lower the score.

    "machines" here is NOT in the cross-field set, so the strip leaves it in
    place and the title difference is preserved — the change is surgical, not a
    blanket extra-token amnesty.
    """
    ctx = _with_cross_field(scorer_context, frozenset({"albuquerqu"}))
    ev = score_title("study widget", "study widget machines", ctx)
    assert ev.score < 100.0


def test_score_title_cross_field_does_not_strip_shared_title_token(
    scorer_context: ScorerContext,
) -> None:
    """A title token that also appears in a MARC field is retained, not stripped.

    "widget" is in both the MARC title and the (contrived) cross-field set; it
    must survive so the genuine title overlap still counts.
    """
    ctx = _with_cross_field(scorer_context, frozenset({"widget"}))
    ev = score_title("study widget", "study widget", ctx)
    assert ev.score == 100.0


def test_score_title_window_credits_contained_distinctive_short_title(
    scorer_context: ScorerContext,
) -> None:
    """A distinctive short title contained in a long one is lifted by the window (#133).

    "Widgets Albuquerque" (2 tokens) sits inside "Studies of widgets in
    Albuquerque machines" (4 tokens); the length ratio 0.5 clears the 0.5
    trigger. The best 2-token window lands exactly on {widget, albuquerqu}, both
    distinctive, so containment is credited to a full score and the Evidence
    carries the window note.
    """
    ev = score_title(
        "Studies of widgets in Albuquerque machines", "Widgets Albuquerque", scorer_context
    )
    assert ev.score == 100.0
    assert ev.note == TITLE_WINDOW_NOTE


def test_score_title_window_lifts_above_symmetric_score(
    scorer_context: ScorerContext,
) -> None:
    """The window only ever raises the score; the same pair scores lower windowless."""
    windowed = score_title(
        "Studies of widgets in Albuquerque machines", "Widgets Albuquerque", scorer_context
    )
    windowless = score_title(
        "Studies of widgets in Albuquerque machines",
        "Widgets Albuquerque",
        _windowless(scorer_context),
    )
    assert windowed.score > windowless.score
    assert windowless.note is None


def test_score_title_window_is_side_agnostic(scorer_context: ScorerContext) -> None:
    """Containment is credited whether the long side is MARC or CCE."""
    long_side = "Studies of widgets in Albuquerque machines"
    short_side = "Widgets Albuquerque"
    long_marc = score_title(long_side, short_side, scorer_context)
    long_cce = score_title(short_side, long_side, scorer_context)
    assert long_marc.score == 100.0
    assert long_cce.score == 100.0
    assert long_marc.note == TITLE_WINDOW_NOTE
    assert long_cce.note == TITLE_WINDOW_NOTE


def test_score_title_window_generic_token_discounted_below_distinctive(
    scorer_context: ScorerContext,
) -> None:
    """A window matched on a common token scores below one matched on a rare token.

    IDF weighting is the intrinsic generic-title guard: a lone contained
    "american" (idf 1.0) carries half the mass floor, so its window confidence —
    and score — stays well below a lone contained "widget" (idf 3.0), which
    saturates. Containment alone never buys a full score on filler.
    """
    generic = score_title("Studies of american machines", "american", scorer_context)
    distinctive = score_title("Studies of widget machines", "widget", scorer_context)
    assert generic.score < distinctive.score
    assert distinctive.score == 100.0


def test_score_title_window_not_triggered_above_ratio(scorer_context: ScorerContext) -> None:
    """A length ratio above the trigger (3 vs 2 tokens = 0.667 > 0.5) skips the window."""
    ev = score_title("Studies of widgets in Albuquerque", "Widgets Albuquerque", scorer_context)
    assert ev.note is None
    assert ev.score < 100.0


def test_score_title_window_disabled_by_zero_trigger(scorer_context: ScorerContext) -> None:
    """A ``title_window_trigger_ratio`` of 0.0 disables the window entirely."""
    ev = score_title(
        "Studies of widgets in Albuquerque machines",
        "Widgets Albuquerque",
        _windowless(scorer_context),
    )
    assert ev.note is None
    assert ev.score < 100.0


def test_best_window_score_empty_side_returns_zero(idf_table: IdfTable) -> None:
    """An empty token sequence yields no window score."""
    assert _best_window_score((), ("widget", "studi"), idf_table, 2.0, 0.6) == 0.0


def test_best_window_score_equal_length_returns_zero(idf_table: IdfTable) -> None:
    """Equal-length sequences never slide (there is nothing to contain)."""
    assert (
        _best_window_score(("widget", "studi"), ("albuquerqu", "machin"), idf_table, 2.0, 0.6)
        == 0.0
    )


def test_best_window_score_disabled_trigger_returns_zero(idf_table: IdfTable) -> None:
    """A non-positive trigger ratio short-circuits before any sliding."""
    assert (
        _best_window_score(("widget",), ("studi", "widget", "machin"), idf_table, 2.0, 0.0) == 0.0
    )
