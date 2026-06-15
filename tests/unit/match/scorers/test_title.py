"""Tests for :mod:`pd_matcher.match.scorers.title`."""

from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.title import _align_tokens
from pd_matcher.match.scorers.title import _shared_weight
from pd_matcher.match.scorers.title import score_title


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
