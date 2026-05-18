"""Tests for :mod:`pd_matcher.match.scorers.title`."""

from pd_matcher.match.scorers.context import ScorerContext
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
        config=scorer_context.config,
    )
    ev = score_title("unique tokens here", "different ones entirely", ctx)
    assert ev.score == 0.0
    assert ev.skipped is False
