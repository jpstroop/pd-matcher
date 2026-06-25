"""Tests for :mod:`pd_matcher.match.scorers.context`."""

from pytest import raises

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.script import dominant_script


def test_scorer_context_is_frozen(scorer_context: ScorerContext) -> None:
    """ScorerContext must reject attribute mutation."""
    with raises(AttributeError):
        setattr(scorer_context, "language", "fre")


def test_marc_title_script_matches_direct_call(scorer_context: ScorerContext) -> None:
    """The memoized script equals :func:`dominant_script` for the same string."""
    for title in ("Vol. II of the report", "בראשית", "1234"):
        assert scorer_context.marc_title_script(title) == dominant_script(title)


def test_marc_title_script_is_memoized(scorer_context: ScorerContext) -> None:
    """A second lookup reuses the cached entry rather than recomputing."""
    assert scorer_context.marc_title_script("History of Rome") == "LATIN"
    assert scorer_context.marc_title_scripts == {"History of Rome": "LATIN"}
    scorer_context.marc_title_scripts["History of Rome"] = "SENTINEL"
    assert scorer_context.marc_title_script("History of Rome") == "SENTINEL"


def test_normalize_numbers_matches_direct_call(scorer_context: ScorerContext) -> None:
    """The memoized normalization equals :func:`normalize_numbers` for the language."""
    for value in ("Vol. III", "the second part", "plain title", ""):
        assert scorer_context.normalize_numbers(value) == normalize_numbers(value, "eng")


def test_normalize_numbers_uses_context_language(french_scorer_context: ScorerContext) -> None:
    """Normalization honors the context's language table, not English."""
    assert french_scorer_context.normalize_numbers("trois") == normalize_numbers("trois", "fre")


def test_normalize_numbers_is_memoized(scorer_context: ScorerContext) -> None:
    """A second lookup reuses the cached entry rather than recomputing."""
    assert scorer_context.normalize_numbers("Vol. IV") == "volume 4"
    assert scorer_context.normalized_numbers == {"Vol. IV": "volume 4"}
    scorer_context.normalized_numbers["Vol. IV"] = "SENTINEL"
    assert scorer_context.normalize_numbers("Vol. IV") == "SENTINEL"


def test_memo_dicts_are_not_shared_between_instances(
    scorer_context: ScorerContext, french_scorer_context: ScorerContext
) -> None:
    """Each context gets its own memo dicts; mutation does not leak."""
    scorer_context.normalize_numbers("Vol. V")
    scorer_context.marc_title_script("History of Rome")
    assert french_scorer_context.normalized_numbers == {}
    assert french_scorer_context.marc_title_scripts == {}
