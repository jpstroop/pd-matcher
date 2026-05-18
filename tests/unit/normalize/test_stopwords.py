"""Tests for :mod:`pd_matcher.normalize.stopwords`."""

from pd_matcher.normalize.stopwords import StopwordSet
from pd_matcher.normalize.stopwords import load_stopwords


def test_load_stopwords_returns_frozensets_for_english() -> None:
    sw = load_stopwords("eng")
    assert isinstance(sw, StopwordSet)
    assert "the" in sw.title
    assert "and" in sw.author
    assert "of" in sw.publisher


def test_load_stopwords_caches_results() -> None:
    first = load_stopwords("eng")
    second = load_stopwords("eng")
    assert first is second


def test_load_stopwords_supports_each_documented_language() -> None:
    for code in ("eng", "fre", "ger", "spa", "ita"):
        sw = load_stopwords(code)
        assert isinstance(sw.title, frozenset)


def test_load_stopwords_falls_back_to_english_for_unknown_language() -> None:
    fallback = load_stopwords("zzz")
    english = load_stopwords("eng")
    assert fallback is english


def test_stopword_set_is_frozen() -> None:
    sw = load_stopwords("eng")
    try:
        setattr(sw, "title", frozenset())
    except AttributeError:
        return
    raise AssertionError("StopwordSet should reject mutation")
