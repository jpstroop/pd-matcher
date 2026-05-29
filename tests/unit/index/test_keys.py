"""Tests for :mod:`pd_matcher.index.keys`."""

from pd_matcher.index.keys import author_keys
from pd_matcher.index.keys import publisher_keys
from pd_matcher.index.keys import title_keys


def test_title_keys_tokenizes_and_drops_english_stopwords() -> None:
    """``a``/``the``/``of`` are dropped; content tokens survive."""
    assert title_keys("A study of the widgets") == frozenset({"study", "widgets"})


def test_title_keys_drop_is_language_independent() -> None:
    """French ``et`` and German ``und`` are dropped via the combined union.

    Neither is an English title stopword, but both appear in the union of
    all supported languages' title stopwords, so key generation drops them
    regardless of any language code.
    """
    assert title_keys("Crime et chatiment") == frozenset({"crime", "chatiment"})
    assert title_keys("Stahl und Eisen") == frozenset({"stahl", "eisen"})


def test_title_keys_does_not_stem() -> None:
    """Plural/inflected forms are preserved verbatim (no stemming)."""
    assert title_keys("studies widgets running") == frozenset({"studies", "widgets", "running"})


def test_author_keys_drops_author_stopwords_and_keeps_names() -> None:
    assert author_keys("Smith, John") == frozenset({"smith", "john"})


def test_publisher_keys_drops_publisher_stopwords() -> None:
    """Connectives drop; corporate-suffix noise drops; entity tokens survive."""
    assert publisher_keys("Acme Press for the Public") == frozenset({"acme", "public"})


def test_publisher_keys_drops_corporate_suffix_noise() -> None:
    """``& Co``, ``Inc.``, ``Ltd``, ``Publishing`` are publisher-side noise."""
    assert publisher_keys("Gabriel Sons & Co.") == frozenset({"gabriel", "sons"})
    assert publisher_keys("State Art Publishing Inc.") == frozenset({"state", "art"})


def test_keys_return_empty_frozenset_for_none() -> None:
    assert title_keys(None) == frozenset()
    assert author_keys(None) == frozenset()
    assert publisher_keys(None) == frozenset()


def test_keys_return_empty_frozenset_for_blank() -> None:
    assert title_keys("   ") == frozenset()
    assert title_keys("") == frozenset()


def test_keys_return_frozenset_type() -> None:
    keys = title_keys("A study of widgets")
    assert isinstance(keys, frozenset)
