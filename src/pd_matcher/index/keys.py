"""Language-independent token-key generation shared by builder and lookup.

The inverted indexes built in Phase 3 and queried in Phase 4 must derive
their token keys IDENTICALLY on both sides, or the candidate side and the
query side will never line up. This module is the single source of truth
for that derivation.

A field value's key set is :func:`pd_matcher.normalize.text.tokenize` of
the value MINUS a fixed combined stopword set. The combined set for a field
is the UNION, across all five supported languages (``eng``/``fre``/``ger``/
``spa``/``ita``), of that field's per-language stopwords. Using the union
keeps key generation language-independent: a French registration and an
English MARC record both drop ``le`` and ``the`` regardless of the language
code attached to either side, so a shared distinguishing token still
collides. No stemming is applied to keys — stems are a scoring concern, and
applying them here would couple key generation to the per-language stemmer.

The three combined stopword sets are computed once at import time and frozen.
"""

from pd_matcher.normalize.stopwords import load_stopwords
from pd_matcher.normalize.text import tokenize

_SUPPORTED_LANGUAGES: tuple[str, ...] = ("eng", "fre", "ger", "spa", "ita")


def _combined_title_stopwords() -> frozenset[str]:
    """Union of every supported language's title stopwords."""
    return frozenset().union(*(load_stopwords(code).title for code in _SUPPORTED_LANGUAGES))


def _combined_author_stopwords() -> frozenset[str]:
    """Union of every supported language's author stopwords."""
    return frozenset().union(*(load_stopwords(code).author for code in _SUPPORTED_LANGUAGES))


def _combined_publisher_stopwords() -> frozenset[str]:
    """Union of every supported language's publisher stopwords."""
    return frozenset().union(*(load_stopwords(code).publisher for code in _SUPPORTED_LANGUAGES))


_TITLE_STOPWORDS: frozenset[str] = _combined_title_stopwords()
_AUTHOR_STOPWORDS: frozenset[str] = _combined_author_stopwords()
_PUBLISHER_STOPWORDS: frozenset[str] = _combined_publisher_stopwords()


def _keys(value: str | None, stopwords: frozenset[str]) -> frozenset[str]:
    """Tokenize ``value`` and drop the combined ``stopwords`` (no stemming)."""
    if not value:
        return frozenset()
    return frozenset(token for token in tokenize(value) if token not in stopwords)


def title_keys(value: str | None) -> frozenset[str]:
    """Return the inverted-index keys for a title field value."""
    return _keys(value, _TITLE_STOPWORDS)


def author_keys(value: str | None) -> frozenset[str]:
    """Return the inverted-index keys for an author field value."""
    return _keys(value, _AUTHOR_STOPWORDS)


def publisher_keys(value: str | None) -> frozenset[str]:
    """Return the inverted-index keys for a publisher field value."""
    return _keys(value, _PUBLISHER_STOPWORDS)


__all__ = [
    "author_keys",
    "publisher_keys",
    "title_keys",
]
