"""Per-language, per-field stopword sets loaded from packaged JSON resources.

The JSON files originate in ``data/stopwords/`` and are copied into
``src/pd_matcher/normalize/stopwords_data/`` at build time via the
``[tool.pdm.build] includes`` list so that installed wheels always carry
their own stopword tables. :func:`load_stopwords` resolves them through
:mod:`importlib.resources`, which works identically for source checkouts and
installed packages.

Loaded :class:`StopwordSet` instances are cached per language code so the
hot path through the matcher does not re-decode JSON on every record.
"""

from importlib.resources import files
from json import loads

from msgspec import Struct

_RESOURCE_PACKAGE = "pd_matcher.normalize.stopwords_data"

_FILENAME_BY_LANGUAGE: dict[str, str] = {
    "eng": "english_stopwords.json",
    "fre": "french_stopwords.json",
    "ger": "german_stopwords.json",
    "spa": "spanish_stopwords.json",
    "ita": "italian_stopwords.json",
}


class StopwordSet(Struct, frozen=True, forbid_unknown_fields=True):
    """Per-field stopword tables for one language."""

    title: frozenset[str]
    author: frozenset[str]
    publisher: frozenset[str]


_CACHE: dict[str, StopwordSet] = {}


def _load_from_resource(filename: str) -> StopwordSet:
    """Decode a stopword JSON resource into a :class:`StopwordSet`."""
    resource = files(_RESOURCE_PACKAGE).joinpath(filename)
    payload = loads(resource.read_text(encoding="utf-8"))
    title_words = payload.get("title_stopwords", [])
    author_words = payload.get("author_stopwords", [])
    publisher_words = payload.get("publisher_stopwords", [])
    return StopwordSet(
        title=frozenset(title_words),
        author=frozenset(author_words),
        publisher=frozenset(publisher_words),
    )


def load_stopwords(language: str) -> StopwordSet:
    """Return the :class:`StopwordSet` for ``language``, cached per code.

    Args:
        language: MARC 3-letter language code (``eng``, ``fre``, ``ger``,
            ``spa``, ``ita``). Unknown codes fall back to English.

    Returns:
        The frozen :class:`StopwordSet` for the resolved language.
    """
    filename = _FILENAME_BY_LANGUAGE.get(language, _FILENAME_BY_LANGUAGE["eng"])
    cached = _CACHE.get(filename)
    if cached is None:
        cached = _load_from_resource(filename)
        _CACHE[filename] = cached
    return cached


__all__ = [
    "StopwordSet",
    "load_stopwords",
]
