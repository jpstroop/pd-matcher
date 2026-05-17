"""Per-language Snowball stemmers backed by PyStemmer.

PyStemmer's :class:`Stemmer.Stemmer` instances are not cheap to construct in
tight loops (each loads its own language data), so this module caches one
instance per language code in a module-level dictionary. Unknown language
codes fall back to the English stemmer — a defensive choice driven by the
ANALYSIS_RESULTS findings that 86%+ of records are English and that an
imperfect English stem is preferable to no stemming for the typical
single-token noise tokens that motivate the fallback.
"""

from collections.abc import Callable

from Stemmer import Stemmer

_LANGUAGE_MAP: dict[str, str] = {
    "eng": "english",
    "fre": "french",
    "ger": "german",
    "spa": "spanish",
    "ita": "italian",
}

_STEMMER_CACHE: dict[str, Stemmer] = {}


def _stemmer_instance(language: str) -> Stemmer:
    """Return a cached :class:`Stemmer` for ``language`` (falls back to English)."""
    snowball_name = _LANGUAGE_MAP.get(language, "english")
    cached = _STEMMER_CACHE.get(snowball_name)
    if cached is None:
        cached = Stemmer(snowball_name)
        _STEMMER_CACHE[snowball_name] = cached
    return cached


def stemmer_for(language: str) -> Callable[[str], str]:
    """Return a callable that stems a single token in ``language``.

    Args:
        language: MARC 3-letter language code. Unknown codes are coerced to
            English.

    Returns:
        A function mapping a single token to its stem.
    """
    instance = _stemmer_instance(language)

    def _stem(token: str) -> str:
        result = instance.stemWord(token)
        return str(result)

    return _stem


def stem_tokens(tokens: tuple[str, ...], language: str) -> tuple[str, ...]:
    """Return the per-token stems of ``tokens`` in ``language``.

    Args:
        tokens: Already-normalized tokens (see :mod:`pd_matcher.normalize.text`).
        language: MARC 3-letter language code; unknown codes fall back to
            English.

    Returns:
        A tuple of stems in input order.
    """
    if not tokens:
        return ()
    instance = _stemmer_instance(language)
    stems = instance.stemWords(list(tokens))
    return tuple(str(stem) for stem in stems)


__all__ = [
    "stem_tokens",
    "stemmer_for",
]
