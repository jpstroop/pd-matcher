"""rapidfuzz token-set scorers for author and publisher fields.

Author names and publisher names tolerate aggressive reordering ("Smith,
John A." vs "John A. Smith") and partial overlap ("Acme Press" vs "Acme
Press, Inc."), which is exactly what
:func:`rapidfuzz.fuzz.token_set_ratio` is engineered for. The pipeline is
the same on both sides — normalize, drop language-specific stopwords for
the field, then compare.
"""

from rapidfuzz.fuzz import token_set_ratio

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.text import tokenize

_MAX_SCORE: float = 100.0
_AUTHOR_SCORER: str = "name.author"
_PUBLISHER_SCORER: str = "name.publisher"


def _prepare(value: str, language: str, stopwords: frozenset[str]) -> tuple[str, str]:
    """Return ``(joined, original_normalized)`` for fuzzy comparison.

    The first element has stopwords removed and is fed to rapidfuzz; the
    second element preserves the normalized form so that callers can record
    its length as a feature.
    """
    normalized = normalize_numbers(value, language)
    tokens = tokenize(normalized)
    kept = [token for token in tokens if token not in stopwords]
    joined = " ".join(kept)
    return joined, " ".join(tokens)


def _evidence(
    scorer_name: str,
    marc_value: str | None,
    nypl_value: str | None,
    stopwords: frozenset[str],
    ctx: ScorerContext,
) -> Evidence:
    if not marc_value or not nypl_value:
        return Evidence(
            scorer=scorer_name,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=(),
        )
    marc_prepared, marc_normalized = _prepare(marc_value, ctx.language, stopwords)
    nypl_prepared, nypl_normalized = _prepare(nypl_value, ctx.language, stopwords)
    if not marc_prepared or not nypl_prepared:
        return Evidence(
            scorer=scorer_name,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=(),
        )
    score = float(token_set_ratio(marc_prepared, nypl_prepared))
    marc_set = set(marc_prepared.split())
    nypl_set = set(nypl_prepared.split())
    overlap = float(len(marc_set & nypl_set))
    features: tuple[tuple[str, float], ...] = (
        ("normalized_marc_len", float(len(marc_normalized))),
        ("normalized_nypl_len", float(len(nypl_normalized))),
        ("token_overlap", overlap),
    )
    return Evidence(
        scorer=scorer_name,
        score=score,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=features,
    )


def score_author(
    marc_author: str | None,
    nypl_author: str | None,
    ctx: ScorerContext,
) -> Evidence:
    """Return :class:`Evidence` comparing two author strings."""
    return _evidence(_AUTHOR_SCORER, marc_author, nypl_author, ctx.stopwords.author, ctx)


def score_publisher(
    marc_publisher: str | None,
    nypl_publisher: str | None,
    ctx: ScorerContext,
) -> Evidence:
    """Return :class:`Evidence` comparing two publisher strings."""
    return _evidence(
        _PUBLISHER_SCORER,
        marc_publisher,
        nypl_publisher,
        ctx.stopwords.publisher,
        ctx,
    )


__all__ = [
    "score_author",
    "score_publisher",
]
