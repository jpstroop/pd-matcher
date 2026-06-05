"""rapidfuzz token-set scorers for author and publisher fields.

Author names and publisher names tolerate aggressive reordering ("Smith,
John A." vs "John A. Smith") and partial overlap ("Acme Press" vs "Acme
Press, Inc."), which is exactly what
:func:`rapidfuzz.fuzz.token_set_ratio` is engineered for. The pipeline is
the same on both sides — normalize, drop language-specific stopwords for
the field, then compare.

When the two token sets are disjoint, ``token_set_ratio`` silently falls
back to character-level Levenshtein and emits noise scores (20-40) for
strings that share nothing semantically. We collapse those to 0 below
:data:`_DISJOINT_FUZZY_FLOOR`. A cutoff of 50 sits above the observed
unrelated-name cluster (16-36, e.g. ``Maruzen`` vs ``Peter Chiarulli``
≈36) while preserving the (50, 70) band where borderline real signal
lives; a stricter cutoff of 70 was measured and cost ~3% recall on the
locked regression set, so 50 is the chosen floor.

The publisher scorer additionally consults a curated alias index when
one is supplied (either via :class:`ScorerContext` or as an explicit
kwarg). When both sides of a comparison resolve to the same canonical
house in the index, the score is lifted to at least
:data:`_ALIAS_HIT_FLOOR`. The floor is a ``max`` rather than a hard
replacement so perfect literal matches retain their 100.0.
"""

from rapidfuzz.fuzz import token_set_ratio

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.publishers import normalize_publisher
from pd_matcher.normalize.text import tokenize

_MAX_SCORE: float = 100.0
_AUTHOR_SCORER: str = "name.author"
_PUBLISHER_SCORER: str = "name.publisher"
_DISJOINT_FUZZY_FLOOR: float = 50.0
_ALIAS_HIT_FLOOR: float = 95.0


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
    marc_set = set(marc_prepared.split())
    nypl_set = set(nypl_prepared.split())
    score = float(token_set_ratio(marc_prepared, nypl_prepared))
    if not (marc_set & nypl_set) and score < _DISJOINT_FUZZY_FLOOR:
        score = 0.0
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
    *,
    alias_index: dict[str, str] | None = None,
) -> Evidence:
    """Return :class:`Evidence` comparing two publisher strings.

    Args:
        marc_publisher: Publisher string from the MARC record.
        nypl_publisher: Publisher string from the CCE record.
        ctx: Per-record scorer context. When ``alias_index`` is ``None``
            and ``ctx.publisher_alias_index`` is populated, the latter is
            used instead.
        alias_index: Optional explicit ``{normalized_name: human_canonical}``
            lookup. When supplied (or available on ``ctx``), both sides
            are run through :func:`normalize_publisher` and resolved; on
            a canonical hit the returned score is at least
            :data:`_ALIAS_HIT_FLOOR`, leaving fuzzy-or-higher scores
            unchanged, and the canonical house name is stamped on
            :attr:`pd_matcher.match.evidence.Evidence.note`.
    """
    base = _evidence(
        _PUBLISHER_SCORER,
        marc_publisher,
        nypl_publisher,
        ctx.stopwords.publisher,
        ctx,
    )
    effective_index = alias_index if alias_index is not None else ctx.publisher_alias_index
    if effective_index is None or base.skipped:
        return base
    marc_canonical = effective_index.get(normalize_publisher(marc_publisher or ""))
    if marc_canonical is None:
        return base
    nypl_canonical = effective_index.get(normalize_publisher(nypl_publisher or ""))
    if nypl_canonical is None or marc_canonical != nypl_canonical:
        return base
    lifted = max(base.score, _ALIAS_HIT_FLOOR)
    if lifted == base.score:
        return base
    return Evidence(
        scorer=base.scorer,
        score=lifted,
        max=base.max,
        skipped=base.skipped,
        decisive=base.decisive,
        features=base.features,
        weight_multiplier=base.weight_multiplier,
        note=marc_canonical,
    )


__all__ = [
    "score_author",
    "score_publisher",
]
