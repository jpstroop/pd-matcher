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

``token_set_ratio`` also has no notion of which shared tokens are
*distinctive*. When two otherwise-unrelated names share only a generic
word — e.g. MARC "Oxford University Press for the Royal Institute…" vs CCE
"University of Hawaii Press", whose only common post-stopword token is
"university" — the shared token dominates the shorter string and the raw
ratio inflates to ~74. To remove that class of false signal we gate the
ratio by the IDF distinctiveness of the shared tokens — the larger of an
IDF-weighted Jaccard (coverage) and the most-distinctive shared token's IDF
normalized by ``default_idf`` (a distinctive-hit term). See
:func:`_idf_gate` for the full rationale. An identical pair gates to
``1.0`` (literal match keeps its raw 100); a genuinely shared distinctive
house token ("knopf") keeps the gate high; overlap on generics alone drives
it toward zero. When exactly one token is shared and the sets differ — a
short side wholly contained in a longer one, which the raw ratio inflates to
100 — the gate drops coverage's short-string credit and crushes a lone
*common* token (a shared given name, surname, or initial) toward zero while
preserving a lone *distinctive* token (issue #83). The gate is applied only
when the token sets actually intersect — the disjoint fuzzy path above is
left untouched so OCR/transcription typos on a distinctive token ("Macmillan"
vs "Macmillian") still score high.

The publisher scorer additionally consults a curated alias index when
one is supplied (either via :class:`ScorerContext` or as an explicit
kwarg). When both sides of a comparison resolve to the same canonical
house in the index, the score is lifted to at least
:data:`_ALIAS_HIT_FLOOR`. The floor is a ``max`` rather than a hard
replacement so perfect literal matches retain their 100.0.
"""

from rapidfuzz.fuzz import token_set_ratio

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.normalize.publishers import normalize_publisher
from pd_matcher.normalize.text import tokenize

_MAX_SCORE: float = 100.0
_AUTHOR_SCORER: str = "name.author"
_PUBLISHER_SCORER: str = "name.publisher"
_DISJOINT_FUZZY_FLOOR: float = 50.0
_ALIAS_HIT_FLOOR: float = 95.0
_SINGLE_TOKEN_IDF_FLOOR: float = 0.6


def _prepare(value: str, ctx: ScorerContext, stopwords: frozenset[str]) -> tuple[str, str]:
    """Return ``(joined, original_normalized)`` for fuzzy comparison.

    The first element has stopwords removed and is fed to rapidfuzz; the
    second element preserves the normalized form so that callers can record
    its length as a feature. Number-normalization is routed through
    :meth:`ScorerContext.normalize_numbers` so a MARC field re-scored against
    every candidate is normalized once per record; the result is byte-identical
    to calling :func:`pd_matcher.normalize.numbers.normalize_numbers` directly.
    """
    normalized = ctx.normalize_numbers(value)
    tokens = tokenize(normalized)
    kept = [token for token in tokens if token not in stopwords]
    joined = " ".join(kept)
    return joined, " ".join(tokens)


def _idf_gate(
    shared: set[str],
    marc_set: set[str],
    nypl_set: set[str],
    idf: IdfTable,
) -> float:
    """Return the IDF distinctiveness gate for two token sets in ``[0, 1]``.

    The gate is the **larger** of two complementary distinctiveness views,
    so a pair clears it if *either* holds:

    * **Coverage** — the IDF-weighted Jaccard
      ``sum(idf over shared) / sum(idf over union)``, the same measure the
      title scorer uses. It is ``1.0`` for identical sets (a literal match
      keeps its raw ``token_set_ratio`` of 100) and falls toward ``0`` as
      unshared distinctive tokens accumulate.
    * **Distinctive hit** — the most distinctive shared token's IDF
      normalized by ``default_idf`` (the IDF a once-seen token carries),
      clamped to ``1.0``. A single rare shared token ("knopf") keeps the
      gate high even when each side carries its own extra distinctive
      tokens, which Jaccard alone would over-penalize.

    Taken together: overlap on a generic token only ("oxford",
    "university") yields a low value on *both* views — low coverage (the
    generic carries little of the union's mass) and a low distinctive-hit
    ratio (the generic's own IDF is small) — so the pair-#5 class of false
    signal is driven toward zero. A genuinely shared distinctive house token
    keeps the gate high through the distinctive-hit term.

    **Single-shared-token floor (issue #83).** When exactly one token is
    shared *and the two sets are not identical* — i.e. a one-word side is a
    strict subset of a longer side, the case ``token_set_ratio`` inflates to
    100 — coverage still credits the shared token's slice of the union mass,
    so a lone *common* given name or surname ("nicholas", "roy",
    "montgomery") or a bare initial ("d") survives at 0.25-0.42. That is the
    dominant labeling complaint: a one-word author should not match a
    multi-word author on a single common token alone. In this regime the gate
    is driven by distinctiveness only — coverage's short-string credit is
    dropped — and a token below :data:`_SINGLE_TOKEN_IDF_FLOOR` (as a
    fraction of ``default_idf``) is linearly crushed toward zero. A lone
    *distinctive* shared token (a rare surname / house) clears the floor and
    keeps its full distinctive-hit value, so genuine mononym and corporate
    matches are preserved. Identical single-token names are exempt: the sets
    are equal, coverage is ``1.0``, and an exact one-word match stays at 100.

    Callers only invoke this when ``shared`` is non-empty, so the union is
    non-empty and its IDF sum is strictly positive; no zero-division guard
    is needed.
    """
    union_mass = sum(idf.score(token) for token in (marc_set | nypl_set))
    shared_mass = sum(idf.score(token) for token in shared)
    coverage = shared_mass / union_mass
    best_shared = max(idf.score(token) for token in shared)
    distinctive_hit = min(best_shared / idf.default_idf, 1.0)
    if len(shared) == 1 and marc_set != nypl_set:
        if distinctive_hit >= _SINGLE_TOKEN_IDF_FLOOR:
            return distinctive_hit
        return distinctive_hit * (distinctive_hit / _SINGLE_TOKEN_IDF_FLOOR)
    return max(coverage, distinctive_hit)


def _evidence(
    scorer_name: str,
    marc_value: str | None,
    nypl_value: str | None,
    stopwords: frozenset[str],
    idf: IdfTable,
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
    marc_prepared, marc_normalized = _prepare(marc_value, ctx, stopwords)
    nypl_prepared, nypl_normalized = _prepare(nypl_value, ctx, stopwords)
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
    shared = marc_set & nypl_set
    score = float(token_set_ratio(marc_prepared, nypl_prepared))
    if shared:
        score *= _idf_gate(shared, marc_set, nypl_set, idf)
    elif score < _DISJOINT_FUZZY_FLOOR:
        score = 0.0
    overlap = float(len(shared))
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
    return _evidence(
        _AUTHOR_SCORER,
        marc_author,
        nypl_author,
        ctx.stopwords.author,
        ctx.author_idf,
        ctx,
    )


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
        ctx.publisher_idf,
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
