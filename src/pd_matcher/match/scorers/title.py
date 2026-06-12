"""IDF-weighted token-set similarity over title fields.

The scorer applies the same normalization pipeline to both sides:

* :func:`pd_matcher.normalize.text.normalize_text` (already implicit in
  :func:`tokenize`) — NFKD-strip + lowercase + punctuation collapse.
* :func:`pd_matcher.normalize.numbers.normalize_numbers` — Roman/word/
  ordinal → Arabic digits in the record's language.
* Stopword removal using the language's title stopword set.
* Snowball stemming for the record's language.

Then the IDF-weighted Jaccard ratio is computed: the sum of IDF over the
stem intersection divided by the sum over the union. This down-weights
common bibliographic filler ("the", "and", "of") and up-weights rare
distinguishing tokens, so two titles that differ only in noise still score
high while two titles that happen to share filler do not.
"""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.signals.script import is_script_mismatch
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.text import tokenize

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "title.token_set"


def _prepare(value: str, ctx: ScorerContext) -> tuple[str, ...]:
    """Tokenize, drop stopwords, and stem ``value`` for the context language."""
    normalized = normalize_numbers(value, ctx.language)
    tokens = tokenize(normalized)
    filtered = [token for token in tokens if token not in ctx.stopwords.title]
    return tuple(ctx.stemmer(token) for token in filtered)


def _skipped() -> Evidence:
    return Evidence(
        scorer=_SCORER_NAME,
        score=0.0,
        max=_MAX_SCORE,
        skipped=True,
        decisive=False,
        features=(),
    )


def _script_mismatch_zero() -> Evidence:
    return Evidence(
        scorer=_SCORER_NAME,
        score=0.0,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=(("script_mismatch", 1.0),),
    )


def score_title(marc_title: str | None, nypl_title: str | None, ctx: ScorerContext) -> Evidence:
    """Return :class:`Evidence` for one (marc_title, nypl_title) pairing.

    Args:
        marc_title: MARC 245 ``$a$b`` value or ``None``.
        nypl_title: NYPL registration title or ``None``.
        ctx: Per-record :class:`ScorerContext`.

    Returns:
        An :class:`Evidence` whose ``score`` lies in ``[0, 100]``. The
        ``skipped`` flag is set when either input is empty or unusable.
        When the two sides use different dominant Unicode scripts, the
        scorer emits a non-skipped zero so the pair contributes to the
        combiner's denominator instead of silently dropping out.
    """
    if not marc_title or not nypl_title:
        return _skipped()
    if is_script_mismatch(marc_title, nypl_title):
        return _script_mismatch_zero()
    marc_tokens = _prepare(marc_title, ctx)
    nypl_tokens = _prepare(nypl_title, ctx)
    if not marc_tokens or not nypl_tokens:
        return _skipped()
    marc_set = set(marc_tokens)
    nypl_set = set(nypl_tokens)
    intersection = marc_set & nypl_set
    union = marc_set | nypl_set
    weighted_intersection = sum(ctx.idf.score(token) for token in intersection)
    weighted_union = sum(ctx.idf.score(token) for token in union)
    raw = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
    score = raw * _MAX_SCORE
    avg_idf = (weighted_union / len(union)) if union else 0.0
    features: tuple[tuple[str, float], ...] = (
        ("token_overlap", float(len(intersection))),
        ("token_total", float(len(union))),
        ("unique_to_marc", float(len(marc_set - nypl_set))),
        ("unique_to_nypl", float(len(nypl_set - marc_set))),
        ("avg_token_idf", avg_idf),
        ("marc_token_len", float(len(marc_set))),
        ("nypl_token_len", float(len(nypl_set))),
    )
    return Evidence(
        scorer=_SCORER_NAME,
        score=score,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=features,
    )


__all__ = [
    "score_title",
]
