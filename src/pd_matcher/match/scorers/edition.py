"""Edition statement compatibility scorer.

Edition fields are short, structured strings ("1st ed.", "Second edition",
"3", "rev.") that the matcher must handle without false positives. After
running :func:`normalize_numbers` the leading edition integer is usually
recoverable; if it is, we compare integers and emit a perfect or zero
score depending on whether they match. When only one side has an
extractable number we fall back to fuzzy token-set comparison; when both
sides lack a number we likewise fall back to fuzzy comparison rather than
emit a misleading zero.
"""

from re import compile as re_compile

from rapidfuzz.fuzz import token_set_ratio

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.text import normalize_text

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "edition.compat"
_LEADING_INT_RE = re_compile(r"(\d{1,4})")


def _extract_edition_number(value: str) -> int | None:
    match = _LEADING_INT_RE.search(value)
    if match is None:
        return None
    return int(match.group(1))


def score_edition(
    marc_edition: str | None,
    nypl_edition: str | None,
    ctx: ScorerContext,
) -> Evidence:
    """Return :class:`Evidence` describing edition compatibility."""
    if not marc_edition or not nypl_edition:
        return Evidence(
            scorer=_SCORER_NAME,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=(),
        )
    marc_normalized = normalize_text(normalize_numbers(marc_edition, ctx.language))
    nypl_normalized = normalize_text(normalize_numbers(nypl_edition, ctx.language))
    if not marc_normalized or not nypl_normalized:
        return Evidence(
            scorer=_SCORER_NAME,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=(),
        )
    marc_num = _extract_edition_number(marc_normalized)
    nypl_num = _extract_edition_number(nypl_normalized)
    explicit_mismatch = 0.0
    if marc_num is not None and nypl_num is not None:
        if marc_num == nypl_num:
            score = _MAX_SCORE
        else:
            score = 0.0
            explicit_mismatch = 1.0
    else:
        score = float(token_set_ratio(marc_normalized, nypl_normalized))
    features: tuple[tuple[str, float], ...] = (
        ("marc_edition_num", float(marc_num) if marc_num is not None else -1.0),
        ("nypl_edition_num", float(nypl_num) if nypl_num is not None else -1.0),
        ("explicit_mismatch", explicit_mismatch),
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
    "score_edition",
]
