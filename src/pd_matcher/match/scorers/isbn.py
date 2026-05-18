"""ISBN exact-match scorer (currently always skipped).

NYPL's Catalog-of-Copyright-Entries transcriptions cover registrations
filed 1923-1977, predating widespread ISBN adoption. The corpus therefore
contains no ISBN data and this scorer always returns
:class:`Evidence` with ``skipped=True``. The module is intentionally
present so the Phase 4 scorer surface is stable: when a future NYPL data
source exposes ISBNs, only this file changes.
"""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.models import IndexedNyplRegRecord

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "isbn.exact"


def score_isbn(
    marc_isbns: tuple[str, ...],
    nypl_record: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> Evidence:
    """Return a permanently-skipped :class:`Evidence` for the ISBN scorer."""
    del nypl_record, ctx
    features: tuple[tuple[str, float], ...] = (("marc_isbn_count", float(len(marc_isbns))),)
    return Evidence(
        scorer=_SCORER_NAME,
        score=0.0,
        max=_MAX_SCORE,
        skipped=True,
        decisive=False,
        features=features,
    )


__all__ = [
    "score_isbn",
]
