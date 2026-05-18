"""Soft-signal year delta scorer.

The Cornell matrix and the ground truth agree that publication, copyright
registration, and renewal years can drift by 1-2 against the canonical
publication year on either side. We model this as a soft signal rather
than a hard gate (the gate happens earlier, in the LMDB year bucket
window): ``0`` years apart scores 100, each additional year subtracts 25,
and any delta of 4 or more years scores 0.
"""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext

_MAX_SCORE: float = 100.0
_PENALTY_PER_YEAR: float = 25.0
_SCORER_NAME: str = "year.delta"


def score_year(
    marc_year: int | None,
    nypl_year: int | None,
    ctx: ScorerContext,
) -> Evidence:
    """Return :class:`Evidence` for a (marc_year, nypl_year) pair.

    Args:
        marc_year: MARC ``008``/``260$c`` derived publication year, or
            ``None`` if it could not be extracted.
        nypl_year: NYPL ``regDate`` year, or ``None``.
        ctx: Scorer context; unused for year scoring but kept in the
            signature so every scorer has the same shape.
    """
    del ctx
    if marc_year is None or nypl_year is None:
        return Evidence(
            scorer=_SCORER_NAME,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=(),
        )
    delta = abs(marc_year - nypl_year)
    score = max(0.0, _MAX_SCORE - delta * _PENALTY_PER_YEAR)
    return Evidence(
        scorer=_SCORER_NAME,
        score=score,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=(("delta_years", float(delta)),),
    )


__all__ = [
    "score_year",
]
