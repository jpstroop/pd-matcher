"""Opportunistic LCCN ↔ NYPL ``regnum`` exact-match scorer.

NYPL registration entries do not carry a separate LCCN field, but the
``regnum`` is the same authoritative copyright office identifier that some
MARC records record as their LCCN. When both sides report the same number
(after collapsing whitespace and leading zeros) we emit Evidence at max
score with ``decisive=True`` — the decisive flag is preserved purely for
audit and ML feature inspection; it does **not** short-circuit the
combiner, which weights this scorer alongside the heuristic ones because
transcription/OCR errors give standard identifiers a non-trivial
false-positive rate. When the IDs disagree we mark the Evidence
``skipped`` rather than fall through to a fuzzy compare — half-matching
identifiers are noise.
"""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.models import IndexedNyplRegRecord

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "lccn.exact"


def _canonical(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip().lstrip("0")
    return stripped or None


def score_lccn(
    marc_lccn: str | None,
    nypl_record: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> Evidence:
    """Return Evidence flagged ``decisive`` when the IDs match exactly."""
    del ctx
    canonical_marc = _canonical(marc_lccn)
    canonical_nypl = _canonical(nypl_record.regnum)
    features: tuple[tuple[str, float], ...] = (
        ("marc_lccn", 1.0 if canonical_marc else 0.0),
        ("nypl_regnum_present", 1.0 if canonical_nypl else 0.0),
    )
    if canonical_marc is None or canonical_nypl is None:
        return Evidence(
            scorer=_SCORER_NAME,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=features,
        )
    if canonical_marc == canonical_nypl:
        return Evidence(
            scorer=_SCORER_NAME,
            score=_MAX_SCORE,
            max=_MAX_SCORE,
            skipped=False,
            decisive=True,
            features=features,
        )
    return Evidence(
        scorer=_SCORER_NAME,
        score=0.0,
        max=_MAX_SCORE,
        skipped=True,
        decisive=False,
        features=features,
    )


__all__ = [
    "score_lccn",
]
