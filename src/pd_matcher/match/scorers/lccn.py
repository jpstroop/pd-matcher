"""LCCN exact-match scorer (MARC ``lccn`` ↔ CCE ``lccn``).

The Library of Congress Control Number is the same identifier on both
sides — MARC carries it in field 010$a, and the NYPL CCE transcription
mirrors the ``<lccn>`` element from the original Copyright Office entry
onto :attr:`IndexedNyplRegRecord.lccn`. Equality after canonicalisation
is therefore the only meaningful comparison. When the IDs match we emit
Evidence at max score with ``decisive=True``; the decisive flag is
preserved purely for audit and ML feature inspection (it does **not**
short-circuit the combiner — in this corpus, transcription/OCR errors
give standard identifiers a non-trivial false-positive rate, so the
Platt calibrator owns the actual ``P(true match)``). When the IDs
disagree we mark the Evidence ``skipped`` rather than fall through to a
fuzzy compare — half-matching identifiers are noise.

Canonicalisation lives in :mod:`pd_matcher.normalize.lccn` so the
review UI and any other caller can render the same normalized form.
"""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.normalize.lccn import canonical

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "lccn.exact"


def score_lccn(
    marc_lccn: str | None,
    nypl_record: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> Evidence:
    """Return Evidence flagged ``decisive`` when the IDs match exactly."""
    del ctx
    canonical_marc = canonical(marc_lccn)
    canonical_nypl = canonical(nypl_record.lccn)
    features: tuple[tuple[str, float], ...] = (
        ("marc_lccn", 1.0 if canonical_marc else 0.0),
        ("nypl_lccn_present", 1.0 if canonical_nypl else 0.0),
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
