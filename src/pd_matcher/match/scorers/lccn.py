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

Canonicalisation follows the LoC LCCN namespace algorithm
(https://www.loc.gov/marc/lccn-namespace.html):

1. Remove all blanks (whitespace).
2. If a forward slash is present, drop it and everything to the right.
3. If a hyphen is present, drop it; left-pad the substring to the right
   of the (removed) hyphen with leading zeros until it is exactly six
   digits.

Inputs whose right-of-hyphen substring exceeds six digits are
malformed under the spec but are kept as-is rather than truncated: the
spec says the substring "should be 6 digits or less", and truncating
would silently merge distinct identifiers. Inputs with more than one
hyphen are also outside the spec; they are returned with whitespace
removed but otherwise unchanged. Either way the canonical form simply
fails to equal any well-formed LCCN.
"""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.models import IndexedNyplRegRecord

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "lccn.exact"
_SUFFIX_WIDTH: int = 6


def _canonical(value: str | None) -> str | None:
    """Apply the LoC LCCN canonicalisation algorithm.

    Returns ``None`` for ``None`` or whitespace-only input.
    """
    if value is None:
        return None
    no_blanks = "".join(value.split())
    if not no_blanks:
        return None
    slash_index = no_blanks.find("/")
    if slash_index != -1:
        no_blanks = no_blanks[:slash_index]
        if not no_blanks:
            return None
    if no_blanks.count("-") != 1:
        return no_blanks
    hyphen_index = no_blanks.index("-")
    left = no_blanks[:hyphen_index]
    right = no_blanks[hyphen_index + 1 :]
    if len(right) < _SUFFIX_WIDTH:
        right = right.rjust(_SUFFIX_WIDTH, "0")
    return left + right


def score_lccn(
    marc_lccn: str | None,
    nypl_record: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> Evidence:
    """Return Evidence flagged ``decisive`` when the IDs match exactly."""
    del ctx
    canonical_marc = _canonical(marc_lccn)
    canonical_nypl = _canonical(nypl_record.lccn)
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
