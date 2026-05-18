"""Structured input fed into the copyright rule engine.

The :class:`Facts` struct collects every datum the Cornell predicates need.
It is the single seam between everything earlier phases produce (MARC
record, the matcher's verdict, today's date) and everything Phase 5
evaluates. Keeping it frozen and explicit means rules can be unit-tested
without touching parsers or the index.

Scope: published books only (Cornell Categories 2 and 3). Unpublished-work
fields (Category 1) and the sound-recording / architectural toggles
(Categories 4 and 5) are intentionally absent.
"""

from datetime import date

from msgspec import Struct

from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


class Facts(Struct, frozen=True, forbid_unknown_fields=True):
    """Everything a predicate may inspect about a single record."""

    pub_year: int | None
    pub_country_code: str | None
    language_code: str | None
    publisher_text: str | None
    was_registered: bool
    was_renewed: bool
    match_confidence: float
    today: date


def _join_publisher_text(
    marc: MarcRecord,
    nypl: IndexedNyplRegRecord | None,
) -> str | None:
    """Return a single lowercase string containing every publisher-like token.

    The result is the union of MARC's ``publisher`` field plus every entry
    of the matched NYPL record's ``publisher_names`` and ``claimants``.
    Used by the inference layer to detect US-government works via regex.

    Args:
        marc: The MARC bibliographic record under evaluation.
        nypl: The matched NYPL registration, if any.

    Returns:
        A space-joined lowercase string, or ``None`` when no publisher
        information is available on either side.
    """
    parts: list[str] = []
    if marc.publisher:
        parts.append(marc.publisher)
    if nypl is not None:
        parts.extend(nypl.publisher_names)
        parts.extend(nypl.claimants)
    if not parts:
        return None
    return " | ".join(parts).lower()


def build_facts(
    marc: MarcRecord,
    match: MatchResult | None,
    *,
    today: date,
    matched_nypl: IndexedNyplRegRecord | None = None,
) -> Facts:
    """Assemble a :class:`Facts` from a MARC record and its match verdict.

    Args:
        marc: The MARC bibliographic record under evaluation.
        match: The matcher's :class:`MatchResult`, or ``None`` when no
            matching pass has been run yet (e.g. a record published
            after the registration window).
        today: The reference date used by every age-sensitive predicate.
        matched_nypl: The hydrated NYPL registration corresponding to
            ``match.best``. Optional; when omitted the inference layer
            loses access to NYPL-side publisher tokens but the rules
            still evaluate.

    Returns:
        A frozen :class:`Facts` capturing the predicate inputs.
    """
    confidence = 0.0
    was_registered = False
    was_renewed = False
    if match is not None and match.best is not None:
        confidence = match.best.combined.calibrated
        was_registered = True
        if matched_nypl is not None:
            was_renewed = matched_nypl.was_renewed
    return Facts(
        pub_year=marc.publication_year,
        pub_country_code=marc.country_code,
        language_code=marc.language_code,
        publisher_text=_join_publisher_text(marc, matched_nypl),
        was_registered=was_registered,
        was_renewed=was_renewed,
        match_confidence=confidence,
        today=today,
    )


__all__ = [
    "Facts",
    "build_facts",
]
