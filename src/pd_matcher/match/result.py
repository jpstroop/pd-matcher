"""Match result containers exposed by :mod:`pd_matcher.match.pipeline`.

The pipeline returns a :class:`MatchResult` per MARC record. ``best`` is
the top-ranked :class:`CandidateMatch` (``None`` when no candidate cleared
the configured floor); ``alternates`` carries up to three runners-up so a
human reviewer can sanity-check ambiguous near-ties. Every candidate
carries the full set of winning Evidence plus the losing Evidence from
non-winning pairings, which is what makes Phase 4's matcher actually
auditable.
"""

from msgspec import Struct

from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence


class CandidateMatch(Struct, frozen=True, forbid_unknown_fields=True):
    """One scored (MARC, NYPL) candidate pair."""

    nypl_uuid: str
    nypl_year: int | None
    combined: CombinedScore
    evidence: tuple[Evidence, ...]
    losing_evidence: tuple[Evidence, ...]


class MatchResult(Struct, frozen=True, forbid_unknown_fields=True):
    """The matcher's verdict for one MARC record."""

    marc_control_id: str
    best: CandidateMatch | None
    alternates: tuple[CandidateMatch, ...]
    candidates_considered: int


__all__ = [
    "CandidateMatch",
    "MatchResult",
]
