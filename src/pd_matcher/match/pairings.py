"""Bounded field-pair permutations for the matching pipeline.

The pipeline runs every pairing returned here through the relevant scorer
and keeps the best :class:`Evidence` per scorer-class; the runners-up are
recorded as ``losing_evidence`` on the
:class:`pd_matcher.match.result.CandidateMatch`. This solves the
"title-stored-as-series" transposition without exposing the matcher to a
quadratic permutation blow-up.
"""

from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_MAX_TITLE_PAIRINGS: int = 3


def title_pairings(
    marc: MarcRecord,
    nypl: IndexedNyplRegRecord,
) -> tuple[tuple[str, str], ...]:
    """Return up to three ``(marc_title, nypl_title)`` pairings to score.

    The primary pairing is ``(marc.title, nypl.title)``. Up to two
    additional pairings are produced by treating each of the first two
    MARC series titles as the candidate work-title, which lets the matcher
    recover from the common transposition where a publisher records the
    series title where the work title belongs.
    """
    extra = _MAX_TITLE_PAIRINGS - 1
    pairings: list[tuple[str, str]] = [(marc.title, nypl.title)]
    for series_title in marc.series_titles[:extra]:
        pairings.append((series_title, nypl.title))
    return tuple(pairings)


def publisher_pairings(
    marc: MarcRecord,
    nypl: IndexedNyplRegRecord,
) -> tuple[tuple[str | None, str], ...]:
    """Return up to two ``(marc_publisher, nypl_publisher)`` pairings.

    The first pairing compares against the joined NYPL ``publisher_names``;
    the second compares against the joined NYPL ``claimants``, which is
    where some bibliographic records actually record the publisher.
    """
    publisher_names = " ".join(nypl.publisher_names)
    claimants = " ".join(nypl.claimants)
    return (
        (marc.publisher, publisher_names),
        (marc.publisher, claimants),
    )


__all__ = [
    "publisher_pairings",
    "title_pairings",
]
