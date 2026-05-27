"""Detect open-multipart series-level cataloging in MARC records.

A MARC record describes the *abstract series* — not any individual
physical volume — when its 300 ``$a`` extent is the AACR2 bare ``"v."``
form or the RDA equivalent ``"volumes"``, or when its 260/264 ``$c``
publication date carries the open-date convention ``[YYYY-]`` (square
brackets = cataloger-supplied date, trailing hyphen = ongoing
publication). Either cue is sufficient.

Such a record is essentially never the correct linkage target for a
CCE registration (which is always cut against one specific volume) —
the volume cardinality scorer in :mod:`pd_matcher.match.scorers.volume`
uses this same predicate to drive its ``WHOLE_OPEN`` classification,
and the review UI surfaces it as a badge so the labeler can spot the
pattern at a glance without inspecting individual fields.

The MARC parser strips the trailing period from ``"v."``, so the
open-multipart sentinel arrives here as bare ``"v"``. Matching is
case-insensitive but exact (after stripping) so genuine part statements
like ``"v. 1"`` or ``"3 volumes"`` do not get swept in.

Citation: AACR2 1.5B5 / RDA 3.4.5.2 (extent of unknown-cardinality
multipart monographs); MARC 008/06 ``m`` (multipart monograph item) is
the catalog-level companion to this physical-side cue.
"""

from re import compile as re_compile

from pd_matcher.models import MarcRecord

_OPEN_DATE_RE = re_compile(r"\[\d{4}-(?!\d)")


def _is_bare_volume_extent(value: str | None) -> bool:
    """Return ``True`` for the AACR2 bare ``"v"`` / RDA bare ``"volumes"`` extent."""
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered == "v" or lowered == "volumes"


def _is_open_publication_date(value: str | None) -> bool:
    """Return ``True`` when ``value`` is a ``[YYYY-]`` open-date string."""
    if not value:
        return False
    return _OPEN_DATE_RE.search(value) is not None


def is_series_level(marc: MarcRecord) -> bool:
    """Return ``True`` when the MARC record describes an open-multipart series.

    Fires when the extent is the AACR2 bare ``"v"`` / RDA bare ``"volumes"``
    sentinel, or when the raw publication date matches the ``[YYYY-]``
    open-date convention. Either cue alone is sufficient.
    """
    return _is_bare_volume_extent(marc.extent) or _is_open_publication_date(
        marc.publication_date_raw
    )


__all__ = [
    "is_series_level",
]
