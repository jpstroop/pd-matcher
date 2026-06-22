"""Streaming, uncapped eligibility filter over a plain MARCXML file.

Where :mod:`pd_groundtruth.acquire` fetches manifest dumps and keeps a balanced
*labeling sample* (a per-(language, decade) quota), this module takes any local
MARCXML ``<collection>`` and writes out every record that is eligible for
production matching, with **no cap**. The output is a single well-formed MARCXML
``<collection>`` in the MARC21 slim namespace, exactly the shape that
``pd-matcher match --marc`` reads.

Eligibility is delegated wholesale to :func:`pd_groundtruth.filters.classify`,
the same predicate acquire uses, so the two never drift. ``classify`` already
restricts to the five supported 008 languages (eng, fre, ger, spa, ita); the
optional ``languages`` argument here narrows *within* that set (e.g. to
reproduce acquire's English-only slice) and never widens it.

The input may be ~1 GB, so records are streamed with ``iterparse`` and each
element is cleared (with its detached previous siblings) immediately after it is
handled, capping peak memory at one record's worth of parsed tree.
"""

from collections import Counter
from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

from lxml.etree import QName
from lxml.etree import _Element
from lxml.etree import iterparse
from lxml.etree import tostring
from msgspec import Struct

from pd_groundtruth.filters import Ineligibility
from pd_groundtruth.filters import classify
from pd_groundtruth.filters import language_of

_LOGGER = getLogger(__name__)

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_XML_DECLARATION = b'<?xml version="1.0" encoding="UTF-8"?>\n'
_COLLECTION_OPEN = f'<collection xmlns="{_MARC_NS}">\n'.encode()
_COLLECTION_CLOSE = b"</collection>\n"

_LANGUAGE_NOT_REQUESTED = "language_not_requested"
"""Report key for records that are eligible but excluded by ``--languages``.

This is deliberately *not* an :class:`Ineligibility` value: those records are
fully eligible (a supported language, in window, a monograph) and were dropped
only because the caller restricted the output to a narrower language set. Using
``UNSUPPORTED_LANGUAGE`` would wrongly imply the language is out of scope.
"""


class FilterReport(Struct, frozen=True):
    """Outcome of a single filter run."""

    scanned: int
    kept: int
    dropped: int
    dropped_by_reason: dict[str, int]


def filter_marcxml(
    *,
    input_path: Path,
    output_path: Path,
    min_year: int,
    languages: frozenset[str] | None = None,
) -> FilterReport:
    """Stream a MARCXML file and write only the eligible records to one file.

    Args:
        input_path: A plain MARCXML ``<collection>`` (the format
            ``pd-matcher match --marc`` reads). Streamed record-by-record.
        output_path: Destination MARCXML ``<collection>``; the parent directory
            is created if absent. Records are serialized verbatim, so the input
            serialization (namespaced or not) is preserved.
        min_year: Inclusive lower bound for the publication year (the moving
            wall). Passed straight through to
            :func:`pd_groundtruth.filters.classify`.
        languages: When ``None`` (the default), every record that passes
            ``classify`` is kept (any of the five supported 008 languages). When
            a set is given, an otherwise-eligible record is dropped unless its
            008 language code is in the set; this narrows within, and never
            widens, ``classify``'s language check.

    Returns:
        A :class:`FilterReport` with the scanned, kept, and dropped counts and a
        per-reason breakdown keyed by :class:`Ineligibility` value.
    """
    dropped_by_reason: Counter[str] = Counter()
    scanned = 0
    kept = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(_XML_DECLARATION)
        handle.write(_COLLECTION_OPEN)
        for record in _iter_records(input_path):
            scanned += 1
            reason = _drop_reason(record, min_year, languages)
            if reason is None:
                handle.write(tostring(record, encoding="UTF-8", xml_declaration=False))
                handle.write(b"\n")
                kept += 1
            else:
                dropped_by_reason[reason] += 1
            _release(record)
        handle.write(_COLLECTION_CLOSE)
    dropped = scanned - kept
    _LOGGER.info(
        "filter complete: scanned=%d kept=%d dropped=%d -> %s",
        scanned,
        kept,
        dropped,
        output_path,
    )
    return FilterReport(
        scanned=scanned,
        kept=kept,
        dropped=dropped,
        dropped_by_reason=dict(dropped_by_reason),
    )


def _drop_reason(
    record: _Element,
    min_year: int,
    languages: frozenset[str] | None,
) -> str | None:
    """Return the reason a record is dropped, or ``None`` when it is kept.

    The base verdict comes entirely from :func:`pd_groundtruth.filters.classify`
    so the eligibility criteria stay identical to acquire. The language
    allow-list is applied only to records ``classify`` already deemed eligible,
    so a record dropped for being out of range or not a book keeps its original,
    more specific :class:`Ineligibility` reason.
    """
    reason = classify(record, min_year)
    if reason is not Ineligibility.ELIGIBLE:
        return str(reason.value)
    if languages is None or language_of(record) in languages:
        return None
    return _LANGUAGE_NOT_REQUESTED


def _iter_records(input_path: Path) -> Iterator[_Element]:
    """Yield ``<record>`` elements from a plain MARCXML file by local name."""
    context = iterparse(str(input_path), events=("end",))
    for _event, element in context:
        if QName(element).localname == "record":
            yield element


def _release(element: _Element) -> None:
    """Clear an emitted record and detach its already-processed siblings."""
    element.clear()
    previous = element.getprevious()
    parent = element.getparent()
    while previous is not None and parent is not None:
        del parent[0]
        previous = element.getprevious()
