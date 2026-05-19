"""Streaming parser for NYPL's XML transcription of the CCE registration corpus.

The source data is the U.S. Copyright Office's Catalog of Copyright Entries
(CCE), published by the Library of Congress and transcribed into XML by
NYPL. The CCE registration corpus is split into one XML file per
year/volume/issue, and the project ships sixty-three year directories.
``iter_nypl_reg_records`` parses a single file using
:func:`lxml.etree.iterparse` over ``<copyrightEntry>`` elements, mirroring
the streaming/clearing pattern used in :mod:`pd_matcher.parsers.marc`.
``iter_nypl_reg_directory`` walks the year tree in sorted order and chains
records from every file beneath it.

The canonical registration number is the ``regnum`` attribute on the
``<copyrightEntry>`` element; the inline ``<regNum>`` text often contains
spaces ("A 125487" vs "A125487") and is therefore preferred only as a
fallback when the attribute is absent.

A :class:`NyplRegParseStats` counter tracks how many subfield values were
repaired by the encoding-hygiene pass (see
:mod:`pd_matcher.normalize.encoding`). Callers wanting visibility supply
their own instance; otherwise an internal counter is used and discarded.
"""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from re import compile as re_compile

from lxml.etree import _Element
from lxml.etree import iterparse

from pd_matcher.models import NyplRegRecord
from pd_matcher.normalize.encoding import clean_text

_ENTRY_TAG = "copyrightEntry"
_YEAR_RE = re_compile(r"(\d{4})")


class NyplRegParseStats:
    """Mutable counters surfaced to callers after a parse run."""

    __slots__ = ("emitted", "mojibake_fixed_count")

    def __init__(self) -> None:
        self.emitted = 0
        self.mojibake_fixed_count = 0


def _text(element: _Element | None, stats: NyplRegParseStats) -> str | None:
    """Return stripped, encoding-cleaned text from ``element`` or ``None``."""
    if element is None or element.text is None:
        return None
    stripped = element.text.strip()
    if not stripped:
        return None
    cleaned = clean_text(stripped)
    if cleaned.mojibake_fixed:
        stats.mojibake_fixed_count += 1
    return cleaned.text or None


def _parse_date(raw: str | None) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` string to a :class:`date`.

    Returns ``None`` for missing input or values that fail to parse.
    """
    if raw is None:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _extract_reg_date(entry: _Element) -> date | None:
    """Read the ``date`` attribute of the first ``<regDate>`` child."""
    reg_date_elem = entry.find("regDate")
    if reg_date_elem is None:
        return None
    return _parse_date(reg_date_elem.get("date"))


def _year_from_text(value: str | None) -> int | None:
    """Extract the first 4-digit run from ``value`` and return it as ``int``."""
    if value is None:
        return None
    match = _YEAR_RE.search(value)
    if match is None:
        return None
    return int(match.group(1))


def _collect_publisher_names(
    entry: _Element, stats: NyplRegParseStats
) -> tuple[list[str], list[str]]:
    """Return ``(pub_names, claimant_publisher_names)`` from ``<publisher>``."""
    names: list[str] = []
    claimants: list[str] = []
    for publisher in entry.iterfind("publisher"):
        for name_elem in publisher.iterfind("pubName"):
            value = _text(name_elem, stats)
            if value is None:
                continue
            names.append(value)
            if name_elem.get("claimant") == "yes":
                claimants.append(value)
    return names, claimants


def _collect_publication_places(entry: _Element, stats: NyplRegParseStats) -> list[str]:
    """Return all ``<pubPlace>`` text values nested under any ``<publisher>``."""
    out: list[str] = []
    for publisher in entry.iterfind("publisher"):
        for place_elem in publisher.iterfind("pubPlace"):
            value = _text(place_elem, stats)
            if value is not None:
                out.append(value)
    return out


def _collect_explicit_claimants(entry: _Element, stats: NyplRegParseStats) -> list[str]:
    """Return text of stand-alone ``<claimant>`` elements (if any)."""
    out: list[str] = []
    for claim in entry.iterfind("claimant"):
        value = _text(claim, stats)
        if value is not None:
            out.append(value)
    return out


def _build_record(entry: _Element, stats: NyplRegParseStats) -> NyplRegRecord | None:
    """Translate one ``<copyrightEntry>`` into a :class:`NyplRegRecord`.

    Entries missing the ``id`` attribute or a non-empty ``<title>`` are
    skipped (returns ``None``); these are unusable for matching because we
    lack either a stable id to key off or a title to compare against.
    """
    uuid = entry.get("id")
    if uuid is None or not uuid.strip():
        return None

    title = _text(entry.find("title"), stats)
    if title is None:
        return None

    regnum = entry.get("regnum") or _text(entry.find("regNum"), stats)
    reg_date = _extract_reg_date(entry)
    reg_year: int | None
    if reg_date is not None:
        reg_year = reg_date.year
    else:
        reg_date_elem = entry.find("regDate")
        reg_year = (
            _year_from_text(_text(reg_date_elem, stats)) if reg_date_elem is not None else None
        )

    author_name = _text(entry.find("author/authorName"), stats)
    edition = _text(entry.find("edition"), stats)
    publisher_names, claimants_from_pub = _collect_publisher_names(entry, stats)
    publication_places = _collect_publication_places(entry, stats)
    explicit_claimants = _collect_explicit_claimants(entry, stats)
    combined_claimants = (*claimants_from_pub, *explicit_claimants)

    stats.emitted += 1
    return NyplRegRecord(
        uuid=uuid,
        title=title,
        regnum=regnum,
        reg_date=reg_date,
        reg_year=reg_year,
        author_name=author_name,
        edition=edition,
        publisher_names=tuple(publisher_names),
        publication_places=tuple(publication_places),
        claimants=combined_claimants,
    )


def iter_nypl_reg_records(
    path: Path, stats: NyplRegParseStats | None = None
) -> Iterator[NyplRegRecord]:
    """Yield :class:`NyplRegRecord` objects streamed from a single XML file.

    Args:
        path: Filesystem path to one CCE registration XML file (NYPL transcription).
        stats: Optional :class:`NyplRegParseStats` counters mutated as
            records are emitted and as the encoding-hygiene pass repairs
            individual subfield values. A fresh stats object is created
            when none is supplied.

    Yields:
        Validated :class:`NyplRegRecord` instances in document order.
    """
    counters = stats if stats is not None else NyplRegParseStats()
    context = iterparse(str(path), events=("end",), tag=_ENTRY_TAG)
    for _event, elem in context:
        record = _build_record(elem, counters)
        if record is not None:
            yield record
        elem.clear()
        previous = elem.getprevious()
        while previous is not None:
            del elem.getparent()[0]
            previous = elem.getprevious()


def iter_nypl_reg_directory(
    root: Path, stats: NyplRegParseStats | None = None
) -> Iterator[NyplRegRecord]:
    """Yield records from every ``*.xml`` file beneath ``root`` in sorted order.

    Args:
        root: Directory containing year subdirectories (e.g. ``data/nypl-reg/xml``).
        stats: Optional shared :class:`NyplRegParseStats` counters; when
            supplied, counts accumulate across all walked files.

    Yields:
        :class:`NyplRegRecord` instances streamed across all discovered files.
    """
    counters = stats if stats is not None else NyplRegParseStats()
    for xml_path in sorted(root.rglob("*.xml")):
        yield from iter_nypl_reg_records(xml_path, counters)


__all__ = [
    "NyplRegParseStats",
    "iter_nypl_reg_directory",
    "iter_nypl_reg_records",
]
