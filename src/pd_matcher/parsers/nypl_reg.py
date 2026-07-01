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

``reg_date`` is sourced strictly from ``<regDate>``. ``reg_year`` uses a
``regDate → copyDate → pubDate`` fallback chain so entries with no
registration date (notably *ad interim* registrations) still land in a
year bucket and stay reachable by the year-blocked matcher;
``affDate``/``noticeDate`` are excluded because they are procedural
(affidavit/notice) rather than registration, copyright, or publication
events. See :func:`_derive_reg_year` for the rationale.

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
_ADDITIONAL_ENTRY_TAG = "additionalEntry"
_YEAR_RE = re_compile(r"(\d{4})")

_MIN_PLAUSIBLE_YEAR = 1700
"""Floor used to reject impossible year derivations from CCE date elements.

The CCE registration corpus runs 1891-1977 (1909 Act regime); 1700 is
well outside the corpus boundary and only screens out parser garbage
(e.g. ``date="0159-01-01"``) without rejecting any plausible entry.
"""

_MAX_PLAUSIBLE_YEAR = 2100
"""Ceiling matching :data:`_MIN_PLAUSIBLE_YEAR`.

Catches malformed five-digit years (e.g. ``date="5764-01-01"`` observed
in the live corpus) without rejecting any real entry.
"""


class NyplRegParseStats:
    """Mutable counters surfaced to callers after a parse run."""

    __slots__ = ("emitted", "mojibake_fixed_count", "years_rejected_out_of_range")

    def __init__(self) -> None:
        self.emitted = 0
        self.mojibake_fixed_count = 0
        self.years_rejected_out_of_range = 0


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
    """Read the ``date`` attribute of the first ``<regDate>`` child.

    This stays strictly the registration date: it is sourced ONLY from
    ``<regDate>`` and is left ``None`` when that element is absent. We do
    NOT fill it from ``copyDate``/``pubDate`` (the year fallback chain in
    :func:`_derive_reg_year`) because those are different events — a
    copyright (©) or publication date is not a registration date, and
    conflating them would misrepresent the record's provenance.
    """
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


def _year_from_date_element(elem: _Element | None, stats: NyplRegParseStats) -> int | None:
    """Return the best-available 4-digit year from a CCE date element.

    The CCE date elements (``regDate``, ``copyDate``, ``pubDate``) carry a
    normalized ``date`` attribute that is either ``YYYY`` or ``YYYY-MM-DD``
    per the CopyrightEntries DTD/guide; both forms start with the year, so
    we read the leading 4-digit run from the attribute first. When the
    attribute is missing or malformed we fall back to the same 4-digit
    scan over the element's display text.

    Years outside ``[_MIN_PLAUSIBLE_YEAR, _MAX_PLAUSIBLE_YEAR]`` are
    rejected as parser garbage and increment
    :attr:`NyplRegParseStats.years_rejected_out_of_range`; callers see
    ``None`` and can move to the next fallback in the chain.
    """
    if elem is None:
        return None
    candidate = _year_from_text(elem.get("date"))
    if candidate is None:
        candidate = _year_from_text(_text(elem, stats))
    if candidate is None:
        return None
    if not _MIN_PLAUSIBLE_YEAR <= candidate <= _MAX_PLAUSIBLE_YEAR:
        stats.years_rejected_out_of_range += 1
        return None
    return candidate


def _derive_reg_year(entry: _Element, stats: NyplRegParseStats) -> int | None:
    """Resolve a registration year via the ``regDate → copyDate → pubDate`` chain.

    Many entries — notably *ad interim* registrations (``regnum`` like
    "AI…") — carry no ``<regDate>`` at all; their date lives in
    ``<copyDate>`` (the copyright © event) and/or ``<pubDate>`` (the
    imprint/publication date). Without a fallback those records get
    ``reg_year=None``, fall into no year bucket, and become unreachable by
    the year-blocked matcher.

    The chain is ordered by semantic closeness to a registration event:

    * ``regDate`` — the registration date itself (canonical).
    * ``copyDate`` — the copyright (©) date; the closest analog to
      registration when no ``regDate`` exists.
    * ``pubDate`` — the publication/imprint date; most aligned with the
      MARC publication year we block against.

    ``affDate`` (affidavit — a procedural printing/manufacture date) and
    ``noticeDate`` are deliberately EXCLUDED: neither is a registration,
    copyright, or publication event, so using them as a year source would
    introduce dates unrelated to the registration we are matching.

    For ``pubDate`` we prefer a direct child or a publisher-level child of
    this entry over any ``pubDate`` buried in nested ``additionalEntry`` /
    ``prevPub`` blocks, so the recovered year reflects this entry's own
    publication rather than a referenced one.
    """
    for tag in ("regDate", "copyDate"):
        year = _year_from_date_element(entry.find(tag), stats)
        if year is not None:
            return year
    return _year_from_date_element(_find_entry_pub_date(entry), stats)


def _find_entry_pub_date(entry: _Element) -> _Element | None:
    """Return this entry's own ``<pubDate>``, ignoring nested-entry copies.

    A direct child is preferred, then a publisher-level child; both belong
    to the entry itself. ``pubDate`` elements inside ``additionalEntry`` or
    ``prevPub`` describe a different (referenced) work and are skipped.
    """
    direct = entry.find("pubDate")
    if direct is not None:
        return direct
    for publisher in entry.iterfind("publisher"):
        publisher_pub_date = publisher.find("pubDate")
        if publisher_pub_date is not None:
            return publisher_pub_date
    return None


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


def _extract_date_attr(entry: _Element, tag: str) -> date | None:
    """Return the ``date`` attribute of ``entry``'s first ``<tag>`` child as a date."""
    elem = entry.find(tag)
    if elem is None:
        return None
    return _parse_date(elem.get("date"))


def _collect_notes(entry: _Element, stats: NyplRegParseStats) -> tuple[str, ...]:
    """Return the text of every ``<note>`` child of ``entry`` in document order."""
    out: list[str] = []
    for note in entry.iterfind("note"):
        value = _text(note, stats)
        if value is not None:
            out.append(value)
    return tuple(out)


def _extract_lccn(entry: _Element, stats: NyplRegParseStats) -> str | None:
    """Return the LCCN for ``entry`` or ``None`` when no usable value exists.

    Prefers the ``normalized`` attribute (an 8-digit canonical form, e.g.
    ``"28000854"``) when present and non-empty; falls back to the element's
    encoding-cleaned text (e.g. ``"28-854"``). Takes the first ``<lccn>``
    when several exist (same convention as ``<author>``).
    """
    elem = entry.find("lccn")
    if elem is None:
        return None
    normalized = elem.get("normalized")
    if normalized is not None:
        stripped = normalized.strip()
        if stripped:
            return stripped
    return _text(elem, stats)


def _collect_prev_regnums(entry: _Element, stats: NyplRegParseStats) -> tuple[str, ...]:
    """Return the text of every ``<prev-regNum>`` child of ``entry`` in order."""
    out: list[str] = []
    for prev in entry.iterfind("prev-regNum"):
        value = _text(prev, stats)
        if value is not None:
            out.append(value)
    return tuple(out)


def _collect_additional_join_keys(
    entry: _Element, stats: NyplRegParseStats
) -> tuple[tuple[str, int], ...]:
    """Return one ``(regnum, year)`` join key per ``<additionalEntry>`` child.

    A single ``<copyrightEntry>`` can bundle several separate registrations as
    ``<additionalEntry>`` children (guide: "Multiple claims in a single entry"),
    each carrying its own ``<regNum>`` and ``<regDate>``. Those interior numbers
    are dropped by :func:`_build_record`, so a renewal citing one cannot join.
    This harvests just enough — the regnum and its year — to add each as an
    extra join key on the parent registration.

    The regnum is read attribute-first (``regnum``) then inline ``<regNum>``
    text, mirroring the top-level convention. The year is **strict**: it is
    derived solely from the additionalEntry's own ``<regDate>``; an
    additionalEntry with no own ``<regDate>`` (or an implausible one) is skipped
    rather than inheriting the parent entry's date, because the parent date is a
    different registration event and would manufacture spurious joins. An
    additionalEntry with no usable regnum is likewise skipped.

    ``<renewalEntry>`` blocks (standalone renewals transcribed in the
    registration XML) are a separate follow-up and are intentionally not
    handled here; see the additionalEntry join-yield finding for the split.
    """
    out: list[tuple[str, int]] = []
    for additional in entry.iterfind(_ADDITIONAL_ENTRY_TAG):
        regnum = additional.get("regnum") or _text(additional.find("regNum"), stats)
        if regnum is None:
            continue
        year = _year_from_date_element(additional.find("regDate"), stats)
        if year is None:
            continue
        out.append((regnum, year))
    return tuple(out)


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
    reg_year = _derive_reg_year(entry, stats)

    first_author = entry.find("author")
    author_name = (
        _text(first_author.find("authorName"), stats) if first_author is not None else None
    )
    author_place = (
        _text(first_author.find("authorPlace"), stats) if first_author is not None else None
    )
    author_is_claimant = first_author is not None and first_author.get("claimant") == "yes"
    edition = _text(entry.find("edition"), stats)
    publisher_names, claimants_from_pub = _collect_publisher_names(entry, stats)
    publication_places = _collect_publication_places(entry, stats)
    explicit_claimants = _collect_explicit_claimants(entry, stats)
    combined_claimants = (*claimants_from_pub, *explicit_claimants)

    copies = _text(entry.find("copies"), stats)
    aff_date = _extract_date_attr(entry, "affDate")
    desc = _text(entry.find("desc"), stats)
    notes = _collect_notes(entry, stats)
    new_matter_claimed = _text(entry.find("newMatterClaimed"), stats)
    copy_date = _extract_date_attr(entry, "copyDate")
    notice_date = _extract_date_attr(entry, "noticeDate")
    lccn = _extract_lccn(entry, stats)
    prev_regnums = _collect_prev_regnums(entry, stats)
    additional_join_keys = _collect_additional_join_keys(entry, stats)

    stats.emitted += 1
    return NyplRegRecord(
        uuid=uuid,
        title=title,
        regnum=regnum,
        reg_date=reg_date,
        reg_year=reg_year,
        author_name=author_name,
        author_place=author_place,
        author_is_claimant=author_is_claimant,
        edition=edition,
        publisher_names=tuple(publisher_names),
        publication_places=tuple(publication_places),
        claimants=combined_claimants,
        copies=copies,
        aff_date=aff_date,
        desc=desc,
        notes=notes,
        new_matter_claimed=new_matter_claimed,
        copy_date=copy_date,
        notice_date=notice_date,
        lccn=lccn,
        prev_regnums=prev_regnums,
        additional_join_keys=additional_join_keys,
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
