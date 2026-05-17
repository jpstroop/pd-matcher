"""Streaming MARCXML parser yielding :class:`MarcRecord` instances.

The Library of Congress MARCXML files we ingest are routinely several gigabytes
each, so this parser uses :func:`lxml.etree.iterparse` with explicit element
clearing after every emitted record. ``pymarc.parse_xml`` was evaluated and
rejected because it materializes the whole document; the iterparse pattern
below holds at most one ``<record>`` element in memory at a time. A small
warning counter exposes how many records were skipped for missing required
fields so callers can surface dataset quality issues without re-reading the
file.
"""

from collections.abc import Iterator
from logging import getLogger
from pathlib import Path
from re import compile as re_compile

from lxml.etree import _Element
from lxml.etree import iterparse

from pd_matcher.models import MarcRecord

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_RECORD_TAG = f"{{{_MARC_NS}}}record"
_CONTROLFIELD_TAG = f"{{{_MARC_NS}}}controlfield"
_DATAFIELD_TAG = f"{{{_MARC_NS}}}datafield"
_SUBFIELD_TAG = f"{{{_MARC_NS}}}subfield"

_TRAILING_PUNCT = " /:;,="
_YEAR_RE = re_compile(r"(\d{4})")
_YEAR_MIN = 1450
_YEAR_MAX = 2050

_LOGGER = getLogger(__name__)


class MarcParseStats:
    """Mutable counters surfaced to callers after a parse run."""

    __slots__ = ("emitted", "skipped_missing_001", "skipped_missing_245a")

    def __init__(self) -> None:
        self.emitted = 0
        self.skipped_missing_001 = 0
        self.skipped_missing_245a = 0


def _clean(value: str | None) -> str | None:
    """Strip MARC trailing punctuation and surrounding whitespace.

    Args:
        value: Raw subfield text or ``None``.

    Returns:
        Cleaned text, or ``None`` if ``value`` is ``None`` or empties out.
    """
    if value is None:
        return None
    stripped = value.strip().rstrip(_TRAILING_PUNCT).strip()
    return stripped or None


def _extract_year(raw_260c: str | None, control_008: str | None) -> int | None:
    """Pull a 4-digit year first from 260/264 ``$c``, then from 008 7-10.

    Args:
        raw_260c: Untrimmed value of 260/264 ``$c`` if present.
        control_008: Raw ``008`` controlfield value if present.

    Returns:
        A year in ``[_YEAR_MIN, _YEAR_MAX]`` or ``None`` if neither source
        yields a plausible 4-digit run.
    """
    if raw_260c is not None:
        match = _YEAR_RE.search(raw_260c)
        if match is not None:
            year = int(match.group(1))
            if _YEAR_MIN <= year <= _YEAR_MAX:
                return year
    if control_008 is not None and len(control_008) >= 11:
        candidate = control_008[7:11]
        if candidate.isdigit():
            year = int(candidate)
            if _YEAR_MIN <= year <= _YEAR_MAX:
                return year
    return None


def _slice_008(control_008: str | None, start: int, end: int) -> str | None:
    """Return ``control_008[start:end]`` if available and non-blank."""
    if control_008 is None or len(control_008) < end:
        return None
    chunk = control_008[start:end].strip()
    return chunk or None


def _subfield_texts(field: _Element, code: str) -> list[str]:
    """Collect all ``$code`` subfield text nodes within ``field``."""
    out: list[str] = []
    for sub in field.iterfind(_SUBFIELD_TAG):
        if sub.get("code") == code and sub.text is not None:
            out.append(sub.text)
    return out


def _first_subfield(field: _Element, code: str) -> str | None:
    """Return the first ``$code`` subfield text within ``field`` or ``None``."""
    for sub in field.iterfind(_SUBFIELD_TAG):
        if sub.get("code") == code and sub.text is not None:
            return sub.text
    return None


def _build_record(record_elem: _Element, stats: MarcParseStats) -> MarcRecord | None:
    """Translate one ``<record>`` element into a :class:`MarcRecord`.

    Records missing a 001 control number or a 245 ``$a`` title are skipped and
    a counter on ``stats`` is incremented. The element itself is left to the
    caller to clear.

    Args:
        record_elem: The MARC ``<record>`` element to read.
        stats: Counters mutated to track skipped records.

    Returns:
        A populated :class:`MarcRecord`, or ``None`` if required fields are
        missing.
    """
    control_id: str | None = None
    control_008: str | None = None
    lccn: str | None = None
    isbns: list[str] = []
    main_author: str | None = None
    added_authors: list[str] = []
    title_a: str | None = None
    title_b: str | None = None
    sor: str | None = None
    edition: str | None = None
    pub_place: str | None = None
    publisher: str | None = None
    pub_date_raw: str | None = None
    extent: str | None = None
    series_titles: list[str] = []

    for child in record_elem:
        if child.tag == _CONTROLFIELD_TAG:
            tag = child.get("tag")
            text = child.text
            if tag == "001" and text is not None:
                control_id = text.strip() or None
            elif tag == "008" and text is not None:
                control_008 = text
            continue
        if child.tag != _DATAFIELD_TAG:
            continue
        tag = child.get("tag")
        if tag == "010":
            if lccn is None:
                lccn = _first_subfield(child, "a")
        elif tag == "020":
            isbns.extend(_subfield_texts(child, "a"))
        elif tag in {"100", "110", "111"}:
            if main_author is None:
                main_author = _first_subfield(child, "a")
        elif tag == "245":
            if title_a is None:
                title_a = _first_subfield(child, "a")
                title_b = _first_subfield(child, "b")
                sor = _first_subfield(child, "c")
        elif tag == "250":
            if edition is None:
                edition = _first_subfield(child, "a")
        elif tag in {"260", "264"}:
            if pub_place is None:
                pub_place = _first_subfield(child, "a")
            if publisher is None:
                publisher = _first_subfield(child, "b")
            if pub_date_raw is None:
                pub_date_raw = _first_subfield(child, "c")
        elif tag == "300":
            if extent is None:
                extent = _first_subfield(child, "a")
        elif tag in {"440", "490", "830"}:
            series_titles.extend(_subfield_texts(child, "a"))
        elif tag == "700":
            added_authors.extend(_subfield_texts(child, "a"))

    if control_id is None:
        stats.skipped_missing_001 += 1
        _LOGGER.warning("marc.skip", extra={"reason": "missing_001"})
        return None
    cleaned_title_a = _clean(title_a)
    if cleaned_title_a is None:
        stats.skipped_missing_245a += 1
        _LOGGER.warning("marc.skip", extra={"reason": "missing_245a", "control_id": control_id})
        return None

    cleaned_title_b = _clean(title_b)
    if cleaned_title_b is None:
        full_title = cleaned_title_a
    else:
        full_title = f"{cleaned_title_a} {cleaned_title_b}"

    stats.emitted += 1
    return MarcRecord(
        control_id=control_id,
        title=full_title,
        lccn=_clean(lccn),
        isbns=tuple(cleaned for cleaned in (_clean(v) for v in isbns) if cleaned is not None),
        main_author=_clean(main_author),
        added_authors=tuple(
            cleaned for cleaned in (_clean(v) for v in added_authors) if cleaned is not None
        ),
        statement_of_responsibility=_clean(sor),
        edition=_clean(edition),
        publication_place=_clean(pub_place),
        publisher=_clean(publisher),
        publication_date_raw=_clean(pub_date_raw),
        publication_year=_extract_year(pub_date_raw, control_008),
        extent=_clean(extent),
        series_titles=tuple(
            cleaned for cleaned in (_clean(v) for v in series_titles) if cleaned is not None
        ),
        language_code=_slice_008(control_008, 35, 38),
        country_code=_slice_008(control_008, 15, 18),
    )


def iter_marc_records(path: Path, stats: MarcParseStats | None = None) -> Iterator[MarcRecord]:
    """Yield :class:`MarcRecord` objects streamed from a MARCXML file.

    The implementation follows the canonical lxml streaming pattern: each
    ``<record>`` element is cleared after emission and all previous siblings
    are detached from the parent, capping peak memory at one record's worth
    of parsed tree regardless of input file size.

    Args:
        path: MARCXML file to parse.
        stats: Optional :class:`MarcParseStats` counters mutated as records
            are emitted or skipped. A fresh stats object is created when
            none is supplied; callers wanting visibility into skip counts
            must pass their own.

    Yields:
        One :class:`MarcRecord` per ``<record>`` element with a valid 001
        and 245 ``$a``.
    """
    counters = stats if stats is not None else MarcParseStats()
    context = iterparse(str(path), events=("end",), tag=_RECORD_TAG)
    for _event, elem in context:
        record = _build_record(elem, counters)
        if record is not None:
            yield record
        elem.clear()
        previous = elem.getprevious()
        while previous is not None:
            del elem.getparent()[0]
            previous = elem.getprevious()


__all__ = [
    "MarcParseStats",
    "iter_marc_records",
]
