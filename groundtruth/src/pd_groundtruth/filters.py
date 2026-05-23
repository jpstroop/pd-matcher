"""Pure record-eligibility predicates over raw MARCXML ``<record>`` elements.

A record is *eligible* for the ground-truth corpus when it is a monograph in a
supported language, published within the CCE-relevant window, and carries a
title. All predicates read directly off the raw leader, the 008 control field,
and the 245 data field so that no normalization or model construction is
required before the decision is made.

Records may arrive in the MARC21 slim namespace
(``http://www.loc.gov/MARC21/slim``) or with no namespace at all, depending on
the source serialization. Every element lookup matches by *local name* so both
serializations behave identically.
"""

from enum import Enum

from lxml.etree import QName
from lxml.etree import _Element

_LEADER_TYPE_POSITION = 6
_LEADER_BIBLIOGRAPHIC_LEVEL_POSITION = 7
_LEADER_BOOK_TYPE = "a"
_LEADER_MONOGRAPH_LEVEL = "m"

_CONTROLFIELD_008_MIN_LENGTH = 38
_LANGUAGE_START = 35
_LANGUAGE_END = 38
_YEAR_START = 7
_YEAR_END = 11
_GOVERNMENT_PUBLICATION_POSITION = 28

_CCE_MAX_YEAR = 1977
_SUPPORTED_LANGUAGES = frozenset({"eng", "fre", "ger", "spa", "ita"})
_NON_GOVERNMENT_CODES = frozenset({" ", "|"})

_FIELD_007_ELECTRONIC_RESOURCE_CODE = "c"
_FIELD_338_ONLINE_RESOURCE_CARRIER = "cr"
_FIELD_245_ELECTRONIC_RESOURCE_MARKER = "electronic resource"
_FIELD_300_ONLINE_RESOURCE_MARKER = "online resource"


class Ineligibility(Enum):
    """Reason a record was rejected, suitable for logging and counters."""

    ELIGIBLE = "eligible"
    MISSING_LEADER = "missing_leader"
    NOT_A_BOOK = "not_a_book"
    NOT_A_MONOGRAPH = "not_a_monograph"
    ELECTRONIC_RESOURCE = "electronic_resource"
    MISSING_008 = "missing_008"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    YEAR_OUT_OF_RANGE = "year_out_of_range"
    INVALID_YEAR = "invalid_year"
    GOVERNMENT_PUBLICATION = "government_publication"
    MISSING_TITLE = "missing_title"


def _local_name(element: _Element) -> str:
    """Return the namespace-agnostic local name of an element."""
    return QName(element).localname


def _find_local(record: _Element, local_name: str) -> _Element | None:
    """Return the first child whose local name matches, regardless of namespace."""
    for child in record:
        if _local_name(child) == local_name:
            return child
    return None


def _find_controlfield(record: _Element, tag: str) -> _Element | None:
    """Return the first ``controlfield`` with the given ``tag`` attribute."""
    for child in record:
        if _local_name(child) == "controlfield" and child.get("tag") == tag:
            return child
    return None


def _find_datafield(record: _Element, tag: str) -> _Element | None:
    """Return the first ``datafield`` with the given ``tag`` attribute."""
    for child in record:
        if _local_name(child) == "datafield" and child.get("tag") == tag:
            return child
    return None


def _find_all_controlfields(record: _Element, tag: str) -> list[_Element]:
    """Return every ``controlfield`` with the given ``tag`` attribute."""
    return [
        child
        for child in record
        if _local_name(child) == "controlfield" and child.get("tag") == tag
    ]


def _find_all_datafields(record: _Element, tag: str) -> list[_Element]:
    """Return every ``datafield`` with the given ``tag`` attribute."""
    return [
        child for child in record if _local_name(child) == "datafield" and child.get("tag") == tag
    ]


def _leader_text(record: _Element) -> str | None:
    """Return the leader text, or ``None`` if absent or empty."""
    leader = _find_local(record, "leader")
    if leader is None or leader.text is None:
        return None
    return leader.text


def is_monograph(record: _Element) -> Ineligibility:
    """Check leader positions 6 and 7 for a book-format monograph."""
    leader = _leader_text(record)
    if leader is None or len(leader) <= _LEADER_BIBLIOGRAPHIC_LEVEL_POSITION:
        return Ineligibility.MISSING_LEADER
    if leader[_LEADER_TYPE_POSITION] != _LEADER_BOOK_TYPE:
        return Ineligibility.NOT_A_BOOK
    if leader[_LEADER_BIBLIOGRAPHIC_LEVEL_POSITION] != _LEADER_MONOGRAPH_LEVEL:
        return Ineligibility.NOT_A_MONOGRAPH
    return Ineligibility.ELIGIBLE


def is_electronic_resource(record: _Element) -> Ineligibility:
    """Detect digital-reissue MARC records via belt-and-suspenders indicators.

    A record is flagged as an electronic resource when *any* of the following
    hold:

    1. A ``<controlfield tag="007">`` whose first byte is ``c`` (the dedicated
       electronic-resource indicator in MARC 007 byte 0).
    2. A ``<datafield tag="338">`` with a ``<subfield code="b">`` whose stripped
       lowercase text equals ``cr`` (the RDA carrier code for "online
       resource").
    3. A ``<datafield tag="245">`` with a ``<subfield code="h">`` whose
       lowercase text contains ``electronic resource`` (the AACR2 general
       material designation, typically rendered as ``[electronic resource]``).
    4. A ``<datafield tag="300">`` with a ``<subfield code="a">`` whose
       lowercase text contains ``online resource`` (the extent-string proxy
       used by many cataloging chains).

    Args:
        record: A raw MARCXML ``<record>`` element (namespaced or not).

    Returns:
        ``Ineligibility.ELECTRONIC_RESOURCE`` when any indicator fires,
        otherwise ``Ineligibility.ELIGIBLE``.
    """
    for controlfield in _find_all_controlfields(record, "007"):
        text = controlfield.text
        if text is None:
            continue
        stripped = text.strip()
        if len(stripped) >= 1 and stripped[0] == _FIELD_007_ELECTRONIC_RESOURCE_CODE:
            return Ineligibility.ELECTRONIC_RESOURCE

    for datafield in _find_all_datafields(record, "338"):
        for child in datafield:
            if (
                _local_name(child) == "subfield"
                and child.get("code") == "b"
                and child.text is not None
                and child.text.strip().lower() == _FIELD_338_ONLINE_RESOURCE_CARRIER
            ):
                return Ineligibility.ELECTRONIC_RESOURCE

    for datafield in _find_all_datafields(record, "245"):
        for child in datafield:
            if (
                _local_name(child) == "subfield"
                and child.get("code") == "h"
                and child.text is not None
                and _FIELD_245_ELECTRONIC_RESOURCE_MARKER in child.text.lower()
            ):
                return Ineligibility.ELECTRONIC_RESOURCE

    for datafield in _find_all_datafields(record, "300"):
        for child in datafield:
            if (
                _local_name(child) == "subfield"
                and child.get("code") == "a"
                and child.text is not None
                and _FIELD_300_ONLINE_RESOURCE_MARKER in child.text.lower()
            ):
                return Ineligibility.ELECTRONIC_RESOURCE

    return Ineligibility.ELIGIBLE


def _control_008_text(record: _Element) -> str | None:
    """Return the 008 control field text, or ``None`` if absent or too short."""
    field = _find_controlfield(record, "008")
    if field is None or field.text is None or len(field.text) < _CONTROLFIELD_008_MIN_LENGTH:
        return None
    return field.text


def language_of(record: _Element) -> str | None:
    """Return the 008 language code (positions 35:38), or ``None`` if unavailable.

    Args:
        record: A raw MARCXML ``<record>`` element (namespaced or not).

    Returns:
        The three-character language code, or ``None`` when the 008 control
        field is missing or shorter than the minimum length.
    """
    text = _control_008_text(record)
    if text is None:
        return None
    return text[_LANGUAGE_START:_LANGUAGE_END]


def year_of(record: _Element) -> int | None:
    """Return the 008 publication year (positions 7:11), or ``None``.

    Args:
        record: A raw MARCXML ``<record>`` element (namespaced or not).

    Returns:
        The four-digit publication year as an integer, or ``None`` when the 008
        control field is missing, too short, or the year is not four digits.
    """
    text = _control_008_text(record)
    if text is None:
        return None
    year_text = text[_YEAR_START:_YEAR_END]
    if not year_text.isdigit():
        return None
    return int(year_text)


def check_language_and_year(record: _Element, min_year: int) -> Ineligibility:
    """Validate the 008 language (35:38), year (7:11), and gov-publication (28).

    Args:
        record: A raw MARCXML ``<record>`` element (namespaced or not).
        min_year: Inclusive lower bound for the publication year (the moving
            wall). Records before this year are public domain by age and carry
            no matching signal.

    Returns:
        ``Ineligibility.ELIGIBLE`` when the language, year, and gov-publication
        checks all pass, otherwise the first criterion that fails.
    """
    text = _control_008_text(record)
    if text is None:
        return Ineligibility.MISSING_008
    language = text[_LANGUAGE_START:_LANGUAGE_END]
    if language not in _SUPPORTED_LANGUAGES:
        return Ineligibility.UNSUPPORTED_LANGUAGE
    year_text = text[_YEAR_START:_YEAR_END]
    if not year_text.isdigit():
        return Ineligibility.INVALID_YEAR
    year = int(year_text)
    if year < min_year or year > _CCE_MAX_YEAR:
        return Ineligibility.YEAR_OUT_OF_RANGE
    if text[_GOVERNMENT_PUBLICATION_POSITION] not in _NON_GOVERNMENT_CODES:
        return Ineligibility.GOVERNMENT_PUBLICATION
    return Ineligibility.ELIGIBLE


def has_title(record: _Element) -> Ineligibility:
    """Check for a 245 data field with a non-empty subfield ``a``."""
    field = _find_datafield(record, "245")
    if field is None:
        return Ineligibility.MISSING_TITLE
    for child in field:
        if (
            _local_name(child) == "subfield"
            and child.get("code") == "a"
            and child.text is not None
            and child.text.strip() != ""
        ):
            return Ineligibility.ELIGIBLE
    return Ineligibility.MISSING_TITLE


def classify(record: _Element, min_year: int) -> Ineligibility:
    """Return the first failing criterion, or ``ELIGIBLE`` when all pass.

    Args:
        record: A raw MARCXML ``<record>`` element (namespaced or not).
        min_year: Inclusive lower bound for the publication year (the moving
            wall).

    Returns:
        ``Ineligibility.ELIGIBLE`` if every criterion holds, otherwise the
        first criterion that fails.
    """
    monograph = is_monograph(record)
    if monograph is not Ineligibility.ELIGIBLE:
        return monograph
    electronic = is_electronic_resource(record)
    if electronic is not Ineligibility.ELIGIBLE:
        return electronic
    language_and_year = check_language_and_year(record, min_year)
    if language_and_year is not Ineligibility.ELIGIBLE:
        return language_and_year
    return has_title(record)


def is_eligible(record: _Element, min_year: int) -> bool:
    """Return whether a record satisfies every eligibility criterion.

    Args:
        record: A raw MARCXML ``<record>`` element (namespaced or not).
        min_year: Inclusive lower bound for the publication year (the moving
            wall).

    Returns:
        ``True`` when every eligibility criterion holds, otherwise ``False``.
    """
    return classify(record, min_year) is Ineligibility.ELIGIBLE
