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

_CCE_MIN_YEAR = 1923
_CCE_MAX_YEAR = 1977
_SUPPORTED_LANGUAGES = frozenset({"eng", "fre", "ger", "spa", "ita"})
_NON_GOVERNMENT_CODES = frozenset({" ", "|"})


class Ineligibility(Enum):
    """Reason a record was rejected, suitable for logging and counters."""

    ELIGIBLE = "eligible"
    MISSING_LEADER = "missing_leader"
    NOT_A_BOOK = "not_a_book"
    NOT_A_MONOGRAPH = "not_a_monograph"
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


def check_language_and_year(record: _Element) -> Ineligibility:
    """Validate the 008 language (35:38), year (7:11), and gov-publication (28)."""
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
    if year < _CCE_MIN_YEAR or year > _CCE_MAX_YEAR:
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


def classify(record: _Element) -> Ineligibility:
    """Return the first failing criterion, or ``ELIGIBLE`` when all pass.

    Args:
        record: A raw MARCXML ``<record>`` element (namespaced or not).

    Returns:
        ``Ineligibility.ELIGIBLE`` if every criterion holds, otherwise the
        first criterion that fails.
    """
    monograph = is_monograph(record)
    if monograph is not Ineligibility.ELIGIBLE:
        return monograph
    language_and_year = check_language_and_year(record)
    if language_and_year is not Ineligibility.ELIGIBLE:
        return language_and_year
    return has_title(record)


def is_eligible(record: _Element) -> bool:
    """Return whether a record satisfies every eligibility criterion."""
    return classify(record) is Ineligibility.ELIGIBLE
