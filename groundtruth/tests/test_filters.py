"""Unit tests for record-eligibility predicates."""

from lxml.etree import _Element
from lxml.etree import fromstring
from pytest import mark

from pd_groundtruth.filters import Ineligibility
from pd_groundtruth.filters import classify
from pd_groundtruth.filters import is_eligible
from pd_groundtruth.filters import language_of

_MARC_NS = "http://www.loc.gov/MARC21/slim"


def _build_record(
    *,
    leader: str | None = "00000nam a2200000 a 4500",
    field_008: str | None = "750101s1950    xxu           000 0 eng d",
    title_a: str | None = "A valid title",
    namespaced: bool = False,
) -> _Element:
    """Construct a MARCXML ``<record>`` element from raw component strings."""
    parts: list[str] = []
    if leader is not None:
        parts.append(f"<leader>{leader}</leader>")
    if field_008 is not None:
        parts.append(f'<controlfield tag="008">{field_008}</controlfield>')
    if title_a is not None:
        parts.append(f'<datafield tag="245"><subfield code="a">{title_a}</subfield></datafield>')
    inner = "".join(parts)
    if namespaced:
        xml = f'<record xmlns="{_MARC_NS}">{inner}</record>'
    else:
        xml = f"<record>{inner}</record>"
    return fromstring(xml.encode("utf-8"))


@mark.parametrize("namespaced", [False, True])
def test_fully_eligible_record(namespaced: bool) -> None:
    record = _build_record(namespaced=namespaced)
    assert is_eligible(record) is True
    assert classify(record) is Ineligibility.ELIGIBLE


@mark.parametrize("language", ["eng", "fre", "ger", "spa", "ita"])
def test_each_supported_language_passes(language: str) -> None:
    field = f"750101s1950    xxu           000 0 {language} d"
    record = _build_record(field_008=field)
    assert is_eligible(record) is True


def test_unsupported_language_fails() -> None:
    field = "750101s1950    xxu           000 0 lat d"
    record = _build_record(field_008=field)
    assert classify(record) is Ineligibility.UNSUPPORTED_LANGUAGE


def test_wrong_leader_type_fails() -> None:
    record = _build_record(leader="00000ncm a2200000 a 4500")
    assert classify(record) is Ineligibility.NOT_A_BOOK


def test_wrong_leader_level_fails() -> None:
    record = _build_record(leader="00000nas a2200000 a 4500")
    assert classify(record) is Ineligibility.NOT_A_MONOGRAPH


def test_missing_leader_fails() -> None:
    record = _build_record(leader=None)
    assert classify(record) is Ineligibility.MISSING_LEADER


def test_short_leader_fails() -> None:
    record = _build_record(leader="00000n")
    assert classify(record) is Ineligibility.MISSING_LEADER


def test_empty_leader_text_fails() -> None:
    record = fromstring(b"<record><leader></leader></record>")
    assert classify(record) is Ineligibility.MISSING_LEADER


def test_year_below_range_fails() -> None:
    field = "750101s1922    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert classify(record) is Ineligibility.YEAR_OUT_OF_RANGE


def test_year_above_range_fails() -> None:
    field = "750101s1978    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert classify(record) is Ineligibility.YEAR_OUT_OF_RANGE


@mark.parametrize("year", ["1923", "1977", "1950"])
def test_year_boundaries_in_range_pass(year: str) -> None:
    field = f"750101s{year}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert is_eligible(record) is True


@mark.parametrize("year", ["uuuu", "nnnn", "||||", "    ", "19 5"])
def test_invalid_year_string_fails(year: str) -> None:
    field = f"750101s{year}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert classify(record) is Ineligibility.INVALID_YEAR


def test_missing_008_fails() -> None:
    record = _build_record(field_008=None)
    assert classify(record) is Ineligibility.MISSING_008


def test_short_008_fails() -> None:
    record = _build_record(field_008="750101s1950")
    assert classify(record) is Ineligibility.MISSING_008


def test_empty_008_text_fails() -> None:
    record = fromstring(
        b"<record><leader>00000nam a2200000 a 4500</leader>"
        b'<controlfield tag="008"></controlfield></record>'
    )
    assert classify(record) is Ineligibility.MISSING_008


def test_missing_245a_fails() -> None:
    record = _build_record(title_a=None)
    assert classify(record) is Ineligibility.MISSING_TITLE


def test_blank_245a_fails() -> None:
    record = _build_record(title_a="   ")
    assert classify(record) is Ineligibility.MISSING_TITLE


def test_245_without_subfield_a_fails() -> None:
    record = fromstring(
        b"<record><leader>00000nam a2200000 a 4500</leader>"
        b'<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
        b'<datafield tag="245"><subfield code="b">subtitle only</subfield></datafield></record>'
    )
    assert classify(record) is Ineligibility.MISSING_TITLE


def test_245a_with_no_text_node_fails() -> None:
    record = fromstring(
        b"<record><leader>00000nam a2200000 a 4500</leader>"
        b'<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
        b'<datafield tag="245"><subfield code="a"/></datafield></record>'
    )
    assert classify(record) is Ineligibility.MISSING_TITLE


def _field_008_with_gov(code: str) -> str:
    """Return a valid 1950/eng 008 string with ``code`` at position 28."""
    base = "750101s1950    xxu           000 0 eng d"
    return base[:28] + code + base[29:]


@mark.parametrize("code", list("acfilmosuz"))
def test_government_publication_codes_fail(code: str) -> None:
    record = _build_record(field_008=_field_008_with_gov(code))
    assert classify(record) is Ineligibility.GOVERNMENT_PUBLICATION


@mark.parametrize("code", [" ", "|"])
def test_blank_and_pipe_government_codes_pass(code: str) -> None:
    record = _build_record(field_008=_field_008_with_gov(code))
    assert is_eligible(record) is True
    assert classify(record) is Ineligibility.ELIGIBLE


@mark.parametrize("language", ["eng", "fre", "ger", "spa", "ita", "lat"])
def test_language_of_returns_code(language: str) -> None:
    field = f"750101s1950    xxu           000 0 {language} d"
    record = _build_record(field_008=field)
    assert language_of(record) == language


def test_language_of_missing_008_returns_none() -> None:
    record = _build_record(field_008=None)
    assert language_of(record) is None


def test_language_of_short_008_returns_none() -> None:
    record = _build_record(field_008="750101s1950")
    assert language_of(record) is None
