"""Unit tests for record-eligibility predicates."""

from lxml.etree import _Element
from lxml.etree import fromstring
from pytest import mark

from pd_groundtruth.filters import Ineligibility
from pd_groundtruth.filters import classify
from pd_groundtruth.filters import is_electronic_resource
from pd_groundtruth.filters import is_eligible
from pd_groundtruth.filters import language_of
from pd_groundtruth.filters import year_of

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_MIN_YEAR = 1931


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
    assert is_eligible(record, _MIN_YEAR) is True
    assert classify(record, _MIN_YEAR) is Ineligibility.ELIGIBLE


@mark.parametrize("language", ["eng", "fre", "ger", "spa", "ita"])
def test_each_supported_language_passes(language: str) -> None:
    field = f"750101s1950    xxu           000 0 {language} d"
    record = _build_record(field_008=field)
    assert is_eligible(record, _MIN_YEAR) is True


def test_unsupported_language_fails() -> None:
    field = "750101s1950    xxu           000 0 lat d"
    record = _build_record(field_008=field)
    assert classify(record, _MIN_YEAR) is Ineligibility.UNSUPPORTED_LANGUAGE


def test_wrong_leader_type_fails() -> None:
    record = _build_record(leader="00000ncm a2200000 a 4500")
    assert classify(record, _MIN_YEAR) is Ineligibility.NOT_A_BOOK


def test_wrong_leader_level_fails() -> None:
    record = _build_record(leader="00000nas a2200000 a 4500")
    assert classify(record, _MIN_YEAR) is Ineligibility.NOT_A_MONOGRAPH


def test_missing_leader_fails() -> None:
    record = _build_record(leader=None)
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_LEADER


def test_short_leader_fails() -> None:
    record = _build_record(leader="00000n")
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_LEADER


def test_empty_leader_text_fails() -> None:
    record = fromstring(b"<record><leader></leader></record>")
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_LEADER


@mark.parametrize("year", ["1930", "1929", "1922"])
def test_year_below_min_year_fails(year: str) -> None:
    field = f"750101s{year}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert classify(record, _MIN_YEAR) is Ineligibility.YEAR_OUT_OF_RANGE


def test_year_at_min_year_passes() -> None:
    field = f"750101s{_MIN_YEAR}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert is_eligible(record, _MIN_YEAR) is True


def test_year_above_range_fails() -> None:
    field = "750101s1978    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert classify(record, _MIN_YEAR) is Ineligibility.YEAR_OUT_OF_RANGE


@mark.parametrize("year", ["1931", "1977", "1950"])
def test_year_boundaries_in_range_pass(year: str) -> None:
    field = f"750101s{year}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert is_eligible(record, _MIN_YEAR) is True


@mark.parametrize(("year", "expected"), [("1953", 1953), ("1931", 1931), ("1977", 1977)])
def test_year_of_returns_year(year: str, expected: int) -> None:
    field = f"750101s{year}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert year_of(record) == expected


@mark.parametrize("year", ["uuuu", "||||", "    "])
def test_year_of_invalid_returns_none(year: str) -> None:
    field = f"750101s{year}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert year_of(record) is None


def test_year_of_missing_008_returns_none() -> None:
    record = _build_record(field_008=None)
    assert year_of(record) is None


@mark.parametrize("year", ["uuuu", "nnnn", "||||", "    ", "19 5"])
def test_invalid_year_string_fails(year: str) -> None:
    field = f"750101s{year}    xxu           000 0 eng d"
    record = _build_record(field_008=field)
    assert classify(record, _MIN_YEAR) is Ineligibility.INVALID_YEAR


def test_missing_008_fails() -> None:
    record = _build_record(field_008=None)
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_008


def test_short_008_fails() -> None:
    record = _build_record(field_008="750101s1950")
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_008


def test_empty_008_text_fails() -> None:
    record = fromstring(
        b"<record><leader>00000nam a2200000 a 4500</leader>"
        b'<controlfield tag="008"></controlfield></record>'
    )
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_008


def test_missing_245a_fails() -> None:
    record = _build_record(title_a=None)
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_TITLE


def test_blank_245a_fails() -> None:
    record = _build_record(title_a="   ")
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_TITLE


def test_245_without_subfield_a_fails() -> None:
    record = fromstring(
        b"<record><leader>00000nam a2200000 a 4500</leader>"
        b'<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
        b'<datafield tag="245"><subfield code="b">subtitle only</subfield></datafield></record>'
    )
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_TITLE


def test_245a_with_no_text_node_fails() -> None:
    record = fromstring(
        b"<record><leader>00000nam a2200000 a 4500</leader>"
        b'<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
        b'<datafield tag="245"><subfield code="a"/></datafield></record>'
    )
    assert classify(record, _MIN_YEAR) is Ineligibility.MISSING_TITLE


def _field_008_with_gov(code: str) -> str:
    """Return a valid 1950/eng 008 string with ``code`` at position 28."""
    base = "750101s1950    xxu           000 0 eng d"
    return base[:28] + code + base[29:]


@mark.parametrize("code", list("acfilmosuz"))
def test_government_publication_codes_fail(code: str) -> None:
    record = _build_record(field_008=_field_008_with_gov(code))
    assert classify(record, _MIN_YEAR) is Ineligibility.GOVERNMENT_PUBLICATION


@mark.parametrize("code", [" ", "|"])
def test_blank_and_pipe_government_codes_pass(code: str) -> None:
    record = _build_record(field_008=_field_008_with_gov(code))
    assert is_eligible(record, _MIN_YEAR) is True
    assert classify(record, _MIN_YEAR) is Ineligibility.ELIGIBLE


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


_BASELINE_LEADER = "00000nam a2200000 a 4500"
_BASELINE_008 = "750101s1955    xxu           000 0 eng d"


def _wrap_record(inner: str, *, namespaced: bool = False) -> _Element:
    """Wrap raw child XML in a ``<record>``, optionally namespaced."""
    if namespaced:
        xml = f'<record xmlns="{_MARC_NS}">{inner}</record>'
    else:
        xml = f"<record>{inner}</record>"
    return fromstring(xml.encode("utf-8"))


def _baseline_record_inner(extra: str = "") -> str:
    """Return the inner XML of a baseline eligible monograph plus extras."""
    return (
        f"<leader>{_BASELINE_LEADER}</leader>"
        f'<controlfield tag="008">{_BASELINE_008}</controlfield>'
        f'<datafield tag="245"><subfield code="a">A valid title</subfield></datafield>'
        f"{extra}"
    )


def test_is_electronic_resource_007_byte0_c_flags() -> None:
    record = _wrap_record(
        _baseline_record_inner('<controlfield tag="007">cr||||||||||||</controlfield>')
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


def test_is_electronic_resource_338_b_cr_flags() -> None:
    record = _wrap_record(
        _baseline_record_inner('<datafield tag="338"><subfield code="b">cr</subfield></datafield>')
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


def test_is_electronic_resource_245_h_electronic_resource_flags() -> None:
    record = _wrap_record(
        "<leader>" + _BASELINE_LEADER + "</leader>"
        f'<controlfield tag="008">{_BASELINE_008}</controlfield>'
        '<datafield tag="245">'
        '<subfield code="a">A valid title</subfield>'
        '<subfield code="h">[electronic resource]</subfield>'
        "</datafield>"
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


def test_is_electronic_resource_300_a_online_resource_flags() -> None:
    extent = "1 online resource (xii, 245 pages)"
    record = _wrap_record(
        _baseline_record_inner(
            f'<datafield tag="300"><subfield code="a">{extent}</subfield></datafield>'
        )
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


def test_is_electronic_resource_eligible_when_no_indicators() -> None:
    record = _wrap_record(
        _baseline_record_inner(
            '<controlfield tag="007">ta</controlfield>'
            '<datafield tag="338"><subfield code="b">nc</subfield></datafield>'
            '<datafield tag="300"><subfield code="a">245 pages ; 24 cm</subfield></datafield>'
        )
    )
    assert is_electronic_resource(record) is Ineligibility.ELIGIBLE


@mark.parametrize("text", ["Electronic Resource", "ELECTRONIC RESOURCE", "[Electronic Resource]"])
def test_is_electronic_resource_245_h_case_insensitive(text: str) -> None:
    record = _wrap_record(
        "<leader>" + _BASELINE_LEADER + "</leader>"
        f'<controlfield tag="008">{_BASELINE_008}</controlfield>'
        '<datafield tag="245">'
        '<subfield code="a">A valid title</subfield>'
        f'<subfield code="h">{text}</subfield>'
        "</datafield>"
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


@mark.parametrize(
    "text",
    ["Online Resource", "ONLINE RESOURCE", "1 Online Resource (PDF)"],
)
def test_is_electronic_resource_300_a_case_insensitive(text: str) -> None:
    record = _wrap_record(
        _baseline_record_inner(
            f'<datafield tag="300"><subfield code="a">{text}</subfield></datafield>'
        )
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


def test_is_electronic_resource_multi_007_one_electronic_flags() -> None:
    record = _wrap_record(
        _baseline_record_inner(
            '<controlfield tag="007">ta</controlfield>'
            '<controlfield tag="007">cr||||||||||||</controlfield>'
        )
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


def test_is_electronic_resource_multi_338_one_online_flags() -> None:
    record = _wrap_record(
        _baseline_record_inner(
            '<datafield tag="338"><subfield code="b">nc</subfield></datafield>'
            '<datafield tag="338"><subfield code="b">cr</subfield></datafield>'
        )
    )
    assert is_electronic_resource(record) is Ineligibility.ELECTRONIC_RESOURCE


@mark.parametrize("body", ["", "   "])
def test_is_electronic_resource_007_empty_or_blank_does_not_flag(body: str) -> None:
    record = _wrap_record(_baseline_record_inner(f'<controlfield tag="007">{body}</controlfield>'))
    assert is_electronic_resource(record) is Ineligibility.ELIGIBLE


def test_is_electronic_resource_338_b_other_code_does_not_flag() -> None:
    record = _wrap_record(
        _baseline_record_inner('<datafield tag="338"><subfield code="b">nc</subfield></datafield>')
    )
    assert is_electronic_resource(record) is Ineligibility.ELIGIBLE


def test_classify_otherwise_eligible_record_with_007_c_returns_electronic_resource() -> None:
    record = _wrap_record(
        _baseline_record_inner('<controlfield tag="007">cr||||||||||||</controlfield>')
    )
    assert classify(record, _MIN_YEAR) is Ineligibility.ELECTRONIC_RESOURCE
    assert is_eligible(record, _MIN_YEAR) is False
