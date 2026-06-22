"""Unit tests for the streaming, uncapped MARCXML eligibility filter."""

from datetime import date
from pathlib import Path

from lxml.etree import parse

from pd_groundtruth.filter import filter_marcxml
from pd_matcher.parsers.marc import iter_marc_records

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_MIN_YEAR = 1931


def _record(
    *,
    title: str,
    field_008: str = "750101s1950    xxu           000 0 eng d",
    leader: str = "00000nam a2200000 a 4500",
    extra: str = "",
    control_001: str = "001",
) -> str:
    """Return a namespaced ``<record>`` fragment with the given components."""
    return (
        f'<record xmlns="{_MARC_NS}">'
        f"<leader>{leader}</leader>"
        f'<controlfield tag="001">{control_001}</controlfield>'
        f'<controlfield tag="008">{field_008}</controlfield>'
        f'<datafield tag="245"><subfield code="a">{title}</subfield></datafield>'
        f"{extra}"
        "</record>"
    )


def _eligible(
    title: str,
    *,
    language: str = "eng",
    year: int = 1950,
    control_001: str = "001",
) -> str:
    """Return a fully eligible record in the given language and year."""
    field = f"750101s{year}    xxu           000 0 {language} d"
    return _record(title=title, field_008=field, control_001=control_001)


def _collection(records: list[str]) -> bytes:
    """Wrap record fragments in a MARCXML collection document."""
    inner = "".join(records)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><collection xmlns="{_MARC_NS}">{inner}</collection>'
    ).encode()


def _write_input(tmp_path: Path, records: list[str]) -> Path:
    """Write a MARCXML collection to a temp file and return its path."""
    path = tmp_path / "input.marcxml"
    path.write_bytes(_collection(records))
    return path


def _kept_titles(output_path: Path) -> list[str]:
    """Return the 245 $a titles of every record in an output collection."""
    tree = parse(str(output_path))
    titles: list[str] = []
    for subfield in tree.iter(f"{{{_MARC_NS}}}subfield"):
        if subfield.get("code") == "a":
            titles.append(subfield.text or "")
    return titles


def test_eligible_records_are_kept(tmp_path: Path) -> None:
    records = [_eligible("First", control_001="a"), _eligible("Second", control_001="b")]
    input_path = _write_input(tmp_path, records)
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.scanned == 2
    assert report.kept == 2
    assert report.dropped == 0
    assert report.dropped_by_reason == {}
    assert sorted(_kept_titles(output_path)) == ["First", "Second"]


def test_not_a_book_is_dropped(tmp_path: Path) -> None:
    sound = _record(title="A recording", leader="00000njm a2200000 a 4500")
    input_path = _write_input(tmp_path, [_eligible("Book"), sound])
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.kept == 1
    assert report.dropped_by_reason == {"not_a_book": 1}
    assert _kept_titles(output_path) == ["Book"]


def test_electronic_resource_is_dropped(tmp_path: Path) -> None:
    electronic = _record(
        title="An e-book",
        extra='<controlfield tag="007">cr</controlfield>',
    )
    input_path = _write_input(tmp_path, [_eligible("Print"), electronic])
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.kept == 1
    assert report.dropped_by_reason == {"electronic_resource": 1}


def test_year_out_of_range_is_dropped(tmp_path: Path) -> None:
    too_old = _eligible("Ancient", year=1800)
    too_new = _eligible("Modern", year=1990)
    input_path = _write_input(tmp_path, [_eligible("In window"), too_old, too_new])
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.kept == 1
    assert report.dropped_by_reason == {"year_out_of_range": 2}


def test_invalid_year_is_dropped(tmp_path: Path) -> None:
    field = "750101suuuu    xxu           000 0 eng d"
    bad_year = _record(title="No year", field_008=field)
    input_path = _write_input(tmp_path, [_eligible("Good"), bad_year])
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.kept == 1
    assert report.dropped_by_reason == {"invalid_year": 1}


def test_unsupported_language_is_dropped(tmp_path: Path) -> None:
    latin = _eligible("Latina", language="lat")
    input_path = _write_input(tmp_path, [_eligible("English"), latin])
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.kept == 1
    assert report.dropped_by_reason == {"unsupported_language": 1}


def test_languages_restriction_narrows_within_eligible(tmp_path: Path) -> None:
    records = [
        _eligible("English", language="eng", control_001="a"),
        _eligible("French", language="fre", control_001="b"),
        _eligible("German", language="ger", control_001="c"),
    ]
    input_path = _write_input(tmp_path, records)
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(
        input_path=input_path,
        output_path=output_path,
        min_year=_MIN_YEAR,
        languages=frozenset({"eng", "fre"}),
    )

    assert report.kept == 2
    assert report.dropped == 1
    assert report.dropped_by_reason == {"language_not_requested": 1}
    assert sorted(_kept_titles(output_path)) == ["English", "French"]


def test_languages_none_keeps_every_supported_language(tmp_path: Path) -> None:
    records = [
        _eligible("E", language="eng", control_001="a"),
        _eligible("F", language="fre", control_001="b"),
        _eligible("G", language="ger", control_001="c"),
        _eligible("S", language="spa", control_001="d"),
        _eligible("I", language="ita", control_001="e"),
    ]
    input_path = _write_input(tmp_path, records)
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.kept == 5
    assert report.dropped_by_reason == {}


def test_min_year_default_is_the_moving_wall(tmp_path: Path) -> None:
    from pd_groundtruth.acquire import default_min_year

    assert default_min_year() == date.today().year - 95


def test_report_counts_sum_consistently(tmp_path: Path) -> None:
    records = [
        _eligible("Keep me", control_001="a"),
        _record(title="Serial", leader="00000nas a2200000 a 4500"),
        _eligible("Too new", year=1990, control_001="b"),
    ]
    input_path = _write_input(tmp_path, records)
    output_path = tmp_path / "out.marcxml"

    report = filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert report.scanned == 3
    assert report.kept == 1
    assert report.dropped == 2
    assert sum(report.dropped_by_reason.values()) == report.dropped


def test_output_round_trips_through_match_reader(tmp_path: Path) -> None:
    records = [
        _eligible("Round trip one", control_001="rt1"),
        _eligible("Round trip two", language="fre", control_001="rt2"),
        _eligible("Latin dropped", language="lat", control_001="rt3"),
    ]
    input_path = _write_input(tmp_path, records)
    output_path = tmp_path / "out.marcxml"

    filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    parsed = list(iter_marc_records(output_path))
    titles = sorted(record.title for record in parsed)
    assert titles == ["Round trip one", "Round trip two"]
    assert {record.control_id for record in parsed} == {"rt1", "rt2"}


def test_output_directory_is_created(tmp_path: Path) -> None:
    input_path = _write_input(tmp_path, [_eligible("Only")])
    output_path = tmp_path / "nested" / "dir" / "out.marcxml"

    filter_marcxml(input_path=input_path, output_path=output_path, min_year=_MIN_YEAR)

    assert output_path.exists()
    assert _kept_titles(output_path) == ["Only"]
