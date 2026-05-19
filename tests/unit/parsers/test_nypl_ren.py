"""Tests for :mod:`pd_matcher.parsers.nypl_ren`."""

from datetime import date
from pathlib import Path

from pytest import mark
from pytest import raises

from pd_matcher.parsers.nypl_ren import NyplRenHeaderError
from pd_matcher.parsers.nypl_ren import NyplRenParseStats
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_records

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
FIXTURE_PRE_1978 = _FIXTURE_DIR / "tiny_ren.tsv"
FIXTURE_FROM_DB = _FIXTURE_DIR / "tiny_ren_from_db.tsv"

_BOTH_FIXTURES = [FIXTURE_PRE_1978, FIXTURE_FROM_DB]


@mark.parametrize("fixture", _BOTH_FIXTURES)
def test_iter_nypl_ren_records_parses_complete_row(fixture: Path) -> None:
    records = {r.id: r for r in iter_nypl_ren_records(fixture)}
    first = records["R200001"]
    assert first.entry_id == "entry-001"
    assert first.oreg == "A111111"
    assert first.odat == date(1940, 5, 10)
    assert first.rdat == date(1968, 5, 15)
    assert first.author == "Smith, John"
    assert first.title == "A study of widgets"
    assert first.claimants == "Acme Press|PWH"
    assert first.new_matter is None
    assert first.full_text == "Smith, John. A study of widgets. R200001"


@mark.parametrize("fixture", _BOTH_FIXTURES)
def test_iter_nypl_ren_records_handles_blank_author_title(fixture: Path) -> None:
    records = {r.id: r for r in iter_nypl_ren_records(fixture)}
    third = records["R200003"]
    assert third.author is None
    assert third.title is None
    assert third.odat is None


@mark.parametrize("fixture", _BOTH_FIXTURES)
def test_iter_nypl_ren_records_tolerates_invalid_dates(fixture: Path) -> None:
    records = {r.id: r for r in iter_nypl_ren_records(fixture)}
    fourth = records["R200004"]
    assert fourth.odat is None
    assert fourth.rdat == date(1990, 6, 1)


@mark.parametrize("fixture", _BOTH_FIXTURES)
def test_iter_nypl_ren_records_skips_rows_missing_id_or_entry_id(fixture: Path) -> None:
    ids = {r.id for r in iter_nypl_ren_records(fixture)}
    assert "" not in ids
    assert len(ids) == 4


@mark.parametrize("fixture", _BOTH_FIXTURES)
def test_iter_nypl_ren_records_returns_empty_for_header_only_file(
    fixture: Path, tmp_path: Path
) -> None:
    only_header = tmp_path / "empty.tsv"
    only_header.write_text(fixture.read_text().splitlines()[0] + "\n", encoding="utf-8")
    assert list(iter_nypl_ren_records(only_header)) == []


def test_iter_nypl_ren_records_returns_empty_for_completely_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "nothing.tsv"
    empty.write_text("", encoding="utf-8")
    assert list(iter_nypl_ren_records(empty)) == []


@mark.parametrize("fixture", _BOTH_FIXTURES)
def test_iter_nypl_ren_records_skips_short_rows(fixture: Path, tmp_path: Path) -> None:
    bad = tmp_path / "short.tsv"
    header = fixture.read_text().splitlines()[0]
    bad.write_text(header + "\nincomplete\trow\n", encoding="utf-8")
    assert list(iter_nypl_ren_records(bad)) == []


def test_iter_nypl_ren_records_raises_on_unexpected_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad_header.tsv"
    bad.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
    with raises(NyplRenHeaderError) as excinfo:
        list(iter_nypl_ren_records(bad))
    message = str(excinfo.value)
    assert "Unexpected CCE renewal header" in message
    assert "pre-1978" in message
    assert "from-db" in message
    assert "'author'" in message
    assert "'auth'" in message
    assert "'rdat'" in message
    assert "'dreg'" in message
    assert "'notes'" in message
    assert "'note'" in message


def test_iter_nypl_ren_directory_walks_tsv_files(tmp_path: Path) -> None:
    (tmp_path / "a.tsv").write_bytes(FIXTURE_PRE_1978.read_bytes())
    (tmp_path / "b.tsv").write_bytes(FIXTURE_PRE_1978.read_bytes())
    records = list(iter_nypl_ren_directory(tmp_path))
    assert len(records) == 8


def test_iter_nypl_ren_directory_mixes_schemas(tmp_path: Path) -> None:
    (tmp_path / "pre_1978.tsv").write_bytes(FIXTURE_PRE_1978.read_bytes())
    (tmp_path / "from_db.tsv").write_bytes(FIXTURE_FROM_DB.read_bytes())
    records = list(iter_nypl_ren_directory(tmp_path))
    assert len(records) == 8
    ids_per_file = [r.id for r in records]
    assert ids_per_file.count("R200001") == 2
    assert ids_per_file.count("R200002") == 2
    assert ids_per_file.count("R200003") == 2
    assert ids_per_file.count("R200004") == 2


def test_iter_nypl_ren_directory_handles_empty_root(tmp_path: Path) -> None:
    assert list(iter_nypl_ren_directory(tmp_path)) == []


_PRE_1978_HEADER = (
    "entry_id\tvolume\tpart\tnumber\tpage\tauthor\ttitle\toreg\todat\t"
    "id\trdat\tclaimants\tnew_matter\tsee_also_ren\tsee_also_reg\tnotes\tfull_text"
)


def _build_row(title: str | bytes) -> bytes:
    """Compose one TSV data row with ``title`` injected as the title cell."""
    title_bytes = title if isinstance(title, bytes) else title.encode("utf-8")
    return (
        b"entry-001\t4\t14A\t1\t1\tSmith\t"
        + title_bytes
        + b"\tA111111\t1940-05-10\tR200001\t1968-05-15\t\t\t\t\t\trow"
    )


def test_iter_nypl_ren_records_repairs_mojibake_in_text_mode(tmp_path: Path) -> None:
    fixture = tmp_path / "moji.tsv"
    fixture.write_bytes(_PRE_1978_HEADER.encode("utf-8") + b"\n" + _build_row("cafÃ©") + b"\n")
    stats = NyplRenParseStats()
    records = list(iter_nypl_ren_records(fixture, stats=stats))
    assert len(records) == 1
    assert records[0].title == "café"
    assert stats.mojibake_fixed_count >= 1
    assert stats.subfields_decoded_as_cp1255 == 0
    assert stats.subfields_decoded_with_replacement == 0


def test_iter_nypl_ren_records_falls_back_to_bytes_mode_for_cp1255_hebrew(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "hebrew_cp1255.tsv"
    title_cp1255 = "שלום".encode("cp1255")
    fixture.write_bytes(_PRE_1978_HEADER.encode("utf-8") + b"\n" + _build_row(title_cp1255) + b"\n")
    stats = NyplRenParseStats()
    records = list(iter_nypl_ren_records(fixture, stats=stats))
    assert len(records) == 1
    assert records[0].title == "שלום"
    assert stats.subfields_decoded_as_cp1255 >= 1
    assert stats.subfields_decoded_with_replacement == 0


def test_iter_nypl_ren_records_bytes_mode_counts_replacement_decodes(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "junk_bytes.tsv"
    fixture.write_bytes(
        _PRE_1978_HEADER.encode("utf-8") + b"\n" + _build_row(b"junk\x80\x99cell") + b"\n"
    )
    stats = NyplRenParseStats()
    records = list(iter_nypl_ren_records(fixture, stats=stats))
    assert len(records) == 1
    assert stats.subfields_decoded_with_replacement >= 1


def test_iter_nypl_ren_records_bytes_mode_skips_blank_lines(tmp_path: Path) -> None:
    fixture = tmp_path / "blank_line.tsv"
    fixture.write_bytes(
        _PRE_1978_HEADER.encode("utf-8") + b"\n\n" + _build_row("שלום".encode("cp1255")) + b"\n\n"
    )
    records = list(iter_nypl_ren_records(fixture))
    assert len(records) == 1


def test_iter_nypl_ren_records_bytes_mode_skips_short_data_rows(tmp_path: Path) -> None:
    fixture = tmp_path / "short.tsv"
    fixture.write_bytes(
        _PRE_1978_HEADER.encode("utf-8")
        + b"\n"
        + _build_row("שלום".encode("cp1255"))
        + b"\n"
        + b"too\tshort\trow\n"
    )
    records = list(iter_nypl_ren_records(fixture))
    assert len(records) == 1


def test_iter_nypl_ren_records_bytes_mode_raises_on_unrecognized_header(tmp_path: Path) -> None:
    fixture = tmp_path / "all_garbage.tsv"
    fixture.write_bytes(b"\xff")
    # _file_is_utf8 fails, so the bytes path takes over; the single garbage
    # line is treated as an (unrecognized) header and raises.
    with raises(NyplRenHeaderError):
        list(iter_nypl_ren_records(fixture))


def test_iter_nypl_ren_records_text_mode_accepts_caller_supplied_stats(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "clean.tsv"
    fixture.write_bytes(_PRE_1978_HEADER.encode("utf-8") + b"\n" + _build_row("Hello") + b"\n")
    stats = NyplRenParseStats()
    records = list(iter_nypl_ren_records(fixture, stats=stats))
    assert stats.emitted == len(records) == 1
    assert stats.mojibake_fixed_count == 0


def test_iter_nypl_ren_directory_accepts_shared_stats(tmp_path: Path) -> None:
    (tmp_path / "a.tsv").write_bytes(FIXTURE_PRE_1978.read_bytes())
    (tmp_path / "b.tsv").write_bytes(FIXTURE_PRE_1978.read_bytes())
    stats = NyplRenParseStats()
    records = list(iter_nypl_ren_directory(tmp_path, stats=stats))
    assert stats.emitted == len(records)
