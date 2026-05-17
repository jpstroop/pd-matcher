"""Tests for :mod:`pd_matcher.parsers.nypl_ren`."""

from datetime import date
from pathlib import Path

from pytest import raises

from pd_matcher.parsers.nypl_ren import NyplRenHeaderError
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_records

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_ren.tsv"


def test_iter_nypl_ren_records_parses_complete_row() -> None:
    records = {r.id: r for r in iter_nypl_ren_records(FIXTURE)}
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


def test_iter_nypl_ren_records_handles_blank_author_title() -> None:
    records = {r.id: r for r in iter_nypl_ren_records(FIXTURE)}
    third = records["R200003"]
    assert third.author is None
    assert third.title is None
    assert third.odat is None


def test_iter_nypl_ren_records_tolerates_invalid_dates() -> None:
    records = {r.id: r for r in iter_nypl_ren_records(FIXTURE)}
    fourth = records["R200004"]
    assert fourth.odat is None
    assert fourth.rdat == date(1990, 6, 1)


def test_iter_nypl_ren_records_skips_rows_missing_id_or_entry_id() -> None:
    ids = {r.id for r in iter_nypl_ren_records(FIXTURE)}
    assert "" not in ids
    assert len(ids) == 4


def test_iter_nypl_ren_records_returns_empty_for_header_only_file(tmp_path: Path) -> None:
    only_header = tmp_path / "empty.tsv"
    only_header.write_text(FIXTURE.read_text().splitlines()[0] + "\n", encoding="utf-8")
    assert list(iter_nypl_ren_records(only_header)) == []


def test_iter_nypl_ren_records_returns_empty_for_completely_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "nothing.tsv"
    empty.write_text("", encoding="utf-8")
    assert list(iter_nypl_ren_records(empty)) == []


def test_iter_nypl_ren_records_skips_short_rows(tmp_path: Path) -> None:
    bad = tmp_path / "short.tsv"
    header = FIXTURE.read_text().splitlines()[0]
    bad.write_text(header + "\nincomplete\trow\n", encoding="utf-8")
    assert list(iter_nypl_ren_records(bad)) == []


def test_iter_nypl_ren_records_raises_on_unexpected_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad_header.tsv"
    bad.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
    with raises(NyplRenHeaderError, match="Unexpected NYPL renewal header"):
        list(iter_nypl_ren_records(bad))


def test_iter_nypl_ren_directory_walks_tsv_files(tmp_path: Path) -> None:
    (tmp_path / "a.tsv").write_bytes(FIXTURE.read_bytes())
    (tmp_path / "b.tsv").write_bytes(FIXTURE.read_bytes())
    records = list(iter_nypl_ren_directory(tmp_path))
    assert len(records) == 8


def test_iter_nypl_ren_directory_handles_empty_root(tmp_path: Path) -> None:
    assert list(iter_nypl_ren_directory(tmp_path)) == []
