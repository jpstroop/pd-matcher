"""Tests for :mod:`pd_matcher.parsers.nypl_reg`."""

from datetime import date
from pathlib import Path

from pd_matcher.parsers.nypl_reg import NyplRegParseStats
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_records

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_reg.xml"


def test_iter_nypl_reg_records_parses_full_entry() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    first = records["UUID-0001"]
    assert first.title == "A study of widgets."
    assert first.regnum == "A111111"
    assert first.reg_date == date(1940, 5, 10)
    assert first.reg_year == 1940
    assert first.author_name == "Smith, John"
    assert first.edition == "1st ed."
    assert first.publisher_names == ("Acme Press",)
    assert first.publication_places == ("New York",)
    assert first.claimants == ("Acme Press",)


def test_iter_nypl_reg_records_handles_multiple_publishers_and_claimants() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    second = records["UUID-0002"]
    assert second.publisher_names == ("Editions Beta", "Distributor Inc.")
    assert second.publication_places == ("Paris", "London")
    assert second.claimants == ("Editions Beta", "Estate of Dubois")


def test_iter_nypl_reg_records_emits_record_without_author() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    third = records["UUID-0003"]
    assert third.author_name is None
    assert third.publisher_names == ()


def test_iter_nypl_reg_records_extracts_year_from_text_when_no_date_attr() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    fourth = records["UUID-0004"]
    assert fourth.regnum is None
    assert fourth.reg_date is None
    assert fourth.reg_year == 1965


def test_iter_nypl_reg_records_handles_missing_regdate() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    fifth = records["UUID-0005"]
    assert fifth.reg_date is None
    assert fifth.reg_year is None


def test_iter_nypl_reg_records_skips_entries_with_blank_id_or_title() -> None:
    records = list(iter_nypl_reg_records(FIXTURE))
    uuids = {r.uuid for r in records}
    assert "" not in uuids
    assert "UUID-0007" not in uuids


def test_iter_nypl_reg_records_handles_invalid_date_attribute() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    eighth = records["UUID-0008"]
    assert eighth.reg_date is None
    assert eighth.reg_year is None


def test_iter_nypl_reg_records_handles_empty_reg_date_text() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    ninth = records["UUID-0009"]
    assert ninth.reg_date is None
    assert ninth.reg_year is None


def test_iter_nypl_reg_records_ignores_empty_publisher_and_place_elements() -> None:
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE)}
    tenth = records["UUID-0010"]
    assert tenth.publisher_names == ("Real Name",)
    assert tenth.publication_places == ("Real Place",)
    assert tenth.claimants == ()


def test_iter_nypl_reg_directory_walks_nested_xml(tmp_path: Path) -> None:
    year_dir = tmp_path / "1940"
    year_dir.mkdir()
    (year_dir / "first.xml").write_bytes(FIXTURE.read_bytes())
    (year_dir / "second.xml").write_bytes(FIXTURE.read_bytes())
    records = list(iter_nypl_reg_directory(tmp_path))
    # Fixture has 11 entries; 2 (blank id, whitespace title) are skipped, so 9 per copy.
    assert len(records) == 18


def test_iter_nypl_reg_directory_handles_empty_root(tmp_path: Path) -> None:
    assert list(iter_nypl_reg_directory(tmp_path)) == []


def test_iter_nypl_reg_records_repairs_mojibake_in_title_and_increments_counter() -> None:
    stats = NyplRegParseStats()
    records = {r.uuid: r for r in iter_nypl_reg_records(FIXTURE, stats=stats)}
    eleventh = records["UUID-0011"]
    assert eleventh.title == "Histoire de la folie à l'âge classique"
    assert stats.mojibake_fixed_count >= 1
    assert stats.emitted == len(records)


def test_iter_nypl_reg_directory_accepts_shared_stats(tmp_path: Path) -> None:
    (tmp_path / "a.xml").write_bytes(FIXTURE.read_bytes())
    (tmp_path / "b.xml").write_bytes(FIXTURE.read_bytes())
    stats = NyplRegParseStats()
    records = list(iter_nypl_reg_directory(tmp_path, stats=stats))
    assert stats.emitted == len(records)
    assert stats.mojibake_fixed_count >= 2
