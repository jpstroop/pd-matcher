"""Tests for :mod:`pd_matcher.parsers.nypl_reg`."""

from datetime import date
from pathlib import Path

from pd_matcher.models import NyplRegRecord
from pd_matcher.parsers.nypl_reg import NyplRegParseStats
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_records

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_reg.xml"


def _write_entry(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "entry.xml"
    path.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<copyrightEntries>{body}</copyrightEntries>',
        encoding="utf-8",
    )
    return path


def _only_record(tmp_path: Path, body: str) -> NyplRegRecord:
    records = list(iter_nypl_reg_records(_write_entry(tmp_path, body)))
    assert len(records) == 1
    return records[0]


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


def test_reg_year_from_reg_date_keeps_reg_date_set(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry id="R1"><title>Has regDate.</title>'
        '<regDate date="1940-05-10">May 10, 1940</regDate>'
        '<copyDate date="1939">1939</copyDate></copyrightEntry>',
    )
    assert record.reg_date == date(1940, 5, 10)
    assert record.reg_year == 1940


def test_reg_year_falls_back_to_copy_date_when_no_reg_date(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry regnum="AI1234" id="C1"><title>Ad interim.</title>'
        '<copyDate date="1948-02-03">Feb. 3, 1948</copyDate>'
        '<publisher><pubDate date="1947">1947</pubDate></publisher></copyrightEntry>',
    )
    assert record.reg_date is None
    assert record.reg_year == 1948


def test_reg_year_falls_back_to_pub_date_when_no_reg_or_copy_date(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry id="P1"><title>Only pubDate.</title>'
        "<publisher><pubName>Press</pubName>"
        '<pubDate date="1951-09-09">Sept. 9, 1951</pubDate></publisher></copyrightEntry>',
    )
    assert record.reg_date is None
    assert record.reg_year == 1951


def test_reg_year_prefers_direct_pub_date_over_publisher_level(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry id="P2"><title>Direct pubDate wins.</title>'
        '<pubDate date="1953">1953</pubDate>'
        '<publisher><pubDate date="1952">1952</pubDate></publisher></copyrightEntry>',
    )
    assert record.reg_year == 1953


def test_reg_year_ignores_aff_date(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry id="A1"><title>Only affDate.</title>'
        '<affDate date="1944-01-01">Jan. 1, 1944</affDate></copyrightEntry>',
    )
    assert record.reg_date is None
    assert record.reg_year is None


def test_reg_year_uses_text_when_date_attr_missing(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry id="T1"><title>copyDate text only.</title>'
        "<copyDate>circa 1958 issue</copyDate></copyrightEntry>",
    )
    assert record.reg_date is None
    assert record.reg_year == 1958


def test_reg_year_parses_year_only_date_attr_form(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry id="Y1"><title>Year-only attr.</title>'
        '<copyDate date="1961">1961</copyDate></copyrightEntry>',
    )
    assert record.reg_year == 1961


def test_reg_year_none_when_no_usable_date_anywhere(tmp_path: Path) -> None:
    record = _only_record(
        tmp_path,
        '<copyrightEntry id="N1"><title>No dates at all.</title>'
        "<publisher><pubName>Press</pubName></publisher></copyrightEntry>",
    )
    assert record.reg_date is None
    assert record.reg_year is None
