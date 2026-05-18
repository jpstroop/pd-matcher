"""Tests for :mod:`pd_matcher.parsers.marc`."""

from pathlib import Path

from pd_matcher.parsers.marc import MarcParseStats
from pd_matcher.parsers.marc import iter_marc_records

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tiny.marcxml"


def test_iter_marc_records_returns_expected_first_record() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    first = by_id["marc-001"]
    assert first.title == "A study of widgets and other small parts"
    assert first.statement_of_responsibility == "by Alice Alpha"
    assert first.lccn == "40012345"
    assert first.isbns == ("9780000000000", "0000000001")
    assert first.main_author == "Alpha, Alice"
    assert first.added_authors == ("Bravo, Bob", "Charlie, Carol")
    assert first.edition == "First edition"
    assert first.publication_place == "New York"
    assert first.publisher == "Acme Press"
    assert first.publication_date_raw == "1940"
    assert first.publication_year == 1940
    assert first.extent == "200 pages"
    assert first.series_titles == ("Series One",)
    assert first.language_code == "eng"
    assert first.country_code == "nyu"


def test_iter_marc_records_handles_264_field() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    second = by_id["marc-002"]
    assert second.publisher == "Editions Beta"
    assert second.publication_place == "Paris"
    assert second.publication_year == 1955
    assert second.series_titles == ("Petite serie",)
    assert second.language_code == "fre"


def test_iter_marc_records_handles_corporate_author_and_short_008() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    third = by_id["marc-003"]
    assert third.main_author == "Some Corporate Body"
    assert third.publication_year is None
    assert third.language_code is None
    assert third.country_code is None
    assert third.series_titles == ("Reports series",)


def test_iter_marc_records_uses_008_year_fallback_for_meeting_record() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    sixth = by_id["marc-006"]
    assert sixth.title == "Proceedings"
    assert sixth.publication_year == 1968
    assert sixth.main_author == "Some Meeting Name"


def test_iter_marc_records_falls_back_to_008_when_260c_lacks_year() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    seventh = by_id["marc-007-260c-no-year"]
    assert seventh.publication_year == 1959
    assert seventh.publication_date_raw == "n.d"


def test_iter_marc_records_rejects_out_of_range_years_in_both_sources() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    eighth = by_id["marc-008-260c-out-of-range"]
    assert eighth.publication_year is None


def test_iter_marc_records_rejects_out_of_range_008_year() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    ninth = by_id["marc-009-008-year-out-of-range"]
    assert ninth.publication_year is None


def test_iter_marc_records_handles_non_numeric_008_year() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    tenth = by_id["marc-010-008-letters"]
    assert tenth.publication_year is None


def test_iter_marc_records_skips_records_missing_required_fields() -> None:
    stats = MarcParseStats()
    records = list(iter_marc_records(FIXTURE, stats=stats))
    ids = {r.control_id for r in records}
    assert "marc-004-no-245" not in ids
    # Two skips for missing 001 (no controlfield + whitespace-only),
    # two skips for missing 245$a (marc-004 and marc-012).
    assert stats.skipped_missing_245a == 2
    assert stats.skipped_missing_001 == 2
    assert stats.emitted == len(records)


def test_iter_marc_records_without_stats_object_creates_internal_counter() -> None:
    records = list(iter_marc_records(FIXTURE))
    assert isinstance(records, list)
    assert records


def test_iter_marc_records_ignores_repeated_single_value_fields() -> None:
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    first = by_id["marc-001"]
    # Second 010/100/245/250/260/300 instances must be ignored.
    assert first.lccn == "40012345"
    assert first.main_author == "Alpha, Alice"
    assert first.publisher == "Acme Press"
