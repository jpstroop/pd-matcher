"""Tests for :mod:`pd_matcher.parsers.marc`."""

from pathlib import Path

from pd_matcher.parsers.marc import MarcParseStats
from pd_matcher.parsers.marc import iter_marc_records

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tiny.marcxml"

_MARC_NS = "http://www.loc.gov/MARC21/slim"


def _write_xml(path: Path, body: str) -> None:
    path.write_text(
        f"<collection xmlns='{_MARC_NS}'>{body}</collection>",
        encoding="utf-8",
    )


def _record(control_id: str, datafields: str = "") -> str:
    return (
        "<record>"
        f"<controlfield tag='001'>{control_id}</controlfield>"
        "<controlfield tag='008'>200718s1953    nyu           000 0 eng  </controlfield>"
        "<datafield ind1='0' ind2='0' tag='245'>"
        "<subfield code='a'>A Title</subfield>"
        "</datafield>"
        f"{datafields}"
        "</record>"
    )


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


def test_iter_marc_records_keeps_title_main_separate_from_fused_title() -> None:
    """245$a is preserved as ``title_main`` even when $b fuses into ``title``."""
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    first = by_id["marc-001"]
    assert first.title == "A study of widgets and other small parts"
    assert first.title_main == "A study of widgets"
    assert first.title_part_number is None
    assert first.title_part_name is None


def test_iter_marc_records_extracts_245_part_number_and_name() -> None:
    """245$n and $p are surfaced as ``title_part_number`` / ``title_part_name``."""
    records = list(iter_marc_records(FIXTURE))
    by_id = {r.control_id: r for r in records}
    parts = by_id["marc-014-title-parts"]
    assert parts.title == "Collected works essays and letters"
    assert parts.title_main == "Collected works"
    assert parts.title_part_number == "Part 2"
    assert parts.title_part_name == "The later years"


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


def test_iter_marc_records_extracts_oclc_from_035_with_ocolc_prefix(tmp_path: Path) -> None:
    path = tmp_path / "oclc.xml"
    _write_xml(
        path,
        _record(
            "oclc-1",
            "<datafield ind1=' ' ind2=' ' tag='035'>"
            "<subfield code='a'>(OCoLC)00012345</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].oclc == "00012345"


def test_iter_marc_records_oclc_absent_when_no_035(tmp_path: Path) -> None:
    path = tmp_path / "no_oclc.xml"
    _write_xml(path, _record("no-oclc"))
    records = list(iter_marc_records(path))
    assert records[0].oclc is None


def test_iter_marc_records_ignores_non_ocolc_035_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "non_ocolc.xml"
    _write_xml(
        path,
        _record(
            "non-ocolc",
            "<datafield ind1=' ' ind2=' ' tag='035'>"
            "<subfield code='a'>(DLC)sn 90123456</subfield>"
            "</datafield>"
            "<datafield ind1=' ' ind2=' ' tag='035'>"
            "<subfield code='a'>(PrU)9988776655</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].oclc is None


def test_iter_marc_records_skips_empty_ocolc_value_and_uses_next(tmp_path: Path) -> None:
    path = tmp_path / "empty_then_real.xml"
    _write_xml(
        path,
        _record(
            "empty-then-real",
            "<datafield ind1=' ' ind2=' ' tag='035'>"
            "<subfield code='a'>(OCoLC)   </subfield>"
            "<subfield code='a'>(OCoLC)99887766</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].oclc == "99887766"


def test_iter_marc_records_keeps_first_ocolc_when_multiple_present(tmp_path: Path) -> None:
    path = tmp_path / "multi.xml"
    _write_xml(
        path,
        _record(
            "multi",
            "<datafield ind1=' ' ind2=' ' tag='035'>"
            "<subfield code='a'>(OCoLC)11111111</subfield>"
            "</datafield>"
            "<datafield ind1=' ' ind2=' ' tag='035'>"
            "<subfield code='a'>(OCoLC)22222222</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].oclc == "11111111"


def test_iter_marc_records_repairs_mojibake_in_subfields_and_increments_counter() -> None:
    stats = MarcParseStats()
    records = list(iter_marc_records(FIXTURE, stats=stats))
    by_id = {r.control_id: r for r in records}
    mojibake = by_id["marc-013-mojibake"]
    assert mojibake.title == "Histoire de la folie à l'âge classique"
    assert stats.mojibake_fixed_count >= 1


def test_iter_marc_records_extracts_246_ind2_4_with_a_and_b(tmp_path: Path) -> None:
    """246 ind2=4 (cover title) with $a + $b is fused into a single variant."""
    path = tmp_path / "246_a_b.xml"
    _write_xml(
        path,
        _record(
            "246-ind2-4",
            "<datafield ind1='1' ind2='4' tag='246'>"
            "<subfield code='a'>Cover form of the title</subfield>"
            "<subfield code='b'>and its subtitle</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == ("Cover form of the title and its subtitle",)


def test_iter_marc_records_extracts_246_a_only_when_b_absent(tmp_path: Path) -> None:
    """246 ind2=0 with only $a is captured without a $b suffix."""
    path = tmp_path / "246_a_only.xml"
    _write_xml(
        path,
        _record(
            "246-a-only",
            "<datafield ind1='1' ind2='0' tag='246'>"
            "<subfield code='a'>Portion of title</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == ("Portion of title",)


def test_iter_marc_records_excludes_246_ind2_1_parallel_title(tmp_path: Path) -> None:
    """246 ind2=1 (parallel title / language variant) is excluded by design."""
    path = tmp_path / "246_parallel.xml"
    _write_xml(
        path,
        _record(
            "246-parallel",
            "<datafield ind1='1' ind2='1' tag='246'>"
            "<subfield code='a'>Le titre parallele</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == ()


def test_iter_marc_records_excludes_246_blank_ind2(tmp_path: Path) -> None:
    """246 with a blank ind2 (cataloger noise) is excluded by design."""
    path = tmp_path / "246_blank.xml"
    _write_xml(
        path,
        _record(
            "246-blank",
            "<datafield ind1='1' ind2=' ' tag='246'>"
            "<subfield code='a'>Unspecified variant</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == ()


def test_iter_marc_records_skips_246_without_a_subfield(tmp_path: Path) -> None:
    """A 246 with a CCE-likely ind2 but no $a subfield is silently skipped."""
    path = tmp_path / "246_no_a.xml"
    _write_xml(
        path,
        _record(
            "246-no-a",
            "<datafield ind1='1' ind2='3' tag='246'>"
            "<subfield code='i'>Other title:</subfield>"
            "<subfield code='b'>orphan subtitle</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == ()


def test_iter_marc_records_skips_246_when_a_cleans_to_empty(tmp_path: Path) -> None:
    """A whitespace-only $a (cleans to None) yields no variant for that 246."""
    path = tmp_path / "246_empty_a.xml"
    _write_xml(
        path,
        _record(
            "246-empty-a",
            "<datafield ind1='1' ind2='3' tag='246'><subfield code='a'>   </subfield></datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == ()


def test_iter_marc_records_caps_246_variants_at_three(tmp_path: Path) -> None:
    """Five CCE-likely 246s on one record yield exactly three variants, in order."""
    path = tmp_path / "246_many.xml"
    _write_xml(
        path,
        _record(
            "246-many",
            "<datafield ind1='1' ind2='0' tag='246'>"
            "<subfield code='a'>Variant one</subfield>"
            "</datafield>"
            "<datafield ind1='1' ind2='2' tag='246'>"
            "<subfield code='a'>Variant two</subfield>"
            "</datafield>"
            "<datafield ind1='1' ind2='3' tag='246'>"
            "<subfield code='a'>Variant three</subfield>"
            "</datafield>"
            "<datafield ind1='1' ind2='4' tag='246'>"
            "<subfield code='a'>Variant four</subfield>"
            "</datafield>"
            "<datafield ind1='1' ind2='7' tag='246'>"
            "<subfield code='a'>Variant five</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == (
        "Variant one",
        "Variant two",
        "Variant three",
    )


def test_iter_marc_records_246_display_text_subfield_i_is_ignored(tmp_path: Path) -> None:
    """A 246 $i display-text prefix does not contaminate the variant."""
    path = tmp_path / "246_with_i.xml"
    _write_xml(
        path,
        _record(
            "246-with-i",
            "<datafield ind1='1' ind2='3' tag='246'>"
            "<subfield code='i'>Title on spine:</subfield>"
            "<subfield code='a'>Spine title only</subfield>"
            "</datafield>",
        ),
    )
    records = list(iter_marc_records(path))
    assert records[0].title_variants == ("Spine title only",)


def test_iter_marc_records_246_walks_all_ind2_subset_values(tmp_path: Path) -> None:
    """Each of {0, 2, 3, 4, 7, 8} is recognized as a CCE-likely ind2."""
    fields = "".join(
        "<datafield ind1='1' ind2='" + value + "' tag='246'>"
        "<subfield code='a'>variant " + value + "</subfield>"
        "</datafield>"
        for value in ("0", "2", "3", "4", "7", "8")
    )
    path = tmp_path / "246_all_ind2.xml"
    _write_xml(path, _record("246-all-ind2", fields))
    records = list(iter_marc_records(path))
    # Cap at 3 — confirms only the first three (in document order) are kept.
    assert records[0].title_variants == (
        "variant 0",
        "variant 2",
        "variant 3",
    )
