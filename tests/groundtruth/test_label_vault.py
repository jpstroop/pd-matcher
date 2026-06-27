"""Unit tests for the JSONL label vault."""

from datetime import date
from pathlib import Path

from msgspec import DecodeError
from pytest import raises

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import CceFacts
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import MatcherScores
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import cce_facts
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import extract_marc_identifiers
from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.label_vault import renewal_year_of
from pd_groundtruth.label_vault import upsert_entry
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


def _entry(
    *,
    marc_control_id: str = "ctrl-1",
    nypl_uuid: str = "uuid-1",
    verdict: str = "match",
    note: str | None = None,
    labeled_at: str = "2026-05-22T12:00:00+00:00",
    labeler: str = "jpstroop",
    lccn: str | None = "40012345",
    oclc: str | None = "00012345",
    isbns: tuple[str, ...] = ("9780000000000",),
    cce_regnum: str | None = "A12345",
    cce_renewal_id: str | None = "R67890",
    cce_renewal_oreg: str | None = "A12345",
) -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=marc_control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        note=note,
        labeled_at=labeled_at,
        labeler=labeler,
        marc_identifiers=MarcIdentifiers(lccn=lccn, oclc=oclc, isbns=isbns),
        cce_regnum=cce_regnum,
        cce_renewal_id=cce_renewal_id,
        cce_renewal_oreg=cce_renewal_oreg,
    )


def test_iter_entries_returns_nothing_for_missing_file(tmp_path: Path) -> None:
    assert list(iter_entries(tmp_path / "missing.jsonl")) == []


def test_current_entries_returns_empty_dict_for_missing_file(tmp_path: Path) -> None:
    assert current_entries(tmp_path / "missing.jsonl") == {}


def test_upsert_creates_parent_dir_and_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "vault.jsonl"
    upsert_entry(path, _entry())
    assert path.exists()
    assert path.parent.is_dir()


def test_round_trip_single_entry(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    entry = _entry(verdict="no_match", note="title collision")
    upsert_entry(path, entry)
    [read_back] = list(iter_entries(path))
    assert read_back == entry


def test_upsert_writes_one_line_per_distinct_pair_with_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    upsert_entry(path, _entry(marc_control_id="a", nypl_uuid="u-a"))
    upsert_entry(path, _entry(marc_control_id="b", nypl_uuid="u-b"))
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2


def test_iter_entries_skips_empty_trailing_lines(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    upsert_entry(path, _entry())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n   \n")
    assert len(list(iter_entries(path))) == 1


def test_iter_entries_raises_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with raises(DecodeError):
        list(iter_entries(path))


def test_iter_entries_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    path.write_text(
        '{"schema":4,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
        '"note":null,"labeled_at":"2026-01-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"extra":"surprise"}\n',
        encoding="utf-8",
    )
    with raises(Exception, match="extra"):
        list(iter_entries(path))


def test_current_entries_returns_latest_per_key(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    first = _entry(verdict="match", labeled_at="2026-05-01T00:00:00+00:00")
    second = _entry(verdict="no_match", labeled_at="2026-05-02T00:00:00+00:00")
    third = _entry(verdict="unsure", labeled_at="2026-05-03T00:00:00+00:00")
    upsert_entry(path, first)
    upsert_entry(path, second)
    upsert_entry(path, third)
    latest = current_entries(path)
    assert latest[("ctrl-1", "uuid-1")] == third


def test_current_entries_keeps_distinct_keys_separate(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    a = _entry(marc_control_id="a", nypl_uuid="u-a", verdict="match")
    b = _entry(marc_control_id="b", nypl_uuid="u-b", verdict="no_match")
    upsert_entry(path, a)
    upsert_entry(path, b)
    latest = current_entries(path)
    assert latest[("a", "u-a")] == a
    assert latest[("b", "u-b")] == b


def test_upsert_replaces_existing_entry_for_same_pair(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    upsert_entry(path, _entry(verdict="match", labeled_at="2026-05-01T00:00:00+00:00"))
    upsert_entry(path, _entry(verdict="no_match", labeled_at="2026-05-02T00:00:00+00:00"))
    entries = list(iter_entries(path))
    assert len(entries) == 1
    assert entries[0].verdict == "no_match"
    assert entries[0].labeled_at == "2026-05-02T00:00:00+00:00"


def test_upsert_same_verdict_updates_timestamp_in_place(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    upsert_entry(path, _entry(verdict="match", note=None, labeled_at="2026-05-01T00:00:00+00:00"))
    upsert_entry(path, _entry(verdict="match", note=None, labeled_at="2026-05-02T00:00:00+00:00"))
    entries = list(iter_entries(path))
    assert len(entries) == 1
    assert entries[0].verdict == "match"
    assert entries[0].labeled_at == "2026-05-02T00:00:00+00:00"


def test_upsert_same_verdict_updates_note_when_changed(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    upsert_entry(path, _entry(verdict="match", note=None, labeled_at="2026-05-01T00:00:00+00:00"))
    upsert_entry(
        path,
        _entry(verdict="match", note="OCR glitch caught", labeled_at="2026-05-02T00:00:00+00:00"),
    )
    entries = list(iter_entries(path))
    assert len(entries) == 1
    assert entries[0].note == "OCR glitch caught"
    assert entries[0].labeled_at == "2026-05-02T00:00:00+00:00"


def test_upsert_preserves_first_seen_order_across_distinct_pairs(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    upsert_entry(path, _entry(marc_control_id="a", nypl_uuid="u-a"))
    upsert_entry(path, _entry(marc_control_id="b", nypl_uuid="u-b"))
    upsert_entry(path, _entry(marc_control_id="c", nypl_uuid="u-c"))
    upsert_entry(
        path,
        _entry(marc_control_id="a", nypl_uuid="u-a", verdict="no_match"),
    )
    keys = [(e.marc_control_id, e.nypl_uuid) for e in iter_entries(path)]
    assert keys == [("a", "u-a"), ("b", "u-b"), ("c", "u-c")]


def test_upsert_cleans_up_tmp_file_on_success(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    upsert_entry(path, _entry())
    tmp_path_artifact = path.with_name(path.name + ".tmp")
    assert not tmp_path_artifact.exists()


def test_extract_marc_identifiers_pulls_lccn_oclc_isbns() -> None:
    marc = MarcRecord(
        control_id="ctrl-1",
        title="Title",
        title_main="Title",
        lccn=" 40012345 ",
        oclc="00012345",
        isbns=("9780000000000", "0000000001"),
    )
    identifiers = extract_marc_identifiers(marc)
    assert identifiers.lccn == " 40012345 "
    assert identifiers.oclc == "00012345"
    assert identifiers.isbns == ("9780000000000", "0000000001")


def test_extract_marc_identifiers_handles_missing_identifiers() -> None:
    marc = MarcRecord(
        control_id="ctrl-1",
        title="Title",
        title_main="Title",
    )
    identifiers = extract_marc_identifiers(marc)
    assert identifiers.lccn is None
    assert identifiers.oclc is None
    assert identifiers.isbns == ()


def test_schema_version_is_seven() -> None:
    """New vault writes use schema 7 (adds the ``match_source`` pathway field)."""
    assert SCHEMA_VERSION == 7


def test_legacy_schema_entries_with_old_fields_reject_decode(tmp_path: Path) -> None:
    """Schema-1/2 lines carry retired fields; the new VaultEntry refuses them.

    Migration via :func:`migrate_vault_v3` is the supported path forward.
    """
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        '{"schema":2,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
        '"reasons":[],"note":null,"labeled_at":"2026-01-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"field_annotations":[]}\n',
        encoding="utf-8",
    )
    with raises(Exception, match=r"reasons|field_annotations"):
        list(iter_entries(path))


def test_round_trip_preserves_cce_identifier_fields(tmp_path: Path) -> None:
    """All three new CCE-side identifier fields survive encode/decode intact."""
    path = tmp_path / "vault.jsonl"
    entry = _entry(
        cce_regnum="A555",
        cce_renewal_id="R999",
        cce_renewal_oreg="A555",
    )
    upsert_entry(path, entry)
    [read_back] = list(iter_entries(path))
    assert read_back.cce_regnum == "A555"
    assert read_back.cce_renewal_id == "R999"
    assert read_back.cce_renewal_oreg == "A555"
    assert read_back == entry


def test_round_trip_preserves_nulls_for_unrenewed_registration(tmp_path: Path) -> None:
    """A reg with no renewal serializes ``cce_renewal_id``/``cce_renewal_oreg`` as null."""
    path = tmp_path / "vault.jsonl"
    entry = _entry(
        cce_regnum="A111",
        cce_renewal_id=None,
        cce_renewal_oreg=None,
    )
    upsert_entry(path, entry)
    [read_back] = list(iter_entries(path))
    assert read_back.cce_regnum == "A111"
    assert read_back.cce_renewal_id is None
    assert read_back.cce_renewal_oreg is None


def test_schema_3_entry_decodes_with_none_for_new_cce_fields(tmp_path: Path) -> None:
    """Forward-compat: a schema-3 line (no CCE fields) decodes with ``None`` defaults."""
    path = tmp_path / "vault.jsonl"
    path.write_text(
        '{"schema":3,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
        '"note":null,"labeled_at":"2026-01-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]}}\n',
        encoding="utf-8",
    )
    [entry] = list(iter_entries(path))
    assert entry.cce_regnum is None
    assert entry.cce_renewal_id is None
    assert entry.cce_renewal_oreg is None
    assert entry.schema == 3


def test_default_categories_is_empty_tuple() -> None:
    """A freshly constructed ``VaultEntry`` has an empty categories tuple."""
    entry = _entry()
    assert entry.categories == ()


def test_renewal_year_of_returns_year_when_date_present() -> None:
    assert renewal_year_of(date(1981, 4, 1)) == 1981


def test_renewal_year_of_returns_none_when_date_absent() -> None:
    assert renewal_year_of(None) is None


def _cce(
    *,
    reg_year: int | None = 1953,
    was_renewed: bool = True,
    renewal_rdat: date | None = date(1981, 4, 1),
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="uuid-1",
        title="A Studied Title",
        was_renewed=was_renewed,
        reg_year=reg_year,
        renewal_rdat=renewal_rdat,
    )


def test_cce_facts_projects_all_three_static_facts() -> None:
    facts = cce_facts(_cce())
    assert facts == CceFacts(reg_year=1953, renewal_year=1981, was_renewed=True)


def test_cce_facts_leaves_renewal_year_none_when_not_renewed() -> None:
    facts = cce_facts(_cce(was_renewed=False, renewal_rdat=None))
    assert facts.renewal_year is None
    assert facts.was_renewed is False
    assert facts.reg_year == 1953


def test_cce_facts_passes_through_missing_reg_year() -> None:
    facts = cce_facts(_cce(reg_year=None))
    assert facts.reg_year is None


def test_round_trip_preserves_categories(tmp_path: Path) -> None:
    """Categories survive encode/decode through the vault file."""
    path = tmp_path / "vault.jsonl"
    entry = VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id="ctrl-1",
        nypl_uuid="uuid-1",
        verdict="no_match",
        note=None,
        labeled_at="2026-06-01T00:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
        categories=("marc_whole_cce_part", "generic_title"),
    )
    upsert_entry(path, entry)
    [read_back] = list(iter_entries(path))
    assert read_back.categories == ("marc_whole_cce_part", "generic_title")


def test_unknown_category_key_raises_validation_error(tmp_path: Path) -> None:
    """msgspec rejects category keys outside the ``CategoryKey`` Literal."""
    path = tmp_path / "vault.jsonl"
    path.write_text(
        '{"schema":5,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
        '"note":null,"labeled_at":"2026-06-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"cce_regnum":null,"cce_renewal_id":null,"cce_renewal_oreg":null,'
        '"categories":["not_a_real_category"]}\n',
        encoding="utf-8",
    )
    with raises(Exception, match="categories"):
        list(iter_entries(path))


def test_schema_4_entry_decodes_with_empty_categories(tmp_path: Path) -> None:
    """Forward-compat: schema-4 lines without ``categories`` decode with ``()``."""
    path = tmp_path / "vault.jsonl"
    path.write_text(
        '{"schema":4,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
        '"note":null,"labeled_at":"2026-05-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"cce_regnum":"A1","cce_renewal_id":null,"cce_renewal_oreg":null}\n',
        encoding="utf-8",
    )
    [entry] = list(iter_entries(path))
    assert entry.schema == 4
    assert entry.categories == ()


def test_matcher_scores_default_to_none() -> None:
    """A freshly constructed ``MatcherScores`` has both confidences unset."""
    scores = MatcherScores()
    assert scores.weighted_mean is None
    assert scores.learned is None


def test_default_enrichment_fields_are_none() -> None:
    """A freshly constructed schema-6 entry has all derived fields ``None``."""
    entry = _entry()
    assert entry.reg_year is None
    assert entry.renewal_year is None
    assert entry.was_renewed is None
    assert entry.scores is None
    assert entry.matcher_version is None


def test_round_trip_preserves_enrichment_fields(tmp_path: Path) -> None:
    """The schema-6 derived fields survive encode/decode through the vault file."""
    path = tmp_path / "vault.jsonl"
    entry = VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id="ctrl-1",
        nypl_uuid="uuid-1",
        verdict="match",
        note=None,
        labeled_at="2026-06-20T00:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
        reg_year=1953,
        renewal_year=1981,
        was_renewed=True,
        scores=MatcherScores(weighted_mean=0.8421, learned=0.9133),
        matcher_version="abc1234-dirty",
    )
    upsert_entry(path, entry)
    [read_back] = list(iter_entries(path))
    assert read_back == entry
    assert read_back.scores is not None
    assert read_back.scores.weighted_mean == 0.8421
    assert read_back.scores.learned == 0.9133


def test_schema_5_entry_decodes_with_none_enrichment_fields(tmp_path: Path) -> None:
    """Forward-compat: a schema-5 line decodes with ``None`` enrichment defaults."""
    path = tmp_path / "vault.jsonl"
    path.write_text(
        '{"schema":5,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
        '"note":null,"labeled_at":"2026-06-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"cce_regnum":"A1","cce_renewal_id":null,"cce_renewal_oreg":null,"categories":[]}\n',
        encoding="utf-8",
    )
    [entry] = list(iter_entries(path))
    assert entry.schema == 5
    assert entry.reg_year is None
    assert entry.scores is None
    assert entry.matcher_version is None


def test_matcher_scores_rejects_unknown_field(tmp_path: Path) -> None:
    """msgspec rejects unexpected keys inside the nested ``scores`` struct."""
    path = tmp_path / "vault.jsonl"
    path.write_text(
        '{"schema":6,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
        '"note":null,"labeled_at":"2026-06-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"scores":{"weighted_mean":0.5,"bogus":1.0}}\n',
        encoding="utf-8",
    )
    with raises(Exception, match=r"bogus|scores"):
        list(iter_entries(path))


def test_match_source_defaults_to_none_and_round_trips(tmp_path: Path) -> None:
    """``match_source`` defaults to ``None`` and survives an upsert round-trip."""
    path = tmp_path / "vault.jsonl"
    default_entry = _entry(marc_control_id="a", nypl_uuid="u-a")
    assert default_entry.match_source is None
    upsert_entry(path, default_entry)
    upsert_entry(
        path,
        VaultEntry(
            schema=SCHEMA_VERSION,
            marc_control_id="b",
            nypl_uuid="u-b",
            verdict="match",
            note=None,
            labeled_at="2026-06-20T00:00:00+00:00",
            labeler="jpstroop",
            marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
            match_source="renewal",
        ),
    )
    entries = {entry.marc_control_id: entry for entry in iter_entries(path)}
    assert entries["a"].match_source is None
    assert entries["b"].match_source == "renewal"
