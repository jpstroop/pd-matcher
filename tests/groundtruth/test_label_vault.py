"""Unit tests for the JSONL label vault."""

from pathlib import Path

from msgspec import DecodeError
from pytest import raises

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import append_entry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import extract_marc_identifiers
from pd_groundtruth.label_vault import iter_entries
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
    )


def test_iter_entries_returns_nothing_for_missing_file(tmp_path: Path) -> None:
    assert list(iter_entries(tmp_path / "missing.jsonl")) == []


def test_current_entries_returns_empty_dict_for_missing_file(tmp_path: Path) -> None:
    assert current_entries(tmp_path / "missing.jsonl") == {}


def test_append_creates_parent_dir_and_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "vault.jsonl"
    append_entry(path, _entry())
    assert path.exists()
    assert path.parent.is_dir()


def test_round_trip_single_entry(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    entry = _entry(verdict="no_match", note="title collision")
    append_entry(path, entry)
    [read_back] = list(iter_entries(path))
    assert read_back == entry


def test_append_writes_one_line_per_entry_with_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    append_entry(path, _entry(marc_control_id="a", nypl_uuid="u-a"))
    append_entry(path, _entry(marc_control_id="b", nypl_uuid="u-b"))
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2


def test_iter_entries_skips_empty_trailing_lines(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    append_entry(path, _entry())
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
        '{"schema":3,"marc_control_id":"a","nypl_uuid":"u","verdict":"match",'
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
    append_entry(path, first)
    append_entry(path, second)
    append_entry(path, third)
    latest = current_entries(path)
    assert latest[("ctrl-1", "uuid-1")] == third


def test_current_entries_keeps_distinct_keys_separate(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    a = _entry(marc_control_id="a", nypl_uuid="u-a", verdict="match")
    b = _entry(marc_control_id="b", nypl_uuid="u-b", verdict="no_match")
    append_entry(path, a)
    append_entry(path, b)
    latest = current_entries(path)
    assert latest[("a", "u-a")] == a
    assert latest[("b", "u-b")] == b


def test_full_history_preserved_via_iter_entries(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    append_entry(path, _entry(verdict="match", labeled_at="2026-05-01T00:00:00+00:00"))
    append_entry(path, _entry(verdict="no_match", labeled_at="2026-05-02T00:00:00+00:00"))
    history = list(iter_entries(path))
    assert [event.verdict for event in history] == ["match", "no_match"]


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


def test_schema_version_is_three() -> None:
    """New vault writes use schema 3 (drops ``reasons``/``field_annotations``)."""
    assert SCHEMA_VERSION == 3


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
