"""Unit tests for the label-vault migration commands."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from msgspec.json import decode as json_decode
from typer.testing import CliRunner

from pd_groundtruth.cli import app
from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.vault_migration import migrate_vault_v3
from pd_groundtruth.vault_migration import migrate_vault_v4
from pd_groundtruth.vault_migration import migrate_vault_v5
from pd_groundtruth.vault_migration import migrate_vault_v6
from pd_matcher.models import IndexedNyplRegRecord

_RUNNER = CliRunner()

_TARGET_V3_SCHEMA: int = 3


_SCHEMA_2_NO_EXTRA = (
    '{"schema":2,"marc_control_id":"a","nypl_uuid":"u-a","verdict":"match",'
    '"reasons":[],"note":null,"labeled_at":"2026-05-01T00:00:00+00:00",'
    '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
    '"field_annotations":[]}'
)
_SCHEMA_2_WITH_REASONS = (
    '{"schema":2,"marc_control_id":"b","nypl_uuid":"u-b","verdict":"no_match",'
    '"reasons":["diff_work","garbled"],"note":"strange match",'
    '"labeled_at":"2026-05-02T00:00:00+00:00","labeler":"jpstroop",'
    '"marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
    '"field_annotations":[]}'
)
_SCHEMA_2_WITH_ANNOTATIONS = (
    '{"schema":2,"marc_control_id":"c","nypl_uuid":"u-c","verdict":"match",'
    '"reasons":[],"note":null,"labeled_at":"2026-05-03T00:00:00+00:00",'
    '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
    '"field_annotations":[{"field":"title","judgment":"overscored"},'
    '{"field":"author","judgment":"correct"}]}'
)
_SCHEMA_2_WITH_BOTH = (
    '{"schema":2,"marc_control_id":"d","nypl_uuid":"u-d","verdict":"unsure",'
    '"reasons":["edition_unsure"],"note":"maybe a reprint",'
    '"labeled_at":"2026-05-04T00:00:00+00:00","labeler":"jpstroop",'
    '"marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
    '"field_annotations":[{"field":"edition","judgment":"underscored"}]}'
)
_SCHEMA_3_ENTRY = (
    '{"schema":3,"marc_control_id":"e","nypl_uuid":"u-e","verdict":"match",'
    '"note":"already migrated","labeled_at":"2026-05-05T00:00:00+00:00",'
    '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]}}'
)


def _write_vault(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_migrate_missing_vault_returns_zero_report(tmp_path: Path) -> None:
    report = migrate_vault_v3(tmp_path / "missing.jsonl")
    assert report.total_entries == 0
    assert report.reasons_folded == 0
    assert report.annotations_folded == 0
    assert report.archive_path is None


def test_migrate_empty_vault_returns_zero_report(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    report = migrate_vault_v3(path)
    assert report.total_entries == 0
    assert report.archive_path is None


def test_migrate_already_schema_3_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_3_ENTRY])
    original_bytes = path.read_bytes()
    report = migrate_vault_v3(path)
    assert report.total_entries == 1
    assert report.reasons_folded == 0
    assert report.annotations_folded == 0
    assert report.archive_path is None
    assert path.read_bytes() == original_bytes
    assert not (tmp_path / "vault.jsonl.pre-v3").exists()


def test_migrate_archives_original_when_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_2_NO_EXTRA])
    original_bytes = path.read_bytes()
    report = migrate_vault_v3(path)
    assert report.archive_path == path.with_name("vault.jsonl.pre-v3")
    assert report.archive_path.read_bytes() == original_bytes


def test_migrate_folds_reasons_into_note(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_2_WITH_REASONS])
    report = migrate_vault_v3(path)
    assert report.total_entries == 1
    assert report.reasons_folded == 1
    assert report.annotations_folded == 0
    [entry] = list(iter_entries(path))
    assert entry.schema == _TARGET_V3_SCHEMA
    assert entry.note == "[reasons: diff_work, garbled] strange match"


def test_migrate_folds_annotations_into_note(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_2_WITH_ANNOTATIONS])
    report = migrate_vault_v3(path)
    assert report.total_entries == 1
    assert report.reasons_folded == 0
    assert report.annotations_folded == 1
    [entry] = list(iter_entries(path))
    assert entry.note == "[annotations: title:overscored, author:correct]"


def test_migrate_folds_both_reasons_and_annotations(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_2_WITH_BOTH])
    report = migrate_vault_v3(path)
    assert report.total_entries == 1
    assert report.reasons_folded == 1
    assert report.annotations_folded == 1
    [entry] = list(iter_entries(path))
    assert entry.note is not None
    assert "[annotations: edition:underscored]" in entry.note
    assert "[reasons: edition_unsure]" in entry.note
    assert "maybe a reprint" in entry.note


def test_migrate_passes_through_entries_with_no_structured_signal(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_2_NO_EXTRA])
    report = migrate_vault_v3(path)
    assert report.total_entries == 1
    assert report.reasons_folded == 0
    assert report.annotations_folded == 0
    [entry] = list(iter_entries(path))
    assert entry.schema == _TARGET_V3_SCHEMA
    assert entry.note is None


def test_migrate_mixed_vault_keeps_already_schema_3_entries_intact(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [_SCHEMA_2_NO_EXTRA, _SCHEMA_2_WITH_REASONS, _SCHEMA_3_ENTRY, _SCHEMA_2_WITH_BOTH],
    )
    report = migrate_vault_v3(path)
    assert report.total_entries == 4
    assert report.reasons_folded == 2
    assert report.annotations_folded == 1
    entries = list(iter_entries(path))
    assert all(entry.schema == _TARGET_V3_SCHEMA for entry in entries)
    note_lookup = {entry.marc_control_id: entry.note for entry in entries}
    assert note_lookup["a"] is None
    assert note_lookup["b"] == "[reasons: diff_work, garbled] strange match"
    assert note_lookup["e"] == "already migrated"
    note_d = note_lookup["d"]
    assert note_d is not None
    assert "[reasons: edition_unsure]" in note_d


def test_migrate_handles_empty_annotation_list_without_folding(tmp_path: Path) -> None:
    """An entry with reasons=[] and field_annotations=[] should not fold anything."""
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_2_NO_EXTRA])
    report = migrate_vault_v3(path)
    assert report.reasons_folded == 0
    assert report.annotations_folded == 0


def test_migrate_drops_annotation_entries_with_invalid_shape(tmp_path: Path) -> None:
    """A malformed annotation dict (missing field/judgment) is skipped silently."""
    path = tmp_path / "vault.jsonl"
    malformed = (
        '{"schema":2,"marc_control_id":"x","nypl_uuid":"u-x","verdict":"match",'
        '"reasons":[],"note":null,"labeled_at":"2026-05-06T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"field_annotations":[{"field":"title"}]}'
    )
    _write_vault(path, [malformed])
    report = migrate_vault_v3(path)
    assert report.annotations_folded == 0
    [entry] = list(iter_entries(path))
    assert entry.note is None


def test_cli_migrate_vault_v3_runs_and_reports(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    _write_vault(vault_path, [_SCHEMA_2_WITH_REASONS])
    result = _RUNNER.invoke(
        app,
        ["migrate-vault-v3", "--vault", str(vault_path)],
    )
    assert result.exit_code == 0
    assert "migrated 1 entries" in result.stdout
    assert "folded reasons on 1" in result.stdout
    assert "archived original" in result.stdout
    assert vault_path.with_name("vault.jsonl.pre-v3").exists()


def test_cli_migrate_vault_v3_idempotent_message(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    _write_vault(vault_path, [_SCHEMA_3_ENTRY])
    result = _RUNNER.invoke(
        app,
        ["migrate-vault-v3", "--vault", str(vault_path)],
    )
    assert result.exit_code == 0
    assert "no migration needed" in result.stdout


def test_cli_migrate_vault_v3_missing_file_reports_zero(tmp_path: Path) -> None:
    vault_path = tmp_path / "absent.jsonl"
    result = _RUNNER.invoke(
        app,
        ["migrate-vault-v3", "--vault", str(vault_path)],
    )
    assert result.exit_code == 0
    assert "migrated 0 entries" in result.stdout


def test_migrate_skips_blank_lines_in_source_vault(tmp_path: Path) -> None:
    """Blank / whitespace-only lines in the raw vault are silently ignored."""
    path = tmp_path / "vault.jsonl"
    path.write_text(
        f"{_SCHEMA_2_NO_EXTRA}\n\n   \n{_SCHEMA_2_WITH_REASONS}\n",
        encoding="utf-8",
    )
    report = migrate_vault_v3(path)
    assert report.total_entries == 2
    assert report.reasons_folded == 1


def test_migrate_strips_old_keys_from_persisted_lines(tmp_path: Path) -> None:
    """The persisted output must not carry ``reasons`` or ``field_annotations``."""
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_SCHEMA_2_WITH_BOTH])
    migrate_vault_v3(path)
    for raw_line in path.read_bytes().splitlines():
        decoded = json_decode(raw_line, type=dict[str, object])
        assert "reasons" not in decoded
        assert "field_annotations" not in decoded
        assert decoded["schema"] == _TARGET_V3_SCHEMA


# --- migrate_vault_v4 ----------------------------------------------------------------


def _reg(
    *,
    uuid: str,
    regnum: str | None,
    renewal_id: str | None = None,
    renewal_oreg: str | None = None,
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid,
        title="Some title",
        was_renewed=renewal_id is not None,
        regnum=regnum,
        renewal_id=renewal_id,
        renewal_oreg=renewal_oreg,
    )


def _v3_line(*, marc_id: str, nypl_uuid: str) -> str:
    return (
        f'{{"schema":3,"marc_control_id":"{marc_id}","nypl_uuid":"{nypl_uuid}",'
        '"verdict":"match","note":null,"labeled_at":"2026-05-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]}}'
    )


def _v4_line(
    *,
    marc_id: str,
    nypl_uuid: str,
    cce_regnum: str | None,
    cce_renewal_id: str | None,
    cce_renewal_oreg: str | None,
) -> str:
    def _render(value: str | None) -> str:
        return "null" if value is None else f'"{value}"'

    return (
        f'{{"schema":4,"marc_control_id":"{marc_id}","nypl_uuid":"{nypl_uuid}",'
        '"verdict":"match","note":null,"labeled_at":"2026-05-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        f'"cce_regnum":{_render(cce_regnum)},'
        f'"cce_renewal_id":{_render(cce_renewal_id)},'
        f'"cce_renewal_oreg":{_render(cce_renewal_oreg)}}}'
    )


def _make_cce_lookup(
    records: dict[str, IndexedNyplRegRecord],
) -> Callable[[str], IndexedNyplRegRecord | None]:
    def lookup(uuid: str) -> IndexedNyplRegRecord | None:
        return records.get(uuid)

    return lookup


def test_v4_missing_vault_returns_zero_report(tmp_path: Path) -> None:
    report = migrate_vault_v4(tmp_path / "missing.jsonl", lambda _u: None)
    assert report.total_entries == 0
    assert report.enriched == 0
    assert report.orphaned == 0


def test_v4_empty_vault_returns_zero_report(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    report = migrate_vault_v4(path, lambda _u: None)
    assert report.total_entries == 0
    assert report.enriched == 0
    assert report.orphaned == 0


def test_v4_enriches_every_entry_when_all_uuids_resolve(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v3_line(marc_id="a", nypl_uuid="u-a"),
            _v3_line(marc_id="b", nypl_uuid="u-b"),
            _v3_line(marc_id="c", nypl_uuid="u-c"),
        ],
    )
    records = {
        "u-a": _reg(uuid="u-a", regnum="A1", renewal_id="R1", renewal_oreg="A1"),
        "u-b": _reg(uuid="u-b", regnum="A2", renewal_id=None, renewal_oreg=None),
        "u-c": _reg(uuid="u-c", regnum="A3", renewal_id="R3-typo", renewal_oreg="A3"),
    }
    report = migrate_vault_v4(path, _make_cce_lookup(records))
    assert report.total_entries == 3
    assert report.enriched == 3
    assert report.orphaned == 0
    entries = {entry.marc_control_id: entry for entry in iter_entries(path)}
    assert entries["a"].schema == 4
    assert entries["a"].cce_regnum == "A1"
    assert entries["a"].cce_renewal_id == "R1"
    assert entries["a"].cce_renewal_oreg == "A1"
    assert entries["b"].cce_regnum == "A2"
    assert entries["b"].cce_renewal_id is None
    assert entries["b"].cce_renewal_oreg is None
    assert entries["c"].cce_renewal_id == "R3-typo"
    assert entries["c"].cce_renewal_oreg == "A3"


def test_v4_handles_orphaned_uuid_with_none_fields(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v3_line(marc_id="a", nypl_uuid="u-a"),
            _v3_line(marc_id="b", nypl_uuid="u-missing"),
        ],
    )
    records = {
        "u-a": _reg(uuid="u-a", regnum="A1", renewal_id="R1", renewal_oreg="A1"),
    }
    report = migrate_vault_v4(path, _make_cce_lookup(records))
    assert report.total_entries == 2
    assert report.enriched == 1
    assert report.orphaned == 1
    entries = {entry.marc_control_id: entry for entry in iter_entries(path)}
    assert entries["a"].cce_regnum == "A1"
    assert entries["b"].schema == 4
    assert entries["b"].cce_regnum is None
    assert entries["b"].cce_renewal_id is None
    assert entries["b"].cce_renewal_oreg is None


def test_v4_is_idempotent_on_already_v4_vault(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v4_line(
                marc_id="a",
                nypl_uuid="u-a",
                cce_regnum="A1",
                cce_renewal_id="R1",
                cce_renewal_oreg="A1",
            ),
        ],
    )
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns

    def _should_not_be_called(_uuid: str) -> IndexedNyplRegRecord | None:
        raise AssertionError("cce_lookup must not be called when vault is already v4")

    report = migrate_vault_v4(path, _should_not_be_called)
    assert report.total_entries == 1
    assert report.enriched == 0
    assert report.orphaned == 0
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime


def test_v4_writes_atomically_no_archive_file_left_behind(tmp_path: Path) -> None:
    """No ``.pre-v4`` archive and no leftover ``*.tmp`` after the migration."""
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_v3_line(marc_id="a", nypl_uuid="u-a")])
    records = {"u-a": _reg(uuid="u-a", regnum="A1", renewal_id="R1", renewal_oreg="A1")}
    migrate_vault_v4(path, _make_cce_lookup(records))
    siblings = sorted(child.name for child in tmp_path.iterdir())
    assert siblings == ["vault.jsonl"]


def test_v4_bumps_schema_field_in_persisted_lines(tmp_path: Path) -> None:
    """Every persisted line ends up at ``schema=4`` regardless of input schema."""
    path = tmp_path / "vault.jsonl"
    _write_vault(path, [_v3_line(marc_id="a", nypl_uuid="u-a")])
    records = {"u-a": _reg(uuid="u-a", regnum="A1")}
    migrate_vault_v4(path, _make_cce_lookup(records))
    for raw_line in path.read_bytes().splitlines():
        decoded = json_decode(raw_line, type=dict[str, object])
        assert decoded["schema"] == 4


def test_cli_migrate_vault_v4_missing_file_reports_zero(tmp_path: Path) -> None:
    vault_path = tmp_path / "absent.jsonl"
    index_path = tmp_path / "cce.lmdb"
    result = _RUNNER.invoke(
        app,
        ["migrate-vault-v4", "--vault", str(vault_path), "--index", str(index_path)],
    )
    assert result.exit_code == 0
    assert "migrated 0 entries" in result.stdout
    assert "enriched 0" in result.stdout
    assert "orphaned 0" in result.stdout


def test_cli_migrate_vault_v4_runs_and_reports(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    _write_vault(
        vault_path,
        [
            _v3_line(marc_id="a", nypl_uuid="u-a"),
            _v3_line(marc_id="b", nypl_uuid="u-missing"),
        ],
    )
    records = {"u-a": _reg(uuid="u-a", regnum="A1", renewal_id="R1", renewal_oreg="A1")}

    class _FakeLookup:
        def __enter__(self) -> _FakeLookup:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_registration(self, uuid: str) -> IndexedNyplRegRecord | None:
            return records.get(uuid)

    def _fake_factory(_path: Path) -> _FakeLookup:
        return _FakeLookup()

    with patch("pd_groundtruth.cli.NyplIndexLookup", _fake_factory):
        result = _RUNNER.invoke(
            app,
            [
                "migrate-vault-v4",
                "--vault",
                str(vault_path),
                "--index",
                str(tmp_path / "cce.lmdb"),
            ],
        )
    assert result.exit_code == 0, result.stdout
    assert "migrated 2 entries" in result.stdout
    assert "enriched 1" in result.stdout
    assert "orphaned 1" in result.stdout


def test_cli_migrate_vault_v4_idempotent(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    _write_vault(
        vault_path,
        [
            _v4_line(
                marc_id="a",
                nypl_uuid="u-a",
                cce_regnum="A1",
                cce_renewal_id="R1",
                cce_renewal_oreg="A1",
            ),
        ],
    )

    class _UnusedLookup:
        def __enter__(self) -> _UnusedLookup:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_registration(self, _uuid: str) -> IndexedNyplRegRecord | None:
            raise AssertionError("must not be called for an already-v4 vault")

    def _fake_factory(_path: Path) -> _UnusedLookup:
        return _UnusedLookup()

    with patch("pd_groundtruth.cli.NyplIndexLookup", _fake_factory):
        result = _RUNNER.invoke(
            app,
            [
                "migrate-vault-v4",
                "--vault",
                str(vault_path),
                "--index",
                str(tmp_path / "cce.lmdb"),
            ],
        )
    assert result.exit_code == 0, result.stdout
    assert "migrated 1 entries" in result.stdout
    assert "enriched 0" in result.stdout
    assert "orphaned 0" in result.stdout


# ---- v5 migration ----------------------------------------------------------


def _v5_line(
    *,
    marc_id: str,
    nypl_uuid: str,
    categories: tuple[str, ...] = (),
) -> str:
    rendered = "[" + ",".join(f'"{c}"' for c in categories) + "]"
    return (
        f'{{"schema":5,"marc_control_id":"{marc_id}","nypl_uuid":"{nypl_uuid}",'
        '"verdict":"match","note":null,"labeled_at":"2026-06-01T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"cce_regnum":null,"cce_renewal_id":null,"cce_renewal_oreg":null,'
        f'"categories":{rendered}}}'
    )


def test_v5_migrates_v4_vault_to_schema_5_with_empty_categories(tmp_path: Path) -> None:
    """Every v4 entry becomes a v5 entry with ``categories=[]``."""
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v4_line(
                marc_id="a",
                nypl_uuid="u-a",
                cce_regnum="A1",
                cce_renewal_id="R1",
                cce_renewal_oreg="A1",
            ),
            _v4_line(
                marc_id="b",
                nypl_uuid="u-b",
                cce_regnum="A2",
                cce_renewal_id=None,
                cce_renewal_oreg=None,
            ),
        ],
    )
    report = migrate_vault_v5(path)
    assert report.total_entries == 2
    assert report.migrated == 2
    entries = {entry.marc_control_id: entry for entry in iter_entries(path)}
    assert entries["a"].schema == 5
    assert entries["a"].categories == ()
    assert entries["b"].categories == ()
    # Pre-existing v4 fields are preserved.
    assert entries["a"].cce_regnum == "A1"
    assert entries["a"].cce_renewal_id == "R1"
    assert entries["b"].cce_regnum == "A2"


def test_v5_is_idempotent_on_already_v5_vault(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v5_line(marc_id="a", nypl_uuid="u-a"),
            _v5_line(marc_id="b", nypl_uuid="u-b", categories=("translation",)),
        ],
    )
    original_bytes = path.read_bytes()
    report = migrate_vault_v5(path)
    assert report.total_entries == 2
    assert report.migrated == 0
    # The file was not rewritten.
    assert path.read_bytes() == original_bytes


def test_v5_handles_missing_vault_file(tmp_path: Path) -> None:
    """A missing vault is a zero-count no-op and creates no file."""
    path = tmp_path / "absent.jsonl"
    report = migrate_vault_v5(path)
    assert report.total_entries == 0
    assert report.migrated == 0
    assert not path.exists()


def test_v5_handles_empty_vault_file(tmp_path: Path) -> None:
    """An empty vault file is a zero-count no-op."""
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    report = migrate_vault_v5(path)
    assert report.total_entries == 0
    assert report.migrated == 0


def test_v5_preserves_every_v4_field_through_the_bump(tmp_path: Path) -> None:
    """Round-trip every v4 field; only schema + categories should be added/changed."""
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v4_line(
                marc_id="x",
                nypl_uuid="u-x",
                cce_regnum="A99",
                cce_renewal_id="R99",
                cce_renewal_oreg="A99",
            ),
        ],
    )
    migrate_vault_v5(path)
    [entry] = list(iter_entries(path))
    assert entry.marc_control_id == "x"
    assert entry.nypl_uuid == "u-x"
    assert entry.verdict == "match"
    assert entry.note is None
    assert entry.labeled_at == "2026-05-01T00:00:00+00:00"
    assert entry.labeler == "jpstroop"
    assert entry.cce_regnum == "A99"
    assert entry.cce_renewal_id == "R99"
    assert entry.cce_renewal_oreg == "A99"
    assert entry.categories == ()
    assert entry.schema == 5


def test_v5_mixed_vault_only_bumps_the_pre_v5_entries(tmp_path: Path) -> None:
    """A mix of v4 and v5 entries: only the v4 ones get bumped + tagged."""
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v4_line(
                marc_id="old",
                nypl_uuid="u-old",
                cce_regnum=None,
                cce_renewal_id=None,
                cce_renewal_oreg=None,
            ),
            _v5_line(marc_id="new", nypl_uuid="u-new", categories=("ocr_confusion",)),
        ],
    )
    report = migrate_vault_v5(path)
    assert report.total_entries == 2
    assert report.migrated == 1
    entries = {entry.marc_control_id: entry for entry in iter_entries(path)}
    assert entries["old"].categories == ()
    assert entries["new"].categories == ("ocr_confusion",)


def test_cli_migrate_vault_v5_runs_and_reports(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    _write_vault(
        vault_path,
        [
            _v4_line(
                marc_id="a",
                nypl_uuid="u-a",
                cce_regnum="A1",
                cce_renewal_id=None,
                cce_renewal_oreg=None,
            ),
        ],
    )
    result = _RUNNER.invoke(
        app,
        ["migrate-vault-v5", "--vault", str(vault_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert "migrated 1 entries" in result.stdout
    assert "bumped 1 to schema 5" in result.stdout


def test_cli_migrate_vault_v5_missing_file_reports_zero(tmp_path: Path) -> None:
    vault_path = tmp_path / "absent.jsonl"
    result = _RUNNER.invoke(
        app,
        ["migrate-vault-v5", "--vault", str(vault_path)],
    )
    assert result.exit_code == 0
    assert "migrated 0 entries" in result.stdout
    assert "bumped 0 to schema 5" in result.stdout


# ---- v6 migration (-> schema 7, match_source backfill) ---------------------


def _v6_line(
    *,
    marc_id: str,
    nypl_uuid: str,
    categories: tuple[str, ...] = (),
) -> str:
    rendered = "[" + ",".join(f'"{c}"' for c in categories) + "]"
    return (
        f'{{"schema":6,"marc_control_id":"{marc_id}","nypl_uuid":"{nypl_uuid}",'
        '"verdict":"match","note":null,"labeled_at":"2026-06-10T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"cce_regnum":null,"cce_renewal_id":null,"cce_renewal_oreg":null,'
        f'"categories":{rendered},"reg_year":1953,"renewal_year":null,'
        '"was_renewed":false,"scores":null,"matcher_version":null}'
    )


def _v7_line(*, marc_id: str, nypl_uuid: str, match_source: str = "registration") -> str:
    return (
        f'{{"schema":7,"marc_control_id":"{marc_id}","nypl_uuid":"{nypl_uuid}",'
        '"verdict":"match","note":null,"labeled_at":"2026-06-11T00:00:00+00:00",'
        '"labeler":"jpstroop","marc_identifiers":{"lccn":null,"oclc":null,"isbns":[]},'
        '"cce_regnum":null,"cce_renewal_id":null,"cce_renewal_oreg":null,'
        '"categories":[],"reg_year":null,"renewal_year":null,"was_renewed":null,'
        f'"scores":null,"matcher_version":null,"match_source":"{match_source}"}}'
    )


def test_v6_migrates_pre_v7_vault_to_schema_7_with_registration_source(tmp_path: Path) -> None:
    """Every pre-v7 entry becomes schema 7 with ``match_source="registration"``."""
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v6_line(marc_id="a", nypl_uuid="u-a"),
            _v6_line(marc_id="b", nypl_uuid="u-b", categories=("translation",)),
        ],
    )
    report = migrate_vault_v6(path)
    assert report.total_entries == 2
    assert report.migrated == 2
    entries = {entry.marc_control_id: entry for entry in iter_entries(path)}
    assert entries["a"].schema == 7
    assert entries["a"].match_source == "registration"
    assert entries["b"].match_source == "registration"
    # Pre-existing fields survive the bump.
    assert entries["b"].categories == ("translation",)
    assert entries["a"].reg_year == 1953
    assert entries["a"].was_renewed is False


def test_v6_is_idempotent_on_already_v7_vault(tmp_path: Path) -> None:
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v7_line(marc_id="a", nypl_uuid="u-a"),
            _v7_line(marc_id="b", nypl_uuid="u-b", match_source="renewal"),
        ],
    )
    original_bytes = path.read_bytes()
    report = migrate_vault_v6(path)
    assert report.total_entries == 2
    assert report.migrated == 0
    assert path.read_bytes() == original_bytes


def test_v6_handles_missing_vault_file(tmp_path: Path) -> None:
    path = tmp_path / "absent.jsonl"
    report = migrate_vault_v6(path)
    assert report.total_entries == 0
    assert report.migrated == 0
    assert not path.exists()


def test_v6_handles_empty_vault_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    report = migrate_vault_v6(path)
    assert report.total_entries == 0
    assert report.migrated == 0


def test_v6_mixed_vault_only_bumps_pre_v7_entries(tmp_path: Path) -> None:
    """A mix of v6 and v7 entries: only the v6 ones get bumped + tagged."""
    path = tmp_path / "vault.jsonl"
    _write_vault(
        path,
        [
            _v6_line(marc_id="old", nypl_uuid="u-old"),
            _v7_line(marc_id="new", nypl_uuid="u-new", match_source="renewal"),
        ],
    )
    report = migrate_vault_v6(path)
    assert report.total_entries == 2
    assert report.migrated == 1
    entries = {entry.marc_control_id: entry for entry in iter_entries(path)}
    assert entries["old"].match_source == "registration"
    # The already-v7 renewal entry is untouched.
    assert entries["new"].match_source == "renewal"


def test_cli_migrate_vault_v6_runs_and_reports(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault.jsonl"
    _write_vault(vault_path, [_v6_line(marc_id="a", nypl_uuid="u-a")])
    result = _RUNNER.invoke(app, ["migrate-vault-v6", "--vault", str(vault_path)])
    assert result.exit_code == 0, result.stdout
    assert "migrated 1 entries" in result.stdout
    assert "bumped 1 to schema 7" in result.stdout


def test_cli_migrate_vault_v6_missing_file_reports_zero(tmp_path: Path) -> None:
    vault_path = tmp_path / "absent.jsonl"
    result = _RUNNER.invoke(app, ["migrate-vault-v6", "--vault", str(vault_path)])
    assert result.exit_code == 0
    assert "migrated 0 entries" in result.stdout
    assert "bumped 0 to schema 7" in result.stdout
