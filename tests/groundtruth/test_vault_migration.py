"""Unit tests for the schema-2 -> schema-3 vault migration."""

from pathlib import Path

from msgspec.json import decode as json_decode
from typer.testing import CliRunner

from pd_groundtruth.cli import app
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.vault_migration import migrate_vault_v3

_RUNNER = CliRunner()


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
    assert entry.schema == SCHEMA_VERSION
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
    assert entry.schema == SCHEMA_VERSION
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
    assert all(entry.schema == SCHEMA_VERSION for entry in entries)
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
        assert decoded["schema"] == SCHEMA_VERSION
