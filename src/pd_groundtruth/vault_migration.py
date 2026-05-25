"""One-shot migration of label-vault JSONL from schema 2 to schema 3.

Schema 3 drops the structured ``reasons`` and ``field_annotations`` fields in
favor of a single free-text ``note``. The migration preserves the historical
signal by folding any pre-schema-3 reason codes / field annotations into the
note text as ``[reasons: ...]`` and ``[annotations: field:judgment, ...]``
prefixes so downstream readers of accumulated notes can still see what the
labeler flagged. The original file is archived to ``<vault>.pre-v3`` before
the migrated content overwrites the canonical path.

Run idempotently: an already-migrated vault (every line at ``schema=3``)
passes through unchanged, no archive is created, and the report shows zero
folds.
"""

from collections.abc import Iterator
from pathlib import Path
from shutil import copyfile

from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode

from pd_groundtruth.label_vault import SCHEMA_VERSION

_ARCHIVE_SUFFIX: str = ".pre-v3"


class MigrationReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`migrate_vault_v3` for the CLI to print."""

    total_entries: int
    reasons_folded: int
    annotations_folded: int
    archive_path: Path | None


def _iter_raw_entries(path: Path) -> Iterator[dict[str, object]]:
    """Stream raw (untyped) JSONL dict entries from ``path``.

    Old entries carry fields the new :class:`VaultEntry` would reject; this
    bypasses the typed decode so the migration can read them.
    """
    with path.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json_decode(stripped, type=dict[str, object])


def _fold_reasons(note: str | None, reasons: list[str]) -> tuple[str | None, bool]:
    """Prepend ``[reasons: ...]`` to ``note`` when ``reasons`` is non-empty.

    Returns ``(new_note, folded)`` where ``folded`` flags whether a fold
    happened so the caller can update the counter.
    """
    if not reasons:
        return note, False
    joined = ", ".join(reasons)
    base = note or ""
    merged = f"[reasons: {joined}] {base}".rstrip()
    return merged, True


def _fold_annotations(
    note: str | None,
    annotations: list[dict[str, object]],
) -> tuple[str | None, bool]:
    """Prepend ``[annotations: field:judgment, ...]`` to ``note`` when present.

    Each annotation is rendered as ``field:judgment``; the function tolerates
    legacy entries whose annotation dicts came from msgspec encoding (same
    shape as the now-deleted ``FieldAnnotation`` struct).
    """
    if not annotations:
        return note, False
    parts: list[str] = []
    for annotation in annotations:
        field = annotation.get("field")
        judgment = annotation.get("judgment")
        if isinstance(field, str) and isinstance(judgment, str):
            parts.append(f"{field}:{judgment}")
    if not parts:
        return note, False
    joined = ", ".join(parts)
    base = note or ""
    merged = f"[annotations: {joined}] {base}".rstrip()
    return merged, True


def _migrate_entry(raw: dict[str, object]) -> tuple[dict[str, object], bool, bool]:
    """Return the schema-3 form of one raw entry plus fold flags.

    ``(new_dict, reasons_folded, annotations_folded)``. An entry already at
    schema 3 passes through untouched and both flags are ``False``.
    """
    schema = raw.get("schema")
    if schema == SCHEMA_VERSION:
        return raw, False, False
    note_raw = raw.get("note")
    note: str | None = note_raw if isinstance(note_raw, str) else None
    reasons_raw = raw.get("reasons", [])
    reasons: list[str] = (
        [r for r in reasons_raw if isinstance(r, str)] if isinstance(reasons_raw, list) else []
    )
    annotations_raw = raw.get("field_annotations", [])
    annotations: list[dict[str, object]] = (
        [a for a in annotations_raw if isinstance(a, dict)]
        if isinstance(annotations_raw, list)
        else []
    )
    note, reasons_folded = _fold_reasons(note, reasons)
    note, annotations_folded = _fold_annotations(note, annotations)
    migrated: dict[str, object] = {
        key: value
        for key, value in raw.items()
        if key not in {"reasons", "field_annotations", "note", "schema"}
    }
    migrated["schema"] = SCHEMA_VERSION
    migrated["note"] = note
    return migrated, reasons_folded, annotations_folded


def migrate_vault_v3(vault_path: Path) -> MigrationReport:
    """Migrate ``vault_path`` in place to schema 3, archiving the original.

    Reads each line raw (bypassing the typed decode), folds pre-schema-3
    structured fields into the note text, archives the original to
    ``<vault>.pre-v3``, and writes the migrated entries back. A missing or
    empty vault returns a zero-count report and creates no archive. A vault
    already entirely at schema 3 is a no-op: no archive, no rewrite.
    """
    if not vault_path.exists() or vault_path.stat().st_size == 0:
        return MigrationReport(
            total_entries=0,
            reasons_folded=0,
            annotations_folded=0,
            archive_path=None,
        )
    raw_entries = list(_iter_raw_entries(vault_path))
    if all(entry.get("schema") == SCHEMA_VERSION for entry in raw_entries):
        return MigrationReport(
            total_entries=len(raw_entries),
            reasons_folded=0,
            annotations_folded=0,
            archive_path=None,
        )
    archive_path = vault_path.with_name(vault_path.name + _ARCHIVE_SUFFIX)
    copyfile(vault_path, archive_path)
    reasons_folded = 0
    annotations_folded = 0
    migrated_entries: list[dict[str, object]] = []
    for raw in raw_entries:
        migrated, folded_r, folded_a = _migrate_entry(raw)
        migrated_entries.append(migrated)
        reasons_folded += int(folded_r)
        annotations_folded += int(folded_a)
    payload = b"".join(json_encode(entry) + b"\n" for entry in migrated_entries)
    vault_path.write_bytes(payload)
    return MigrationReport(
        total_entries=len(raw_entries),
        reasons_folded=reasons_folded,
        annotations_folded=annotations_folded,
        archive_path=archive_path,
    )


__all__ = [
    "MigrationReport",
    "migrate_vault_v3",
]
