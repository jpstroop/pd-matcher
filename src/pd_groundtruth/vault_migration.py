"""One-shot migrations for the label-vault JSONL across schema bumps.

Two migrations live here:

* :func:`migrate_vault_v3` folds the pre-schema-3 ``reasons`` /
  ``field_annotations`` structured signals into the free-text ``note``,
  archiving the original to ``<vault>.pre-v3`` before rewriting.
* :func:`migrate_vault_v4` backfills the three flat top-level CCE-side
  identifier fields (``cce_regnum``, ``cce_renewal_id``, ``cce_renewal_oreg``)
  introduced by schema 4 by looking each entry's ``nypl_uuid`` up in the CCE
  index and copying the values across. The pre-migration vault lives in git
  history; no on-disk ``.pre-v4`` archive is written.

Both migrations run idempotently: a vault already entirely at the target
schema passes through untouched and the report shows zero changes.
"""

from collections.abc import Callable
from collections.abc import Iterator
from os import replace as os_replace
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile

from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode

from pd_matcher.models import IndexedNyplRegRecord

CceLookupFn = Callable[[str], IndexedNyplRegRecord | None]

_SCHEMA_V3: int = 3
_SCHEMA_V4: int = 4
_SCHEMA_V5: int = 5
_ARCHIVE_SUFFIX: str = ".pre-v3"


class MigrationReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`migrate_vault_v3` for the CLI to print."""

    total_entries: int
    reasons_folded: int
    annotations_folded: int
    archive_path: Path | None


class MigrationReportV4(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`migrate_vault_v4` for the CLI to print.

    ``orphaned`` counts entries whose ``nypl_uuid`` no longer resolves in the
    CCE index (data drift): those entries still get bumped to ``schema=4`` but
    keep ``None`` for the three new CCE-side identifier fields.
    """

    total_entries: int
    enriched: int
    orphaned: int


class MigrationReportV5(Struct, frozen=True, forbid_unknown_fields=True):
    """Counts emitted by :func:`migrate_vault_v5` for the CLI to print.

    ``migrated`` is the number of entries whose ``schema`` was below 5 and
    that received the empty-tuple ``categories`` backfill.
    """

    total_entries: int
    migrated: int


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


def _schema_at_least(raw: dict[str, object], minimum: int) -> bool:
    """Return ``True`` when ``raw['schema']`` is an int ``>= minimum``."""
    schema = raw.get("schema")
    return isinstance(schema, int) and schema >= minimum


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
    schema 3 or later passes through untouched and both flags are ``False``.
    """
    schema = raw.get("schema")
    if isinstance(schema, int) and schema >= _SCHEMA_V3:
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
    migrated["schema"] = _SCHEMA_V3
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
    if all(_schema_at_least(entry, _SCHEMA_V3) for entry in raw_entries):
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


def _atomic_write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    """Write ``entries`` (one JSON dict per line) to ``path`` atomically.

    Streams to a temp file in the same directory then ``os.replace`` swaps it
    into place so concurrent readers see either the old or the new content,
    never a truncated half-write.
    """
    payload = b"".join(json_encode(entry) + b"\n" for entry in entries)
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os_replace(temp_path, path)


def migrate_vault_v4(
    vault_path: Path,
    cce_lookup: CceLookupFn,
) -> MigrationReportV4:
    """Backfill the schema-4 CCE-side identifier fields in place.

    Reads each line raw (bypassing the typed decode), looks ``nypl_uuid`` up
    via ``cce_lookup`` to obtain an :class:`IndexedNyplRegRecord`, copies its
    ``regnum`` / ``renewal_id`` / ``renewal_oreg`` onto the entry, and bumps
    the ``schema`` field to 4. Entries whose ``nypl_uuid`` no longer resolves
    in the index (data drift) keep ``None`` for the three new fields but still
    get ``schema=4``; they are counted in :attr:`MigrationReportV4.orphaned`.

    The pre-migration vault is preserved in git history rather than on disk
    (no ``.pre-v4`` archive is created). The write itself is atomic: a temp
    file in the same directory is renamed over the canonical path.

    A missing or empty vault returns a zero-count report and creates no file.
    A vault already entirely at schema 4 is a no-op: no rewrite.

    Args:
        vault_path: The JSONL label vault to migrate.
        cce_lookup: ``nypl_uuid -> IndexedNyplRegRecord | None`` resolver,
            typically :meth:`pd_matcher.index.lookup.NyplIndexLookup.get_registration`.
            Injected so tests can drive the migration without standing up a
            real LMDB env.
    """
    if not vault_path.exists() or vault_path.stat().st_size == 0:
        return MigrationReportV4(total_entries=0, enriched=0, orphaned=0)
    raw_entries = list(_iter_raw_entries(vault_path))
    if all(_schema_at_least(entry, _SCHEMA_V4) for entry in raw_entries):
        return MigrationReportV4(total_entries=len(raw_entries), enriched=0, orphaned=0)
    enriched = 0
    orphaned = 0
    migrated_entries: list[dict[str, object]] = []
    for raw in raw_entries:
        nypl_uuid = raw.get("nypl_uuid")
        record = cce_lookup(nypl_uuid) if isinstance(nypl_uuid, str) else None
        if record is None:
            orphaned += 1
            raw["cce_regnum"] = None
            raw["cce_renewal_id"] = None
            raw["cce_renewal_oreg"] = None
        else:
            enriched += 1
            raw["cce_regnum"] = record.regnum
            raw["cce_renewal_id"] = record.renewal_id
            raw["cce_renewal_oreg"] = record.renewal_oreg
        raw["schema"] = _SCHEMA_V4
        migrated_entries.append(raw)
    _atomic_write_jsonl(vault_path, migrated_entries)
    return MigrationReportV4(
        total_entries=len(raw_entries),
        enriched=enriched,
        orphaned=orphaned,
    )


def migrate_vault_v5(vault_path: Path) -> MigrationReportV5:
    """Bump every entry to schema 5 and backfill ``categories`` with ``()``.

    Schema 5 adds a ``categories: tuple[CategoryKey, ...]`` field to
    :class:`VaultEntry`. The migration is uniform — no external lookup —
    so every pre-v5 entry receives an empty tuple and a bump to ``schema=5``.

    Idempotent: a vault already entirely at schema 5 returns a zero-count
    report and is not rewritten. A missing or empty vault is also a
    zero-count no-op.

    The write itself is atomic: a temp file in the same directory is renamed
    over the canonical path. The pre-migration vault is preserved in git
    history rather than on disk.
    """
    if not vault_path.exists() or vault_path.stat().st_size == 0:
        return MigrationReportV5(total_entries=0, migrated=0)
    raw_entries = list(_iter_raw_entries(vault_path))
    if all(_schema_at_least(entry, _SCHEMA_V5) for entry in raw_entries):
        return MigrationReportV5(total_entries=len(raw_entries), migrated=0)
    migrated = 0
    migrated_entries: list[dict[str, object]] = []
    for raw in raw_entries:
        if not _schema_at_least(raw, _SCHEMA_V5):
            raw["schema"] = _SCHEMA_V5
            raw["categories"] = []
            migrated += 1
        migrated_entries.append(raw)
    _atomic_write_jsonl(vault_path, migrated_entries)
    return MigrationReportV5(total_entries=len(raw_entries), migrated=migrated)


__all__ = [
    "CceLookupFn",
    "MigrationReport",
    "MigrationReportV4",
    "MigrationReportV5",
    "migrate_vault_v3",
    "migrate_vault_v4",
    "migrate_vault_v5",
]
