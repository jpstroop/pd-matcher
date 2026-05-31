"""Durable, git-tracked JSONL vault for human verdicts.

The vault is the *source of truth* for ground-truth labels.
:class:`~pd_groundtruth.review_db.ReviewDb` is a transient working queue: it is
rebuilt each time ``acquire`` and ``build-queue`` re-run (for example after a
new filter lands). The vault, ``data/label_vault.jsonl``, lives in the repo and is
committed to git so the human labor invested in adjudicating pairs survives
those rebuilds.

The file is an upsert table: exactly one :class:`VaultEntry` per
``(marc_control_id, nypl_uuid)`` pair, encoded with
:func:`msgspec.json.encode` (compact, deterministic) and terminated with
``"\\n"``. Every line carries a ``schema`` integer for forward-compat.
Re-submitting a verdict for the same pair replaces the existing entry in
place; relabel history is not preserved. The latest state IS the entry.

Identifiers persisted alongside the verdict (under
:class:`MarcIdentifiers`) let downstream tooling re-pair a labeled MARC record
with its rebuilt index even if the ``marc_control_id`` ever shifts. The same
principle applies to the CCE side: ``cce_regnum``, ``cce_renewal_id``, and
``cce_renewal_oreg`` are baked into every schema-4 entry as flat top-level
fields so the published JSONL is a complete, self-contained linkage table that
consumers can cross-reference back to Copyright Office data without joining
anything else. ``cce_renewal_oreg`` exists alongside ``cce_regnum`` so future
work matching against the renewal index independently can compare the
renewal's transcribed original-registration cite to the matched registration's
``regnum`` and surface NYPL OCR errors.
"""

from collections.abc import Iterator
from os import fsync
from pathlib import Path

from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode

from pd_matcher.models import MarcRecord

SCHEMA_VERSION: int = 4


class MarcIdentifiers(Struct, frozen=True, forbid_unknown_fields=True):
    """All available stable identifiers for a MARC record at label time."""

    lccn: str | None
    oclc: str | None
    isbns: tuple[str, ...]


class VaultEntry(Struct, frozen=True, forbid_unknown_fields=True):
    """The current verdict for one ``(marc_control_id, nypl_uuid)`` pair.

    Exactly one entry exists per pair; re-submitting a verdict replaces the
    existing entry in place. Free-text ``note`` is the only structured
    signal the labeler carries alongside the verdict — the pre-schema-3
    ``reasons`` and ``field_annotations`` fields have been retired in favor
    of letting accumulated notes surface patterns naturally.

    Schema 4 adds three flat top-level CCE-side identifier fields:
    ``cce_regnum`` (the registration's Copyright Office number),
    ``cce_renewal_id`` (the NYPL renewal record id when the registration was
    renewed), and ``cce_renewal_oreg`` (the original registration cite copied
    from the renewal). All three default to ``None`` so schema-3 entries
    decode cleanly during forward-compat reads.
    """

    schema: int
    marc_control_id: str
    nypl_uuid: str
    verdict: str
    note: str | None
    labeled_at: str
    labeler: str
    marc_identifiers: MarcIdentifiers
    cce_regnum: str | None = None
    cce_renewal_id: str | None = None
    cce_renewal_oreg: str | None = None


def upsert_entry(path: Path, entry: VaultEntry) -> None:
    """Insert or replace the entry for ``entry``'s pair in the vault.

    The vault holds at most one entry per ``(marc_control_id, nypl_uuid)``.
    If an entry for that pair already exists, it is replaced in place by
    ``entry`` (the new entry's fields win wholesale — verdict, note,
    timestamp, labeler, identifiers). Otherwise ``entry`` is appended at the
    end. Insertion order across distinct pairs is preserved.

    Written atomically via a temp file + ``os.replace`` so a crash mid-write
    cannot corrupt the vault. ``os.fsync`` runs on the temp file before the
    rename because the vault is precious data and a crash between buffered
    write and OS flush would otherwise silently lose the label.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (entry.marc_control_id, entry.nypl_uuid)
    existing = list(iter_entries(path))
    replaced = False
    for index, current in enumerate(existing):
        if (current.marc_control_id, current.nypl_uuid) == key:
            existing[index] = entry
            replaced = True
            break
    if not replaced:
        existing.append(entry)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        for item in existing:
            handle.write(json_encode(item))
            handle.write(b"\n")
        handle.flush()
        fsync(handle.fileno())
    tmp_path.replace(path)


def iter_entries(path: Path) -> Iterator[VaultEntry]:
    """Stream :class:`VaultEntry` records from ``path`` lazily.

    Empty lines (including trailing blanks) are skipped. Malformed JSON raises
    immediately rather than being silently dropped: vault integrity matters and
    corruption must surface, not accumulate. A missing path yields nothing
    (no error) because an unlabeled project legitimately has no vault file.
    """
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json_decode(stripped, type=VaultEntry)


def current_entries(path: Path) -> dict[tuple[str, str], VaultEntry]:
    """Return one :class:`VaultEntry` per ``(marc_control_id, nypl_uuid)``.

    Since the vault upsert semantics guarantee one entry per pair, this is
    a straight ``(key, entry)`` projection of the file. A missing file
    returns an empty dict.
    """
    return {(entry.marc_control_id, entry.nypl_uuid): entry for entry in iter_entries(path)}


def extract_marc_identifiers(marc: MarcRecord) -> MarcIdentifiers:
    """Project a :class:`MarcRecord` into the identifiers carried by the vault.

    LCCN and OCLC are taken as-is; ISBNs preserve the parser's order. No
    additional normalization is performed — the parser has already cleaned
    these values.
    """
    return MarcIdentifiers(
        lccn=marc.lccn,
        oclc=marc.oclc,
        isbns=marc.isbns,
    )


__all__ = [
    "SCHEMA_VERSION",
    "MarcIdentifiers",
    "VaultEntry",
    "current_entries",
    "extract_marc_identifiers",
    "iter_entries",
    "upsert_entry",
]
