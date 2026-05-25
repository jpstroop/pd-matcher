"""Durable, git-tracked JSONL vault for human verdicts.

The vault is the *source of truth* for ground-truth labels.
:class:`~pd_groundtruth.review_db.ReviewDb` is a transient working queue: it is
rebuilt each time ``acquire`` and ``build-queue`` re-run (for example after a
new filter lands). The vault, ``data/label_vault.jsonl``, lives in the repo and is
committed to git so the human labor invested in adjudicating pairs survives
those rebuilds.

The file is append-only JSONL: one :class:`VaultEntry` per line, encoded with
:func:`msgspec.json.encode` (compact, deterministic) and terminated with
``"\\n"``. Every line carries a ``schema`` integer for forward-compat. The
*current* verdict for a pair is the last entry with a given
``(marc_control_id, nypl_uuid)`` key, so re-labels do not overwrite history —
the full audit trail is preserved.

Identifiers persisted alongside the verdict (under
:class:`MarcIdentifiers`) let downstream tooling re-pair a labeled MARC record
with its rebuilt index even if the ``marc_control_id`` ever shifts.
"""

from collections.abc import Iterator
from os import fsync
from pathlib import Path

from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode

from pd_matcher.models import MarcRecord

SCHEMA_VERSION: int = 3


class MarcIdentifiers(Struct, frozen=True, forbid_unknown_fields=True):
    """All available stable identifiers for a MARC record at label time."""

    lccn: str | None
    oclc: str | None
    isbns: tuple[str, ...]


class VaultEntry(Struct, frozen=True, forbid_unknown_fields=True):
    """One verdict event persisted to the label vault.

    A ``(marc_control_id, nypl_uuid)`` pair may appear in the vault multiple
    times; the latest entry by file order wins. Free-text ``note`` is the only
    structured signal the labeler carries alongside the verdict — the
    pre-schema-3 ``reasons`` and ``field_annotations`` fields have been retired
    in favor of letting accumulated notes surface patterns naturally.
    """

    schema: int
    marc_control_id: str
    nypl_uuid: str
    verdict: str
    note: str | None
    labeled_at: str
    labeler: str
    marc_identifiers: MarcIdentifiers


def append_entry(path: Path, entry: VaultEntry) -> None:
    """Append one :class:`VaultEntry` to ``path`` as a single JSONL line.

    Creates the parent directory and the file if either is missing. Calls
    :func:`os.fsync` after the write because the vault is precious data — a
    process crash between ``write`` and the OS buffer flush would otherwise
    silently lose the label.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_encode(entry).decode("utf-8") + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        fsync(handle.fileno())


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
    """Return the latest :class:`VaultEntry` per ``(marc_control_id, nypl_uuid)``.

    Later lines in the file overwrite earlier ones, matching the "last entry
    wins" semantics of the vault. A missing file returns an empty dict.
    """
    latest: dict[tuple[str, str], VaultEntry] = {}
    for entry in iter_entries(path):
        latest[(entry.marc_control_id, entry.nypl_uuid)] = entry
    return latest


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
    "append_entry",
    "current_entries",
    "extract_marc_identifiers",
    "iter_entries",
]
