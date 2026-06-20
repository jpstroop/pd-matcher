"""Durable, git-tracked JSONL vault for human verdicts.

The vault is the *source of truth* for ground-truth labels.
:class:`~pd_groundtruth.review_db.ReviewDb` is a transient working queue: it is
rebuilt each time ``acquire`` and ``build-queue`` re-run (for example after a
new filter lands). The vault, ``data/training/label_vault.jsonl``, lives in the
``cce-marc-linkage`` submodule and is committed to git so the human labor invested
in adjudicating pairs survives those rebuilds.

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

Schema 5 adds a ``categories: tuple[CategoryKey, ...]`` field that captures
recurring rationale patterns the labeler used to type into ``note``
(series-vs-volume mismatches, translations, OCR confusion, etc.). The
vocabulary is fixed in code via the :data:`CategoryKey` ``Literal`` type;
msgspec rejects unknown keys at decode. The default is the empty tuple,
so v4 data still decodes via the v5 struct.

Schema 6 adds machine-derived fields, all defaulting to ``None`` so schema-5
entries still decode via the v6 struct. They split by when they are written:
the three static CCE facts — ``reg_year`` and ``renewal_year`` (the CCE
registration and renewal years) and ``was_renewed`` (whether a renewal joined
the registration) — never change for a pair, so the review server stamps them
at label time via :func:`cce_facts`. The version-bound fields — a
:class:`MatcherScores` pair of matcher confidences and the ``matcher_version``
that produced them — stay ``None`` on label-time writes and are filled by the
``enrich-vault`` command on publish. Neither path alters a human-entered field.
"""

from collections.abc import Iterator
from datetime import date
from os import fsync
from pathlib import Path
from typing import Literal

from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode

from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

SCHEMA_VERSION: int = 6

CategoryKey = Literal[
    "marc_whole_cce_part",
    "cce_whole_marc_part",
    "translation",
    "different_edition",
    "ocr_confusion",
    "same_title_different_work",
    "generic_title",
]


class MarcIdentifiers(Struct, frozen=True, forbid_unknown_fields=True):
    """All available stable identifiers for a MARC record at label time."""

    lccn: str | None
    oclc: str | None
    isbns: tuple[str, ...]


class MatcherScores(Struct, frozen=True, forbid_unknown_fields=True):
    """Both matchers' confidence in the pair, as recorded by ``enrich-vault``.

    Each value is a confidence in ``[0.0, 1.0]`` where higher means more
    confident the pair is a true match, rounded to 4 decimals. The two are
    independently scaled and need NOT share a derivation: ``weighted_mean`` is
    the weighted-mean combiner's raw score divided by 100, while ``learned`` is
    the LightGBM combiner's calibrated probability. Both are only meaningful
    relative to ``1.0``, not to each other. ``learned`` is ``None`` when no
    learned-model artifact was available at enrichment time.
    """

    weighted_mean: float | None = None
    learned: float | None = None


class CceFacts(Struct, frozen=True, forbid_unknown_fields=True):
    """The three static, version-independent CCE facts for a pair.

    These derive only from the joined CCE registration and never change for a
    given pair, so they are stamped onto a :class:`VaultEntry` both at label
    time (by the review server) and by ``enrich-vault``. They are the
    schema-6 ``reg_year`` / ``renewal_year`` / ``was_renewed`` fields, kept
    apart from the version-bound ``scores`` / ``matcher_version`` that only
    ``enrich-vault`` writes.
    """

    reg_year: int | None
    renewal_year: int | None
    was_renewed: bool | None


def renewal_year_of(renewal_rdat: date | None) -> int | None:
    """Return the renewal-recording year, or ``None`` when no renewal joined.

    The renewal year is the year of ``renewal_rdat`` (the renewal-recording
    date copied onto the indexed registration during the index-build join);
    when no renewal joined the registration the date — and the year — are
    ``None``. The single source of the rdat-to-year derivation shared by
    :func:`cce_facts` and the review server's label-time write path.
    """
    if renewal_rdat is None:
        return None
    return renewal_rdat.year


def cce_facts(cce: IndexedNyplRegRecord) -> CceFacts:
    """Project the static CCE facts off a joined registration record.

    Args:
        cce: The indexed registration, carrying ``reg_year``, ``was_renewed``,
            and the joined ``renewal_rdat`` from which the renewal year is
            derived.
    """
    return CceFacts(
        reg_year=cce.reg_year,
        renewal_year=renewal_year_of(cce.renewal_rdat),
        was_renewed=cce.was_renewed,
    )


class VaultEntry(Struct, frozen=True, forbid_unknown_fields=True):
    """The current verdict for one ``(marc_control_id, nypl_uuid)`` pair.

    Exactly one entry exists per pair; re-submitting a verdict replaces the
    existing entry in place. Free-text ``note`` is the labeler's open
    rationale; ``categories`` (schema 5) is the structured complement —
    a multi-select list of recurring patterns the verdict reflects.

    Schema 4 adds three flat top-level CCE-side identifier fields:
    ``cce_regnum`` (the registration's Copyright Office number),
    ``cce_renewal_id`` (the NYPL renewal record id when the registration was
    renewed), and ``cce_renewal_oreg`` (the original registration cite copied
    from the renewal). All three default to ``None`` so schema-3 entries
    decode cleanly during forward-compat reads.

    Schema 5 adds the ``categories`` tuple — zero or more
    :data:`CategoryKey` values capturing recurring rationale patterns:

    * ``marc_whole_cce_part`` — MARC describes a whole series/set; CCE
      registers a single member (usually ``no_match``).
    * ``cce_whole_marc_part`` — CCE is the series-level registration;
      MARC is one member (``match`` by inference per the labeling guide).
    * ``translation`` — one side registers a translation, the other the
      original.
    * ``different_edition`` — same work, different edition / printing
      (typically year or publisher mismatch on otherwise identical title
      + author).
    * ``ocr_confusion`` — match obscured by an OCR transcription error.
    * ``same_title_different_work`` — full title agreement with author /
      publisher / year all contradicting.
    * ``generic_title`` — the title is generic enough that title-only
      match is unreliable.

    Unknown category keys raise ``msgspec.ValidationError`` at decode time
    because ``CategoryKey`` is a ``Literal`` type; the v1 vocabulary is
    extended by appending new keys to that type, which is forward-compat
    for old data (which has empty tuples) but not backward-compat for
    new keys read by old code.

    Schema 6 adds five machine-derived fields, all defaulting to ``None`` so
    schema-5 entries decode cleanly. The three static CCE facts are stamped at
    label time by the review server; the two version-bound fields are filled
    by ``enrich-vault`` on publish:

    * ``reg_year`` — the CCE registration year.
    * ``renewal_year`` — the renewal-recording year, present only when a
      renewal joined the registration.
    * ``was_renewed`` — ``True`` when a renewal joined this registration.
    * ``scores`` — both matchers' confidence (:class:`MatcherScores`).
    * ``matcher_version`` — the matcher build that produced ``scores``.
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
    categories: tuple[CategoryKey, ...] = ()
    reg_year: int | None = None
    renewal_year: int | None = None
    was_renewed: bool | None = None
    scores: MatcherScores | None = None
    matcher_version: str | None = None


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
    "CategoryKey",
    "CceFacts",
    "MarcIdentifiers",
    "MatcherScores",
    "VaultEntry",
    "cce_facts",
    "current_entries",
    "extract_marc_identifiers",
    "iter_entries",
    "renewal_year_of",
    "upsert_entry",
]
