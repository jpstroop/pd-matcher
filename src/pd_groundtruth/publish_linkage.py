"""Reshape vault entries into the published-linkage JSONL pair.

The vault is keyed internally by ``(marc_control_id, nypl_uuid)``, which is
convenient against Princeton's MARC pool but misleading to external consumers
— Princeton's control IDs aren't useful to anyone else, and leading with them
overstates their value as the linkage. The published artifact instead leads
with universal bibliographic identifiers (LCCN, OCLC, ISBN), follows with
CCE-side identifiers, and demotes ``marc_control_id`` to a provenance trace
at the tail of each row.

The labeler's free-text note is intentionally dropped on publication —
the verdict is the durable signal; notes are working-state for our own
analysis.

Two files are emitted on each invocation, sharing the same schema:

* The full set at ``training_output_path`` (default
  ``data/published/training.jsonl``) carries every adjudicated verdict
  (``match``, ``no_match``, ``unsure``). It's the natural training input
  for a learned matcher — positives, hard negatives, and ambiguous calls
  all in one place.
* The matches-only subset at ``matches_output_path`` (default
  ``data/published/matches.jsonl``) carries only ``match`` rows — the
  curated linkage table for consumers who only need confirmed pairs.

Both files live inside the in-tree clone of the ``jpstroop/cce-marc-linkage``
data repository (gitignored from this code repo). Each file is written
atomically via temp file + ``os.replace`` so a crash mid-write cannot
corrupt the output.
"""

from os import fsync
from pathlib import Path

from msgspec import Struct
from msgspec.json import encode as json_encode

from pd_groundtruth.label_vault import current_entries


class PublishedRow(Struct, frozen=True, forbid_unknown_fields=True):
    """One published linkage row: identifiers + verdict + provenance.

    Field declaration order is the JSONL serialization order — universal
    identifiers lead, Princeton-local ``marc_control_id`` is at the tail.
    """

    lccn: str | None
    oclc: str | None
    isbns: tuple[str, ...]
    cce_regnum: str | None
    cce_renewal_id: str | None
    cce_renewal_oreg: str | None
    nypl_uuid: str
    verdict: str
    labeled_at: str
    labeler: str
    marc_control_id: str


class PublishReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of one :func:`publish_linkage` invocation."""

    rows_written: int
    matches: int
    no_matches: int
    unsures: int


def publish_linkage(
    vault_path: Path,
    training_output_path: Path,
    matches_output_path: Path,
) -> PublishReport:
    """Write the published-linkage JSONL pair from ``vault_path``.

    Emits two files sharing the same schema: ``training_output_path``
    carries every adjudicated verdict, ``matches_output_path`` carries
    only ``match`` rows. Rows are emitted in ``labeled_at`` ascending
    order so successive regenerations produce stable, diff-friendly
    output. Each file is written atomically via a temp file +
    ``os.replace`` so a crash mid-write cannot leave a half-written
    published artifact.
    """
    training_output_path.parent.mkdir(parents=True, exist_ok=True)
    matches_output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(
        current_entries(vault_path).values(),
        key=lambda entry: entry.labeled_at,
    )
    matches = 0
    no_matches = 0
    unsures = 0
    training_tmp = training_output_path.with_name(training_output_path.name + ".tmp")
    matches_tmp = matches_output_path.with_name(matches_output_path.name + ".tmp")
    with training_tmp.open("wb") as full_handle, matches_tmp.open("wb") as matches_handle:
        for entry in entries:
            row = PublishedRow(
                lccn=entry.marc_identifiers.lccn,
                oclc=entry.marc_identifiers.oclc,
                isbns=entry.marc_identifiers.isbns,
                cce_regnum=entry.cce_regnum,
                cce_renewal_id=entry.cce_renewal_id,
                cce_renewal_oreg=entry.cce_renewal_oreg,
                nypl_uuid=entry.nypl_uuid,
                verdict=entry.verdict,
                labeled_at=entry.labeled_at,
                labeler=entry.labeler,
                marc_control_id=entry.marc_control_id,
            )
            payload = json_encode(row)
            full_handle.write(payload)
            full_handle.write(b"\n")
            if entry.verdict == "match":
                matches += 1
                matches_handle.write(payload)
                matches_handle.write(b"\n")
            elif entry.verdict == "no_match":
                no_matches += 1
            else:
                unsures += 1
        full_handle.flush()
        fsync(full_handle.fileno())
        matches_handle.flush()
        fsync(matches_handle.fileno())
    training_tmp.replace(training_output_path)
    matches_tmp.replace(matches_output_path)
    return PublishReport(
        rows_written=len(entries),
        matches=matches,
        no_matches=no_matches,
        unsures=unsures,
    )


__all__ = [
    "PublishReport",
    "PublishedRow",
    "publish_linkage",
]
