"""Stream the whole catalog and write the full in-scope matching corpus.

Where :mod:`pd_groundtruth.acquire` fetches the manifest dumps and keeps a
balanced *labeling sample* (a per-(language, decade) quota), and
:mod:`pd_groundtruth.filter` filters a single local MARCXML file, this module
streams **every** dump in the manifest and writes **every** in-scope record —
uncapped — to one MARCXML collection suitable for ``pd-matcher match --marc``.

The raw catalog is far too large to persist: each dump is downloaded to a
temporary archive (md5 verified), its records are streamed with ``iterparse``,
the survivors are appended to the output, and the temp download is deleted
before the next dump begins. At most one dump's compressed archive is ever on
disk, and nothing is extracted.

Eligibility is delegated wholesale to :func:`pd_groundtruth.filter._drop_reason`
(itself built on :func:`pd_groundtruth.filters.classify`), the same predicate
``acquire`` and ``filter`` use, so the three never drift. The optional
``languages`` argument narrows *within* the five supported 008 languages and
never widens that set.
"""

from collections import Counter
from logging import getLogger
from pathlib import Path

from msgspec import Struct

from pd_groundtruth.acquire import stream_dump_records
from pd_groundtruth.filter import _drop_reason
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
from pd_groundtruth.manifest import DumpEntry
from pd_groundtruth.manifest import fetch_manifest
from pd_groundtruth.writer import MarcxmlCollectionWriter

_LOGGER = getLogger(__name__)


class CorpusReport(Struct, frozen=True):
    """Outcome of a corpus-extraction run."""

    dumps_processed: int
    records_scanned: int
    kept: int
    dropped: int
    dropped_by_reason: dict[str, int]


def build_corpus(
    *,
    output_path: Path,
    min_year: int,
    languages: frozenset[str] | None = None,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    max_dumps: int | None = None,
) -> CorpusReport:
    """Stream every manifest dump and write all in-scope records to one file.

    Args:
        output_path: Destination MARCXML ``<collection>`` (the format
            ``pd-matcher match --marc`` reads); the parent directory is created.
            In-scope records are appended across dumps as they stream, so the
            full corpus is never buffered in memory.
        min_year: Inclusive lower bound for the publication year (the moving
            wall). Passed straight through to the shared eligibility predicate.
        languages: When ``None`` (the default), every record that passes
            eligibility is kept (any of the five supported 008 languages). When
            a set is given, an otherwise-eligible record is dropped unless its
            008 language code is in the set; this narrows within, and never
            widens, the language check.
        manifest_url: Absolute URL of the dump manifest JSON.
        max_dumps: Optional cap on the number of dumps processed (for testing or
            partial runs).

    Returns:
        A :class:`CorpusReport` with the dumps processed, records scanned, kept
        (in-scope), and dropped counts plus a per-reason breakdown.
    """
    entries = fetch_manifest(manifest_url)
    return _build_corpus_entries(
        entries,
        output_path=output_path,
        min_year=min_year,
        languages=languages,
        max_dumps=max_dumps,
    )


def _build_corpus_entries(
    entries: tuple[DumpEntry, ...],
    *,
    output_path: Path,
    min_year: int,
    languages: frozenset[str] | None,
    max_dumps: int | None,
) -> CorpusReport:
    """Run corpus extraction over an already-resolved set of dump entries."""
    dropped_by_reason: Counter[str] = Counter()
    dumps_processed = 0
    records_scanned = 0
    kept = 0
    with MarcxmlCollectionWriter(output_path) as writer:
        for entry in entries:
            if max_dumps is not None and dumps_processed >= max_dumps:
                break
            dump_scanned = 0
            dump_kept = 0
            for record in stream_dump_records(entry):
                dump_scanned += 1
                reason = _drop_reason(record, min_year, languages)
                if reason is None:
                    writer.write(record)
                    dump_kept += 1
                else:
                    dropped_by_reason[reason] += 1
                record.clear()
            dumps_processed += 1
            records_scanned += dump_scanned
            kept += dump_kept
            _LOGGER.info(
                "dump done: scanned=%d kept=%d running_kept=%d",
                dump_scanned,
                dump_kept,
                kept,
            )
        kept = writer.records_written
    dropped = records_scanned - kept
    _LOGGER.info(
        "corpus complete: dumps=%d scanned=%d kept=%d dropped=%d -> %s",
        dumps_processed,
        records_scanned,
        kept,
        dropped,
        output_path,
    )
    return CorpusReport(
        dumps_processed=dumps_processed,
        records_scanned=records_scanned,
        kept=kept,
        dropped=dropped,
        dropped_by_reason=dict(dropped_by_reason),
    )
