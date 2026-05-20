"""Streaming acquisition orchestration.

Each dump is downloaded to a temporary ``.tar.gz`` (with its md5 verified
against the manifest), opened in streaming gzip mode, and its single MARCXML
member is parsed incrementally with ``iterparse`` so that neither the
compressed archive nor the multi-hundred-megabyte member is ever fully
materialized in memory. Eligible records flow into the shard writer until the
survivor cap is reached.
"""

from collections.abc import Iterator
from hashlib import md5
from logging import getLogger
from pathlib import Path
from tarfile import open as tar_open
from tempfile import NamedTemporaryFile

from lxml.etree import QName
from lxml.etree import _Element
from lxml.etree import iterparse
from msgspec import Struct
from requests import get

from pd_groundtruth.filters import is_eligible
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
from pd_groundtruth.manifest import DumpEntry
from pd_groundtruth.manifest import fetch_manifest
from pd_groundtruth.writer import MarcxmlShardWriter

_LOGGER = getLogger(__name__)

_DEFAULT_MAX_RECORDS = 50000
_DOWNLOAD_CHUNK_BYTES = 1 << 20
_REQUEST_TIMEOUT_SECONDS = 300


class AcquireReport(Struct, frozen=True):
    """Outcome of an acquisition run."""

    dumps_processed: int
    records_scanned: int
    records_kept: int
    shards_written: int
    stopped_reason: str


class Md5MismatchError(RuntimeError):
    """Raised when a downloaded dump's md5 does not match the manifest."""


def acquire(
    *,
    out_dir: Path,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    max_records: int = _DEFAULT_MAX_RECORDS,
    max_dumps: int | None = None,
) -> AcquireReport:
    """Stream Princeton dumps and write eligible records as MARCXML shards.

    Args:
        out_dir: Directory for the MARCXML shard output.
        manifest_url: Absolute URL of the dump manifest JSON.
        max_records: Stop once this many eligible records have been written.
        max_dumps: Optional cap on the number of dumps processed.

    Returns:
        A report of dumps processed, records scanned/kept, shards written, and
        why the run stopped.
    """
    entries = fetch_manifest(manifest_url)
    return _acquire_entries(
        entries,
        out_dir=out_dir,
        max_records=max_records,
        max_dumps=max_dumps,
    )


def _acquire_entries(
    entries: tuple[DumpEntry, ...],
    *,
    out_dir: Path,
    max_records: int,
    max_dumps: int | None,
) -> AcquireReport:
    """Run acquisition over an already-resolved set of dump entries."""
    dumps_processed = 0
    records_scanned = 0
    records_kept = 0
    stopped_reason = "dumps_exhausted"

    writer = MarcxmlShardWriter(out_dir)
    try:
        for entry in entries:
            if max_dumps is not None and dumps_processed >= max_dumps:
                stopped_reason = "max_dumps"
                break
            if records_kept >= max_records:
                stopped_reason = "max_records"
                break

            scanned, kept = _process_dump(
                entry,
                writer=writer,
                remaining=max_records - records_kept,
            )
            dumps_processed += 1
            records_scanned += scanned
            records_kept += kept
            _LOGGER.info(
                "dump done: scanned=%d kept=%d running_total=%d",
                scanned,
                kept,
                records_kept,
            )
            if records_kept >= max_records:
                stopped_reason = "max_records"
                break
    finally:
        writer.close()

    return AcquireReport(
        dumps_processed=dumps_processed,
        records_scanned=records_scanned,
        records_kept=records_kept,
        shards_written=writer.shards_written,
        stopped_reason=stopped_reason,
    )


def _process_dump(
    entry: DumpEntry,
    *,
    writer: MarcxmlShardWriter,
    remaining: int,
) -> tuple[int, int]:
    """Download, verify, and scan a single dump, returning (scanned, kept)."""
    temp_path = _download_and_verify(entry)
    try:
        scanned = 0
        kept = 0
        for record in _iter_records(temp_path):
            scanned += 1
            if kept < remaining and is_eligible(record):
                writer.write(record)
                kept += 1
            record.clear()
        return scanned, kept
    finally:
        temp_path.unlink(missing_ok=True)


def _download_and_verify(entry: DumpEntry) -> Path:
    """Stream a dump to a temp file, verifying its md5 against the manifest.

    Raises:
        Md5MismatchError: If the computed md5 differs from ``entry.md5``.
    """
    _LOGGER.info("downloading dump: %s", entry.url)
    digest = md5()
    response = get(entry.url, stream=True, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    with NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp:
        temp_path = Path(temp.name)
        for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
            if chunk:
                digest.update(chunk)
                temp.write(chunk)
    computed = digest.hexdigest()
    if computed != entry.md5:
        temp_path.unlink(missing_ok=True)
        raise Md5MismatchError(
            f"md5 mismatch for {entry.url}: expected {entry.md5}, got {computed}"
        )
    return temp_path


def _iter_records(archive_path: Path) -> Iterator[_Element]:
    """Yield ``<record>`` elements from the single member of a gzip tar.

    The archive is opened in streaming mode (``r|gz``); the member's file
    object is fed straight to ``iterparse`` so the member is never extracted to
    disk. Records are matched by local name to tolerate either MARCXML
    serialization.
    """
    with tar_open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            fileobj = archive.extractfile(member)
            if fileobj is None:
                continue
            for _event, element in iterparse(fileobj, events=("end",)):
                if QName(element).localname == "record":
                    yield element
