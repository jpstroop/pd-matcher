"""Streaming acquisition orchestration.

Each dump is downloaded to a temporary ``.tar.gz`` (with its md5 verified
against the manifest), opened in streaming gzip mode, and its single MARCXML
member is parsed incrementally with ``iterparse`` so that neither the
compressed archive nor the multi-hundred-megabyte member is ever fully
materialized in memory. Eligible records are routed by 008 language code to a
per-language shard writer until every configured language has reached its cap.
"""

from collections.abc import Iterator
from collections.abc import Mapping
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
from pd_groundtruth.filters import language_of
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
from pd_groundtruth.manifest import DumpEntry
from pd_groundtruth.manifest import fetch_manifest
from pd_groundtruth.writer import MarcxmlShardWriter

_LOGGER = getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 20
_REQUEST_TIMEOUT_SECONDS = 300


class AcquireReport(Struct, frozen=True):
    """Outcome of an acquisition run."""

    dumps_processed: int
    records_scanned: int
    kept_by_language: dict[str, int]
    shards_written: int
    stopped_reason: str


class Md5MismatchError(RuntimeError):
    """Raised when a downloaded dump's md5 does not match the manifest."""


def acquire(
    *,
    out_dir: Path,
    caps: Mapping[str, int],
    manifest_url: str = DEFAULT_MANIFEST_URL,
    max_dumps: int | None = None,
) -> AcquireReport:
    """Stream Princeton dumps and write eligible records as per-language shards.

    Args:
        out_dir: Root directory; each language writes into ``out_dir/<lang>/``.
        caps: Mapping of 008 language code to the maximum records to keep for
            that language. Languages absent from this mapping are never kept.
        manifest_url: Absolute URL of the dump manifest JSON.
        max_dumps: Optional cap on the number of dumps processed.

    Returns:
        A report of dumps processed, records scanned, records kept per language,
        total shards written, and why the run stopped.
    """
    entries = fetch_manifest(manifest_url)
    return _acquire_entries(
        entries,
        out_dir=out_dir,
        caps=caps,
        max_dumps=max_dumps,
    )


def _acquire_entries(
    entries: tuple[DumpEntry, ...],
    *,
    out_dir: Path,
    caps: Mapping[str, int],
    max_dumps: int | None,
) -> AcquireReport:
    """Run acquisition over an already-resolved set of dump entries."""
    dumps_processed = 0
    records_scanned = 0
    kept_by_language: dict[str, int] = dict.fromkeys(caps, 0)
    full_languages: set[str] = set()
    stopped_reason = "dumps_exhausted"

    writers = {language: MarcxmlShardWriter(out_dir / language) for language in caps}
    try:
        for entry in entries:
            if max_dumps is not None and dumps_processed >= max_dumps:
                stopped_reason = "max_dumps"
                break
            if _all_full(kept_by_language, caps):
                stopped_reason = "caps_reached"
                break

            scanned = _process_dump(
                entry,
                writers=writers,
                caps=caps,
                kept_by_language=kept_by_language,
            )
            dumps_processed += 1
            records_scanned += scanned
            _log_newly_full(kept_by_language, caps, full_languages)
            _LOGGER.info(
                "dump done: scanned=%d running_total=%d %s%s",
                scanned,
                records_scanned,
                _format_progress(kept_by_language, caps),
                _format_full(full_languages),
            )
            if _all_full(kept_by_language, caps):
                stopped_reason = "caps_reached"
                break
    finally:
        shards_written = 0
        for writer in writers.values():
            writer.close()
            shards_written += writer.shards_written

    _LOGGER.info(
        "acquisition complete: dumps=%d scanned=%d shards=%d reason=%s %s",
        dumps_processed,
        records_scanned,
        shards_written,
        stopped_reason,
        _format_progress(kept_by_language, caps),
    )

    return AcquireReport(
        dumps_processed=dumps_processed,
        records_scanned=records_scanned,
        kept_by_language=kept_by_language,
        shards_written=shards_written,
        stopped_reason=stopped_reason,
    )


def _all_full(kept_by_language: Mapping[str, int], caps: Mapping[str, int]) -> bool:
    """Return whether every configured language has reached its cap."""
    return all(kept_by_language[language] >= cap for language, cap in caps.items())


def _format_progress(kept_by_language: Mapping[str, int], caps: Mapping[str, int]) -> str:
    """Render ``lang=kept/cap`` pairs for every configured language."""
    return " ".join(
        f"{language}={kept_by_language[language]}/{caps[language]}" for language in caps
    )


def _format_full(full_languages: set[str]) -> str:
    """Render a trailing ``full=[...]`` segment when any language is full."""
    if not full_languages:
        return ""
    return " full=[" + ",".join(sorted(full_languages)) + "]"


def _log_newly_full(
    kept_by_language: Mapping[str, int],
    caps: Mapping[str, int],
    full_languages: set[str],
) -> None:
    """Log each language that newly reached its cap since the last check."""
    for language, cap in caps.items():
        if language not in full_languages and kept_by_language[language] >= cap:
            full_languages.add(language)
            _LOGGER.info("language full: %s reached cap %d", language, cap)


def _process_dump(
    entry: DumpEntry,
    *,
    writers: Mapping[str, MarcxmlShardWriter],
    caps: Mapping[str, int],
    kept_by_language: dict[str, int],
) -> int:
    """Download, verify, and scan a single dump, returning the records scanned."""
    temp_path = _download_and_verify(entry)
    try:
        scanned = 0
        for record in _iter_records(temp_path):
            scanned += 1
            if is_eligible(record):
                language = language_of(record)
                if (
                    language is not None
                    and language in caps
                    and kept_by_language[language] < caps[language]
                ):
                    writers[language].write(record)
                    kept_by_language[language] += 1
            record.clear()
        return scanned
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
