"""Streaming acquisition orchestration.

Each dump is downloaded to a temporary ``.tar.gz`` (with its md5 verified
against the manifest), opened in streaming gzip mode, and its single MARCXML
member is parsed incrementally with ``iterparse`` so that neither the
compressed archive nor the multi-hundred-megabyte member is ever fully
materialized in memory. Eligible records are routed by 008 language code to a
per-language shard writer, but sampling is constrained by a per-(language,
decade) quota so that no single decade can dominate a language's slice of the
corpus.
"""

from collections.abc import Iterator
from datetime import date
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

from pd_groundtruth.filters import _CCE_MAX_YEAR
from pd_groundtruth.filters import is_eligible
from pd_groundtruth.filters import language_of
from pd_groundtruth.filters import year_of
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
from pd_groundtruth.manifest import DumpEntry
from pd_groundtruth.manifest import fetch_manifest
from pd_groundtruth.writer import MarcxmlShardWriter

_LOGGER = getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 20
_REQUEST_TIMEOUT_SECONDS = 300

_TARGET_LANGUAGES = ("eng", "fre", "ger", "spa", "ita")
_MOVING_WALL_AGE = 95


def default_min_year() -> int:
    """Return the moving-wall lower bound (``today.year - 95``)."""
    return date.today().year - _MOVING_WALL_AGE


def _decade_of(year: int) -> int:
    """Return the decade bucket for a year (e.g. 1953 -> 1950)."""
    return (year // 10) * 10


def _decade_buckets(min_year: int) -> tuple[int, ...]:
    """Return the ascending decade buckets spanned by ``min_year..1977``."""
    first = _decade_of(min_year)
    last = _decade_of(_CCE_MAX_YEAR)
    return tuple(range(first, last + 10, 10))


class AcquireReport(Struct, frozen=True):
    """Outcome of an acquisition run."""

    dumps_processed: int
    records_scanned: int
    kept_by_language_decade: dict[str, dict[int, int]]
    kept_by_language: dict[str, int]
    shards_written: int
    stopped_reason: str


class Md5MismatchError(RuntimeError):
    """Raised when a downloaded dump's md5 does not match the manifest."""


def acquire(
    *,
    out_dir: Path,
    per_decade_cap: int,
    min_year: int,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    max_dumps: int | None = None,
) -> AcquireReport:
    """Stream Princeton dumps and write eligible records as per-language shards.

    Args:
        out_dir: Root directory; each language writes into ``out_dir/<lang>/``.
        per_decade_cap: Maximum records to keep per (target language, decade)
            bucket. A record is kept only while its ``(language, decade)`` bucket
            is below this quota.
        min_year: Inclusive lower bound for the publication year (the moving
            wall). Also fixes the set of decade buckets (``min_year``..1977).
        manifest_url: Absolute URL of the dump manifest JSON.
        max_dumps: Optional cap on the number of dumps processed.

    Returns:
        A report of dumps processed, records scanned, records kept per
        (language, decade) and per language, total shards written, and why the
        run stopped.
    """
    entries = fetch_manifest(manifest_url)
    return _acquire_entries(
        entries,
        out_dir=out_dir,
        per_decade_cap=per_decade_cap,
        min_year=min_year,
        max_dumps=max_dumps,
    )


def _acquire_entries(
    entries: tuple[DumpEntry, ...],
    *,
    out_dir: Path,
    per_decade_cap: int,
    min_year: int,
    max_dumps: int | None,
) -> AcquireReport:
    """Run acquisition over an already-resolved set of dump entries."""
    decades = _decade_buckets(min_year)
    kept: dict[str, dict[int, int]] = {
        language: dict.fromkeys(decades, 0) for language in _TARGET_LANGUAGES
    }
    full_buckets: set[tuple[str, int]] = set()
    dumps_processed = 0
    records_scanned = 0
    stopped_reason = "dumps_exhausted"

    writers = {language: MarcxmlShardWriter(out_dir / language) for language in _TARGET_LANGUAGES}
    try:
        for entry in entries:
            if max_dumps is not None and dumps_processed >= max_dumps:
                stopped_reason = "max_dumps"
                break
            if _all_full(kept, per_decade_cap):
                stopped_reason = "quotas_reached"
                break

            scanned = _process_dump(
                entry,
                writers=writers,
                kept=kept,
                per_decade_cap=per_decade_cap,
                min_year=min_year,
            )
            dumps_processed += 1
            records_scanned += scanned
            _log_newly_full(kept, per_decade_cap, full_buckets)
            _LOGGER.info(
                "dump done: scanned=%d running_total=%d %s",
                scanned,
                records_scanned,
                _format_progress(kept, per_decade_cap),
            )
            if _all_full(kept, per_decade_cap):
                stopped_reason = "quotas_reached"
                break
    finally:
        shards_written = 0
        for writer in writers.values():
            writer.close()
            shards_written += writer.shards_written

    kept_by_language = {language: sum(buckets.values()) for language, buckets in kept.items()}
    _LOGGER.info(
        "acquisition complete: dumps=%d scanned=%d shards=%d reason=%s\n%s",
        dumps_processed,
        records_scanned,
        shards_written,
        stopped_reason,
        _format_summary(kept, per_decade_cap),
    )

    return AcquireReport(
        dumps_processed=dumps_processed,
        records_scanned=records_scanned,
        kept_by_language_decade=kept,
        kept_by_language=kept_by_language,
        shards_written=shards_written,
        stopped_reason=stopped_reason,
    )


def _all_full(kept: dict[str, dict[int, int]], per_decade_cap: int) -> bool:
    """Return whether every (target language, decade) bucket is at the quota."""
    return all(count >= per_decade_cap for buckets in kept.values() for count in buckets.values())


def _format_progress(kept: dict[str, dict[int, int]], per_decade_cap: int) -> str:
    """Render every target language's per-decade fill on one line.

    Each language is shown with its per-decade buckets (not just a running
    total), since all languages are decade-bucketed identically; segments
    are separated by ``|`` to keep the single-line dump-progress readable.
    """
    segments = [
        f"{language} "
        + " ".join(
            f"[{decade}]={count}/{per_decade_cap}" for decade, count in kept[language].items()
        )
        for language in _TARGET_LANGUAGES
    ]
    return " | ".join(segments)


def _format_summary(kept: dict[str, dict[int, int]], per_decade_cap: int) -> str:
    """Render a per-language, per-decade fill table."""
    lines: list[str] = []
    for language in _TARGET_LANGUAGES:
        cells = " ".join(
            f"[{decade}]={count}/{per_decade_cap}" for decade, count in kept[language].items()
        )
        lines.append(f"  {language}: {cells} total={sum(kept[language].values())}")
    return "\n".join(lines)


def _log_newly_full(
    kept: dict[str, dict[int, int]],
    per_decade_cap: int,
    full_buckets: set[tuple[str, int]],
) -> None:
    """Log each (language, decade) bucket that newly reached the quota."""
    for language, buckets in kept.items():
        for decade, count in buckets.items():
            key = (language, decade)
            if key not in full_buckets and count >= per_decade_cap:
                full_buckets.add(key)
                _LOGGER.info(
                    "bucket full: %s[%d] reached quota %d", language, decade, per_decade_cap
                )


def _process_dump(
    entry: DumpEntry,
    *,
    writers: dict[str, MarcxmlShardWriter],
    kept: dict[str, dict[int, int]],
    per_decade_cap: int,
    min_year: int,
) -> int:
    """Download, verify, and scan a single dump, returning the records scanned."""
    temp_path = _download_and_verify(entry)
    try:
        scanned = 0
        for record in _iter_records(temp_path):
            scanned += 1
            if is_eligible(record, min_year):
                language = language_of(record)
                year = year_of(record)
                if (  # pragma: no branch  # is_eligible already validates these
                    language is not None and language in kept and year is not None
                ):
                    decade = _decade_of(year)
                    if kept[language][decade] < per_decade_cap:
                        writers[language].write(record)
                        kept[language][decade] += 1
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
            if chunk:  # pragma: no branch  # responses mock never yields empty keepalive chunks
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
            if (
                fileobj is None
            ):  # pragma: no cover  # isfile() above already excludes non-regular members
                continue
            for _event, element in iterparse(fileobj, events=("end",)):
                if QName(element).localname == "record":
                    yield element
