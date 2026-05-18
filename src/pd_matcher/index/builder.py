"""Build the LMDB index from streaming parser output.

The builder owns the orchestration that turns the on-disk CCE corpus
(the U.S. Copyright Office's Catalog of Copyright Entries, published by
the Library of Congress and transcribed into XML/TSV by NYPL) into the
LMDB env that Phase 4's matcher reads. Renewals stream first so the
``ren_by_oreg`` lookup is fully populated by the time registrations are
ingested; that way every :class:`IndexedNyplRegRecord` can have its
``was_renewed`` flag resolved with a single ``ren_by_oreg.get`` call instead
of a separate join pass at the end.

Idempotency is achieved by hashing source-file paths, sizes, and mtimes
(content hashes are too slow on the 1.4 GB registration corpus). When the
hash plus the schema version stored in the ``meta`` sub-DB match the current
source tree the builder short-circuits and returns ``BuildReport.skipped =
True``; passing ``force=True`` bypasses the check and rebuilds in place.
"""

from datetime import UTC
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from time import perf_counter

from msgspec import Struct
from structlog import get_logger
from structlog.contextvars import bind_contextvars
from structlog.contextvars import unbind_contextvars

from pd_matcher.index.codec import encode_reg
from pd_matcher.index.codec import encode_ren
from pd_matcher.index.codec import encode_uuid_list
from pd_matcher.index.codec import encode_year_key
from pd_matcher.index.codec import make_renewal_key
from pd_matcher.index.store import NyplIndexStore
from pd_matcher.models import index_reg
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory

_LOGGER = get_logger(__name__)

_META_SCHEMA_VERSION_KEY = b"schema_version"
_META_SOURCE_HASH_KEY = b"source_hash"
_META_BUILD_TIMESTAMP_KEY = b"build_timestamp"
_META_REG_COUNT_KEY = b"registrations_written"
_META_REN_COUNT_KEY = b"renewals_written"
_META_RENEWAL_JOINS_KEY = b"renewal_joins"
_META_YEAR_BUCKETS_KEY = b"year_buckets"


class BuildReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of one :func:`build_index` invocation.

    ``skipped`` is ``True`` when the on-disk env is already current with the
    sources (matching schema version and source hash). The numeric counters
    reflect a fresh build; they remain zero on a skipped run.
    """

    skipped: bool
    registrations_written: int
    renewals_written: int
    renewal_joins: int
    year_buckets: int
    duration_seconds: float


def _hash_directory(root: Path, suffix: str) -> str:
    """Compute a deterministic hash of files beneath ``root`` matching ``suffix``.

    Hash inputs are (path-relative-to-root, size, mtime-ns) tuples sorted in
    a stable order. Full content hashing would be authoritative but is too
    slow on the 1.4 GB registration tree; size+mtime is the standard cheap
    proxy and is fine for cache invalidation against a curated submodule.
    """
    hasher = sha256()
    paths = sorted(root.rglob(f"*{suffix}"))
    for path in paths:
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        chunk = f"{relative}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        hasher.update(chunk)
    return hasher.hexdigest()


def _compute_source_hash(reg_dir: Path, ren_dir: Path) -> str:
    """Combine the reg and ren directory hashes into one source fingerprint."""
    hasher = sha256()
    hasher.update(_hash_directory(reg_dir, ".xml").encode("ascii"))
    hasher.update(b"|")
    hasher.update(_hash_directory(ren_dir, ".tsv").encode("ascii"))
    return hasher.hexdigest()


def _read_existing_meta(store: NyplIndexStore) -> tuple[int | None, str | None]:
    """Return ``(schema_version, source_hash)`` from an existing env, if any."""
    schema_blob = store.meta.get(_META_SCHEMA_VERSION_KEY)
    hash_blob = store.meta.get(_META_SOURCE_HASH_KEY)
    schema_version = int(schema_blob.decode("ascii")) if schema_blob is not None else None
    source_hash = hash_blob.decode("ascii") if hash_blob is not None else None
    return schema_version, source_hash


def _is_current(out_path: Path, expected_hash: str, schema_version: int) -> bool:
    """Return ``True`` when ``out_path`` already holds a matching build."""
    if not out_path.exists():
        return False
    with NyplIndexStore(out_path, readonly=True) as readonly_store:
        existing_schema, existing_hash = _read_existing_meta(readonly_store)
    return existing_schema == schema_version and existing_hash == expected_hash


def _ingest_renewals(
    store: NyplIndexStore,
    ren_dir: Path,
) -> int:
    """Stream renewals into ``ren_by_id`` and ``ren_by_oreg``; return count."""
    bind_contextvars(phase="renewals")
    try:
        written = 0
        with store.write_transaction():
            for record in iter_nypl_ren_directory(ren_dir):
                store.ren_by_id.put(record.entry_id.encode("utf-8"), encode_ren(record))
                if record.oreg is not None and record.odat is not None:
                    join_key = make_renewal_key(record.oreg, record.odat)
                    store.ren_by_oreg.put(join_key, record.entry_id.encode("utf-8"))
                written += 1
        _LOGGER.info("index.renewals.ingested", count=written)
        return written
    finally:
        unbind_contextvars("phase")


def _ingest_registrations(
    store: NyplIndexStore,
    reg_dir: Path,
) -> tuple[int, int, dict[int, list[str]]]:
    """Stream registrations into ``reg_by_id`` and collect year buckets.

    Returns a tuple of ``(records_written, renewal_joins, year_buckets)``.
    Records with no ``reg_year`` are skipped from the year-bucket index but
    still written to ``reg_by_id`` so they can be looked up by uuid.
    """
    bind_contextvars(phase="registrations")
    try:
        written = 0
        joins = 0
        year_buckets: dict[int, list[str]] = {}
        with store.write_transaction():
            for record in iter_nypl_reg_directory(reg_dir):
                was_renewed = False
                if record.regnum is not None:
                    join_key = make_renewal_key(record.regnum, record.reg_date)
                    was_renewed = store.ren_by_oreg.get(join_key) is not None
                if was_renewed:
                    joins += 1
                indexed = index_reg(record, was_renewed=was_renewed)
                store.reg_by_id.put(record.uuid.encode("utf-8"), encode_reg(indexed))
                if record.reg_year is not None:
                    year_buckets.setdefault(record.reg_year, []).append(record.uuid)
                written += 1
        _LOGGER.info(
            "index.registrations.ingested",
            count=written,
            renewal_joins=joins,
            year_buckets=len(year_buckets),
        )
        return written, joins, year_buckets
    finally:
        unbind_contextvars("phase")


def _flush_year_buckets(store: NyplIndexStore, year_buckets: dict[int, list[str]]) -> int:
    """Write each year bucket to ``reg_by_year`` and return the bucket count."""
    bind_contextvars(phase="year_buckets")
    try:
        with store.write_transaction():
            for year in sorted(year_buckets):
                uuids = tuple(year_buckets[year])
                store.reg_by_year.put(encode_year_key(year), encode_uuid_list(uuids))
        bucket_count = len(year_buckets)
        _LOGGER.info("index.year_buckets.flushed", count=bucket_count)
        return bucket_count
    finally:
        unbind_contextvars("phase")


def _write_meta(
    store: NyplIndexStore,
    *,
    schema_version: int,
    source_hash: str,
    registrations: int,
    renewals: int,
    renewal_joins: int,
    year_buckets: int,
) -> None:
    """Persist the build metadata used by lookups and the info CLI."""
    bind_contextvars(phase="meta")
    try:
        timestamp = datetime.now(tz=UTC).isoformat()
        with store.write_transaction():
            store.meta.put(_META_SCHEMA_VERSION_KEY, str(schema_version).encode("ascii"))
            store.meta.put(_META_SOURCE_HASH_KEY, source_hash.encode("ascii"))
            store.meta.put(_META_BUILD_TIMESTAMP_KEY, timestamp.encode("ascii"))
            store.meta.put(_META_REG_COUNT_KEY, str(registrations).encode("ascii"))
            store.meta.put(_META_REN_COUNT_KEY, str(renewals).encode("ascii"))
            store.meta.put(_META_RENEWAL_JOINS_KEY, str(renewal_joins).encode("ascii"))
            store.meta.put(_META_YEAR_BUCKETS_KEY, str(year_buckets).encode("ascii"))
        _LOGGER.info(
            "index.meta.written",
            schema_version=schema_version,
            registrations=registrations,
            renewals=renewals,
            renewal_joins=renewal_joins,
            year_buckets=year_buckets,
        )
    finally:
        unbind_contextvars("phase")


def _purge_directory(path: Path) -> None:
    """Remove ``path`` and everything beneath it if it exists."""
    if path.exists():
        rmtree(path)


def build_index(
    *,
    reg_dir: Path,
    ren_dir: Path,
    out_path: Path,
    schema_version: int = 1,
    force: bool = False,
) -> BuildReport:
    """Materialise the LMDB index from the two CCE source directories.

    Args:
        reg_dir: Directory holding the CCE registration XML tree
            (NYPL transcription).
        ren_dir: Directory holding the CCE renewal TSV files
            (NYPL transcription).
        out_path: LMDB env directory to create or overwrite.
        schema_version: Bumped whenever the stored record shape changes.
        force: When ``True`` always rebuild even if the existing env matches.

    Returns:
        A :class:`BuildReport` describing the outcome.
    """
    start = perf_counter()
    source_hash = _compute_source_hash(reg_dir, ren_dir)

    if not force and _is_current(out_path, source_hash, schema_version):
        _LOGGER.info(
            "index.build.skipped",
            reason="source_hash_match",
            out_path=str(out_path),
        )
        return BuildReport(
            skipped=True,
            registrations_written=0,
            renewals_written=0,
            renewal_joins=0,
            year_buckets=0,
            duration_seconds=perf_counter() - start,
        )

    _purge_directory(out_path)

    with NyplIndexStore(out_path, readonly=False) as store:
        renewals_written = _ingest_renewals(store, ren_dir)
        registrations_written, renewal_joins, year_buckets = _ingest_registrations(store, reg_dir)
        bucket_count = _flush_year_buckets(store, year_buckets)
        _write_meta(
            store,
            schema_version=schema_version,
            source_hash=source_hash,
            registrations=registrations_written,
            renewals=renewals_written,
            renewal_joins=renewal_joins,
            year_buckets=bucket_count,
        )

    duration = perf_counter() - start
    _LOGGER.info(
        "index.build.complete",
        registrations=registrations_written,
        renewals=renewals_written,
        renewal_joins=renewal_joins,
        year_buckets=bucket_count,
        duration_seconds=duration,
    )
    return BuildReport(
        skipped=False,
        registrations_written=registrations_written,
        renewals_written=renewals_written,
        renewal_joins=renewal_joins,
        year_buckets=bucket_count,
        duration_seconds=duration,
    )


__all__ = [
    "BuildReport",
    "build_index",
]
