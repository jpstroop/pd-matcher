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
(content hashes are too slow on the 1.4 GB registration corpus), and by
hashing the bytes of every parser/model/codec module that contributes to the
stored record shape. When the source hash, the schema version, and the
parser fingerprint stored in the ``meta`` sub-DB all match the current
source tree and code the builder short-circuits and returns
``BuildReport.skipped = True``; passing ``force=True`` bypasses the check
and rebuilds in place. Any drift in the parsers, in ``models.py``, in the
LMDB codec, or in this module itself invalidates the cache automatically so
nobody has to remember to bump ``schema_version`` for a code-only change.
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

from pd_matcher.index.codec import decode_ren
from pd_matcher.index.codec import encode_reg
from pd_matcher.index.codec import encode_ren
from pd_matcher.index.codec import encode_uuid_list
from pd_matcher.index.codec import encode_year_key
from pd_matcher.index.codec import make_renewal_key
from pd_matcher.index.keys import author_keys
from pd_matcher.index.keys import publisher_keys
from pd_matcher.index.keys import title_keys
from pd_matcher.index.store import NyplIndexStore
from pd_matcher.index.store import _SubDb
from pd_matcher.models import index_reg
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory

_LOGGER = get_logger(__name__)

_META_SCHEMA_VERSION_KEY = b"schema_version"
_META_SOURCE_HASH_KEY = b"source_hash"
_META_PARSER_FINGERPRINT_KEY = b"parser_fingerprint"
_META_BUILD_TIMESTAMP_KEY = b"build_timestamp"
_META_REG_COUNT_KEY = b"registrations_written"
_META_REN_COUNT_KEY = b"renewals_written"
_META_RENEWAL_JOINS_KEY = b"renewal_joins"
_META_YEAR_BUCKETS_KEY = b"year_buckets"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PARSER_FINGERPRINT_FILES: tuple[Path, ...] = (
    _PACKAGE_ROOT / "parsers" / "nypl_reg.py",
    _PACKAGE_ROOT / "parsers" / "nypl_ren.py",
    _PACKAGE_ROOT / "models.py",
    _PACKAGE_ROOT / "index" / "codec.py",
    _PACKAGE_ROOT / "index" / "builder.py",
)


class _IngestResult(Struct, frozen=True, forbid_unknown_fields=True):
    """Aggregates collected during a single registration ingest pass.

    ``year_buckets`` and the three token-postings maps are accumulated in
    memory while ``reg_by_id`` is written, then flushed to their respective
    sub-DBs once the streaming pass completes.
    """

    written: int
    joins: int
    year_buckets: dict[int, list[str]]
    title_postings: dict[str, list[str]]
    author_postings: dict[str, list[str]]
    publisher_postings: dict[str, list[str]]


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


def _compute_parser_fingerprint() -> str:
    """Return a sha256 over the bytes of every parser/model/codec module.

    The fingerprint protects against silent staleness when the source XML/TSV
    files are unchanged but the code that turns them into stored records has
    drifted (a new field on :class:`NyplRegRecord`, a tweak to
    :func:`index_reg`, an encoder change, ...). Each file's bytes are sha256'd
    and the per-file digests are concatenated as ``path|hexdigest\\n`` in
    path-sorted order before the final sha256, so the result is deterministic
    and stable across check-outs of the same code at the same paths.
    """
    hasher = sha256()
    for path in sorted(_PARSER_FINGERPRINT_FILES):
        relative = path.relative_to(_PACKAGE_ROOT).as_posix()
        file_digest = sha256(path.read_bytes()).hexdigest()
        hasher.update(f"{relative}|{file_digest}\n".encode("ascii"))
    return hasher.hexdigest()


class _ExistingMeta(Struct, frozen=True, forbid_unknown_fields=True):
    """Snapshot of cache-validity inputs read from an existing index env."""

    schema_version: int | None
    source_hash: str | None
    parser_fingerprint: str | None


def _read_existing_meta(store: NyplIndexStore) -> _ExistingMeta:
    """Return the cache-validity inputs persisted in ``store``'s ``meta`` sub-DB."""
    schema_blob = store.meta.get(_META_SCHEMA_VERSION_KEY)
    hash_blob = store.meta.get(_META_SOURCE_HASH_KEY)
    fingerprint_blob = store.meta.get(_META_PARSER_FINGERPRINT_KEY)
    return _ExistingMeta(
        schema_version=int(schema_blob.decode("ascii")) if schema_blob is not None else None,
        source_hash=hash_blob.decode("ascii") if hash_blob is not None else None,
        parser_fingerprint=(
            fingerprint_blob.decode("ascii") if fingerprint_blob is not None else None
        ),
    )


def _cache_mismatch_reason(
    existing: _ExistingMeta,
    *,
    expected_source_hash: str,
    expected_schema_version: int,
    expected_parser_fingerprint: str,
) -> str | None:
    """Return the first mismatched cache key, or ``None`` when all match.

    Source-hash drift is reported first because it is by far the most common
    cause of a rebuild on a developer's box; schema version and parser
    fingerprint follow in declaration order. Missing keys (e.g., an index
    written before the parser fingerprint existed) collapse to the relevant
    ``*_missing`` reason so the rebuild log line is self-explanatory.
    """
    if existing.source_hash is None:
        return "source_hash_missing"
    if existing.source_hash != expected_source_hash:
        return "source_hash_changed"
    if existing.schema_version is None:
        return "schema_version_missing"
    if existing.schema_version != expected_schema_version:
        return "schema_version_changed"
    if existing.parser_fingerprint is None:
        return "parser_fingerprint_missing"
    if existing.parser_fingerprint != expected_parser_fingerprint:
        return "parser_fingerprint_changed"
    return None


def _check_cache(
    out_path: Path,
    *,
    expected_source_hash: str,
    expected_schema_version: int,
    expected_parser_fingerprint: str,
) -> str | None:
    """Return ``None`` when the on-disk env is current, else a mismatch reason."""
    if not out_path.exists():
        return "no_existing_env"
    with NyplIndexStore(out_path, readonly=True) as readonly_store:
        existing = _read_existing_meta(readonly_store)
    return _cache_mismatch_reason(
        existing,
        expected_source_hash=expected_source_hash,
        expected_schema_version=expected_schema_version,
        expected_parser_fingerprint=expected_parser_fingerprint,
    )


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


def _accumulate_postings(
    postings: dict[str, list[str]],
    tokens: frozenset[str],
    uuid: str,
) -> None:
    """Append ``uuid`` to the posting list of every token in ``tokens``."""
    for token in tokens:
        postings.setdefault(token, []).append(uuid)


def _ingest_registrations(
    store: NyplIndexStore,
    reg_dir: Path,
) -> _IngestResult:
    """Stream registrations into ``reg_by_id`` and collect index aggregates.

    Returns an :class:`_IngestResult` carrying the record/join counts, the
    year buckets, and the three token-postings maps (title, author, and
    publisher). Records with no ``reg_year`` are skipped from the year-bucket
    index but still written to ``reg_by_id`` so they can be looked up by uuid.
    Publisher postings draw tokens from both ``publisher_names`` and
    ``claimants`` so either side of a registration's publisher signal can
    retrieve the record.
    """
    bind_contextvars(phase="registrations")
    try:
        written = 0
        joins = 0
        year_buckets: dict[int, list[str]] = {}
        title_postings: dict[str, list[str]] = {}
        author_postings: dict[str, list[str]] = {}
        publisher_postings: dict[str, list[str]] = {}
        with store.write_transaction():
            for record in iter_nypl_reg_directory(reg_dir):
                was_renewed = False
                renewal = None
                if record.regnum is not None:
                    join_key = make_renewal_key(record.regnum, record.reg_date)
                    entry_id_blob = store.ren_by_oreg.get(join_key)
                    if entry_id_blob is not None:
                        was_renewed = True
                        ren_blob = store.ren_by_id.get(entry_id_blob)
                        if ren_blob is not None:  # pragma: no branch
                            renewal = decode_ren(ren_blob)
                if was_renewed:
                    joins += 1
                indexed = index_reg(record, was_renewed=was_renewed, renewal=renewal)
                store.reg_by_id.put(record.uuid.encode("utf-8"), encode_reg(indexed))
                if record.reg_year is not None:
                    year_buckets.setdefault(record.reg_year, []).append(record.uuid)
                _accumulate_postings(title_postings, title_keys(record.title), record.uuid)
                _accumulate_postings(author_postings, author_keys(record.author_name), record.uuid)
                publisher_tokens: frozenset[str] = frozenset().union(
                    *(publisher_keys(value) for value in record.publisher_names),
                    *(publisher_keys(value) for value in record.claimants),
                )
                _accumulate_postings(publisher_postings, publisher_tokens, record.uuid)
                written += 1
        _LOGGER.info(
            "index.registrations.ingested",
            count=written,
            renewal_joins=joins,
            year_buckets=len(year_buckets),
            title_tokens=len(title_postings),
            author_tokens=len(author_postings),
            publisher_tokens=len(publisher_postings),
        )
        return _IngestResult(
            written=written,
            joins=joins,
            year_buckets=year_buckets,
            title_postings=title_postings,
            author_postings=author_postings,
            publisher_postings=publisher_postings,
        )
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


def _flush_token_index(
    store: NyplIndexStore,
    sub_db: _SubDb,
    postings: dict[str, list[str]],
    *,
    name: str,
) -> int:
    """Write each token's posting list to ``sub_db`` and return the token count.

    Mirrors :func:`_flush_year_buckets`: a single write transaction stores
    ``token (utf-8) -> encode_uuid_list(uuids)`` for every accumulated token.
    """
    bind_contextvars(phase=f"token_index.{name}")
    try:
        with store.write_transaction():
            for token in sorted(postings):
                uuids = tuple(postings[token])
                sub_db.put(token.encode("utf-8"), encode_uuid_list(uuids))
        token_count = len(postings)
        _LOGGER.info("index.token_index.flushed", index=name, count=token_count)
        return token_count
    finally:
        unbind_contextvars("phase")


def _write_meta(
    store: NyplIndexStore,
    *,
    schema_version: int,
    source_hash: str,
    parser_fingerprint: str,
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
            store.meta.put(_META_PARSER_FINGERPRINT_KEY, parser_fingerprint.encode("ascii"))
            store.meta.put(_META_BUILD_TIMESTAMP_KEY, timestamp.encode("ascii"))
            store.meta.put(_META_REG_COUNT_KEY, str(registrations).encode("ascii"))
            store.meta.put(_META_REN_COUNT_KEY, str(renewals).encode("ascii"))
            store.meta.put(_META_RENEWAL_JOINS_KEY, str(renewal_joins).encode("ascii"))
            store.meta.put(_META_YEAR_BUCKETS_KEY, str(year_buckets).encode("ascii"))
        _LOGGER.info(
            "index.meta.written",
            schema_version=schema_version,
            parser_fingerprint=parser_fingerprint,
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
    schema_version: int = 4,
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
    parser_fingerprint = _compute_parser_fingerprint()

    if force:
        mismatch_reason: str | None = "force"
    else:
        mismatch_reason = _check_cache(
            out_path,
            expected_source_hash=source_hash,
            expected_schema_version=schema_version,
            expected_parser_fingerprint=parser_fingerprint,
        )
    if mismatch_reason is None:
        _LOGGER.info(
            "index.build.skipped",
            reason="cache_current",
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

    _LOGGER.info(
        "index.build.rebuilding",
        reason=mismatch_reason,
        out_path=str(out_path),
    )
    _purge_directory(out_path)

    with NyplIndexStore(out_path, readonly=False) as store:
        renewals_written = _ingest_renewals(store, ren_dir)
        ingest = _ingest_registrations(store, reg_dir)
        bucket_count = _flush_year_buckets(store, ingest.year_buckets)
        _flush_token_index(store, store.title_index, ingest.title_postings, name="title")
        _flush_token_index(store, store.author_index, ingest.author_postings, name="author")
        _flush_token_index(
            store, store.publisher_index, ingest.publisher_postings, name="publisher"
        )
        _write_meta(
            store,
            schema_version=schema_version,
            source_hash=source_hash,
            parser_fingerprint=parser_fingerprint,
            registrations=ingest.written,
            renewals=renewals_written,
            renewal_joins=ingest.joins,
            year_buckets=bucket_count,
        )

    duration = perf_counter() - start
    _LOGGER.info(
        "index.build.complete",
        registrations=ingest.written,
        renewals=renewals_written,
        renewal_joins=ingest.joins,
        year_buckets=bucket_count,
        duration_seconds=duration,
    )
    return BuildReport(
        skipped=False,
        registrations_written=ingest.written,
        renewals_written=renewals_written,
        renewal_joins=ingest.joins,
        year_buckets=bucket_count,
        duration_seconds=duration,
    )


__all__ = [
    "BuildReport",
    "build_index",
]
