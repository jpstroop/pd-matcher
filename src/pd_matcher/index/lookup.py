"""Read-only query API over the LMDB-backed NYPL index.

Phase 4's matcher and Phase 7's ``pd-matcher index info`` CLI consume the
index through :class:`NyplIndexLookup`; it opens the underlying
:class:`NyplIndexStore` in read-only mode and exposes a small,
caller-friendly API. Read-only mode is the LMDB sweet spot — many worker
processes can mmap the same env directory without any contention, which is
the whole reason this project picked LMDB.
"""

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Self

from msgspec import Struct

from pd_matcher.index.codec import decode_reg
from pd_matcher.index.codec import decode_ren
from pd_matcher.index.codec import decode_uuid_list
from pd_matcher.index.codec import encode_year_key
from pd_matcher.index.store import NyplIndexStore
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import NyplRenRecord

_META_SCHEMA_VERSION_KEY = b"schema_version"
_META_SOURCE_HASH_KEY = b"source_hash"
_META_BUILD_TIMESTAMP_KEY = b"build_timestamp"
_META_REG_COUNT_KEY = b"registrations_written"
_META_REN_COUNT_KEY = b"renewals_written"
_META_RENEWAL_JOINS_KEY = b"renewal_joins"
_META_YEAR_BUCKETS_KEY = b"year_buckets"


class IndexStats(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of an existing on-disk index, returned by :meth:`stats`."""

    schema_version: int
    source_hash: str
    build_timestamp: str
    registrations_written: int
    renewals_written: int
    renewal_joins: int
    year_buckets: int


class NyplIndexLookup:
    """Read-only LMDB lookup over the indexed NYPL corpus.

    The store is opened read-only on construction and held for the lifetime
    of the instance; callers should use the class as a context manager to
    guarantee the underlying env handle is released.
    """

    __slots__ = ("_store",)

    def __init__(self, path: Path) -> None:
        self._store = NyplIndexStore(path, readonly=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying LMDB env."""
        self._store.close()

    def get_registration(self, uuid: str) -> IndexedNyplRegRecord | None:
        """Return the indexed registration for ``uuid`` or ``None``."""
        blob = self._store.reg_by_id.get(uuid.encode("utf-8"))
        if blob is None:
            return None
        return decode_reg(blob)

    def get_renewal(self, entry_id: str) -> NyplRenRecord | None:
        """Return the renewal record for ``entry_id`` or ``None``."""
        blob = self._store.ren_by_id.get(entry_id.encode("utf-8"))
        if blob is None:
            return None
        return decode_ren(blob)

    def iter_registrations(self) -> Iterator[IndexedNyplRegRecord]:
        """Yield every registration in the index in storage order.

        Used by Phase 4's IDF builder to scan the entire corpus once and
        compute per-token document frequencies. The walk is read-only and
        streams from a single LMDB cursor, so it is safe to invoke while
        other readers are open.
        """
        for _key, blob in self._store.reg_by_id.iter_items():
            yield decode_reg(blob)

    def candidates_for_year(
        self,
        year: int,
        window: int = 0,
    ) -> Iterator[IndexedNyplRegRecord]:
        """Yield every registration whose ``reg_year`` is in ``[year-window, year+window]``.

        Args:
            year: Centre of the year window.
            window: Inclusive radius; ``0`` yields just ``year`` itself.

        Yields:
            :class:`IndexedNyplRegRecord` instances, deduplicated by uuid in
            case a registration somehow appears in multiple buckets.
        """
        seen: set[str] = set()
        for candidate_year in range(year - window, year + window + 1):
            blob = self._store.reg_by_year.get(encode_year_key(candidate_year))
            if blob is None:
                continue
            for uuid in decode_uuid_list(blob):
                if uuid in seen:
                    continue
                seen.add(uuid)
                record = self.get_registration(uuid)
                if record is not None:
                    yield record

    def stats(self) -> IndexStats:
        """Return the build metadata persisted alongside the index.

        Raises:
            RuntimeError: If the env does not contain a complete metadata
                record (i.e. it was not produced by
                :func:`pd_matcher.index.builder.build_index`).
        """
        meta = self._store.meta
        schema_blob = meta.get(_META_SCHEMA_VERSION_KEY)
        hash_blob = meta.get(_META_SOURCE_HASH_KEY)
        timestamp_blob = meta.get(_META_BUILD_TIMESTAMP_KEY)
        reg_blob = meta.get(_META_REG_COUNT_KEY)
        ren_blob = meta.get(_META_REN_COUNT_KEY)
        joins_blob = meta.get(_META_RENEWAL_JOINS_KEY)
        buckets_blob = meta.get(_META_YEAR_BUCKETS_KEY)
        if (
            schema_blob is None
            or hash_blob is None
            or timestamp_blob is None
            or reg_blob is None
            or ren_blob is None
            or joins_blob is None
            or buckets_blob is None
        ):
            raise RuntimeError("index meta sub-DB is incomplete; rebuild required")
        return IndexStats(
            schema_version=int(schema_blob.decode("ascii")),
            source_hash=hash_blob.decode("ascii"),
            build_timestamp=timestamp_blob.decode("ascii"),
            registrations_written=int(reg_blob.decode("ascii")),
            renewals_written=int(ren_blob.decode("ascii")),
            renewal_joins=int(joins_blob.decode("ascii")),
            year_buckets=int(buckets_blob.decode("ascii")),
        )


__all__ = [
    "IndexStats",
    "NyplIndexLookup",
]
