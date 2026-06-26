"""Read-only query API over the LMDB-backed CCE index.

Phase 4's matcher and Phase 7's ``pd-matcher index info`` CLI consume the
index through :class:`NyplIndexLookup`; it opens the underlying
:class:`NyplIndexStore` in read-only mode and exposes a small,
caller-friendly API. The indexed data is the U.S. Copyright Office's
Catalog of Copyright Entries (CCE), published by the Library of Congress
and transcribed into XML/TSV by NYPL. Read-only mode is the LMDB sweet
spot — many worker processes can mmap the same env directory without any
contention, which is the whole reason this project picked LMDB.
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
from pd_matcher.index.keys import author_keys
from pd_matcher.index.keys import publisher_keys
from pd_matcher.index.keys import title_keys
from pd_matcher.index.store import NyplIndexStore
from pd_matcher.index.store import _SubDb
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord

_META_SCHEMA_VERSION_KEY = b"schema_version"
_META_SOURCE_HASH_KEY = b"source_hash"
_META_BUILD_TIMESTAMP_KEY = b"build_timestamp"
_META_REG_COUNT_KEY = b"registrations_written"
_META_REN_COUNT_KEY = b"renewals_written"
_META_RENEWAL_JOINS_KEY = b"renewal_joins"
_META_YEAR_BUCKETS_KEY = b"year_buckets"
_META_REN_YEAR_BUCKETS_KEY = b"renewal_year_buckets"
_META_REN_TITLE_TOKENS_KEY = b"renewal_title_tokens"
_META_REN_AUTHOR_TOKENS_KEY = b"renewal_author_tokens"
_META_REN_CLAIMANTS_TOKENS_KEY = b"renewal_claimants_tokens"


class IndexStats(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of an existing on-disk index, returned by :meth:`stats`."""

    schema_version: int
    source_hash: str
    build_timestamp: str
    registrations_written: int
    renewals_written: int
    renewal_joins: int
    year_buckets: int
    renewal_year_buckets: int
    renewal_title_tokens: int
    renewal_author_tokens: int
    renewal_claimants_tokens: int


class NyplIndexLookup:
    """Read-only LMDB lookup over the indexed CCE corpus.

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

    def iter_renewals(self) -> Iterator[NyplRenRecord]:
        """Yield every renewal in the index in storage order.

        Mirrors :meth:`iter_registrations` over the ``ren_by_id`` sub-DB so the
        renewal-side IDF builders can scan the entire renewal corpus once. The
        walk is read-only and streams from a single LMDB cursor.
        """
        for _key, blob in self._store.ren_by_id.iter_items():
            yield decode_ren(blob)

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

    def _year_candidates(self, year: int, window: int) -> set[str]:
        """Return the union of uuids in every year bucket across the window."""
        candidates: set[str] = set()
        for candidate_year in range(year - window, year + window + 1):
            blob = self._store.reg_by_year.get(encode_year_key(candidate_year))
            if blob is None:
                continue
            candidates.update(decode_uuid_list(blob))
        return candidates

    @staticmethod
    def _postings(sub_db: _SubDb, token: str) -> tuple[str, ...]:
        """Return the posting list for ``token`` in ``sub_db`` (empty if absent)."""
        blob = sub_db.get(token.encode("utf-8"))
        if blob is None:
            return ()
        return decode_uuid_list(blob)

    def _token_candidates(self, marc: MarcRecord) -> set[str]:
        """Return the union of every registration sharing a query token.

        Title tokens come from ``marc.title``; author tokens from both
        ``marc.main_author`` and ``marc.statement_of_responsibility``;
        publisher tokens from ``marc.publisher``. Each token's posting list
        is fetched from the matching inverted sub-DB and unioned together.
        """
        candidates: set[str] = set()
        for token in title_keys(marc.title):
            candidates.update(self._postings(self._store.title_index, token))
        author_tokens = author_keys(marc.main_author) | author_keys(
            marc.statement_of_responsibility
        )
        for token in author_tokens:
            candidates.update(self._postings(self._store.author_index, token))
        for token in publisher_keys(marc.publisher):
            candidates.update(self._postings(self._store.publisher_index, token))
        return candidates

    def candidates_for(
        self,
        marc: MarcRecord,
        window: int = 0,
    ) -> Iterator[IndexedNyplRegRecord]:
        """Yield registrations sharing a year AND at least one field token.

        Candidate retrieval (cheap) is separated from scoring (expensive):
        rather than scoring an entire year bucket, only registrations that
        both fall inside the year window and share a title/author/publisher
        token with ``marc`` are returned. The token side is a UNION across
        all query tokens (favouring recall); the final candidate set is the
        INTERSECTION of the year set and the token set.

        Args:
            marc: The MARC record to retrieve candidates for.
            window: Inclusive year radius; ``0`` restricts to the exact year.

        Yields:
            :class:`IndexedNyplRegRecord` instances, deduplicated by uuid.
            Nothing is yielded when ``marc`` has no ``publication_year``,
            when the year set is empty, or when the record shares no token.
        """
        if marc.publication_year is None:
            return
        year_set = self._year_candidates(marc.publication_year, window)
        if not year_set:
            return
        token_set = self._token_candidates(marc)
        if not token_set:
            return
        for uuid in year_set & token_set:
            record = self.get_registration(uuid)
            if record is not None:
                yield record

    def _renewal_year_candidates(self, year: int, window: int) -> set[str]:
        """Return the union of renewal entry ids across the ``ren_by_year`` window buckets."""
        candidates: set[str] = set()
        for candidate_year in range(year - window, year + window + 1):
            blob = self._store.ren_by_year.get(encode_year_key(candidate_year))
            if blob is None:
                continue
            candidates.update(decode_uuid_list(blob))
        return candidates

    def _renewal_token_candidates(self, marc: MarcRecord) -> set[str]:
        """Return the union of every renewal sharing a query token.

        Mirrors :meth:`_token_candidates` over the renewal-side inverted
        indexes: title tokens from ``marc.title`` hit ``ren_title_index``,
        author tokens from ``marc.main_author`` and
        ``marc.statement_of_responsibility`` hit ``ren_author_index``, and
        publisher tokens from ``marc.publisher`` hit ``ren_claimants_index``
        (the renewal side carries claimants, not a publisher field).
        """
        candidates: set[str] = set()
        for token in title_keys(marc.title):
            candidates.update(self._postings(self._store.ren_title_index, token))
        author_tokens = author_keys(marc.main_author) | author_keys(
            marc.statement_of_responsibility
        )
        for token in author_tokens:
            candidates.update(self._postings(self._store.ren_author_index, token))
        for token in publisher_keys(marc.publisher):
            candidates.update(self._postings(self._store.ren_claimants_index, token))
        return candidates

    def candidates_for_renewal(
        self,
        marc: MarcRecord,
        window: int = 0,
    ) -> Iterator[NyplRenRecord]:
        """Yield renewals sharing an original-registration year AND a field token.

        The renewal-side mirror of :meth:`candidates_for`: a MARC record is
        retrieved directly against renewal records (for books with no
        registration match) by intersecting the renewal year set (bucketed on
        the original-registration ``odat`` year so it aligns with
        ``marc.publication_year``) with the union of renewals sharing a
        title/author/claimants token. This is retrieval only — scoring is the
        caller's concern.

        Args:
            marc: The MARC record to retrieve renewal candidates for.
            window: Inclusive year radius; ``0`` restricts to the exact year.

        Yields:
            :class:`NyplRenRecord` instances, deduplicated by entry id.
            Nothing is yielded when ``marc`` has no ``publication_year``, when
            the year set is empty, or when the record shares no token.
        """
        if marc.publication_year is None:
            return
        year_set = self._renewal_year_candidates(marc.publication_year, window)
        if not year_set:
            return
        token_set = self._renewal_token_candidates(marc)
        if not token_set:
            return
        for entry_id in year_set & token_set:
            record = self.get_renewal(entry_id)
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
        ren_buckets_blob = meta.get(_META_REN_YEAR_BUCKETS_KEY)
        ren_title_blob = meta.get(_META_REN_TITLE_TOKENS_KEY)
        ren_author_blob = meta.get(_META_REN_AUTHOR_TOKENS_KEY)
        ren_claimants_blob = meta.get(_META_REN_CLAIMANTS_TOKENS_KEY)
        if (
            schema_blob is None
            or hash_blob is None
            or timestamp_blob is None
            or reg_blob is None
            or ren_blob is None
            or joins_blob is None
            or buckets_blob is None
            or ren_buckets_blob is None
            or ren_title_blob is None
            or ren_author_blob is None
            or ren_claimants_blob is None
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
            renewal_year_buckets=int(ren_buckets_blob.decode("ascii")),
            renewal_title_tokens=int(ren_title_blob.decode("ascii")),
            renewal_author_tokens=int(ren_author_blob.decode("ascii")),
            renewal_claimants_tokens=int(ren_claimants_blob.decode("ascii")),
        )


__all__ = [
    "IndexStats",
    "NyplIndexLookup",
]
