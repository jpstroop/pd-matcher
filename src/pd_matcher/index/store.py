"""LMDB environment wrapper exposing the named sub-DBs used by the indexer.

LMDB was chosen for this project precisely because a single on-disk env file
can be mmap-shared across an arbitrary number of reader processes with zero
copies; this module owns the env handle and the named sub-DB handles so the
rest of the codebase can stay unaware of ``lmdb.Cursor`` and friends.

The public surface is two classes:

* :class:`NyplIndexStore` — opens the env, exposes five typed sub-DB handles,
  and supports use as a context manager so callers can rely on
  ``__exit__`` to close the env even on failure paths.
* :class:`_SubDb` — a minimal handle returned by the sub-DB properties that
  exposes ``get``, ``put``, ``iter_items`` and ``count``. It deliberately
  hides raw cursor/transaction objects so consumers cannot leak ``Any``-typed
  ``lmdb`` values into the rest of the codebase.

Writes are batched inside a single short-lived ``lmdb.Transaction`` that the
caller scopes via :meth:`NyplIndexStore.write_transaction`; reads use a
fresh read transaction per ``_SubDb.get`` call (LMDB read txns are cheap and
this keeps the read API stateless and process-safe).
"""

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Self
from typing import cast

from lmdb import Environment as LmdbEnvironment
from lmdb import Transaction as LmdbTransaction
from lmdb import _Database as LmdbDatabase


class _SubDb:
    """Typed handle to a single named sub-DB inside :class:`NyplIndexStore`.

    Instances are constructed by :class:`NyplIndexStore` and should not be
    built directly. Each instance keeps a reference to the parent env and
    the opaque LMDB sub-DB handle and provides a small typed surface so
    callers never touch ``lmdb.Cursor`` or ``lmdb.Transaction``.
    """

    __slots__ = ("_db", "_env", "_store")

    def __init__(self, store: NyplIndexStore, db_handle: LmdbDatabase) -> None:
        self._store = store
        self._env: LmdbEnvironment = store._env
        self._db: LmdbDatabase = db_handle

    def get(self, key: bytes) -> bytes | None:
        """Return the value for ``key`` or ``None`` if missing."""
        txn = self._store._active_txn
        if txn is not None:
            value = txn.get(key, db=self._db)
        else:
            with self._env.begin(db=self._db, write=False, buffers=False) as read_txn:
                value = read_txn.get(key)
        # We always open with ``buffers=False`` so the stub's ``bytes |
        # memoryview`` union is in practice always ``bytes``; ``bytes(value)``
        # is the cheap identity path for ``bytes`` and the explicit coercion
        # for ``memoryview``.
        return None if value is None else bytes(value)

    def put(self, key: bytes, value: bytes) -> None:
        """Store ``value`` at ``key`` inside the active write transaction.

        Raises:
            RuntimeError: If no write transaction is currently active. Writes
                must occur inside a :meth:`NyplIndexStore.write_transaction`
                block so the builder can group thousands of writes into a
                single fsync.
        """
        txn = self._store._active_txn
        if txn is None:
            raise RuntimeError("put() requires an active write transaction")
        txn.put(key, value, db=self._db)

    def iter_items(self) -> Iterator[tuple[bytes, bytes]]:
        """Yield every ``(key, value)`` pair in insertion-key order."""
        with self._env.begin(db=self._db, write=False, buffers=False) as txn:
            for key, value in txn.cursor():
                yield bytes(key), bytes(value)

    def count(self) -> int:
        """Return the number of entries currently stored in the sub-DB."""
        with self._env.begin(db=self._db, write=False) as txn:
            stat = txn.stat(self._db)
            return stat["entries"]


class NyplIndexStore:
    """LMDB environment wrapper exposing the named sub-DBs we care about.

    The env is opened with ``max_dbs=8`` so we have headroom for future
    sub-DBs without an env migration. ``subdir=True`` is used because the
    LMDB data directory layout (``data.mdb`` plus ``lock.mdb`` inside one
    directory) is what Phase 6's worker processes will mmap-share.

    Readonly mode opens the env with ``lock=False`` so any number of worker
    processes can share the same env without contending on the writer lock.
    Write mode keeps locking enabled so concurrent builder invocations fail
    fast instead of corrupting the env.
    """

    __slots__ = (
        "_active_txn",
        "_env",
        "_meta_db",
        "_path",
        "_readonly",
        "_reg_by_id_db",
        "_reg_by_year_db",
        "_ren_by_id_db",
        "_ren_by_oreg_db",
    )

    def __init__(
        self,
        path: Path,
        *,
        map_size_bytes: int = 16 * 1024**3,
        readonly: bool = False,
    ) -> None:
        self._path = path
        self._readonly = readonly
        self._active_txn: LmdbTransaction | None = None
        if not readonly:
            path.mkdir(parents=True, exist_ok=True)
        self._env = LmdbEnvironment(
            str(path),
            map_size=map_size_bytes,
            subdir=True,
            readonly=readonly,
            lock=not readonly,
            max_dbs=8,
            create=not readonly,
        )
        self._reg_by_id_db = self._env.open_db(b"reg_by_id", create=not readonly)
        self._ren_by_id_db = self._env.open_db(b"ren_by_id", create=not readonly)
        self._reg_by_year_db = self._env.open_db(b"reg_by_year", create=not readonly)
        self._ren_by_oreg_db = self._env.open_db(b"ren_by_oreg", create=not readonly)
        self._meta_db = self._env.open_db(b"meta", create=not readonly)

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
        """Close the underlying LMDB env and release file handles."""
        self._env.close()

    @property
    def path(self) -> Path:
        """Filesystem location of the LMDB env directory."""
        return self._path

    @property
    def readonly(self) -> bool:
        """``True`` if the env was opened in read-only mode."""
        return self._readonly

    @property
    def reg_by_id(self) -> _SubDb:
        """Sub-DB mapping ``uuid_bytes`` to an encoded ``IndexedNyplRegRecord``."""
        return _SubDb(self, self._reg_by_id_db)

    @property
    def ren_by_id(self) -> _SubDb:
        """Sub-DB mapping ``entry_id_bytes`` to an encoded ``NyplRenRecord``."""
        return _SubDb(self, self._ren_by_id_db)

    @property
    def reg_by_year(self) -> _SubDb:
        """Sub-DB mapping a year key to a tuple of registration uuids."""
        return _SubDb(self, self._reg_by_year_db)

    @property
    def ren_by_oreg(self) -> _SubDb:
        """Sub-DB mapping ``make_renewal_key(...)`` to the renewal entry id."""
        return _SubDb(self, self._ren_by_oreg_db)

    @property
    def meta(self) -> _SubDb:
        """Sub-DB holding free-form build metadata (timestamps, hashes)."""
        return _SubDb(self, self._meta_db)

    def write_transaction(self) -> _WriteTransaction:
        """Return a context manager that scopes a single write transaction.

        Raises:
            RuntimeError: If the env was opened in read-only mode.
        """
        if self._readonly:
            raise RuntimeError("write_transaction() not allowed on readonly store")
        return _WriteTransaction(self)


class _WriteTransaction:
    """Context manager that opens and commits a single LMDB write txn.

    ``__exit__`` is only ever reached after ``__enter__`` succeeded (Python
    skips ``__exit__`` when ``__enter__`` raises), so the active transaction
    is guaranteed to exist on the store by the time we commit or abort.
    """

    __slots__ = ("_store",)

    def __init__(self, store: NyplIndexStore) -> None:
        self._store = store

    def __enter__(self) -> Self:
        if self._store._active_txn is not None:
            raise RuntimeError("write transaction already active on this store")
        self._store._active_txn = self._store._env.begin(write=True, buffers=False)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        txn: LmdbTransaction = cast("LmdbTransaction", self._store._active_txn)
        self._store._active_txn = None
        if exc is None:
            txn.commit()
        else:
            txn.abort()


__all__ = [
    "NyplIndexStore",
]
