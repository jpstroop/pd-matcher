"""Lossless MARCXML writers.

Survivor records are written verbatim into ``<collection>`` files — either a
single streaming file (:class:`MarcxmlCollectionWriter`) or a series of capped
shards (:class:`MarcxmlShardWriter`). Records are serialized exactly as they
arrived (namespaces, subfields, indicators preserved) because the whole point
of this corpus is faithful round-tripping through the later matching and review
phases.
"""

from logging import getLogger
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from lxml.etree import _Element
from lxml.etree import tostring

_LOGGER = getLogger(__name__)

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_XML_DECLARATION = b'<?xml version="1.0" encoding="UTF-8"?>\n'
_COLLECTION_OPEN = f'<collection xmlns="{_MARC_NS}">\n'.encode()
_COLLECTION_CLOSE = b"</collection>\n"

_DEFAULT_SHARD_SIZE = 5000
_SHARD_NAME_TEMPLATE = "candidates_{index:05d}.xml"


class MarcxmlCollectionWriter:
    """Stream ``<record>`` elements into one well-formed MARCXML collection.

    The output is a single ``<collection xmlns="...slim">`` file in the format
    ``pd-matcher match --marc`` consumes. Records are appended as they arrive,
    so an arbitrarily large corpus can be written without ever buffering the
    whole document in memory. The parent directory is created on open.
    """

    __slots__ = ("_handle", "_path", "_records_written")

    def __init__(self, path: Path) -> None:
        """Open ``path`` for writing and emit the collection header.

        Args:
            path: Destination MARCXML file; the parent directory is created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._records_written = 0
        handle = path.open("wb")
        handle.write(_XML_DECLARATION)
        handle.write(_COLLECTION_OPEN)
        self._handle: BinaryIO | None = handle

    @property
    def records_written(self) -> int:
        """Number of records appended to the collection so far."""
        return self._records_written

    def write(self, record: _Element) -> None:
        """Append one record to the collection, serialized verbatim.

        Args:
            record: A MARCXML ``<record>`` element.
        """
        handle = self._require_handle()
        handle.write(tostring(record, encoding="UTF-8", xml_declaration=False))
        handle.write(b"\n")
        self._records_written += 1

    def close(self) -> None:
        """Write the closing tag and close the file handle, if still open."""
        if self._handle is None:
            return
        handle = self._handle
        handle.write(_COLLECTION_CLOSE)
        handle.close()
        self._handle = None
        _LOGGER.info("wrote %d records to %s", self._records_written, self._path)

    def __enter__(self) -> MarcxmlCollectionWriter:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the collection on context exit."""
        self.close()

    def _require_handle(self) -> BinaryIO:
        """Return the open handle, raising if the collection is closed."""
        handle = self._handle
        if handle is None:
            raise RuntimeError("collection writer is closed")
        return handle


class MarcxmlShardWriter:
    """Write ``<record>`` elements into capped, well-formed MARCXML shards."""

    __slots__ = (
        "_handle",
        "_out_dir",
        "_records_in_shard",
        "_shard_index",
        "_shard_size",
        "_shards_written",
        "_total_written",
    )

    def __init__(self, out_dir: Path, shard_size: int = _DEFAULT_SHARD_SIZE) -> None:
        """Initialize the writer.

        Args:
            out_dir: Directory that will hold the shard files; created if absent.
            shard_size: Maximum records per shard before rolling to a new file.

        Raises:
            ValueError: If ``shard_size`` is not a positive integer.
        """
        if shard_size < 1:
            raise ValueError("shard_size must be a positive integer")
        self._out_dir = out_dir
        self._shard_size = shard_size
        self._handle: BinaryIO | None = None
        self._shard_index = 0
        self._records_in_shard = 0
        self._total_written = 0
        self._shards_written = 0

    @property
    def total_written(self) -> int:
        """Total number of records written across all shards."""
        return self._total_written

    @property
    def shards_written(self) -> int:
        """Number of shard files that have been finalized."""
        return self._shards_written

    def write(self, record: _Element) -> None:
        """Append one record to the current shard, rolling over at the cap.

        Args:
            record: A MARCXML ``<record>`` element to serialize verbatim.
        """
        if self._handle is None or self._records_in_shard >= self._shard_size:
            self._open_new_shard()
        serialized = tostring(record, encoding="UTF-8", xml_declaration=False)
        handle = self._require_handle()
        handle.write(serialized)
        handle.write(b"\n")
        self._records_in_shard += 1
        self._total_written += 1

    def close(self) -> None:
        """Finalize the open shard, if any."""
        self._close_open_shard()

    def __enter__(self) -> MarcxmlShardWriter:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the open shard on context exit."""
        self.close()

    def _open_new_shard(self) -> None:
        """Close any open shard and start a fresh one."""
        self._close_open_shard()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._shard_index += 1
        path = self._out_dir / _SHARD_NAME_TEMPLATE.format(index=self._shard_index)
        handle = path.open("wb")
        handle.write(_XML_DECLARATION)
        handle.write(_COLLECTION_OPEN)
        self._handle = handle
        self._records_in_shard = 0
        _LOGGER.info("opened shard: %s", path)

    def _close_open_shard(self) -> None:
        """Write the closing tag and close the current shard handle."""
        if self._handle is None:
            return
        handle = self._handle
        handle.write(_COLLECTION_CLOSE)
        handle.close()
        self._handle = None
        self._shards_written += 1

    def _require_handle(self) -> BinaryIO:
        """Return the open shard handle, raising if it is missing."""
        handle = self._handle
        if handle is None:
            raise RuntimeError("no open shard")
        return handle
