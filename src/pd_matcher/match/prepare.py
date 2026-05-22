"""Prepare MARCXML into pickled record chunks for re-runnable matching.

Running the matcher over the real corpus repeatedly means re-parsing
multi-gigabyte MARCXML on every run — :func:`iter_marc_records` is the
slowest stage of the pipeline. ``prepare-marc`` pays that cost once: it
streams every record out of a MARCXML file (or a directory of ``*.xml``
shards), pickles the :class:`MarcRecord` structs into fixed-size
``chunk_NNNNN.pkl`` files, and writes a :class:`PreparedManifest` next to
them. A subsequent ``match --prepared DIR`` reads the chunks back in
manifest order, skipping the XML parse entirely and giving the reporter a
known total so it can show percent-complete and an ETA.

Preparation is idempotent: the manifest records a ``source_hash`` over the
input files' ``path|size|mtime_ns`` triples, and a re-run with a matching
hash and schema version is a no-op unless ``force`` is passed. A real
rebuild removes any orphaned chunks from a previous (differently sized)
run before writing fresh ones.

The chunk codec is stdlib :mod:`pickle` on purpose: chunks are an internal,
single-machine cache (never shipped, never long-lived across schema
changes), and pickling a tuple of msgspec structs is both simpler and
faster than re-encoding through msgpack for this throwaway artifact. The
manifest, by contrast, is a frozen :class:`msgspec.Struct` so its on-disk
shape is validated on read and rejects schema drift.
"""

from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from pickle import dump
from pickle import load
from time import monotonic
from typing import Final

from msgspec import Struct
from msgspec.json import Decoder
from msgspec.json import Encoder

from pd_matcher.models import MarcRecord
from pd_matcher.parsers.marc import MarcParseStats
from pd_matcher.parsers.marc import iter_marc_records
from pd_matcher.progress import ProgressReporter

_LOGGER = getLogger(__name__)

_MANIFEST_VERSION: Final[int] = 1
_MANIFEST_NAME: Final[str] = "manifest.json"
_CHUNK_GLOB: Final[str] = "chunk_*.pkl"
_CHUNK_DIGITS: Final[int] = 5
_DEFAULT_CHUNK_SIZE: Final[int] = 1000

_MANIFEST_ENCODER: Final[Encoder] = Encoder()


class PreparedManifest(Struct, frozen=True, forbid_unknown_fields=True):
    """Immutable description of one prepared-chunk directory.

    Attributes:
        version: Schema version of the manifest format itself.
        total_records: Count of :class:`MarcRecord` objects across all chunks.
        chunk_files: Chunk filenames in read order (relative to the directory).
        chunk_size: Target records-per-chunk used when the chunks were written.
        source_hash: ``sha256`` fingerprint of the input files; see
            :func:`compute_source_hash`.
        created_at: ISO-8601 UTC timestamp of the build.
    """

    version: int
    total_records: int
    chunk_files: tuple[str, ...]
    chunk_size: int
    source_hash: str
    created_at: str


_MANIFEST_DECODER: Final[Decoder[PreparedManifest]] = Decoder(PreparedManifest)


class PrepareReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary returned by :func:`prepare_marc`."""

    out_dir: Path
    total_records: int
    chunk_count: int
    chunk_size: int
    skipped: bool
    duration_seconds: float


def _iter_source_files(source: Path) -> tuple[Path, ...]:
    """Return the MARCXML files behind ``source`` in stable sorted order.

    A file argument yields a single-element tuple; a directory yields its
    sorted ``*.xml`` children (non-recursive — shards live flat).
    """
    if source.is_dir():
        return tuple(sorted(source.glob("*.xml")))
    return (source,)


def _iter_source_records(source: Path, stats: MarcParseStats) -> Iterator[MarcRecord]:
    """Stream every :class:`MarcRecord` across all files behind ``source``.

    :func:`iter_marc_records` parses a single file, so a directory source is
    fanned out here over its sorted ``*.xml`` shards in stable order.
    """
    for path in _iter_source_files(source):
        yield from iter_marc_records(path, stats)


def compute_source_hash(source: Path) -> str:
    """Fingerprint ``source`` over each file's ``path|size|mtime_ns``.

    The same cheap size+mtime proxy the index builder uses: full content
    hashing would be authoritative but is far too slow on multi-gigabyte
    inputs, and size+mtime reliably catches a changed or re-exported corpus.
    Paths are made relative to the directory (or to the file's own name) so
    the hash is stable across absolute-path differences.
    """
    files = _iter_source_files(source)
    root = source if source.is_dir() else source.parent
    hasher = sha256()
    for path in files:
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        hasher.update(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return hasher.hexdigest()


def _chunk_name(index: int) -> str:
    """Return the zero-padded chunk filename for ordinal ``index``."""
    return f"chunk_{index:0{_CHUNK_DIGITS}d}.pkl"


def _write_chunk(out_dir: Path, index: int, records: tuple[MarcRecord, ...]) -> str:
    """Pickle ``records`` to ``chunk_<index>.pkl`` and return the filename."""
    name = _chunk_name(index)
    with (out_dir / name).open("wb") as handle:
        dump(records, handle, protocol=HIGHEST_PROTOCOL)
    return name


def _remove_orphan_chunks(out_dir: Path) -> None:
    """Delete every existing ``chunk_*.pkl`` so a rebuild leaves no stragglers."""
    for stale in out_dir.glob(_CHUNK_GLOB):
        stale.unlink()


def read_manifest(out_dir: Path) -> PreparedManifest:
    """Load and validate the :class:`PreparedManifest` in ``out_dir``.

    Raises:
        FileNotFoundError: When no manifest file exists in ``out_dir``.
    """
    path = out_dir / _MANIFEST_NAME
    return _MANIFEST_DECODER.decode(path.read_bytes())


def _existing_manifest(out_dir: Path) -> PreparedManifest | None:
    """Return the manifest already in ``out_dir``, or ``None`` if absent."""
    if not (out_dir / _MANIFEST_NAME).is_file():
        return None
    return read_manifest(out_dir)


def iter_prepared_records(out_dir: Path) -> Iterator[MarcRecord]:
    """Yield every :class:`MarcRecord` from ``out_dir`` in manifest order.

    The manifest fixes both the chunk set and their order, so a prepared
    directory replays deterministically regardless of filesystem listing
    quirks.
    """
    manifest = read_manifest(out_dir)
    for name in manifest.chunk_files:
        with (out_dir / name).open("rb") as handle:
            records: tuple[MarcRecord, ...] = load(handle)
        yield from records


def prepare_marc(
    source: Path,
    out_dir: Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    force: bool = False,
) -> PrepareReport:
    """Stream ``source`` MARCXML into pickled chunks under ``out_dir``.

    Args:
        source: A MARCXML file or a directory of ``*.xml`` shards.
        out_dir: Destination directory for the chunks and manifest. Created
            if missing.
        chunk_size: Target number of records per chunk. Must be ``>= 1``.
        force: Rebuild even when an existing manifest matches the current
            ``source_hash`` and schema version.

    Returns:
        A :class:`PrepareReport`; ``skipped=True`` when an up-to-date build
        already existed and ``force`` was not set.

    Raises:
        ValueError: If ``chunk_size < 1``.
        FileNotFoundError: If ``source`` does not exist.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 (got {chunk_size!r})")
    if not source.exists():
        raise FileNotFoundError(f"source does not exist: {source}")
    started_at = monotonic()
    source_hash = compute_source_hash(source)
    existing = _existing_manifest(out_dir)
    if (
        not force
        and existing is not None
        and existing.version == _MANIFEST_VERSION
        and existing.source_hash == source_hash
    ):
        _LOGGER.info(
            "prepare.skip source_hash matches existing manifest (%d records, %d chunks)",
            existing.total_records,
            len(existing.chunk_files),
        )
        return PrepareReport(
            out_dir=out_dir,
            total_records=existing.total_records,
            chunk_count=len(existing.chunk_files),
            chunk_size=existing.chunk_size,
            skipped=True,
            duration_seconds=monotonic() - started_at,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_orphan_chunks(out_dir)
    parse_stats = MarcParseStats()
    reporter = ProgressReporter(logger=_LOGGER, total=0, clock=monotonic)
    chunk_files: list[str] = []
    buffer: list[MarcRecord] = []
    total_records = 0
    for record in _iter_source_records(source, parse_stats):
        buffer.append(record)
        total_records += 1
        if len(buffer) >= chunk_size:
            chunk_files.append(_write_chunk(out_dir, len(chunk_files), tuple(buffer)))
            buffer.clear()
            reporter.update(total_records, detail=f"chunks={len(chunk_files)}")
    if buffer:
        chunk_files.append(_write_chunk(out_dir, len(chunk_files), tuple(buffer)))
    manifest = PreparedManifest(
        version=_MANIFEST_VERSION,
        total_records=total_records,
        chunk_files=tuple(chunk_files),
        chunk_size=chunk_size,
        source_hash=source_hash,
        created_at=datetime.now(UTC).isoformat(),
    )
    (out_dir / _MANIFEST_NAME).write_bytes(_MANIFEST_ENCODER.encode(manifest))
    _LOGGER.info(
        "prepare.complete records=%d chunks=%d skipped_001=%d skipped_245a=%d",
        total_records,
        len(chunk_files),
        parse_stats.skipped_missing_001,
        parse_stats.skipped_missing_245a,
    )
    return PrepareReport(
        out_dir=out_dir,
        total_records=total_records,
        chunk_count=len(chunk_files),
        chunk_size=chunk_size,
        skipped=False,
        duration_seconds=monotonic() - started_at,
    )


__all__ = [
    "PrepareReport",
    "PreparedManifest",
    "compute_source_hash",
    "iter_prepared_records",
    "prepare_marc",
    "read_manifest",
]
