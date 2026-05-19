"""Streaming parser for NYPL's TSV transcription of the CCE renewal corpus.

The source data is the U.S. Copyright Office's Catalog of Copyright Entries
(CCE), published by the Library of Congress and transcribed into TSV by
NYPL. The CCE renewal corpus ships as tab-separated value files with one of
two stable header schemas. Files covering the pre-1978 entries (the vast
majority) use the original column names ``author``, ``title``, ``rdat``,
``notes``. Files named ``*-from-db.tsv`` (1978 and later) use the variant
names ``auth``, ``titl``, ``dreg``, ``note``. Column ordering, count, and
semantics are otherwise identical, and both decode to the same
:class:`NyplRenRecord`.

We use :func:`csv.reader` (not :class:`csv.DictReader`) for two reasons:
(a) iterating positional rows is measurably faster on the large files we
ingest, and (b) the schema is known and stable enough to bind once at the
top of each file. We resolve the column index map at header time and pass
it into the row builder, so per-row work never branches on which schema is
in use. Empty cells become ``None`` for nullable fields, and the two date
columns are coerced to :class:`datetime.date` via ``date.fromisoformat``
with a fall back to ``None`` on parse failure (some historic rows contain
partial or malformed dates).

Encoding hygiene runs at two layers. Every file is first probed for
whole-file UTF-8 validity. The supplied corpus has always passed this
probe, so the hot path uses the standard ``str``-mode reader. If the
probe fails, we fall back to a bytes-level reader that routes each cell
through :func:`pd_matcher.normalize.cp1255_fallback.decode_subfield` —
defensive support for a future ingest that contains Windows-1255 Hebrew
content. In both paths, every finalized cell value also passes through
:func:`pd_matcher.normalize.encoding.clean_text` to repair mojibake and
strip stray bidi/BOM characters. Per-file counters
(:class:`NyplRenParseStats`) expose how many cells were routed through
each path so dataset quality can be reported without re-walking the
files.
"""

from collections.abc import Iterator
from csv import reader as csv_reader
from datetime import date
from pathlib import Path

from pd_matcher.models import NyplRenRecord
from pd_matcher.normalize.cp1255_fallback import decode_subfield
from pd_matcher.normalize.encoding import clean_text

_HEADER_PRE_1978: tuple[str, ...] = (
    "entry_id",
    "volume",
    "part",
    "number",
    "page",
    "author",
    "title",
    "oreg",
    "odat",
    "id",
    "rdat",
    "claimants",
    "new_matter",
    "see_also_ren",
    "see_also_reg",
    "notes",
    "full_text",
)

_HEADER_FROM_DB: tuple[str, ...] = (
    "entry_id",
    "volume",
    "part",
    "number",
    "page",
    "auth",
    "titl",
    "oreg",
    "odat",
    "id",
    "dreg",
    "claimants",
    "new_matter",
    "see_also_ren",
    "see_also_reg",
    "note",
    "full_text",
)

_KNOWN_HEADERS: tuple[tuple[str, ...], ...] = (_HEADER_PRE_1978, _HEADER_FROM_DB)


class NyplRenHeaderError(ValueError):
    """Raised when a renewal TSV's header row does not match either contract."""


class _ColumnMap:
    """Pre-resolved column indices for one of the known renewal header schemas.

    Binding the column positions once per file (rather than per row) keeps the
    hot path branch-free across schema variants.
    """

    __slots__ = (
        "author",
        "claimants",
        "entry_id",
        "full_text",
        "id",
        "new_matter",
        "odat",
        "oreg",
        "rdat",
        "title",
        "width",
    )

    def __init__(self, header: tuple[str, ...]) -> None:
        self.entry_id = header.index("entry_id")
        self.oreg = header.index("oreg")
        self.odat = header.index("odat")
        self.id = header.index("id")
        self.claimants = header.index("claimants")
        self.new_matter = header.index("new_matter")
        self.full_text = header.index("full_text")
        self.author = header.index("author" if "author" in header else "auth")
        self.title = header.index("title" if "title" in header else "titl")
        self.rdat = header.index("rdat" if "rdat" in header else "dreg")
        self.width = len(header)


_COLUMN_MAPS: dict[tuple[str, ...], _ColumnMap] = {
    header: _ColumnMap(header) for header in _KNOWN_HEADERS
}


class NyplRenParseStats:
    """Mutable counters surfaced to callers after a parse run."""

    __slots__ = (
        "emitted",
        "mojibake_fixed_count",
        "subfields_decoded_as_cp1255",
        "subfields_decoded_with_replacement",
    )

    def __init__(self) -> None:
        self.emitted = 0
        self.mojibake_fixed_count = 0
        self.subfields_decoded_as_cp1255 = 0
        self.subfields_decoded_with_replacement = 0


def _file_is_utf8(path: Path) -> bool:
    """Return ``True`` iff every byte in ``path`` decodes cleanly as UTF-8."""
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _clean_cell(value: str, stats: NyplRenParseStats) -> str | None:
    """Strip ``value`` and run encoding hygiene, returning ``None`` if blank."""
    stripped = value.strip()
    if not stripped:
        return None
    cleaned = clean_text(stripped)
    if cleaned.mojibake_fixed:
        stats.mojibake_fixed_count += 1
    return cleaned.text or None


def _parse_iso_date(value: str) -> date | None:
    """Parse ``value`` as ISO date or return ``None`` on blank/invalid input."""
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _row_to_record(
    row: list[str], columns: _ColumnMap, stats: NyplRenParseStats
) -> NyplRenRecord | None:
    """Translate a positional row to a :class:`NyplRenRecord`.

    Rows shorter than the expected column count are skipped (returns ``None``)
    rather than raised on, because partial trailing rows have been observed
    in historic dumps.
    """
    if len(row) < columns.width:
        return None
    renewal_id = row[columns.id].strip()
    entry_id = row[columns.entry_id].strip()
    if not renewal_id or not entry_id:
        return None
    stats.emitted += 1
    return NyplRenRecord(
        id=renewal_id,
        entry_id=entry_id,
        oreg=_clean_cell(row[columns.oreg], stats),
        odat=_parse_iso_date(row[columns.odat]),
        rdat=_parse_iso_date(row[columns.rdat]),
        author=_clean_cell(row[columns.author], stats),
        title=_clean_cell(row[columns.title], stats),
        claimants=_clean_cell(row[columns.claimants], stats),
        new_matter=_clean_cell(row[columns.new_matter], stats),
        full_text=_clean_cell(row[columns.full_text], stats),
    )


def _resolve_columns(path: Path, header: tuple[str, ...]) -> _ColumnMap:
    """Return the column map for ``header`` or raise :class:`NyplRenHeaderError`."""
    columns = _COLUMN_MAPS.get(header)
    if columns is None:
        raise NyplRenHeaderError(
            f"Unexpected CCE renewal header in {path}: got {header!r}; "
            f"expected one of pre-1978 {_HEADER_PRE_1978!r} "
            f"or from-db {_HEADER_FROM_DB!r}"
        )
    return columns


def _decode_row_bytes(raw_row: list[bytes], stats: NyplRenParseStats) -> list[str]:
    """Decode each cell via the defensive ladder, updating fallback counters."""
    decoded: list[str] = []
    for cell in raw_row:
        result = decode_subfield(cell)
        if result.encoding_used == "windows-1255":
            stats.subfields_decoded_as_cp1255 += 1
        elif result.encoding_used == "utf-8-replace":
            stats.subfields_decoded_with_replacement += 1
        decoded.append(result.text)
    return decoded


def _iter_text_mode(path: Path, stats: NyplRenParseStats) -> Iterator[NyplRenRecord]:
    """Hot path: file is whole-file UTF-8, so ``csv.reader`` reads it directly."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv_reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            return
        columns = _resolve_columns(path, tuple(header))
        for row in reader:
            record = _row_to_record(row, columns, stats)
            if record is not None:
                yield record


def _iter_bytes_mode(path: Path, stats: NyplRenParseStats) -> Iterator[NyplRenRecord]:
    """Fallback path: parse raw bytes, decoding each cell via the cp1255 ladder.

    The text path is preferred because csv.reader correctly handles quoting
    rules; the bytes path splits naively on tab. The renewal corpus does not
    use TSV quoting in any shipped row, so this trade-off is safe in
    practice and the fallback exists to preserve rows the text path would
    fail outright on.
    """
    raw_lines = path.read_bytes().splitlines()
    header_cells = [cell.decode("utf-8", errors="replace") for cell in raw_lines[0].split(b"\t")]
    columns = _resolve_columns(path, tuple(header_cells))
    for raw_line in raw_lines[1:]:
        if not raw_line:
            continue
        raw_cells = raw_line.split(b"\t")
        decoded_cells = _decode_row_bytes(raw_cells, stats)
        record = _row_to_record(decoded_cells, columns, stats)
        if record is not None:
            yield record


def iter_nypl_ren_records(
    path: Path, stats: NyplRenParseStats | None = None
) -> Iterator[NyplRenRecord]:
    """Yield :class:`NyplRenRecord` objects streamed from one TSV file.

    Args:
        path: Filesystem path to a single renewal TSV file.
        stats: Optional :class:`NyplRenParseStats` counters mutated as
            records are emitted and as the encoding-hygiene pass repairs
            individual cell values. A fresh stats object is created when
            none is supplied.

    Yields:
        :class:`NyplRenRecord` instances, one per data row.

    Raises:
        NyplRenHeaderError: If the file's header row matches neither the
            pre-1978 nor the ``*-from-db`` column contract.
    """
    counters = stats if stats is not None else NyplRenParseStats()
    if _file_is_utf8(path):
        yield from _iter_text_mode(path, counters)
        return
    yield from _iter_bytes_mode(path, counters)


def iter_nypl_ren_directory(
    root: Path, stats: NyplRenParseStats | None = None
) -> Iterator[NyplRenRecord]:
    """Yield records from every ``*.tsv`` file beneath ``root`` in sorted order.

    Args:
        root: Directory containing renewal TSV files (e.g. ``data/nypl-ren/data``).
        stats: Optional shared :class:`NyplRenParseStats` counters; when
            supplied, counts accumulate across all walked files.

    Yields:
        :class:`NyplRenRecord` instances streamed across all discovered files.
    """
    counters = stats if stats is not None else NyplRenParseStats()
    for tsv_path in sorted(root.rglob("*.tsv")):
        yield from iter_nypl_ren_records(tsv_path, counters)


__all__ = [
    "NyplRenHeaderError",
    "NyplRenParseStats",
    "iter_nypl_ren_directory",
    "iter_nypl_ren_records",
]
