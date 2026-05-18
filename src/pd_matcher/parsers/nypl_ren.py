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
"""

from collections.abc import Iterator
from csv import reader as csv_reader
from datetime import date
from pathlib import Path

from pd_matcher.models import NyplRenRecord

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


def _none_if_blank(value: str) -> str | None:
    """Return ``value.strip() or None`` so empty cells decode to ``None``."""
    stripped = value.strip()
    return stripped or None


def _parse_iso_date(value: str) -> date | None:
    """Parse ``value`` as ISO date or return ``None`` on blank/invalid input."""
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _row_to_record(row: list[str], columns: _ColumnMap) -> NyplRenRecord | None:
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
    return NyplRenRecord(
        id=renewal_id,
        entry_id=entry_id,
        oreg=_none_if_blank(row[columns.oreg]),
        odat=_parse_iso_date(row[columns.odat]),
        rdat=_parse_iso_date(row[columns.rdat]),
        author=_none_if_blank(row[columns.author]),
        title=_none_if_blank(row[columns.title]),
        claimants=_none_if_blank(row[columns.claimants]),
        new_matter=_none_if_blank(row[columns.new_matter]),
        full_text=_none_if_blank(row[columns.full_text]),
    )


def iter_nypl_ren_records(path: Path) -> Iterator[NyplRenRecord]:
    """Yield :class:`NyplRenRecord` objects streamed from one TSV file.

    Args:
        path: Filesystem path to a single renewal TSV file.

    Yields:
        :class:`NyplRenRecord` instances, one per data row.

    Raises:
        NyplRenHeaderError: If the file's header row matches neither the
            pre-1978 nor the ``*-from-db`` column contract.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv_reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            return
        header_tuple = tuple(header)
        columns = _COLUMN_MAPS.get(header_tuple)
        if columns is None:
            raise NyplRenHeaderError(
                f"Unexpected CCE renewal header in {path}: got {header_tuple!r}; "
                f"expected one of pre-1978 {_HEADER_PRE_1978!r} "
                f"or from-db {_HEADER_FROM_DB!r}"
            )
        for row in reader:
            record = _row_to_record(row, columns)
            if record is not None:
                yield record


def iter_nypl_ren_directory(root: Path) -> Iterator[NyplRenRecord]:
    """Yield records from every ``*.tsv`` file beneath ``root`` in sorted order.

    Args:
        root: Directory containing renewal TSV files (e.g. ``data/nypl-ren/data``).

    Yields:
        :class:`NyplRenRecord` instances streamed across all discovered files.
    """
    for tsv_path in sorted(root.rglob("*.tsv")):
        yield from iter_nypl_ren_records(tsv_path)


__all__ = [
    "NyplRenHeaderError",
    "iter_nypl_ren_directory",
    "iter_nypl_ren_records",
]
