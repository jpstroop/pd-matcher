"""Streaming parser for NYPL transcribed copyright renewal TSV files.

The renewal corpus ships as tab-separated value files with a fixed header.
We use :func:`csv.reader` (not :class:`csv.DictReader`) for two reasons:
(a) iterating positional rows is measurably faster on the large files we
ingest, and (b) the header is documented and stable enough to bind once at
the top of each file. Empty cells become ``None`` for nullable fields, and
``odat``/``rdat`` are coerced to :class:`datetime.date` via
``date.fromisoformat`` with a fall back to ``None`` on parse failure (some
historic rows contain partial or malformed dates).
"""

from collections.abc import Iterator
from csv import reader as csv_reader
from datetime import date
from pathlib import Path

from pd_matcher.models import NyplRenRecord

_EXPECTED_HEADER: tuple[str, ...] = (
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


class NyplRenHeaderError(ValueError):
    """Raised when a renewal TSV's header row does not match the contract."""


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


def _row_to_record(row: list[str]) -> NyplRenRecord | None:
    """Translate a positional row to a :class:`NyplRenRecord`.

    Rows shorter than the expected column count are skipped (returns ``None``)
    rather than raised on, because partial trailing rows have been observed
    in historic dumps.
    """
    if len(row) < len(_EXPECTED_HEADER):
        return None
    renewal_id = row[9].strip()
    entry_id = row[0].strip()
    if not renewal_id or not entry_id:
        return None
    return NyplRenRecord(
        id=renewal_id,
        entry_id=entry_id,
        oreg=_none_if_blank(row[7]),
        odat=_parse_iso_date(row[8]),
        rdat=_parse_iso_date(row[10]),
        author=_none_if_blank(row[5]),
        title=_none_if_blank(row[6]),
        claimants=_none_if_blank(row[11]),
        new_matter=_none_if_blank(row[12]),
        full_text=_none_if_blank(row[16]),
    )


def iter_nypl_ren_records(path: Path) -> Iterator[NyplRenRecord]:
    """Yield :class:`NyplRenRecord` objects streamed from one TSV file.

    Args:
        path: Filesystem path to a single renewal TSV file.

    Yields:
        :class:`NyplRenRecord` instances, one per data row.

    Raises:
        NyplRenHeaderError: If the file's header row does not match the
            documented column contract.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv_reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            return
        if tuple(header) != _EXPECTED_HEADER:
            raise NyplRenHeaderError(
                f"Unexpected NYPL renewal header in {path}: got {tuple(header)!r}, "
                f"expected {_EXPECTED_HEADER!r}"
            )
        for row in reader:
            record = _row_to_record(row)
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
