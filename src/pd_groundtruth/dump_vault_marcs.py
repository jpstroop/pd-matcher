"""Dump MARCXML records corresponding to vault entries to a single file.

The published dataset's MARC half: for every distinct ``marc_control_id``
in the label vault, copy the matching ``<record>`` element from
``data/candidates/`` into a single MARCXML file. The default destination is
``data/published/marc.xml`` — a path inside the in-tree clone of the
separate ``jpstroop/cce-marc-linkage`` data repository (gitignored from the
code repo). The output is the durable bibliographic context for the
labeled linkage so downstream consumers can reproduce or refine the
dataset without access to Princeton's full bibdata dump.

The implementation streams: each candidate shard is iterparsed, only the
records whose 001 controlfield matches the wanted set are kept, and they
are written through ``lxml.etree.xmlfile`` so peak memory stays bounded by
the size of one record element, not the whole corpus.
"""

from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

from lxml.etree import _Element
from lxml.etree import iterparse
from lxml.etree import xmlfile
from msgspec import Struct

from pd_groundtruth.label_vault import current_entries

_LOGGER = getLogger(__name__)

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_RECORD_TAG = f"{{{_MARC_NS}}}record"
_CONTROLFIELD_TAG = f"{{{_MARC_NS}}}controlfield"
_COLLECTION_TAG = f"{{{_MARC_NS}}}collection"
_CONTROL_NUMBER_TAG = "001"


class DumpReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of one :func:`dump_vault_marcs` invocation."""

    vault_entries: int
    distinct_marcs_requested: int
    marcs_written: int
    marcs_missing: int


def _wanted_marc_ids(vault_path: Path) -> tuple[int, set[str]]:
    """Return ``(vault_entry_count, set_of_unique_marc_control_ids)``."""
    entries = current_entries(vault_path)
    wanted: set[str] = {marc_id for marc_id, _ in entries}
    return len(entries), wanted


def _iter_shards(pool_path: Path) -> Iterator[Path]:
    """Yield every ``<pool>/<lang>/*.xml`` shard."""
    yield from sorted(pool_path.glob("*/*.xml"))


def _control_number(record: _Element) -> str | None:
    """Return the 001 controlfield value, or ``None`` when absent."""
    for child in record.iterchildren(tag=_CONTROLFIELD_TAG):
        if child.get("tag") == _CONTROL_NUMBER_TAG:
            text = child.text
            return text.strip() if text is not None else None
    return None


def _clear_record(record: _Element) -> None:
    """Release the parsed record from lxml's internal tree."""
    record.clear()
    previous = record.getprevious()
    parent = record.getparent()
    while previous is not None and parent is not None:
        del parent[0]
        previous = record.getprevious()


def dump_vault_marcs(
    vault_path: Path,
    pool_path: Path,
    output_path: Path,
) -> DumpReport:
    """Write a MARCXML collection of every vault MARC to ``output_path``.

    Reads ``vault_path`` for the set of unique ``marc_control_id`` values,
    walks every ``<pool>/<lang>/*.xml`` shard, and writes each matching
    ``<record>`` into a single ``<collection>`` at ``output_path``. Returns a
    :class:`DumpReport` summarising the counts.
    """
    vault_entries, wanted = _wanted_marc_ids(vault_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    with xmlfile(str(output_path), encoding="utf-8") as out:
        out.write_declaration()
        with out.element(_COLLECTION_TAG, nsmap={None: _MARC_NS}):
            for shard in _iter_shards(pool_path):
                context = iterparse(str(shard), events=("end",), tag=_RECORD_TAG)
                for _event, record in context:
                    control_id = _control_number(record)
                    if control_id in wanted and control_id not in written:
                        out.write(record)
                        written.add(control_id)
                    _clear_record(record)
                if wanted.issubset(written):
                    break
    with output_path.open("ab") as appendix:
        appendix.write(b"\n")
    missing = wanted - written
    if missing:
        _LOGGER.warning(
            "dump_vault_marcs.missing_from_pool count=%d sample=%s",
            len(missing),
            sorted(missing)[:5],
        )
    return DumpReport(
        vault_entries=vault_entries,
        distinct_marcs_requested=len(wanted),
        marcs_written=len(written),
        marcs_missing=len(missing),
    )


__all__ = [
    "DumpReport",
    "dump_vault_marcs",
]
