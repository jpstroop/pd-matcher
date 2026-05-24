"""Fetch and parse the Princeton ``bibdata`` dump manifest.

The manifest is a JSON document describing the set of MARC dump files that make
up one full bibliographic export. Each ``bib_records`` entry carries a download
URL and an md5 checksum that we verify after streaming the file to disk.
"""

from logging import getLogger

from msgspec import Struct
from requests import get

_LOGGER = getLogger(__name__)

DEFAULT_MANIFEST_URL = "https://bibdata.princeton.edu/dumps/16368.json"

_REQUEST_TIMEOUT_SECONDS = 60


class DumpEntry(Struct, frozen=True, forbid_unknown_fields=False):
    """One downloadable MARC dump file referenced by the manifest."""

    url: str
    md5: str


def parse_manifest(payload: bytes) -> tuple[DumpEntry, ...]:
    """Parse manifest JSON bytes into an ordered tuple of dump entries.

    Args:
        payload: Raw JSON body of the manifest document.

    Returns:
        Dump entries in manifest order, each with a download URL and md5.

    Raises:
        ValueError: If the manifest is missing ``files.bib_records`` or an
            entry lacks the required ``dump_file``/``md5`` keys.
    """
    from msgspec.json import decode

    document = decode(payload, type=_ManifestDocument)
    entries = document.files.bib_records
    if not entries:
        raise ValueError("manifest contains no files.bib_records entries")
    return tuple(DumpEntry(url=entry.dump_file, md5=entry.md5) for entry in entries)


def fetch_manifest(manifest_url: str = DEFAULT_MANIFEST_URL) -> tuple[DumpEntry, ...]:
    """Download and parse the dump manifest.

    Args:
        manifest_url: Absolute URL of the manifest JSON document.

    Returns:
        Dump entries in manifest order.
    """
    _LOGGER.info("fetching manifest: %s", manifest_url)
    response = get(manifest_url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    entries = parse_manifest(response.content)
    _LOGGER.info("manifest parsed: %d dump entries", len(entries))
    return entries


class _ManifestEntry(Struct, forbid_unknown_fields=False):
    """Raw ``bib_records`` entry as encoded in the manifest JSON."""

    dump_file: str
    md5: str


class _ManifestFiles(Struct, forbid_unknown_fields=False):
    """The ``files`` object of the manifest JSON."""

    bib_records: list[_ManifestEntry]


class _ManifestDocument(Struct, forbid_unknown_fields=False):
    """Top-level manifest JSON document."""

    files: _ManifestFiles
