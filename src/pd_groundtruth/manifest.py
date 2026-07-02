"""Discover, fetch, and parse the Princeton ``bibdata`` dump manifest.

A manifest is a JSON document describing the set of MARC dump files that make up
one full bibliographic export. Each ``bib_records`` entry carries a download URL
and an md5 checksum that we verify after streaming the file to disk.

Princeton rotates dump IDs, so a hardcoded manifest URL goes stale and starts
returning an HTML error page. Instead of pinning an ID, :func:`discover_full_dump_url`
reads the public event log at :data:`DEFAULT_EVENTS_URL` and returns the most
recent successful ``full_dump`` event's manifest URL, which
:func:`fetch_manifest` uses by default.
"""

from logging import getLogger

from msgspec import DecodeError
from msgspec import Struct
from msgspec.json import decode
from requests import get

_LOGGER = getLogger(__name__)

DEFAULT_EVENTS_URL = "https://bibdata.princeton.edu/events.json"

_FULL_DUMP_TYPE = "full_dump"
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
        ValueError: If the body is not valid manifest JSON (e.g. an HTML error
            page from an expired URL), or if it declares no
            ``files.bib_records`` entries.
    """
    try:
        document = decode(payload, type=_ManifestDocument)
    except DecodeError as error:
        raise ValueError(
            f"manifest body is not valid manifest JSON ({len(payload)} bytes); "
            "pass --manifest-url with a current full-dump URL"
        ) from error
    entries = document.files.bib_records
    if not entries:
        raise ValueError("manifest contains no files.bib_records entries")
    return tuple(DumpEntry(url=entry.dump_file, md5=entry.md5) for entry in entries)


def discover_full_dump_url(events_url: str = DEFAULT_EVENTS_URL) -> str:
    """Return the manifest URL of Princeton's most recent successful full dump.

    Reads the bibdata event log, keeps the successful ``full_dump`` events, and
    returns the newest one's ``dump_url`` (events sort by their ISO-8601
    ``finish`` timestamp). This replaces hardcoding a dump ID that expires.

    Args:
        events_url: Absolute URL of the bibdata event-log JSON array.

    Returns:
        The absolute manifest URL of the latest successful full dump.

    Raises:
        ValueError: If the event log is not valid JSON, or lists no successful
            full-dump events.
    """
    _LOGGER.info("discovering full dump: %s", events_url)
    response = get(events_url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    try:
        events = decode(response.content, type=list[_DumpEvent])
    except DecodeError as error:
        raise ValueError(
            f"event log at {events_url} is not the expected JSON array; "
            "pass --manifest-url with a current full-dump URL"
        ) from error
    full_dumps = [event for event in events if event.dump_type == _FULL_DUMP_TYPE and event.success]
    if not full_dumps:
        raise ValueError(
            f"event log at {events_url} lists no successful full_dump events; "
            "pass --manifest-url with a current full-dump URL"
        )
    latest = max(full_dumps, key=lambda event: event.finish or "")
    _LOGGER.info("discovered full dump: %s (finished %s)", latest.dump_url, latest.finish)
    return latest.dump_url


def fetch_manifest(manifest_url: str | None = None) -> tuple[DumpEntry, ...]:
    """Download and parse a dump manifest, discovering the latest full dump by default.

    Args:
        manifest_url: Absolute URL of the manifest JSON document. When ``None``
            (the default), :func:`discover_full_dump_url` resolves the latest
            successful full dump instead of pinning an ID that expires.

    Returns:
        Dump entries in manifest order.
    """
    resolved_url = manifest_url if manifest_url is not None else discover_full_dump_url()
    _LOGGER.info("fetching manifest: %s", resolved_url)
    response = get(resolved_url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    entries = parse_manifest(response.content)
    _LOGGER.info("manifest parsed: %d dump entries", len(entries))
    return entries


class _DumpEvent(Struct, forbid_unknown_fields=False):
    """One bibdata dump event; ``dump_url`` points at that dump's manifest."""

    dump_type: str
    success: bool
    dump_url: str
    finish: str | None = None


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
