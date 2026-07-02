"""Unit tests for manifest discovery and parsing (network mocked)."""

from msgspec.json import encode
from pytest import raises
from responses import GET
from responses import RequestsMock

from pd_groundtruth.manifest import DEFAULT_EVENTS_URL
from pd_groundtruth.manifest import DumpEntry
from pd_groundtruth.manifest import discover_full_dump_url
from pd_groundtruth.manifest import fetch_manifest
from pd_groundtruth.manifest import parse_manifest


def _manifest_payload(entries: list[tuple[str, str]]) -> bytes:
    """Encode a manifest JSON document for the given ``(dump_file, md5)`` pairs."""
    return encode({"files": {"bib_records": [{"dump_file": u, "md5": m} for u, m in entries]}})


def _event(dump_type: str, success: bool, dump_url: str, finish: str | None) -> dict[str, object]:
    """Build one bibdata event dict for the mocked event log."""
    return {
        "dump_type": dump_type,
        "success": success,
        "dump_url": dump_url,
        "finish": finish,
    }


def test_parse_manifest_returns_ordered_entries() -> None:
    payload = _manifest_payload([("https://x/a.tar.gz", "aaa"), ("https://x/b.tar.gz", "bbb")])
    entries = parse_manifest(payload)
    assert entries == (
        DumpEntry(url="https://x/a.tar.gz", md5="aaa"),
        DumpEntry(url="https://x/b.tar.gz", md5="bbb"),
    )


def test_parse_manifest_empty_entries_raises() -> None:
    payload = encode({"files": {"bib_records": []}})
    with raises(ValueError, match=r"no files\.bib_records"):
        parse_manifest(payload)


def test_parse_manifest_non_json_raises_clear_error() -> None:
    with raises(ValueError, match=r"not valid manifest JSON.*--manifest-url"):
        parse_manifest(b"<!DOCTYPE html><title>Error</title>")


def test_discover_full_dump_url_returns_latest_successful_full_dump() -> None:
    events = [
        _event("changed_records", True, "https://x/dumps/1.json", "2026-06-18T00:00:00Z"),
        _event("full_dump", True, "https://x/dumps/2.json", "2026-05-01T00:00:00Z"),
        _event("full_dump", False, "https://x/dumps/3.json", "2026-06-30T00:00:00Z"),
        _event("full_dump", True, "https://x/dumps/4.json", "2026-06-19T00:00:00Z"),
    ]
    with RequestsMock() as mock:
        mock.add(GET, DEFAULT_EVENTS_URL, body=encode(events))
        assert discover_full_dump_url() == "https://x/dumps/4.json"


def test_discover_full_dump_url_no_full_dumps_raises() -> None:
    events = [_event("changed_records", True, "https://x/dumps/1.json", "2026-06-18T00:00:00Z")]
    with RequestsMock() as mock:
        mock.add(GET, DEFAULT_EVENTS_URL, body=encode(events))
        with raises(ValueError, match=r"no successful full_dump events.*--manifest-url"):
            discover_full_dump_url()


def test_discover_full_dump_url_non_json_raises() -> None:
    with RequestsMock() as mock:
        mock.add(GET, DEFAULT_EVENTS_URL, body=b"<!DOCTYPE html><title>Error</title>")
        with raises(ValueError, match=r"not the expected JSON array.*--manifest-url"):
            discover_full_dump_url()


def test_fetch_manifest_with_explicit_url_skips_discovery() -> None:
    manifest_url = "https://x/dumps/9.json"
    with RequestsMock() as mock:
        mock.add(GET, manifest_url, body=_manifest_payload([("https://x/a.tar.gz", "aaa")]))
        entries = fetch_manifest(manifest_url)
    assert entries == (DumpEntry(url="https://x/a.tar.gz", md5="aaa"),)


def test_fetch_manifest_default_discovers_latest_full_dump() -> None:
    dump_url = "https://x/dumps/4.json"
    events = [_event("full_dump", True, dump_url, "2026-06-19T00:00:00Z")]
    with RequestsMock() as mock:
        mock.add(GET, DEFAULT_EVENTS_URL, body=encode(events))
        mock.add(GET, dump_url, body=_manifest_payload([("https://x/z.tar.gz", "zzz")]))
        entries = fetch_manifest()
    assert entries == (DumpEntry(url="https://x/z.tar.gz", md5="zzz"),)
