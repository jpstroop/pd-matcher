"""Unit tests for manifest parsing."""

from msgspec.json import encode
from pytest import raises

from pd_groundtruth.manifest import DumpEntry
from pd_groundtruth.manifest import parse_manifest


def test_parse_manifest_returns_ordered_entries() -> None:
    payload = encode(
        {
            "files": {
                "bib_records": [
                    {"dump_file": "https://x/a.tar.gz", "md5": "aaa"},
                    {"dump_file": "https://x/b.tar.gz", "md5": "bbb"},
                ]
            }
        }
    )
    entries = parse_manifest(payload)
    assert entries == (
        DumpEntry(url="https://x/a.tar.gz", md5="aaa"),
        DumpEntry(url="https://x/b.tar.gz", md5="bbb"),
    )


def test_parse_manifest_empty_entries_raises() -> None:
    payload = encode({"files": {"bib_records": []}})
    with raises(ValueError, match=r"no files\.bib_records"):
        parse_manifest(payload)
