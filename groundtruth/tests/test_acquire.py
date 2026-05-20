"""Unit tests for the streaming acquisition orchestration (network mocked)."""

from hashlib import md5
from io import BytesIO
from pathlib import Path
from tarfile import DIRTYPE
from tarfile import TarInfo
from tarfile import open as tar_open

from lxml.etree import parse
from msgspec.json import encode
from pytest import raises
from responses import GET
from responses import RequestsMock

from pd_groundtruth.acquire import Md5MismatchError
from pd_groundtruth.acquire import acquire

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_MANIFEST_URL = "https://example.test/dumps/1.json"

_ELIGIBLE = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">Eligible {n}</subfield></datafield>'
    "</record>"
)
_NOT_A_BOOK = (
    '<record xmlns="{ns}">'
    "<leader>00000ncm a2200000 a 4500</leader>"
    '<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">Score</subfield></datafield>'
    "</record>"
)
_WRONG_YEAR = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="008">750101s1850    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">Old</subfield></datafield>'
    "</record>"
)


def _collection(records: list[str]) -> bytes:
    """Wrap record XML fragments in a MARCXML collection document."""
    inner = "".join(r.format(ns=_MARC_NS) for r in records)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><collection xmlns="{_MARC_NS}">{inner}</collection>'
    ).encode()


def _make_targz(collection_xml: bytes, *, include_dir_member: bool = False) -> bytes:
    """Build a gzip tar archive containing one MARCXML member."""
    buffer = BytesIO()
    with tar_open(fileobj=buffer, mode="w:gz") as archive:
        if include_dir_member:
            dir_info = TarInfo(name="subdir/")
            dir_info.type = DIRTYPE
            archive.addfile(dir_info)
        info = TarInfo(name="records.xml")
        info.size = len(collection_xml)
        archive.addfile(info, BytesIO(collection_xml))
    return buffer.getvalue()


def _manifest_payload(entries: list[tuple[str, str]]) -> bytes:
    """Encode a manifest JSON document for the given (url, md5) pairs."""
    files = {"bib_records": [{"dump_file": url, "md5": digest} for url, digest in entries]}
    return encode({"files": files})


def _eligible_records(count: int) -> list[str]:
    return [_ELIGIBLE.replace("{n}", str(i)) for i in range(count)]


def test_keeps_only_eligible_records(tmp_path: Path) -> None:
    records = [
        *_eligible_records(2),
        _NOT_A_BOOK,
        _WRONG_YEAR,
    ]
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(out_dir=tmp_path / "out", manifest_url=_MANIFEST_URL)

    assert report.records_scanned == 4
    assert report.records_kept == 2
    assert report.dumps_processed == 1
    assert report.shards_written == 1
    assert report.stopped_reason == "dumps_exhausted"

    tree = parse(str(tmp_path / "out" / "candidates_00001.xml"))
    assert len(tree.getroot()) == 2


def test_max_records_stops_mid_run(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records(10)))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            manifest_url=_MANIFEST_URL,
            max_records=3,
        )

    assert report.records_kept == 3
    assert report.stopped_reason == "max_records"


def test_max_dumps_stops_after_first(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records(2)))
    digest = md5(archive).hexdigest()
    url_a = "https://example.test/dumps/a.tar.gz"
    url_b = "https://example.test/dumps/b.tar.gz"

    with RequestsMock() as mock:
        mock.add(
            GET,
            _MANIFEST_URL,
            body=_manifest_payload([(url_a, digest), (url_b, digest)]),
        )
        mock.add(GET, url_a, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            manifest_url=_MANIFEST_URL,
            max_dumps=1,
        )

    assert report.dumps_processed == 1
    assert report.records_kept == 2
    assert report.stopped_reason == "max_dumps"


def test_non_file_tar_members_are_skipped(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records(2)), include_dir_member=True)
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(out_dir=tmp_path / "out", manifest_url=_MANIFEST_URL)

    assert report.records_kept == 2


def test_md5_mismatch_raises(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records(1)))
    dump_url = "https://example.test/dumps/dump1.tar.gz"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, "deadbeef")]))
        mock.add(GET, dump_url, body=archive)
        with raises(Md5MismatchError, match="md5 mismatch"):
            acquire(out_dir=tmp_path / "out", manifest_url=_MANIFEST_URL)


def test_dumps_exhausted_across_two_dumps(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records(2)))
    digest = md5(archive).hexdigest()
    url_a = "https://example.test/dumps/a.tar.gz"
    url_b = "https://example.test/dumps/b.tar.gz"

    with RequestsMock() as mock:
        mock.add(
            GET,
            _MANIFEST_URL,
            body=_manifest_payload([(url_a, digest), (url_b, digest)]),
        )
        mock.add(GET, url_a, body=archive)
        mock.add(GET, url_b, body=archive)
        report = acquire(out_dir=tmp_path / "out", manifest_url=_MANIFEST_URL)

    assert report.dumps_processed == 2
    assert report.records_kept == 4
    assert report.stopped_reason == "dumps_exhausted"
