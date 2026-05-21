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

_DEFAULT_CAPS = {"eng": 100, "fre": 100, "ger": 100, "spa": 100, "ita": 100}


def _eligible(language: str, label: str) -> str:
    """Return an eligible record in ``language`` with a unique title."""
    field = f"750101s1950    xxu           000 0 {language} d"
    return (
        '<record xmlns="{ns}">'
        "<leader>00000nam a2200000 a 4500</leader>"
        f'<controlfield tag="008">{field}</controlfield>'
        f'<datafield tag="245"><subfield code="a">{label}</subfield></datafield>'
        "</record>"
    )


_GOVERNMENT = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="008">750101s1950    xxu          f000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">Gov report</subfield></datafield>'
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


def _eligible_records(language: str, count: int) -> list[str]:
    return [_eligible(language, f"{language} {i}") for i in range(count)]


def test_routes_by_language_and_drops_ineligible(tmp_path: Path) -> None:
    records = [
        _eligible("eng", "E1"),
        _eligible("eng", "E2"),
        _eligible("fre", "F1"),
        _eligible("ger", "G1"),
        _GOVERNMENT,
        _WRONG_YEAR,
    ]
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(out_dir=tmp_path / "out", caps=_DEFAULT_CAPS, manifest_url=_MANIFEST_URL)

    assert report.records_scanned == 6
    assert report.kept_by_language == {"eng": 2, "fre": 1, "ger": 1, "spa": 0, "ita": 0}
    assert report.dumps_processed == 1
    assert report.shards_written == 3
    assert report.stopped_reason == "dumps_exhausted"

    eng_tree = parse(str(tmp_path / "out" / "eng" / "candidates_00001.xml"))
    fre_tree = parse(str(tmp_path / "out" / "fre" / "candidates_00001.xml"))
    assert len(eng_tree.getroot()) == 2
    assert len(fre_tree.getroot()) == 1
    assert not (tmp_path / "out" / "spa").exists()


def test_government_publication_is_dropped(tmp_path: Path) -> None:
    archive = _make_targz(_collection([_GOVERNMENT]))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(out_dir=tmp_path / "out", caps=_DEFAULT_CAPS, manifest_url=_MANIFEST_URL)

    assert report.records_scanned == 1
    assert report.kept_by_language["eng"] == 0
    assert report.shards_written == 0


def test_per_language_cap_stops_adding(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 10)))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()
    caps = {"eng": 3, "fre": 5}

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(out_dir=tmp_path / "out", caps=caps, manifest_url=_MANIFEST_URL)

    assert report.kept_by_language == {"eng": 3, "fre": 0}
    assert report.stopped_reason == "dumps_exhausted"
    eng_tree = parse(str(tmp_path / "out" / "eng" / "candidates_00001.xml"))
    assert len(eng_tree.getroot()) == 3


def test_all_caps_reached_stops_run(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 5)))
    digest = md5(archive).hexdigest()
    url_a = "https://example.test/dumps/a.tar.gz"
    url_b = "https://example.test/dumps/b.tar.gz"
    caps = {"eng": 2}

    with RequestsMock() as mock:
        mock.add(
            GET,
            _MANIFEST_URL,
            body=_manifest_payload([(url_a, digest), (url_b, digest)]),
        )
        mock.add(GET, url_a, body=archive)
        report = acquire(out_dir=tmp_path / "out", caps=caps, manifest_url=_MANIFEST_URL)

    assert report.dumps_processed == 1
    assert report.kept_by_language == {"eng": 2}
    assert report.stopped_reason == "caps_reached"


def test_max_dumps_stops_after_first(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 2)))
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
            caps=_DEFAULT_CAPS,
            manifest_url=_MANIFEST_URL,
            max_dumps=1,
        )

    assert report.dumps_processed == 1
    assert report.kept_by_language["eng"] == 2
    assert report.stopped_reason == "max_dumps"


def test_non_file_tar_members_are_skipped(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 2)), include_dir_member=True)
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(out_dir=tmp_path / "out", caps=_DEFAULT_CAPS, manifest_url=_MANIFEST_URL)

    assert report.kept_by_language["eng"] == 2


def test_md5_mismatch_raises(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 1)))
    dump_url = "https://example.test/dumps/dump1.tar.gz"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, "deadbeef")]))
        mock.add(GET, dump_url, body=archive)
        with raises(Md5MismatchError, match="md5 mismatch"):
            acquire(out_dir=tmp_path / "out", caps=_DEFAULT_CAPS, manifest_url=_MANIFEST_URL)


def test_dumps_exhausted_across_two_dumps(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 2)))
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
        report = acquire(out_dir=tmp_path / "out", caps=_DEFAULT_CAPS, manifest_url=_MANIFEST_URL)

    assert report.dumps_processed == 2
    assert report.kept_by_language["eng"] == 4
    assert report.stopped_reason == "dumps_exhausted"
