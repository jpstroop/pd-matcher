"""Unit tests for the streaming acquisition orchestration (network mocked)."""

from collections import namedtuple
from hashlib import md5
from io import BytesIO
from pathlib import Path
from tarfile import DIRTYPE
from tarfile import TarInfo
from tarfile import open as tar_open

from lxml.etree import parse
from msgspec.json import encode
from pytest import MonkeyPatch
from pytest import raises
from responses import GET
from responses import RequestsMock

from pd_groundtruth.acquire import Md5MismatchError
from pd_groundtruth.acquire import acquire
from pd_groundtruth.disk_guard import InsufficientDiskSpaceError

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_MANIFEST_URL = "https://example.test/dumps/1.json"
_MIN_YEAR = 1931
_MB = 1 << 20

_Usage = namedtuple("_Usage", ("total", "used", "free"))


def _eligible(language: str, year: int, label: str) -> str:
    """Return an eligible record in ``language``/``year`` with a unique title."""
    field = f"750101s{year}    xxu           000 0 {language} d"
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
_BELOW_MIN_YEAR = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="008">750101s1929    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">Too old</subfield></datafield>'
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


def _eligible_records(language: str, year: int, count: int) -> list[str]:
    return [_eligible(language, year, f"{language} {year} {i}") for i in range(count)]


def test_routes_by_language_decade_and_drops_ineligible(tmp_path: Path) -> None:
    records = [
        _eligible("eng", 1935, "E1930s"),
        _eligible("eng", 1955, "E1950sA"),
        _eligible("eng", 1958, "E1950sB"),
        _eligible("fre", 1945, "F1940s"),
        _eligible("ger", 1972, "G1970s"),
        _GOVERNMENT,
        _BELOW_MIN_YEAR,
    ]
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=100,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.records_scanned == 7
    assert report.kept_by_language_decade["eng"] == {
        1930: 1,
        1940: 0,
        1950: 2,
        1960: 0,
        1970: 0,
    }
    assert report.kept_by_language_decade["fre"][1940] == 1
    assert report.kept_by_language_decade["ger"][1970] == 1
    assert report.kept_by_language == {"eng": 3, "fre": 1, "ger": 1, "spa": 0, "ita": 0}
    assert report.dumps_processed == 1
    assert report.shards_written == 3
    assert report.stopped_reason == "dumps_exhausted"

    eng_tree = parse(str(tmp_path / "out" / "eng" / "candidates_00001.xml"))
    fre_tree = parse(str(tmp_path / "out" / "fre" / "candidates_00001.xml"))
    assert len(eng_tree.getroot()) == 3
    assert len(fre_tree.getroot()) == 1
    assert not (tmp_path / "out" / "spa").exists()


def test_below_min_year_is_dropped(tmp_path: Path) -> None:
    archive = _make_targz(_collection([_BELOW_MIN_YEAR]))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=100,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.records_scanned == 1
    assert report.kept_by_language["eng"] == 0
    assert report.shards_written == 0


def test_government_publication_is_dropped(tmp_path: Path) -> None:
    archive = _make_targz(_collection([_GOVERNMENT]))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=100,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.records_scanned == 1
    assert report.kept_by_language["eng"] == 0
    assert report.shards_written == 0


def test_per_decade_quota_stops_adding(tmp_path: Path) -> None:
    records = _eligible_records("eng", 1955, 10)
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=2,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.kept_by_language_decade["eng"][1950] == 2
    assert report.kept_by_language["eng"] == 2
    assert report.stopped_reason == "dumps_exhausted"
    eng_tree = parse(str(tmp_path / "out" / "eng" / "candidates_00001.xml"))
    assert len(eng_tree.getroot()) == 2


def test_quota_applies_per_decade_within_language(tmp_path: Path) -> None:
    records = _eligible_records("eng", 1955, 5) + _eligible_records("eng", 1965, 5)
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=2,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.kept_by_language_decade["eng"][1950] == 2
    assert report.kept_by_language_decade["eng"][1960] == 2
    assert report.kept_by_language["eng"] == 4


def test_all_quotas_reached_stops_run(tmp_path: Path) -> None:
    decades = [1935, 1945, 1955, 1965, 1975]
    languages = ("eng", "fre", "ger", "spa", "ita")
    records: list[str] = []
    for language in languages:
        for year in decades:
            records.extend(_eligible_records(language, year, 2))
    archive = _make_targz(_collection(records))
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
            per_decade_cap=2,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.dumps_processed == 1
    assert report.stopped_reason == "quotas_reached"
    assert report.kept_by_language == dict.fromkeys(languages, 10)


def test_max_dumps_stops_after_first(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 1950, 2)))
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
            per_decade_cap=100,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
            max_dumps=1,
        )

    assert report.dumps_processed == 1
    assert report.kept_by_language["eng"] == 2
    assert report.stopped_reason == "max_dumps"


def test_non_file_tar_members_are_skipped(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 1950, 2)), include_dir_member=True)
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=100,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.kept_by_language["eng"] == 2


def test_md5_mismatch_raises(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 1950, 1)))
    dump_url = "https://example.test/dumps/dump1.tar.gz"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, "deadbeef")]))
        mock.add(GET, dump_url, body=archive)
        with raises(Md5MismatchError, match="md5 mismatch"):
            acquire(
                out_dir=tmp_path / "out",
                per_decade_cap=100,
                min_year=_MIN_YEAR,
                manifest_url=_MANIFEST_URL,
            )


def test_zero_per_decade_cap_stops_immediately_with_quotas_reached(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 1950, 1)))
    digest = md5(archive).hexdigest()
    dump_url = "https://example.test/dumps/dump1.tar.gz"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=0,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.dumps_processed == 0
    assert report.stopped_reason == "quotas_reached"
    assert report.shards_written == 0


def test_dumps_exhausted_across_two_dumps(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_records("eng", 1950, 2)))
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
        report = acquire(
            out_dir=tmp_path / "out",
            per_decade_cap=100,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.dumps_processed == 2
    assert report.kept_by_language["eng"] == 4
    assert report.stopped_reason == "dumps_exhausted"


def test_preflight_abort_finalizes_valid_partial_shards(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    archive_a = _make_targz(_collection(_eligible_records("eng", 1950, 3)))
    archive_b = _make_targz(_collection(_eligible_records("fre", 1960, 3)))
    url_a = "https://example.test/dumps/a.tar.gz"
    url_b = "https://example.test/dumps/b.tar.gz"
    out_dir = tmp_path / "out"

    free_values = iter([10 * 1024 * _MB, 10 * 1024 * _MB, 1 * _MB, 1 * _MB])

    def _usage(_path: object) -> _Usage:
        free = next(free_values)
        return _Usage(total=free * 2, used=free, free=free)

    monkeypatch.setattr("pd_groundtruth.disk_guard.disk_usage", _usage)

    with RequestsMock() as mock:
        mock.assert_all_requests_are_fired = False
        mock.add(
            GET,
            _MANIFEST_URL,
            body=_manifest_payload(
                [(url_a, md5(archive_a).hexdigest()), (url_b, md5(archive_b).hexdigest())]
            ),
        )
        mock.add(GET, url_a, body=archive_a)
        mock.add(GET, url_b, body=archive_b)
        with raises(InsufficientDiskSpaceError) as excinfo:
            acquire(
                out_dir=out_dir,
                per_decade_cap=100,
                min_year=_MIN_YEAR,
                manifest_url=_MANIFEST_URL,
                min_free_space_mb=2048,
            )

    assert excinfo.value.records_written == 3
    assert excinfo.value.dumps_written == 1
    eng_shard = out_dir / "eng" / "candidates_00001.xml"
    root = parse(str(eng_shard)).getroot()
    assert root.tag == f"{{{_MARC_NS}}}collection"
    assert len(root) == 3
