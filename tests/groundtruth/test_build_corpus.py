"""Unit tests for the uncapped, streaming corpus extractor (network mocked)."""

from hashlib import md5
from io import BytesIO
from pathlib import Path
from tarfile import TarInfo
from tarfile import open as tar_open
from unittest.mock import patch

from lxml.etree import parse
from msgspec.json import encode
from pytest import raises
from responses import GET
from responses import RequestsMock

from pd_groundtruth.acquire import Md5MismatchError
from pd_groundtruth.build_corpus import build_corpus
from pd_matcher.parsers.marc import iter_marc_records

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_MANIFEST_URL = "https://example.test/dumps/1.json"
_MIN_YEAR = 1931


def _eligible(language: str, year: int, label: str, control_001: str) -> str:
    """Return an eligible record in ``language``/``year`` with a unique id."""
    field = f"750101s{year}    xxu           000 0 {language} d"
    return (
        '<record xmlns="{ns}">'
        "<leader>00000nam a2200000 a 4500</leader>"
        f'<controlfield tag="001">{control_001}</controlfield>'
        f'<controlfield tag="008">{field}</controlfield>'
        f'<datafield tag="245"><subfield code="a">{label}</subfield></datafield>'
        "</record>"
    )


_GOVERNMENT = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">gov1</controlfield>'
    '<controlfield tag="008">750101s1950    xxu          f000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">Gov report</subfield></datafield>'
    "</record>"
)
_BELOW_MIN_YEAR = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">old1</controlfield>'
    '<controlfield tag="008">750101s1929    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">Too old</subfield></datafield>'
    "</record>"
)
_SOUND_RECORDING = (
    '<record xmlns="{ns}">'
    "<leader>00000njm a2200000 a 4500</leader>"
    '<controlfield tag="001">snd1</controlfield>'
    '<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">A recording</subfield></datafield>'
    "</record>"
)
_ELECTRONIC = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">ebk1</controlfield>'
    '<controlfield tag="007">cr</controlfield>'
    '<controlfield tag="008">750101s1950    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">An e-book</subfield></datafield>'
    "</record>"
)
_UNSUPPORTED_LANGUAGE = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">lat1</controlfield>'
    '<controlfield tag="008">750101s1950    xxu           000 0 lat d</controlfield>'
    '<datafield tag="245"><subfield code="a">Latina</subfield></datafield>'
    "</record>"
)
_INVALID_YEAR = (
    '<record xmlns="{ns}">'
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">bad1</controlfield>'
    '<controlfield tag="008">750101suuuu    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245"><subfield code="a">No year</subfield></datafield>'
    "</record>"
)


def _collection(records: list[str]) -> bytes:
    """Wrap record XML fragments in a MARCXML collection document."""
    inner = "".join(r.format(ns=_MARC_NS) for r in records)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><collection xmlns="{_MARC_NS}">{inner}</collection>'
    ).encode()


def _make_targz(collection_xml: bytes) -> bytes:
    """Build a gzip tar archive containing one MARCXML member."""
    buffer = BytesIO()
    with tar_open(fileobj=buffer, mode="w:gz") as archive:
        info = TarInfo(name="records.xml")
        info.size = len(collection_xml)
        archive.addfile(info, BytesIO(collection_xml))
    return buffer.getvalue()


def _manifest_payload(entries: list[tuple[str, str]]) -> bytes:
    """Encode a manifest JSON document for the given (url, md5) pairs."""
    files = {"bib_records": [{"dump_file": url, "md5": digest} for url, digest in entries]}
    return encode({"files": files})


def _eligible_many(language: str, year: int, count: int) -> list[str]:
    """Return ``count`` eligible records all in the same language/decade."""
    return [
        _eligible(language, year, f"{language} {year} {i}", f"{language}{year}{i}")
        for i in range(count)
    ]


def _titles(output_path: Path) -> list[str]:
    """Return the 245 $a titles of every record in an output collection."""
    tree = parse(str(output_path))
    titles: list[str] = []
    for subfield in tree.iter(f"{{{_MARC_NS}}}subfield"):
        if subfield.get("code") == "a":
            titles.append(subfield.text or "")
    return titles


def test_keeps_every_eligible_record_uncapped(tmp_path: Path) -> None:
    records = _eligible_many("eng", 1955, 50)
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()
    output_path = tmp_path / "corpus.marcxml"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = build_corpus(
            output_path=output_path,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.dumps_processed == 1
    assert report.records_scanned == 50
    assert report.kept == 50
    assert report.dropped == 0
    assert report.dropped_by_reason == {}
    assert len(parse(str(output_path)).getroot()) == 50


def test_drops_each_reason(tmp_path: Path) -> None:
    records = [
        _eligible("eng", 1955, "Keep", "k1"),
        _BELOW_MIN_YEAR,
        _GOVERNMENT,
        _SOUND_RECORDING,
        _ELECTRONIC,
        _UNSUPPORTED_LANGUAGE,
        _INVALID_YEAR,
    ]
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()
    output_path = tmp_path / "corpus.marcxml"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = build_corpus(
            output_path=output_path,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.records_scanned == 7
    assert report.kept == 1
    assert report.dropped == 6
    assert report.dropped_by_reason == {
        "year_out_of_range": 1,
        "government_publication": 1,
        "not_a_book": 1,
        "electronic_resource": 1,
        "unsupported_language": 1,
        "invalid_year": 1,
    }
    assert _titles(output_path) == ["Keep"]


def test_languages_narrows_within_eligible(tmp_path: Path) -> None:
    records = [
        _eligible("eng", 1955, "English", "e1"),
        _eligible("fre", 1955, "French", "f1"),
        _eligible("ger", 1955, "German", "g1"),
    ]
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()
    output_path = tmp_path / "corpus.marcxml"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        report = build_corpus(
            output_path=output_path,
            min_year=_MIN_YEAR,
            languages=frozenset({"eng", "fre"}),
            manifest_url=_MANIFEST_URL,
        )

    assert report.kept == 2
    assert report.dropped == 1
    assert report.dropped_by_reason == {"language_not_requested": 1}
    assert sorted(_titles(output_path)) == ["English", "French"]


def test_streams_and_appends_across_dumps(tmp_path: Path) -> None:
    archive_a = _make_targz(_collection(_eligible_many("eng", 1950, 3)))
    archive_b = _make_targz(_collection(_eligible_many("fre", 1960, 4)))
    url_a = "https://example.test/dumps/a.tar.gz"
    url_b = "https://example.test/dumps/b.tar.gz"
    output_path = tmp_path / "corpus.marcxml"

    with RequestsMock() as mock:
        mock.add(
            GET,
            _MANIFEST_URL,
            body=_manifest_payload(
                [(url_a, md5(archive_a).hexdigest()), (url_b, md5(archive_b).hexdigest())]
            ),
        )
        mock.add(GET, url_a, body=archive_a)
        mock.add(GET, url_b, body=archive_b)
        report = build_corpus(
            output_path=output_path,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    assert report.dumps_processed == 2
    assert report.records_scanned == 7
    assert report.kept == 7
    assert len(parse(str(output_path)).getroot()) == 7


def test_max_dumps_stops_after_first(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_many("eng", 1950, 2)))
    digest = md5(archive).hexdigest()
    url_a = "https://example.test/dumps/a.tar.gz"
    url_b = "https://example.test/dumps/b.tar.gz"
    output_path = tmp_path / "corpus.marcxml"

    with RequestsMock() as mock:
        mock.add(
            GET,
            _MANIFEST_URL,
            body=_manifest_payload([(url_a, digest), (url_b, digest)]),
        )
        mock.add(GET, url_a, body=archive)
        report = build_corpus(
            output_path=output_path,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
            max_dumps=1,
        )

    assert report.dumps_processed == 1
    assert report.kept == 2


def test_output_round_trips_through_match_reader(tmp_path: Path) -> None:
    records = [
        _eligible("eng", 1955, "Round trip one", "rt1"),
        _eligible("fre", 1955, "Round trip two", "rt2"),
        _UNSUPPORTED_LANGUAGE,
    ]
    archive = _make_targz(_collection(records))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    digest = md5(archive).hexdigest()
    output_path = tmp_path / "corpus.marcxml"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, digest)]))
        mock.add(GET, dump_url, body=archive)
        build_corpus(
            output_path=output_path,
            min_year=_MIN_YEAR,
            manifest_url=_MANIFEST_URL,
        )

    parsed = list(iter_marc_records(output_path))
    assert sorted(record.title for record in parsed) == ["Round trip one", "Round trip two"]
    assert {record.control_id for record in parsed} == {"rt1", "rt2"}


def test_md5_mismatch_raises(tmp_path: Path) -> None:
    archive = _make_targz(_collection(_eligible_many("eng", 1950, 1)))
    dump_url = "https://example.test/dumps/dump1.tar.gz"
    output_path = tmp_path / "corpus.marcxml"

    with RequestsMock() as mock:
        mock.add(GET, _MANIFEST_URL, body=_manifest_payload([(dump_url, "deadbeef")]))
        mock.add(GET, dump_url, body=archive)
        with raises(Md5MismatchError, match="md5 mismatch"):
            build_corpus(
                output_path=output_path,
                min_year=_MIN_YEAR,
                manifest_url=_MANIFEST_URL,
            )


def test_temp_downloads_are_deleted(tmp_path: Path) -> None:
    archive_a = _make_targz(_collection(_eligible_many("eng", 1950, 2)))
    archive_b = _make_targz(_collection(_eligible_many("fre", 1960, 2)))
    url_a = "https://example.test/dumps/a.tar.gz"
    url_b = "https://example.test/dumps/b.tar.gz"
    output_path = tmp_path / "corpus.marcxml"
    captured: list[Path] = []

    from pd_groundtruth.acquire import _download_and_verify
    from pd_groundtruth.manifest import DumpEntry

    def _spy(entry: DumpEntry) -> Path:
        temp = _download_and_verify(entry)
        captured.append(temp)
        return temp

    with RequestsMock() as mock:
        mock.add(
            GET,
            _MANIFEST_URL,
            body=_manifest_payload(
                [(url_a, md5(archive_a).hexdigest()), (url_b, md5(archive_b).hexdigest())]
            ),
        )
        mock.add(GET, url_a, body=archive_a)
        mock.add(GET, url_b, body=archive_b)
        with patch("pd_groundtruth.acquire._download_and_verify", side_effect=_spy):
            build_corpus(
                output_path=output_path,
                min_year=_MIN_YEAR,
                manifest_url=_MANIFEST_URL,
            )

    assert len(captured) == 2
    assert all(not temp.exists() for temp in captured)
