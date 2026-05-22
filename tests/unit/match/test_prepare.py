"""Tests for :mod:`pd_matcher.match.prepare`."""

from pathlib import Path

from pytest import raises

from pd_matcher.match.prepare import PreparedManifest
from pd_matcher.match.prepare import PrepareReport
from pd_matcher.match.prepare import compute_source_hash
from pd_matcher.match.prepare import iter_prepared_records
from pd_matcher.match.prepare import prepare_marc
from pd_matcher.match.prepare import read_manifest
from pd_matcher.parsers.marc import iter_marc_records

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_MARC = _FIXTURES / "tiny.marcxml"


def _expected_record_count() -> int:
    """Count records the parser actually emits from the tiny fixture."""
    return sum(1 for _ in iter_marc_records(_MARC))


def test_prepare_marc_writes_chunks_and_manifest(tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    report = prepare_marc(_MARC, out, chunk_size=5)
    assert isinstance(report, PrepareReport)
    expected = _expected_record_count()
    assert report.total_records == expected
    assert report.skipped is False
    assert report.chunk_count == (expected + 4) // 5
    manifest = read_manifest(out)
    assert isinstance(manifest, PreparedManifest)
    assert manifest.total_records == expected
    assert len(manifest.chunk_files) == report.chunk_count
    for name in manifest.chunk_files:
        assert (out / name).is_file()


def test_prepare_marc_rejects_zero_chunk_size(tmp_path: Path) -> None:
    with raises(ValueError, match="chunk_size must be >= 1"):
        prepare_marc(_MARC, tmp_path / "p", chunk_size=0)


def test_prepare_marc_rejects_missing_source(tmp_path: Path) -> None:
    with raises(FileNotFoundError, match="source does not exist"):
        prepare_marc(tmp_path / "nope.xml", tmp_path / "p")


def test_prepare_marc_single_chunk_when_chunk_size_exceeds_records(tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    report = prepare_marc(_MARC, out, chunk_size=10_000)
    assert report.chunk_count == 1
    assert read_manifest(out).chunk_files == ("chunk_00000.pkl",)


def test_prepare_marc_is_idempotent_on_second_run(tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    first = prepare_marc(_MARC, out, chunk_size=5)
    assert first.skipped is False
    second = prepare_marc(_MARC, out, chunk_size=5)
    assert second.skipped is True
    assert second.total_records == first.total_records
    assert second.chunk_count == first.chunk_count


def test_prepare_marc_force_rebuilds(tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    prepare_marc(_MARC, out, chunk_size=5)
    rebuilt = prepare_marc(_MARC, out, chunk_size=5, force=True)
    assert rebuilt.skipped is False


def test_prepare_marc_rebuild_removes_orphan_chunks(tmp_path: Path) -> None:
    """A re-run with a smaller chunk count leaves no stale chunks behind."""
    out = tmp_path / "prepared"
    many = prepare_marc(_MARC, out, chunk_size=2)
    assert many.chunk_count > 1
    few = prepare_marc(_MARC, out, chunk_size=10_000, force=True)
    assert few.chunk_count == 1
    on_disk = sorted(p.name for p in out.glob("chunk_*.pkl"))
    assert on_disk == ["chunk_00000.pkl"]


def test_prepare_marc_accepts_directory_source(tmp_path: Path) -> None:
    source = tmp_path / "shards"
    source.mkdir()
    (source / "a.xml").write_bytes(_MARC.read_bytes())
    (source / "b.xml").write_bytes(_MARC.read_bytes())
    out = tmp_path / "prepared"
    report = prepare_marc(source, out, chunk_size=1000)
    assert report.total_records == 2 * _expected_record_count()


def test_iter_prepared_records_roundtrips_parser_output(tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    prepare_marc(_MARC, out, chunk_size=3)
    original = list(iter_marc_records(_MARC))
    replayed = list(iter_prepared_records(out))
    assert replayed == original


def test_compute_source_hash_changes_with_content(tmp_path: Path) -> None:
    one = tmp_path / "one.xml"
    one.write_bytes(_MARC.read_bytes())
    before = compute_source_hash(one)
    one.write_bytes(_MARC.read_bytes() + b"<!-- changed -->")
    after = compute_source_hash(one)
    assert before != after


def test_compute_source_hash_directory_is_relative_path_stable(tmp_path: Path) -> None:
    """Two directories with identically named/sized files hash the same."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    payload = _MARC.read_bytes()
    (dir_a / "x.xml").write_bytes(payload)
    (dir_b / "x.xml").write_bytes(payload)
    import os

    mtime = (dir_a / "x.xml").stat().st_mtime_ns
    os.utime((dir_b / "x.xml"), ns=(mtime, mtime))
    assert compute_source_hash(dir_a) == compute_source_hash(dir_b)


def test_read_manifest_missing_raises(tmp_path: Path) -> None:
    with raises(FileNotFoundError):
        read_manifest(tmp_path)
