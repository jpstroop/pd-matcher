"""Tests for :mod:`pd_matcher.index.builder`."""

from datetime import date
from pathlib import Path
from time import sleep

from pd_matcher.index.builder import build_index
from pd_matcher.index.codec import decode_reg
from pd_matcher.index.codec import decode_uuid_list
from pd_matcher.index.codec import encode_year_key
from pd_matcher.index.codec import make_renewal_key
from pd_matcher.index.store import NyplIndexStore

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _seed_sources(root: Path) -> tuple[Path, Path]:
    """Copy the tiny reg/ren fixtures into isolated source directories."""
    reg_dir = root / "reg"
    ren_dir = root / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    return reg_dir, ren_dir


def test_build_index_writes_records_year_buckets_and_meta(tmp_path: Path) -> None:
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    report = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    assert report.skipped is False
    assert report.registrations_written == 8
    assert report.renewals_written == 4
    assert report.renewal_joins == 2
    assert report.year_buckets == 4
    assert report.duration_seconds >= 0.0

    with NyplIndexStore(out_path, readonly=True) as store:
        widgets_blob = store.reg_by_id.get(b"UUID-0001")
        assert widgets_blob is not None
        widgets = decode_reg(widgets_blob)
        assert widgets.was_renewed is True
        assert widgets.reg_year == 1940

        bad_date_blob = store.reg_by_id.get(b"UUID-0008")
        assert bad_date_blob is not None
        assert decode_reg(bad_date_blob).was_renewed is False

        bucket_1940 = store.reg_by_year.get(encode_year_key(1940))
        assert bucket_1940 is not None
        assert decode_uuid_list(bucket_1940) == ("UUID-0001",)

        join_key = make_renewal_key("A111111", date(1940, 5, 10))
        assert store.ren_by_oreg.get(join_key) == b"entry-001"


def test_build_index_is_idempotent_without_force(tmp_path: Path) -> None:
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    first = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    assert first.skipped is False

    second = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    assert second.skipped is True
    assert second.registrations_written == 0
    assert second.renewals_written == 0
    assert second.renewal_joins == 0
    assert second.year_buckets == 0


def test_build_index_force_rebuilds_existing_env(tmp_path: Path) -> None:
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    forced = build_index(
        reg_dir=reg_dir,
        ren_dir=ren_dir,
        out_path=out_path,
        force=True,
    )
    assert forced.skipped is False
    assert forced.registrations_written == 8


def test_build_index_rebuilds_when_schema_version_changes(tmp_path: Path) -> None:
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path, schema_version=1)
    rebuilt = build_index(
        reg_dir=reg_dir,
        ren_dir=ren_dir,
        out_path=out_path,
        schema_version=2,
    )
    assert rebuilt.skipped is False
    assert rebuilt.registrations_written == 8


def test_build_index_rebuilds_when_source_files_change(tmp_path: Path) -> None:
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    # Touching the file changes its mtime so the source hash drifts.
    sleep(0.01)
    target = ren_dir / "tiny_ren.tsv"
    target.write_bytes(target.read_bytes())

    rebuilt = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    assert rebuilt.skipped is False


def test_build_index_skips_registrations_without_regnum(tmp_path: Path) -> None:
    """UUID-0004 lacks ``regnum`` so its renewal lookup is bypassed entirely."""
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    with NyplIndexStore(out_path, readonly=True) as store:
        blob = store.reg_by_id.get(b"UUID-0004")
        assert blob is not None
        record = decode_reg(blob)
        assert record.regnum is None
        assert record.was_renewed is False
