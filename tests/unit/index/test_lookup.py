"""Tests for :mod:`pd_matcher.index.lookup`."""

from pathlib import Path

from pytest import raises

from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.index.store import NyplIndexStore

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _build_tiny_index(root: Path) -> Path:
    reg_dir = root / "reg"
    ren_dir = root / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = root / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def test_lookup_get_registration_returns_indexed_record(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        record = lookup.get_registration("UUID-0001")
        assert record is not None
        assert record.regnum == "A111111"
        assert record.was_renewed is True


def test_lookup_get_registration_returns_none_for_missing(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        assert lookup.get_registration("does-not-exist") is None


def test_lookup_get_renewal_returns_record(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        renewal = lookup.get_renewal("entry-001")
        assert renewal is not None
        assert renewal.id == "R200001"
        assert renewal.oreg == "A111111"


def test_lookup_get_renewal_returns_none_for_missing(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        assert lookup.get_renewal("missing-entry") is None


def test_candidates_for_year_zero_window(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        records = list(lookup.candidates_for_year(1940))
    assert [r.uuid for r in records] == ["UUID-0001"]


def test_candidates_for_year_with_window_walks_buckets_in_ascending_year_order(
    tmp_path: Path,
) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        records = list(lookup.candidates_for_year(1960, window=5))
    # Year buckets in range 1955..1965 contain UUID-0002 (1955), UUID-0003
    # (1962), UUID-0011 (1962), and UUID-0004 (1965); they are walked in
    # ascending year order.
    uuids = [r.uuid for r in records]
    assert uuids == ["UUID-0002", "UUID-0003", "UUID-0011", "UUID-0004"]


def test_candidates_for_year_returns_nothing_for_empty_bucket(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        assert list(lookup.candidates_for_year(1800, window=10)) == []


def test_candidates_for_year_dedupes_when_uuid_appears_in_two_buckets(tmp_path: Path) -> None:
    """If a uuid appears in two year buckets it is only yielded once."""
    out_path = _build_tiny_index(tmp_path)
    # Manually plant the same uuid into a second year bucket so the dedupe
    # path inside `candidates_for_year` is actually exercised.
    from pd_matcher.index.codec import encode_uuid_list
    from pd_matcher.index.codec import encode_year_key

    with NyplIndexStore(out_path) as store, store.write_transaction():
        store.reg_by_year.put(encode_year_key(1941), encode_uuid_list(("UUID-0001",)))

    with NyplIndexLookup(out_path) as lookup:
        records = list(lookup.candidates_for_year(1940, window=1))
    assert [r.uuid for r in records] == ["UUID-0001"]


def test_candidates_for_year_skips_uuid_with_no_registration(tmp_path: Path) -> None:
    """A bucket entry that points at a missing registration is skipped."""
    out_path = _build_tiny_index(tmp_path)
    from pd_matcher.index.codec import encode_uuid_list
    from pd_matcher.index.codec import encode_year_key

    with NyplIndexStore(out_path) as store, store.write_transaction():
        store.reg_by_year.put(
            encode_year_key(1999),
            encode_uuid_list(("UUID-MISSING",)),
        )

    with NyplIndexLookup(out_path) as lookup:
        records = list(lookup.candidates_for_year(1999))
    assert records == []


def test_stats_reflect_build_report(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        stats = lookup.stats()
    assert stats.schema_version == 2
    assert stats.registrations_written == 9
    assert stats.renewals_written == 4
    assert stats.renewal_joins == 2
    assert stats.year_buckets == 4
    assert stats.build_timestamp.endswith("+00:00")
    assert len(stats.source_hash) == 64


def test_stats_raises_when_meta_is_incomplete(tmp_path: Path) -> None:
    env_path = tmp_path / "idx.lmdb"
    # Create an env with the schema/sub-DBs but no meta records at all.
    with NyplIndexStore(env_path):
        pass
    with NyplIndexLookup(env_path) as lookup, raises(RuntimeError, match="meta sub-DB"):
        lookup.stats()
