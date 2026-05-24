"""Tests for :mod:`pd_matcher.index.lookup`."""

from pathlib import Path

from pytest import raises

from pd_matcher.copyright.coverage import LEGACY_COVERAGE
from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.index.store import NyplIndexStore
from pd_matcher.models import MarcRecord

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


def _marc(
    *,
    title: str = "",
    main_author: str | None = None,
    statement_of_responsibility: str | None = None,
    publisher: str | None = None,
    publication_year: int | None = None,
) -> MarcRecord:
    return MarcRecord(
        control_id="m",
        title=title,
        title_main=title,
        main_author=main_author,
        statement_of_responsibility=statement_of_responsibility,
        publisher=publisher,
        publication_year=publication_year,
    )


def test_candidates_for_returns_year_and_title_token_sharer(tmp_path: Path) -> None:
    """A title token shared with UUID-0001 in the 1940 bucket retrieves it."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="An illustrated study of widgets", publication_year=1940)
        records = list(lookup.candidates_for(marc))
    assert [r.uuid for r in records] == ["UUID-0001"]


def test_candidates_for_matches_via_author_token(tmp_path: Path) -> None:
    """An author token (no shared title token) still retrieves the candidate."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="Totally different heading", main_author="Smith", publication_year=1940)
        records = list(lookup.candidates_for(marc))
    assert [r.uuid for r in records] == ["UUID-0001"]


def test_candidates_for_matches_via_sor_author_token(tmp_path: Path) -> None:
    """Statement-of-responsibility tokens feed the author retrieval path."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(
            title="Totally different heading",
            statement_of_responsibility="by John Smith",
            publication_year=1940,
        )
        records = list(lookup.candidates_for(marc))
    assert [r.uuid for r in records] == ["UUID-0001"]


def test_candidates_for_matches_via_publisher_token(tmp_path: Path) -> None:
    """A publisher token (no shared title/author token) retrieves the candidate."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="Totally different heading", publisher="Acme", publication_year=1940)
        records = list(lookup.candidates_for(marc))
    assert [r.uuid for r in records] == ["UUID-0001"]


def test_candidates_for_intersects_year_and_token(tmp_path: Path) -> None:
    """A shared token but a non-matching year yields nothing (intersection)."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="A study of widgets", publication_year=1962)
        records = list(lookup.candidates_for(marc))
    assert records == []


def test_candidates_for_yields_nothing_without_shared_token(tmp_path: Path) -> None:
    """A correct year but no shared token yields nothing (token set empty)."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="Zzz qqq xyzzy", publication_year=1940)
        records = list(lookup.candidates_for(marc))
    assert records == []


def test_candidates_for_yields_nothing_without_publication_year(tmp_path: Path) -> None:
    """No publication_year short-circuits to an empty iterator."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="A study of widgets", publication_year=None)
        records = list(lookup.candidates_for(marc))
    assert records == []


def test_candidates_for_yields_nothing_when_year_bucket_empty(tmp_path: Path) -> None:
    """A year with no registrations yields nothing even with shared tokens."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="A study of widgets", publication_year=1800)
        records = list(lookup.candidates_for(marc))
    assert records == []


def test_candidates_for_window_widens_year_set(tmp_path: Path) -> None:
    """A non-zero window pulls in a token sharer from an adjacent year.

    UUID-0011 ("Histoire de la folie", 1962) shares the ``folie`` token; a
    1963 query with window 1 reaches the 1962 bucket and retrieves it.
    """
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="folie", publication_year=1963)
        zero_window = list(lookup.candidates_for(marc, window=0))
        widened = list(lookup.candidates_for(marc, window=1))
    assert zero_window == []
    assert [r.uuid for r in widened] == ["UUID-0011"]


def test_candidates_for_skips_uuid_with_no_registration(tmp_path: Path) -> None:
    """A posting that points at a missing registration is skipped."""
    out_path = _build_tiny_index(tmp_path)
    from pd_matcher.index.codec import encode_uuid_list
    from pd_matcher.index.codec import encode_year_key

    with NyplIndexStore(out_path) as store, store.write_transaction():
        # Plant a uuid in both the 1940 year bucket and the title index so the
        # intersection contains it, but never write a reg_by_id record for it.
        store.reg_by_year.put(encode_year_key(1940), encode_uuid_list(("UUID-GHOST",)))
        store.title_index.put(b"widgets", encode_uuid_list(("UUID-GHOST",)))

    with NyplIndexLookup(out_path) as lookup:
        marc = _marc(title="widgets", publication_year=1940)
        records = list(lookup.candidates_for(marc))
    assert records == []


def test_stats_reflect_build_report(tmp_path: Path) -> None:
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        stats = lookup.stats()
    assert stats.schema_version == 5
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


def test_coverage_derived_from_built_index(tmp_path: Path) -> None:
    """A built index exposes a coverage struct derived from the year histograms."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        coverage = lookup.coverage()
    assert coverage.reg_min_year >= 1900
    assert coverage.reg_max_year <= 1977
    assert coverage.ren_min_year >= 1900
    assert coverage.ren_max_year <= 2005


def test_coverage_falls_back_to_legacy_when_meta_missing(tmp_path: Path) -> None:
    """An env without per-year count blobs returns :data:`LEGACY_COVERAGE`."""
    env_path = tmp_path / "idx.lmdb"
    with NyplIndexStore(env_path):
        pass
    with NyplIndexLookup(env_path) as lookup:
        assert lookup.coverage() == LEGACY_COVERAGE
