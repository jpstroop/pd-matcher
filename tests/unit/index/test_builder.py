"""Tests for :mod:`pd_matcher.index.builder`."""

from datetime import date
from pathlib import Path
from time import sleep

from pytest import MonkeyPatch

from pd_matcher.index.builder import _META_PARSER_FINGERPRINT_KEY
from pd_matcher.index.builder import _PACKAGE_ROOT
from pd_matcher.index.builder import _PARSER_FINGERPRINT_FILES
from pd_matcher.index.builder import _cache_mismatch_reason
from pd_matcher.index.builder import _ExistingMeta
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


def _seed_regnum_variant_sources(root: Path) -> tuple[Path, Path]:
    """Copy the regnum-variant reg/ren fixtures into isolated source dirs."""
    reg_dir = root / "reg"
    ren_dir = root / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "variant_reg.xml").write_bytes((_FIXTURES / "regnum_variant_reg.xml").read_bytes())
    (ren_dir / "variant_ren.tsv").write_bytes((_FIXTURES / "regnum_variant_ren.tsv").read_bytes())
    return reg_dir, ren_dir


def _seed_range_regnum_sources(root: Path) -> tuple[Path, Path]:
    """Copy the range-regnum reg/ren fixtures into isolated source dirs."""
    reg_dir = root / "reg"
    ren_dir = root / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "range_reg.xml").write_bytes((_FIXTURES / "range_regnum_reg.xml").read_bytes())
    (ren_dir / "range_ren.tsv").write_bytes((_FIXTURES / "range_regnum_ren.tsv").read_bytes())
    return reg_dir, ren_dir


def _seed_year_join_sources(root: Path) -> tuple[Path, Path]:
    """Copy the year-join recovery reg/ren fixtures into isolated source dirs."""
    reg_dir = root / "reg"
    ren_dir = root / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "year_join_reg.xml").write_bytes((_FIXTURES / "year_join_reg.xml").read_bytes())
    (ren_dir / "year_join_ren.tsv").write_bytes((_FIXTURES / "year_join_ren.tsv").read_bytes())
    return reg_dir, ren_dir


def test_build_index_writes_records_year_buckets_and_meta(tmp_path: Path) -> None:
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    report = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    assert report.skipped is False
    assert report.registrations_written == 9
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

        join_key = make_renewal_key("A111111", 1940)
        assert store.ren_by_oreg.get(join_key) == b"entry-001"


def test_build_index_writes_token_inverted_indexes(tmp_path: Path) -> None:
    """Title/author/publisher tokens map to their registration uuids.

    UUID-0001 has title "A study of widgets.", author "Smith, John", and
    publisher "Acme Press". UUID-0002 carries publisher names and a claimant;
    the publisher index draws tokens from both, so "estate" (from the
    claimant "Estate of Dubois") retrieves UUID-0002.
    """
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    with NyplIndexStore(out_path, readonly=True) as store:
        widgets = store.title_index.get(b"widgets")
        assert widgets is not None
        assert "UUID-0001" in decode_uuid_list(widgets)

        smith = store.author_index.get(b"smith")
        assert smith is not None
        assert decode_uuid_list(smith) == ("UUID-0001",)

        acme = store.publisher_index.get(b"acme")
        assert acme is not None
        assert "UUID-0001" in decode_uuid_list(acme)

        # Claimant tokens feed the publisher index too.
        estate = store.publisher_index.get(b"estate")
        assert estate is not None
        assert "UUID-0002" in decode_uuid_list(estate)


def test_build_index_writes_renewal_token_indexes_and_year_buckets(tmp_path: Path) -> None:
    """Renewal title/author/claimants tokens and odat-year buckets are built.

    entry-001 (odat 1940) has title "A study of widgets", author "Smith, John",
    and claimants "Acme Press|PWH"; its tokens retrieve its entry id from the
    three renewal inverted indexes, and the renewal year bucket is keyed on the
    original-registration year (1940), not the renewal date (1968).
    """
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    report = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    assert report.renewal_year_buckets == 2
    assert report.renewal_title_tokens > 0
    assert report.renewal_author_tokens > 0
    assert report.renewal_claimants_tokens > 0

    with NyplIndexStore(out_path, readonly=True) as store:
        widgets = store.ren_title_index.get(b"widgets")
        assert widgets is not None
        assert decode_uuid_list(widgets) == ("entry-001",)

        smith = store.ren_author_index.get(b"smith")
        assert smith is not None
        assert decode_uuid_list(smith) == ("entry-001",)

        acme = store.ren_claimants_index.get(b"acme")
        assert acme is not None
        assert decode_uuid_list(acme) == ("entry-001",)

        # Bucketed by odat (1940), not rdat (1968).
        bucket_1940 = store.ren_by_year.get(encode_year_key(1940))
        assert bucket_1940 is not None
        assert decode_uuid_list(bucket_1940) == ("entry-001",)
        assert store.ren_by_year.get(encode_year_key(1968)) is None


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
    assert forced.registrations_written == 9


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
    assert rebuilt.registrations_written == 9


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


def test_build_index_projects_renewal_fields_onto_matched_registration(tmp_path: Path) -> None:
    """When a registration's renewal joins, the indexed record carries the renewal projection."""
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    with NyplIndexStore(out_path, readonly=True) as store:
        blob = store.reg_by_id.get(b"UUID-0001")
        assert blob is not None
        record = decode_reg(blob)

    assert record.was_renewed is True
    assert record.renewal_id == "R200001"
    assert record.renewal_oreg == "A111111"
    assert record.renewal_rdat == date(1968, 5, 15)
    assert record.renewal_author == "Smith, John"
    assert record.renewal_title == "A study of widgets"
    assert record.renewal_claimants == "Acme Press|PWH"
    assert record.renewal_new_matter is None


def test_build_index_renewal_fields_none_for_unrenewed_registration(tmp_path: Path) -> None:
    """Registrations without a renewal join carry ``None`` across the renewal projection."""
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    with NyplIndexStore(out_path, readonly=True) as store:
        blob = store.reg_by_id.get(b"UUID-0003")
        assert blob is not None
        record = decode_reg(blob)

    assert record.was_renewed is False
    assert record.renewal_id is None
    assert record.renewal_oreg is None
    assert record.renewal_rdat is None
    assert record.renewal_author is None
    assert record.renewal_title is None
    assert record.renewal_claimants is None
    assert record.renewal_new_matter is None


def test_build_index_persists_parser_fingerprint(tmp_path: Path) -> None:
    """A fresh build writes the parser fingerprint into the meta sub-DB."""
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    with NyplIndexStore(out_path, readonly=True) as store:
        fingerprint = store.meta.get(_META_PARSER_FINGERPRINT_KEY)
    assert fingerprint is not None
    assert len(fingerprint.decode("ascii")) == 64


def test_build_index_rebuilds_when_parser_fingerprint_changes(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A change in any tracked parser/model/codec file invalidates the cache.

    Simulated by pointing ``_PARSER_FINGERPRINT_FILES`` at an extra ``.py``
    file under the package root whose bytes we mutate between builds. The
    second build must rebuild even though sources and schema version are
    unchanged.
    """
    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    extra_file = _PACKAGE_ROOT / "_test_parser_fingerprint_probe.py"
    extra_file.write_text("# original\n", encoding="utf-8")
    monkeypatch.setattr(
        "pd_matcher.index.builder._PARSER_FINGERPRINT_FILES",
        (*_PARSER_FINGERPRINT_FILES, extra_file),
    )
    try:
        first = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
        assert first.skipped is False

        extra_file.write_text("# mutated\n", encoding="utf-8")

        rebuilt = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
        assert rebuilt.skipped is False
        assert rebuilt.registrations_written == 9
    finally:
        extra_file.unlink()


def test_build_index_rebuilds_when_parser_fingerprint_missing(tmp_path: Path) -> None:
    """An older env without ``parser_fingerprint`` is treated as a miss and rebuilt.

    We build once, delete the persisted fingerprint key directly through the
    LMDB env to mimic an index written before this feature existed, and
    confirm the next build sees the absence and reruns the full pipeline.
    """
    from lmdb import Environment as LmdbEnvironment

    reg_dir, ren_dir = _seed_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)

    env = LmdbEnvironment(str(out_path), max_dbs=16, readonly=False)
    try:
        meta_db = env.open_db(b"meta")
        with env.begin(db=meta_db, write=True) as txn:
            assert txn.delete(_META_PARSER_FINGERPRINT_KEY) is True
    finally:
        env.close()

    rebuilt = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    assert rebuilt.skipped is False
    assert rebuilt.registrations_written == 9


def test_parser_fingerprint_files_includes_registration_numbers_module() -> None:
    """The regnum normalizer is fingerprinted so edits to it force a rebuild."""
    assert _PACKAGE_ROOT / "normalize" / "registration_numbers.py" in _PARSER_FINGERPRINT_FILES


def test_build_index_joins_format_variant_regnum_against_renewal(tmp_path: Path) -> None:
    """A registration whose regnum is a format-variant of a renewal's oreg joins.

    The renewal carries the canonical ``AI9217``; UUID-V001's ``regnum`` is the
    hyphenated ``AI-9217`` in the same registration year (1927), so it joins
    only because :func:`make_renewal_key` normalizes both sides identically —
    under the raw key (``AI-9217|…`` vs ``AI9217|…``) it would not. UUID-V002
    shares the normalized regnum but registers in a different year (1930), so it
    must not join, proving the year stays part of the key.
    """
    reg_dir, ren_dir = _seed_regnum_variant_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    report = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    assert report.renewal_joins == 1

    with NyplIndexStore(out_path, readonly=True) as store:
        variant_blob = store.reg_by_id.get(b"UUID-V001")
        assert variant_blob is not None
        variant = decode_reg(variant_blob)
        assert variant.was_renewed is True
        assert variant.renewal_id == "R129296"

        mismatch_blob = store.reg_by_id.get(b"UUID-V002")
        assert mismatch_blob is not None
        assert decode_reg(mismatch_blob).was_renewed is False


def test_build_index_joins_multi_number_range_registration_on_interior_number(
    tmp_path: Path,
) -> None:
    """A multi-number range registration joins a renewal citing an interior number.

    UUID-R001's ``regnum`` is the space-separated whole ``"A692774 A692775"``;
    the renewal cites the interior number ``A692775`` at the same original date.
    Under the single mashed key (``A692774A692775|…``) the renewal can never
    collide, so the join only lands because :func:`make_renewal_keys` fans the
    range out into one per-number key on both sides. The registration is flagged
    ``is_range_registration`` and carries the projected renewal.
    """
    reg_dir, ren_dir = _seed_range_regnum_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    report = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    assert report.renewal_joins == 1

    odat_year = date(1950, 3, 15).year
    with NyplIndexStore(out_path, readonly=True) as store:
        blob = store.reg_by_id.get(b"UUID-R001")
        assert blob is not None
        record = decode_reg(blob)
        assert record.was_renewed is True
        assert record.is_range_registration is True
        assert record.renewal_id == "R301045"
        assert record.renewal_oreg == "A692775"

        # The mashed whole-string key never resolves; the interior number does.
        assert store.ren_by_oreg.get(make_renewal_key("A692774 A692775", odat_year)) is None
        assert store.ren_by_oreg.get(make_renewal_key("A692775", odat_year)) == b"entry-200"


def test_build_index_year_join_recovers_same_year_and_dateless_registrations(
    tmp_path: Path,
) -> None:
    """Year-level keying joins where exact-date keying could not.

    Three registrations exercise the year-based join:

    * UUID-Y001 registers ``A700001`` on ``1951-06-20`` while its renewal's
      ``odat`` is ``1951-11-05`` — the same year, a different day. Exact-date
      keying would split them; the year key ``A700001|1951`` joins them.
    * UUID-Y002 carries no ``<regDate>`` at all, only a ``<copyDate>`` of
      ``1952-04-01`` from which ``reg_year`` is derived. It keys on
      ``A700002|1952`` and joins its 1952 renewal — a registration exact-date
      keying could never reach.
    * UUID-Y003 has neither a date nor a derivable year, so ``reg_year`` is
      ``None``; it produces no year suffix, matches no renewal, and stays
      unrenewed.
    """
    reg_dir, ren_dir = _seed_year_join_sources(tmp_path)
    out_path = tmp_path / "idx.lmdb"

    report = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    assert report.renewal_joins == 2

    with NyplIndexStore(out_path, readonly=True) as store:
        same_year_blob = store.reg_by_id.get(b"UUID-Y001")
        assert same_year_blob is not None
        same_year = decode_reg(same_year_blob)
        assert same_year.reg_date == date(1951, 6, 20)
        assert same_year.was_renewed is True
        assert same_year.renewal_id == "R400300"

        dateless_blob = store.reg_by_id.get(b"UUID-Y002")
        assert dateless_blob is not None
        dateless = decode_reg(dateless_blob)
        assert dateless.reg_date is None
        assert dateless.reg_year == 1952
        assert dateless.was_renewed is True
        assert dateless.renewal_id == "R400301"

        no_year_blob = store.reg_by_id.get(b"UUID-Y003")
        assert no_year_blob is not None
        no_year = decode_reg(no_year_blob)
        assert no_year.reg_year is None
        assert no_year.was_renewed is False

        assert store.ren_by_oreg.get(make_renewal_key("A700001", 1951)) == b"entry-300"
        assert store.ren_by_oreg.get(make_renewal_key("A700002", 1952)) == b"entry-301"


def test_cache_mismatch_reason_reports_each_field_in_declaration_order() -> None:
    """Each missing/mismatched cache key resolves to its named reason.

    The on-disk env always writes all three meta keys together, so the only
    in-the-wild ``*_missing`` case is ``parser_fingerprint`` on a legacy env.
    These checks pin the defensive ``is None`` branches against future
    refactors that might silently let a partial meta read crash.
    """
    expected_source = "src-hash"
    expected_schema = 4
    expected_fingerprint = "fp"

    assert (
        _cache_mismatch_reason(
            _ExistingMeta(schema_version=None, source_hash=None, parser_fingerprint=None),
            expected_source_hash=expected_source,
            expected_schema_version=expected_schema,
            expected_parser_fingerprint=expected_fingerprint,
        )
        == "source_hash_missing"
    )
    assert (
        _cache_mismatch_reason(
            _ExistingMeta(
                schema_version=None,
                source_hash=expected_source,
                parser_fingerprint=None,
            ),
            expected_source_hash=expected_source,
            expected_schema_version=expected_schema,
            expected_parser_fingerprint=expected_fingerprint,
        )
        == "schema_version_missing"
    )
    assert (
        _cache_mismatch_reason(
            _ExistingMeta(
                schema_version=expected_schema,
                source_hash=expected_source,
                parser_fingerprint=expected_fingerprint,
            ),
            expected_source_hash=expected_source,
            expected_schema_version=expected_schema,
            expected_parser_fingerprint=expected_fingerprint,
        )
        is None
    )
