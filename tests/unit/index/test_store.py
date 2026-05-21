"""Tests for :mod:`pd_matcher.index.store`."""

from pathlib import Path

from pytest import raises

from pd_matcher.index.store import NyplIndexStore


def test_store_writes_and_reads_back_inside_transaction(tmp_path: Path) -> None:
    with NyplIndexStore(tmp_path / "env") as store:
        with store.write_transaction():
            store.reg_by_id.put(b"k1", b"v1")
            store.ren_by_id.put(b"k2", b"v2")
            # get inside the active txn must see the freshly-written value.
            assert store.reg_by_id.get(b"k1") == b"v1"
        # After commit the value is still readable via a fresh read txn.
        assert store.reg_by_id.get(b"k1") == b"v1"
        assert store.ren_by_id.get(b"k2") == b"v2"


def test_store_get_returns_none_for_missing_key(tmp_path: Path) -> None:
    with NyplIndexStore(tmp_path / "env") as store:
        assert store.reg_by_id.get(b"absent") is None


def test_store_put_outside_transaction_raises(tmp_path: Path) -> None:
    with (
        NyplIndexStore(tmp_path / "env") as store,
        raises(RuntimeError, match="active write transaction"),
    ):
        store.reg_by_id.put(b"k", b"v")


def test_store_count_reflects_entries(tmp_path: Path) -> None:
    with NyplIndexStore(tmp_path / "env") as store:
        assert store.reg_by_id.count() == 0
        with store.write_transaction():
            store.reg_by_id.put(b"a", b"1")
            store.reg_by_id.put(b"b", b"2")
        assert store.reg_by_id.count() == 2


def test_store_iter_items_yields_all_pairs_in_key_order(tmp_path: Path) -> None:
    with NyplIndexStore(tmp_path / "env") as store:
        with store.write_transaction():
            store.reg_by_id.put(b"b", b"2")
            store.reg_by_id.put(b"a", b"1")
            store.reg_by_id.put(b"c", b"3")
        items = list(store.reg_by_id.iter_items())
    assert items == [(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]


def test_store_sub_dbs_are_isolated(tmp_path: Path) -> None:
    with NyplIndexStore(tmp_path / "env") as store:
        with store.write_transaction():
            store.reg_by_id.put(b"shared_key", b"reg_value")
            store.ren_by_id.put(b"shared_key", b"ren_value")
            store.reg_by_year.put(b"shared_key", b"year_value")
            store.ren_by_oreg.put(b"shared_key", b"oreg_value")
            store.title_index.put(b"shared_key", b"title_value")
            store.author_index.put(b"shared_key", b"author_value")
            store.publisher_index.put(b"shared_key", b"publisher_value")
            store.meta.put(b"shared_key", b"meta_value")
        assert store.reg_by_id.get(b"shared_key") == b"reg_value"
        assert store.ren_by_id.get(b"shared_key") == b"ren_value"
        assert store.reg_by_year.get(b"shared_key") == b"year_value"
        assert store.ren_by_oreg.get(b"shared_key") == b"oreg_value"
        assert store.title_index.get(b"shared_key") == b"title_value"
        assert store.author_index.get(b"shared_key") == b"author_value"
        assert store.publisher_index.get(b"shared_key") == b"publisher_value"
        assert store.meta.get(b"shared_key") == b"meta_value"


def test_store_token_indexes_round_trip_uuid_list_postings(tmp_path: Path) -> None:
    """The three inverted indexes round-trip encoded uuid-list postings."""
    from pd_matcher.index.codec import decode_uuid_list
    from pd_matcher.index.codec import encode_uuid_list

    with NyplIndexStore(tmp_path / "env") as store:
        with store.write_transaction():
            store.title_index.put(b"widgets", encode_uuid_list(("u1", "u2")))
            store.author_index.put(b"smith", encode_uuid_list(("u1",)))
            store.publisher_index.put(b"acme", encode_uuid_list(("u3", "u4", "u1")))
        title_blob = store.title_index.get(b"widgets")
        author_blob = store.author_index.get(b"smith")
        publisher_blob = store.publisher_index.get(b"acme")
        assert title_blob is not None
        assert author_blob is not None
        assert publisher_blob is not None
        assert decode_uuid_list(title_blob) == ("u1", "u2")
        assert decode_uuid_list(author_blob) == ("u1",)
        assert decode_uuid_list(publisher_blob) == ("u3", "u4", "u1")
        assert store.title_index.get(b"absent") is None


def test_store_readonly_rejects_writes(tmp_path: Path) -> None:
    env_path = tmp_path / "env"
    with NyplIndexStore(env_path) as store, store.write_transaction():
        store.reg_by_id.put(b"k", b"v")

    with NyplIndexStore(env_path, readonly=True) as readonly_store:
        assert readonly_store.readonly is True
        assert readonly_store.path == env_path
        assert readonly_store.reg_by_id.get(b"k") == b"v"
        with raises(RuntimeError, match="readonly"):
            readonly_store.write_transaction()


def test_store_disallows_nested_write_transactions(tmp_path: Path) -> None:
    with (
        NyplIndexStore(tmp_path / "env") as store,
        store.write_transaction(),
        raises(RuntimeError, match="already active"),
        store.write_transaction(),
    ):
        pass


def test_store_aborts_write_transaction_on_exception(tmp_path: Path) -> None:
    env_path = tmp_path / "env"
    error_raised = False
    with NyplIndexStore(env_path) as store:
        try:
            with store.write_transaction():
                store.reg_by_id.put(b"k", b"v")
                raise ValueError("boom")
        except ValueError:
            error_raised = True
    assert error_raised is True

    with NyplIndexStore(env_path) as store:
        # Aborted writes must not be visible to a subsequent reader.
        assert store.reg_by_id.get(b"k") is None
