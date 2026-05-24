"""Tests for :mod:`pd_matcher.index.codec`."""

from datetime import date

from pd_matcher.index.codec import decode_reg
from pd_matcher.index.codec import decode_ren
from pd_matcher.index.codec import decode_uuid_list
from pd_matcher.index.codec import decode_year_key
from pd_matcher.index.codec import encode_reg
from pd_matcher.index.codec import encode_ren
from pd_matcher.index.codec import encode_uuid_list
from pd_matcher.index.codec import encode_year_key
from pd_matcher.index.codec import make_renewal_key
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import NyplRenRecord


def test_encode_reg_round_trips_full_record() -> None:
    record = IndexedNyplRegRecord(
        uuid="UUID-1",
        title="A Study of Widgets",
        was_renewed=True,
        regnum="A111111",
        reg_date=date(1940, 5, 10),
        reg_year=1940,
        author_name="Smith, John",
        author_place="Cambridge, Mass.",
        author_is_claimant=True,
        edition="1st ed.",
        publisher_names=("Acme Press",),
        publication_places=("New York",),
        claimants=("Acme Press",),
        copies="2c.",
        aff_date=date(1940, 6, 1),
        desc="vi, 200 p.",
        notes=("note one", "note two"),
        new_matter_claimed="ch. 5 added",
        copy_date=date(1940, 4, 1),
        notice_date=date(1940, 4, 2),
    )
    assert decode_reg(encode_reg(record)) == record


def test_encode_reg_round_trips_minimal_record() -> None:
    record = IndexedNyplRegRecord(uuid="UUID-1", title="t", was_renewed=False)
    assert decode_reg(encode_reg(record)) == record


def test_decode_reg_accepts_legacy_record_missing_new_cce_fields() -> None:
    from msgspec.msgpack import encode as msgpack_encode

    legacy_payload: dict[str, object] = {
        "uuid": "UUID-LEGACY",
        "title": "Pre-fields record",
        "was_renewed": True,
        "regnum": "A111111",
        "reg_date": "1940-05-10",
        "reg_year": 1940,
        "author_name": "Smith, John",
        "edition": "1st ed.",
        "publisher_names": ["Acme Press"],
        "publication_places": ["New York"],
        "claimants": ["Acme Press"],
    }
    decoded = decode_reg(msgpack_encode(legacy_payload))
    assert decoded.uuid == "UUID-LEGACY"
    assert decoded.author_place is None
    assert decoded.author_is_claimant is False
    assert decoded.copies is None
    assert decoded.aff_date is None
    assert decoded.desc is None
    assert decoded.notes == ()
    assert decoded.new_matter_claimed is None
    assert decoded.copy_date is None
    assert decoded.notice_date is None


def test_encode_ren_round_trips_full_record() -> None:
    record = NyplRenRecord(
        id="R200001",
        entry_id="entry-001",
        oreg="A111111",
        odat=date(1940, 5, 10),
        rdat=date(1968, 5, 15),
        author="Smith, John",
        title="A study of widgets",
        claimants="Acme Press|PWH",
        new_matter=None,
        full_text="Smith, John. A study of widgets. R200001",
    )
    assert decode_ren(encode_ren(record)) == record


def test_encode_uuid_list_round_trips_tuple() -> None:
    uuids = ("UUID-1", "UUID-2", "UUID-3")
    assert decode_uuid_list(encode_uuid_list(uuids)) == uuids


def test_encode_uuid_list_round_trips_empty() -> None:
    assert decode_uuid_list(encode_uuid_list(())) == ()


def test_year_key_is_two_bytes_big_endian() -> None:
    key = encode_year_key(1940)
    assert key == b"\x07\x94"
    assert decode_year_key(key) == 1940


def test_year_key_preserves_numeric_ordering() -> None:
    earlier = encode_year_key(1923)
    later = encode_year_key(1977)
    assert earlier < later


def test_make_renewal_key_includes_iso_date() -> None:
    key = make_renewal_key("A111111", date(1940, 5, 10))
    assert key == b"A111111|1940-05-10"


def test_make_renewal_key_handles_missing_date() -> None:
    key = make_renewal_key("A111111", None)
    assert key == b"A111111|"
