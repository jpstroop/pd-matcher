"""Tests for :mod:`pd_matcher.models`."""

from datetime import date

from msgspec import ValidationError
from msgspec import convert
from msgspec import to_builtins
from pytest import raises

from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRegRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.models import index_reg


def test_marc_record_minimal_fields_default_optional_to_none() -> None:
    record = MarcRecord(control_id="m1", title="Sample title")
    assert record.lccn is None
    assert record.isbns == ()
    assert record.added_authors == ()


def test_marc_record_is_frozen() -> None:
    record = MarcRecord(control_id="m1", title="t")
    with raises(AttributeError):
        setattr(record, "title", "other")


def test_marc_record_forbids_extra_fields() -> None:
    with raises(ValidationError):
        convert(
            {"control_id": "m1", "title": "t", "unknown": 1},
            type=MarcRecord,
        )


def test_marc_record_roundtrips_through_msgspec_builtins() -> None:
    original = MarcRecord(
        control_id="m1",
        title="Widgets",
        lccn="40012345",
        isbns=("9780000000000",),
        publication_year=1940,
    )
    again = convert(to_builtins(original), type=MarcRecord)
    assert original == again


def test_nypl_reg_record_minimal_required_fields() -> None:
    rec = NyplRegRecord(uuid="UUID-1", title="t")
    assert rec.publisher_names == ()
    assert rec.claimants == ()


def test_nypl_reg_record_is_frozen() -> None:
    rec = NyplRegRecord(uuid="UUID-1", title="t")
    with raises(AttributeError):
        setattr(rec, "title", "other")


def test_nypl_reg_record_roundtrips() -> None:
    rec = NyplRegRecord(
        uuid="UUID-1",
        title="t",
        regnum="A123",
        reg_date=date(1940, 5, 10),
        reg_year=1940,
        publisher_names=("Acme",),
        claimants=("Acme",),
    )
    again = convert(to_builtins(rec), type=NyplRegRecord)
    assert rec == again


def test_nypl_ren_record_minimal_required_fields() -> None:
    rec = NyplRenRecord(id="R1", entry_id="e1")
    assert rec.author is None
    assert rec.title is None


def test_nypl_ren_record_is_frozen() -> None:
    rec = NyplRenRecord(id="R1", entry_id="e1")
    with raises(AttributeError):
        setattr(rec, "id", "other")


def test_index_reg_copies_all_fields_and_adds_renewal_flag() -> None:
    parsed = NyplRegRecord(
        uuid="UUID-1",
        title="t",
        regnum="A1",
        reg_date=date(1940, 5, 10),
        reg_year=1940,
        author_name="Smith",
        edition="1st",
        publisher_names=("Acme",),
        publication_places=("NY",),
        claimants=("Acme",),
    )
    indexed = index_reg(parsed, was_renewed=True)
    assert isinstance(indexed, IndexedNyplRegRecord)
    assert indexed.was_renewed is True
    assert indexed.uuid == parsed.uuid
    assert indexed.title == parsed.title
    assert indexed.regnum == parsed.regnum
    assert indexed.reg_date == parsed.reg_date
    assert indexed.reg_year == parsed.reg_year
    assert indexed.author_name == parsed.author_name
    assert indexed.edition == parsed.edition
    assert indexed.publisher_names == parsed.publisher_names
    assert indexed.publication_places == parsed.publication_places
    assert indexed.claimants == parsed.claimants


def test_indexed_nypl_reg_record_is_frozen() -> None:
    rec = IndexedNyplRegRecord(uuid="UUID-1", title="t", was_renewed=False)
    with raises(AttributeError):
        setattr(rec, "was_renewed", True)
