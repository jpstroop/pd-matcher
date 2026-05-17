"""Tests for :mod:`pd_matcher.models`."""

from datetime import date

from msgspec import ValidationError
from msgspec import convert
from msgspec import to_builtins
from pytest import raises

from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRegRecord
from pd_matcher.models import NyplRenRecord


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
