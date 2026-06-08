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
    record = MarcRecord(control_id="m1", title="Sample title", title_main="Sample title")
    assert record.lccn is None
    assert record.isbns == ()
    assert record.added_authors == ()
    assert record.title_part_number is None
    assert record.title_part_name is None
    assert record.title_variants == ()


def test_marc_record_is_frozen() -> None:
    record = MarcRecord(control_id="m1", title="t", title_main="t")
    with raises(AttributeError):
        setattr(record, "title", "other")


def test_marc_record_forbids_extra_fields() -> None:
    with raises(ValidationError):
        convert(
            {"control_id": "m1", "title": "t", "title_main": "t", "unknown": 1},
            type=MarcRecord,
        )


def test_marc_record_roundtrips_through_msgspec_builtins() -> None:
    original = MarcRecord(
        control_id="m1",
        title="Widgets",
        title_main="Widgets",
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
        author_place="Cambridge, Mass.",
        author_is_claimant=True,
        edition="1st",
        publisher_names=("Acme",),
        publication_places=("NY",),
        claimants=("Acme",),
        copies="2c.",
        aff_date=date(1940, 6, 1),
        desc="vi, 200 p.",
        notes=("note one", "note two"),
        new_matter_claimed="ch. 5 added",
        copy_date=date(1940, 4, 1),
        notice_date=date(1940, 4, 2),
        lccn="28000854",
        prev_regnums=("A100000",),
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
    assert indexed.author_place == parsed.author_place
    assert indexed.author_is_claimant == parsed.author_is_claimant
    assert indexed.edition == parsed.edition
    assert indexed.publisher_names == parsed.publisher_names
    assert indexed.publication_places == parsed.publication_places
    assert indexed.claimants == parsed.claimants
    assert indexed.copies == parsed.copies
    assert indexed.aff_date == parsed.aff_date
    assert indexed.desc == parsed.desc
    assert indexed.notes == parsed.notes
    assert indexed.new_matter_claimed == parsed.new_matter_claimed
    assert indexed.copy_date == parsed.copy_date
    assert indexed.notice_date == parsed.notice_date
    assert indexed.lccn == parsed.lccn
    assert indexed.prev_regnums == parsed.prev_regnums


def test_index_reg_preserves_defaults_for_new_cce_fields() -> None:
    parsed = NyplRegRecord(uuid="UUID-1", title="t")
    indexed = index_reg(parsed, was_renewed=False)
    assert indexed.author_place is None
    assert indexed.author_is_claimant is False
    assert indexed.copies is None
    assert indexed.aff_date is None
    assert indexed.desc is None
    assert indexed.notes == ()
    assert indexed.new_matter_claimed is None
    assert indexed.copy_date is None
    assert indexed.notice_date is None
    assert indexed.lccn is None
    assert indexed.prev_regnums == ()


def test_index_reg_projects_renewal_fields_when_renewal_supplied() -> None:
    parsed = NyplRegRecord(uuid="UUID-1", title="t", regnum="A111111", reg_date=date(1940, 5, 10))
    renewal = NyplRenRecord(
        id="R200001",
        entry_id="entry-001",
        oreg="A111111",
        odat=date(1940, 5, 10),
        rdat=date(1968, 5, 15),
        author="Smith, John",
        title="A study of widgets",
        claimants="Acme Press|PWH",
        new_matter="updated chapter 7",
    )
    indexed = index_reg(parsed, was_renewed=True, renewal=renewal)
    assert indexed.was_renewed is True
    assert indexed.renewal_id == "R200001"
    assert indexed.renewal_oreg == "A111111"
    assert indexed.renewal_rdat == date(1968, 5, 15)
    assert indexed.renewal_author == "Smith, John"
    assert indexed.renewal_title == "A study of widgets"
    assert indexed.renewal_claimants == "Acme Press|PWH"
    assert indexed.renewal_new_matter == "updated chapter 7"


def test_index_reg_renewal_fields_default_to_none_when_renewal_absent() -> None:
    parsed = NyplRegRecord(uuid="UUID-1", title="t")
    indexed = index_reg(parsed, was_renewed=True)
    assert indexed.was_renewed is True
    assert indexed.renewal_id is None
    assert indexed.renewal_oreg is None
    assert indexed.renewal_rdat is None
    assert indexed.renewal_author is None
    assert indexed.renewal_title is None
    assert indexed.renewal_claimants is None
    assert indexed.renewal_new_matter is None


def test_indexed_nypl_reg_record_is_frozen() -> None:
    rec = IndexedNyplRegRecord(uuid="UUID-1", title="t", was_renewed=False)
    with raises(AttributeError):
        setattr(rec, "was_renewed", True)
