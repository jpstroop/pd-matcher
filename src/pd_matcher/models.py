"""Frozen typed records produced by parsers and consumed downstream.

These ``msgspec.Struct`` records are the on-the-wire shape for everything that
flows from the parsers through normalization, indexing and matching. All
structs are ``frozen=True`` (no post-construction mutation), use
``forbid_unknown_fields=True`` (defense against schema drift when records are
decoded from disk by later phases), and rely on msgspec's default ``__slots__``
generation for compact memory layout. Sequence fields are declared as
``tuple[...]`` so that hashing and equality remain cheap and so that no
caller can mutate parser output in-place.
"""

from datetime import date

from msgspec import Struct


class MarcRecord(Struct, frozen=True, forbid_unknown_fields=True):
    """One MARC bibliographic record extracted from a MARCXML source."""

    control_id: str
    title: str
    lccn: str | None = None
    isbns: tuple[str, ...] = ()
    main_author: str | None = None
    added_authors: tuple[str, ...] = ()
    statement_of_responsibility: str | None = None
    edition: str | None = None
    publication_place: str | None = None
    publisher: str | None = None
    publication_date_raw: str | None = None
    publication_year: int | None = None
    extent: str | None = None
    series_titles: tuple[str, ...] = ()
    language_code: str | None = None
    country_code: str | None = None


class NyplRegRecord(Struct, frozen=True, forbid_unknown_fields=True):
    """One NYPL transcription of a Copyright Office registration entry."""

    uuid: str
    title: str
    regnum: str | None = None
    reg_date: date | None = None
    reg_year: int | None = None
    author_name: str | None = None
    edition: str | None = None
    publisher_names: tuple[str, ...] = ()
    publication_places: tuple[str, ...] = ()
    claimants: tuple[str, ...] = ()


class NyplRenRecord(Struct, frozen=True, forbid_unknown_fields=True):
    """One NYPL transcription of a Copyright Office renewal entry."""

    id: str
    entry_id: str
    oreg: str | None = None
    odat: date | None = None
    rdat: date | None = None
    author: str | None = None
    title: str | None = None
    claimants: str | None = None
    new_matter: str | None = None
    full_text: str | None = None


__all__ = [
    "MarcRecord",
    "NyplRegRecord",
    "NyplRenRecord",
]
