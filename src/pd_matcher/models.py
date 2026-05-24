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
    title_main: str
    lccn: str | None = None
    oclc: str | None = None
    isbns: tuple[str, ...] = ()
    title_part_number: str | None = None
    title_part_name: str | None = None
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
    """The registration year, or the best-available copyright/publication
    year (``copyDate`` then ``pubDate``) when no registration date is
    present. ``reg_date`` remains strictly the registration date and stays
    ``None`` when no ``<regDate>`` exists; ``reg_year`` may still be set
    from the fallback chain so the record lands in a year bucket."""
    author_name: str | None = None
    author_place: str | None = None
    author_is_claimant: bool = False
    edition: str | None = None
    publisher_names: tuple[str, ...] = ()
    publication_places: tuple[str, ...] = ()
    claimants: tuple[str, ...] = ()
    copies: str | None = None
    aff_date: date | None = None
    desc: str | None = None
    notes: tuple[str, ...] = ()
    new_matter_claimed: str | None = None
    copy_date: date | None = None
    notice_date: date | None = None


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


class IndexedNyplRegRecord(Struct, frozen=True, forbid_unknown_fields=True):
    """A :class:`NyplRegRecord` augmented with its precomputed renewal flag.

    The matcher always cares whether a registration was renewed, so the join
    against ``ren_by_oreg`` is performed once during index build and baked
    into this struct rather than re-evaluated per candidate at match time.
    All other fields mirror :class:`NyplRegRecord` exactly.
    """

    uuid: str
    title: str
    was_renewed: bool
    regnum: str | None = None
    reg_date: date | None = None
    reg_year: int | None = None
    """The registration year, or the best-available copyright/publication
    year (``copyDate`` then ``pubDate``) when no registration date is
    present. Mirrors :attr:`NyplRegRecord.reg_year`; ``reg_date`` remains
    strictly the registration date."""
    author_name: str | None = None
    author_place: str | None = None
    author_is_claimant: bool = False
    edition: str | None = None
    publisher_names: tuple[str, ...] = ()
    publication_places: tuple[str, ...] = ()
    claimants: tuple[str, ...] = ()
    copies: str | None = None
    aff_date: date | None = None
    desc: str | None = None
    notes: tuple[str, ...] = ()
    new_matter_claimed: str | None = None
    copy_date: date | None = None
    notice_date: date | None = None


def index_reg(record: NyplRegRecord, *, was_renewed: bool) -> IndexedNyplRegRecord:
    """Copy a parsed :class:`NyplRegRecord` into an :class:`IndexedNyplRegRecord`.

    Args:
        record: The parser output to wrap.
        was_renewed: Pre-resolved renewal status; ``True`` when a matching
            renewal entry exists in ``ren_by_oreg``.

    Returns:
        A new :class:`IndexedNyplRegRecord` with the same field values plus
        the supplied ``was_renewed`` flag.
    """
    return IndexedNyplRegRecord(
        uuid=record.uuid,
        title=record.title,
        was_renewed=was_renewed,
        regnum=record.regnum,
        reg_date=record.reg_date,
        reg_year=record.reg_year,
        author_name=record.author_name,
        author_place=record.author_place,
        author_is_claimant=record.author_is_claimant,
        edition=record.edition,
        publisher_names=record.publisher_names,
        publication_places=record.publication_places,
        claimants=record.claimants,
        copies=record.copies,
        aff_date=record.aff_date,
        desc=record.desc,
        notes=record.notes,
        new_matter_claimed=record.new_matter_claimed,
        copy_date=record.copy_date,
        notice_date=record.notice_date,
    )


__all__ = [
    "IndexedNyplRegRecord",
    "MarcRecord",
    "NyplRegRecord",
    "NyplRenRecord",
    "index_reg",
]
