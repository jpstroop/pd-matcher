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
    title_variants: tuple[str, ...] = ()
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
    lccn: str | None = None
    """The Library of Congress Control Number for this registration, when the
    CCE entry carries a ``<lccn>`` element. When the element's ``normalized``
    attribute is present its value (the 8-digit canonical form) is preferred;
    otherwise the element's display text (e.g. ``"28-854"``) is stored."""
    prev_regnums: tuple[str, ...] = ()
    """Every ``<prev-regNum>`` text under the entry, in document order. These
    are back-references to earlier registrations (revised editions, second
    printings, etc.); the DTD permits multiple per entry."""


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
    When the join lands a matching :class:`NyplRenRecord`, a projection of
    that renewal (date, claimants, new matter, etc.) is also copied across
    so downstream consumers can show renewal details without a second LMDB
    lookup. All non-renewal fields mirror :class:`NyplRegRecord` exactly.
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
    lccn: str | None = None
    """Mirrors :attr:`NyplRegRecord.lccn`."""
    prev_regnums: tuple[str, ...] = ()
    """Mirrors :attr:`NyplRegRecord.prev_regnums`."""
    renewal_id: str | None = None
    """Mirrors :attr:`NyplRenRecord.id` when a renewal joined this
    registration; otherwise ``None``."""
    renewal_oreg: str | None = None
    """Mirrors :attr:`NyplRenRecord.oreg` (the original registration number
    the renewal points back at)."""
    renewal_rdat: date | None = None
    """Mirrors :attr:`NyplRenRecord.rdat` (renewal-recording date)."""
    renewal_author: str | None = None
    """Mirrors :attr:`NyplRenRecord.author` as transcribed on the renewal."""
    renewal_title: str | None = None
    """Mirrors :attr:`NyplRenRecord.title` as transcribed on the renewal."""
    renewal_claimants: str | None = None
    """Mirrors :attr:`NyplRenRecord.claimants` as transcribed on the renewal."""
    renewal_new_matter: str | None = None
    """Mirrors :attr:`NyplRenRecord.new_matter` as transcribed on the renewal."""


def index_reg(
    record: NyplRegRecord,
    *,
    was_renewed: bool,
    renewal: NyplRenRecord | None = None,
) -> IndexedNyplRegRecord:
    """Copy a parsed :class:`NyplRegRecord` into an :class:`IndexedNyplRegRecord`.

    Args:
        record: The parser output to wrap.
        was_renewed: Pre-resolved renewal status; ``True`` when a matching
            renewal entry exists in ``ren_by_oreg``.
        renewal: The matching :class:`NyplRenRecord` whose fields should be
            projected onto the indexed record. May be ``None`` even when
            ``was_renewed`` is ``True`` (e.g. legacy indices that did not
            carry the projection); in that case the ``renewal_*`` fields
            default to ``None``.

    Returns:
        A new :class:`IndexedNyplRegRecord` with the same field values plus
        the supplied ``was_renewed`` flag and the renewal projection.
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
        lccn=record.lccn,
        prev_regnums=record.prev_regnums,
        renewal_id=renewal.id if renewal is not None else None,
        renewal_oreg=renewal.oreg if renewal is not None else None,
        renewal_rdat=renewal.rdat if renewal is not None else None,
        renewal_author=renewal.author if renewal is not None else None,
        renewal_title=renewal.title if renewal is not None else None,
        renewal_claimants=renewal.claimants if renewal is not None else None,
        renewal_new_matter=renewal.new_matter if renewal is not None else None,
    )


__all__ = [
    "IndexedNyplRegRecord",
    "MarcRecord",
    "NyplRegRecord",
    "NyplRenRecord",
    "index_reg",
]
