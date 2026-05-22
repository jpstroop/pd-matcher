"""Pure view model: turn a persisted review row into a renderable card.

A :class:`ReviewCard` is the single, fully-typed projection the templates
consume. It decodes the lossless ``marc_json`` blob back into a
:class:`MarcRecord` so the MARC side of the card can show real subfields, and
decodes ``evidence_json`` (a flat ``scorer -> normalized score`` mapping, as
written by :func:`pd_groundtruth.build_queue._evidence_payload`) into an
ordered list of :class:`EvidenceBar` for the per-field evidence bars. The CCE
side mirrors the denormalized columns, with the integer ``cce_was_renewed``
flag rendered as a human-readable renewal label. No ``Any`` crosses this
boundary; this module is the unit-tested heart of the UI.
"""

from msgspec import Struct
from msgspec.json import decode as json_decode
from pd_matcher.models import MarcRecord

from pd_groundtruth.review_db import ReviewPairRow

RENEWAL_RENEWED: str = "Renewed"
RENEWAL_NOT_RENEWED: str = "Not renewed"
RENEWAL_UNKNOWN: str = "unknown"


class EvidenceBar(Struct, frozen=True, forbid_unknown_fields=True):
    """One per-field evidence reading for a horizontal score bar."""

    scorer: str
    normalized: float


class ReviewCard(Struct, frozen=True, forbid_unknown_fields=True):
    """A fully-typed, render-ready projection of one review pair."""

    pair_id: int
    language: str
    decade: int | None
    score: float
    band: str
    marc_control_id: str
    nypl_uuid: str

    marc_title: str
    marc_title_main: str | None
    marc_statement_of_responsibility: str | None
    marc_main_author: str | None
    marc_added_authors: tuple[str, ...]
    marc_publisher: str | None
    marc_year: int | None
    marc_edition: str | None
    marc_series_titles: tuple[str, ...]
    marc_lccn: str | None
    marc_language_code: str | None
    marc_country_code: str | None

    cce_title: str | None
    cce_author: str | None
    cce_publishers: str | None
    cce_claimants: str | None
    cce_reg_year: int | None
    cce_regnum: str | None
    cce_renewal_label: str

    evidence: tuple[EvidenceBar, ...]


def render_renewal_label(was_renewed: int | None) -> str:
    """Map the stored integer renewal flag to a display label.

    Args:
        was_renewed: ``1`` for renewed, ``0`` for not renewed, ``None`` when
            the source registration carried no resolvable renewal status.

    Returns:
        One of :data:`RENEWAL_RENEWED`, :data:`RENEWAL_NOT_RENEWED`, or
        :data:`RENEWAL_UNKNOWN`.
    """
    if was_renewed is None:
        return RENEWAL_UNKNOWN
    return RENEWAL_RENEWED if was_renewed else RENEWAL_NOT_RENEWED


def parse_evidence(evidence_json: str) -> tuple[EvidenceBar, ...]:
    """Decode ``evidence_json`` into an ordered tuple of evidence bars.

    The stored shape is a flat JSON object mapping scorer name to its
    normalized ``[0, 1]`` score; insertion order is preserved so the bars
    render in the matcher's scorer order.
    """
    payload: dict[str, float] = json_decode(evidence_json, type=dict[str, float])
    return tuple(EvidenceBar(scorer=scorer, normalized=score) for scorer, score in payload.items())


def _title_main_if_distinct(marc: MarcRecord) -> str | None:
    """Return ``title_main`` only when it differs from the full ``title``."""
    if marc.title_main and marc.title_main != marc.title:
        return marc.title_main
    return None


def build_card(row: ReviewPairRow) -> ReviewCard:
    """Project a persisted :class:`ReviewPairRow` into a :class:`ReviewCard`.

    Decodes the lossless ``marc_json`` blob into a :class:`MarcRecord` to
    expose full MARC subfields, decodes ``evidence_json`` into ordered
    :class:`EvidenceBar` readings, and renders the renewal flag as a label.
    """
    marc: MarcRecord = json_decode(row.marc_json, type=MarcRecord)
    return ReviewCard(
        pair_id=row.id,
        language=row.language,
        decade=row.decade,
        score=row.score,
        band=row.band,
        marc_control_id=row.marc_control_id,
        nypl_uuid=row.nypl_uuid,
        marc_title=marc.title,
        marc_title_main=_title_main_if_distinct(marc),
        marc_statement_of_responsibility=marc.statement_of_responsibility,
        marc_main_author=marc.main_author,
        marc_added_authors=marc.added_authors,
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        marc_edition=marc.edition,
        marc_series_titles=marc.series_titles,
        marc_lccn=marc.lccn,
        marc_language_code=marc.language_code,
        marc_country_code=marc.country_code,
        cce_title=row.cce_title,
        cce_author=row.cce_author,
        cce_publishers=row.cce_publishers,
        cce_claimants=row.cce_claimants,
        cce_reg_year=row.cce_reg_year,
        cce_regnum=row.cce_regnum,
        cce_renewal_label=render_renewal_label(row.cce_was_renewed),
        evidence=parse_evidence(row.evidence_json),
    )


__all__ = [
    "RENEWAL_NOT_RENEWED",
    "RENEWAL_RENEWED",
    "RENEWAL_UNKNOWN",
    "EvidenceBar",
    "ReviewCard",
    "build_card",
    "parse_evidence",
    "render_renewal_label",
]
