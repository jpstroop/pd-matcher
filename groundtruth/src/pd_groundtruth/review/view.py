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

from datetime import date
from datetime import datetime

from msgspec import Struct
from msgspec.json import decode as json_decode
from pd_matcher.models import MarcRecord

from pd_groundtruth.review.reasons import reasons_for
from pd_groundtruth.review.relative_time import format_relative
from pd_groundtruth.review_db import LabeledPairRow
from pd_groundtruth.review_db import ReviewPairRow

_TITLE_TRUNCATE: int = 60
_ELLIPSIS: str = "…"

RENEWAL_RENEWED: str = "Renewed"
RENEWAL_NOT_RENEWED: str = "Not renewed"
RENEWAL_UNKNOWN: str = "unknown"

CLAIMANT_LABEL: str = "Author is claimant"

_ONLINE_RESOURCE_MARKER: str = "online resource"


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
    marc_is_online_resource: bool

    cce_title: str | None
    cce_author: str | None
    cce_publishers: str | None
    cce_claimants: str | None
    cce_reg_year: int | None
    cce_regnum: str | None
    cce_renewal_label: str

    cce_edition: str | None
    cce_publication_places: tuple[str, ...]
    cce_author_place: str | None
    cce_author_is_claimant: bool
    author_is_claimant_label: str | None
    cce_copies: str | None
    cce_aff_date: date | None
    cce_desc: str | None
    cce_notes: tuple[str, ...]
    cce_new_matter_claimed: str | None
    cce_copy_date: date | None
    cce_notice_date: date | None

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


def author_is_claimant_label(is_claimant: int | None) -> str | None:
    """Return :data:`CLAIMANT_LABEL` when ``is_claimant`` is truthy, else ``None``.

    Args:
        is_claimant: ``1`` when the parser saw ``author claimant="yes"``,
            ``0`` for the DTD default, ``None`` when older data predates the
            field being captured. Both ``0`` and ``None`` collapse to ``None``
            so the card renders nothing when there is no signal to show.
    """
    return CLAIMANT_LABEL if is_claimant else None


def _split_places(raw: str | None) -> tuple[str, ...]:
    """Split a ``"; "``-joined publication-places string into a tuple."""
    if not raw:
        return ()
    return tuple(part for part in (chunk.strip() for chunk in raw.split(";")) if part)


def _split_notes(raw: str | None) -> tuple[str, ...]:
    """Split a newline-joined CCE notes string into a tuple, dropping blanks."""
    if not raw:
        return ()
    return tuple(line for line in raw.splitlines() if line)


def _parse_iso_date(raw: str | None) -> date | None:
    """Parse an ISO-formatted date string or return ``None`` when absent."""
    if raw is None:
        return None
    return date.fromisoformat(raw)


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


def _is_online_resource(extent: str | None) -> bool:
    """Return ``True`` when ``extent`` flags the MARC record as a digital reissue.

    A MARC 300 ``extent`` containing ``online resource`` (case-insensitive)
    marks the record as an e-book / digital reprint, which usually describes
    the wrong artifact for the matcher (year and publisher come from the
    reissue, not the original publication). The card shows a badge so the
    reviewer can adjust expectations accordingly.
    """
    if extent is None:
        return False
    return _ONLINE_RESOURCE_MARKER in extent.lower()


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
        marc_is_online_resource=_is_online_resource(marc.extent),
        cce_title=row.cce_title,
        cce_author=row.cce_author,
        cce_publishers=row.cce_publishers,
        cce_claimants=row.cce_claimants,
        cce_reg_year=row.cce_reg_year,
        cce_regnum=row.cce_regnum,
        cce_renewal_label=render_renewal_label(row.cce_was_renewed),
        cce_edition=row.cce_edition,
        cce_publication_places=_split_places(row.cce_publication_places),
        cce_author_place=row.cce_author_place,
        cce_author_is_claimant=bool(row.cce_author_is_claimant),
        author_is_claimant_label=author_is_claimant_label(row.cce_author_is_claimant),
        cce_copies=row.cce_copies,
        cce_aff_date=_parse_iso_date(row.cce_aff_date),
        cce_desc=row.cce_desc,
        cce_notes=_split_notes(row.cce_notes),
        cce_new_matter_claimed=row.cce_new_matter_claimed,
        cce_copy_date=_parse_iso_date(row.cce_copy_date),
        cce_notice_date=_parse_iso_date(row.cce_notice_date),
        evidence=parse_evidence(row.evidence_json),
    )


class LabeledRow(Struct, frozen=True, forbid_unknown_fields=True):
    """A render-ready projection of one row in the ``/labels`` table."""

    pair_id: int
    language: str
    marc_control_id: str
    marc_title: str
    marc_title_short: str
    cce_title: str
    cce_title_short: str
    verdict: str
    reason_codes: tuple[str, ...]
    reason_labels: tuple[str, ...]
    labeled_at: str
    labeled_relative: str


def _truncate(value: str, limit: int = _TITLE_TRUNCATE) -> str:
    """Truncate ``value`` to ``limit`` chars, appending an ellipsis if cut."""
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + _ELLIPSIS


def _resolve_reason_labels(verdict: str, codes: tuple[str, ...]) -> tuple[str, ...]:
    """Map reason codes to their human-readable labels for ``verdict``.

    Codes outside the verdict's vocabulary fall back to the raw code so the
    display never silently drops a stored value (e.g. an entry preserved from
    an older vocabulary).
    """
    if not codes:
        return ()
    vocabulary = {reason.code: reason.label for reason in reasons_for(verdict)}
    return tuple(vocabulary.get(code, code) for code in codes)


def build_labeled_row(row: LabeledPairRow, now: datetime) -> LabeledRow:
    """Project one :class:`LabeledPairRow` into a render-ready :class:`LabeledRow`.

    Empty / null titles render as the empty string in the table; the truncated
    forms drive what the cell displays while the full strings live in the
    hover ``title`` attribute for disambiguation.
    """
    marc_title = row.marc_title or ""
    cce_title = row.cce_title or ""
    return LabeledRow(
        pair_id=row.pair_id,
        language=row.language,
        marc_control_id=row.marc_control_id,
        marc_title=marc_title,
        marc_title_short=_truncate(marc_title),
        cce_title=cce_title,
        cce_title_short=_truncate(cce_title),
        verdict=row.verdict,
        reason_codes=row.reason_codes,
        reason_labels=_resolve_reason_labels(row.verdict, row.reason_codes),
        labeled_at=row.labeled_at,
        labeled_relative=format_relative(row.labeled_at, now),
    )


__all__ = [
    "CLAIMANT_LABEL",
    "RENEWAL_NOT_RENEWED",
    "RENEWAL_RENEWED",
    "RENEWAL_UNKNOWN",
    "EvidenceBar",
    "LabeledRow",
    "ReviewCard",
    "author_is_claimant_label",
    "build_card",
    "build_labeled_row",
    "parse_evidence",
    "render_renewal_label",
]
