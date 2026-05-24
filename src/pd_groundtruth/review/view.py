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

from pd_groundtruth.review.field_annotations import FieldAnnotation
from pd_groundtruth.review.field_annotations import judgment_symbol
from pd_groundtruth.review.reasons import reasons_for
from pd_groundtruth.review.relative_time import format_relative
from pd_groundtruth.review_db import LabeledPairRow
from pd_groundtruth.review_db import ReviewPairRow
from pd_matcher.models import MarcRecord

_TITLE_TRUNCATE: int = 60
_ELLIPSIS: str = "…"

RENEWAL_RENEWED: str = "Renewed"
RENEWAL_NOT_RENEWED: str = "Not renewed"
RENEWAL_UNKNOWN: str = "unknown"

CLAIMANT_LABEL: str = "Author is claimant"

PREDICTED_STATUS_FAMILY_PD: str = "pd"
PREDICTED_STATUS_FAMILY_IN_COPYRIGHT: str = "in_copyright"
PREDICTED_STATUS_FAMILY_UNKNOWN: str = "unknown"

_ONLINE_RESOURCE_MARKER: str = "online resource"

_LCCN_BASE_URL: str = "https://lccn.loc.gov/"
_OCLC_BASE_URL: str = "https://www.worldcat.org/oclc/"


class EvidenceBar(Struct, frozen=True, forbid_unknown_fields=True):
    """One per-field evidence reading for a horizontal score bar.

    ``source`` is the human-readable ``"marc_field ↔ cce_field"`` label of the
    pairing that produced the winning Evidence for group scorers (title /
    author / publisher). It is ``None`` for non-group scorers (lccn, isbn,
    year, edition) and for rows persisted before evidence-source capture
    landed; the card template suppresses the breadcrumb in either case.
    """

    scorer: str
    normalized: float
    source: str | None = None


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
    marc_title_part_number: str | None
    marc_title_part_name: str | None
    marc_statement_of_responsibility: str | None
    marc_main_author: str | None
    marc_added_authors: tuple[str, ...]
    marc_publisher: str | None
    marc_publication_place: str | None
    marc_year: int | None
    marc_publication_date_raw: str | None
    marc_edition: str | None
    marc_extent: str | None
    marc_series_titles: tuple[str, ...]
    marc_lccn: str | None
    marc_isbns: tuple[str, ...]
    marc_oclc: str | None
    marc_oclc_url: str | None
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
    cce_lccn: str | None
    cce_lccn_url: str | None
    cce_prev_regnums: tuple[str, ...]

    predicted_status: str | None
    predicted_status_family: str

    cce_renewal_id: str | None
    cce_renewal_oreg: str | None
    cce_renewal_rdat: date | None
    cce_renewal_author: str | None
    cce_renewal_title: str | None
    cce_renewal_claimants: str | None
    cce_renewal_new_matter: str | None
    cce_renewal_claimants_differ: bool
    cce_has_renewal_details: bool

    evidence: tuple[EvidenceBar, ...]


def predicted_status_family(status: str | None) -> str:
    """Classify a stored Cornell status into a coarse rendering family.

    Returns one of :data:`PREDICTED_STATUS_FAMILY_PD`,
    :data:`PREDICTED_STATUS_FAMILY_IN_COPYRIGHT`, or
    :data:`PREDICTED_STATUS_FAMILY_UNKNOWN`. The classifier inspects the
    serialized name (``status.name``) so it stays decoupled from the
    ``pd_matcher.copyright.status.CopyrightStatus`` import surface on the
    review side.

    Args:
        status: The stored ``CopyrightStatus`` name (e.g.
            ``"PD_REGISTERED_NOT_RENEWED"``), or ``None`` when the queue
            row predates predicted-status capture.
    """
    if status is None:
        return PREDICTED_STATUS_FAMILY_UNKNOWN
    if status.startswith("PD_"):
        return PREDICTED_STATUS_FAMILY_PD
    if status.startswith("IN_COPYRIGHT"):
        return PREDICTED_STATUS_FAMILY_IN_COPYRIGHT
    return PREDICTED_STATUS_FAMILY_UNKNOWN


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


def _split_prev_regnums(raw: str | None) -> tuple[str, ...]:
    """Split a ``"; "``-joined prev-regnums string into a tuple, dropping blanks."""
    if not raw:
        return ()
    return tuple(part for part in (chunk.strip() for chunk in raw.split(";")) if part)


def _lccn_url(lccn: str | None) -> str | None:
    """Return the public ``lccn.loc.gov`` URL for ``lccn`` or ``None`` when absent.

    The LCCN permalink service accepts both the 8-digit normalized form and
    the human ``NN-NNNN`` form, so no normalization is required here — the
    value is interpolated as stored.
    """
    if not lccn:
        return None
    return f"{_LCCN_BASE_URL}{lccn}"


def _oclc_url(oclc: str | None) -> str | None:
    """Return the WorldCat permalink for ``oclc`` or ``None`` when absent."""
    if not oclc:
        return None
    return f"{_OCLC_BASE_URL}{oclc}"


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


def parse_evidence_sources(evidence_sources_json: str) -> dict[str, str]:
    """Decode ``evidence_sources_json`` into a ``scorer -> "marc ↔ cce"`` map.

    The stored shape is a flat JSON object mapping scorer name to the human
    label of the winning pairing's ``(marc_field, cce_field)`` source. Older
    rows persisted before evidence-source capture landed are missing keys
    here and surface as ``EvidenceBar.source = None`` downstream.
    """
    return json_decode(evidence_sources_json, type=dict[str, str])


def _title_main_if_distinct(marc: MarcRecord) -> str | None:
    """Return ``title_main`` only when it differs from the full ``title``."""
    if marc.title_main and marc.title_main != marc.title:
        return marc.title_main
    return None


def _publication_date_raw_if_distinct(marc: MarcRecord) -> str | None:
    """Return ``publication_date_raw`` only when it adds detail beyond the year.

    The raw 260/264 ``$c`` often duplicates the four-digit ``publication_year``
    (e.g. ``"1953"``); collapsing those cases to ``None`` keeps the rendered
    card free of redundant rows. When the raw form carries extra punctuation,
    qualifiers, or a different value entirely (e.g. ``"c1953."`` or
    ``"[1953?]"``), the row is shown.
    """
    raw = marc.publication_date_raw
    if not raw:
        return None
    year = marc.publication_year
    if year is not None and raw.strip() == str(year):
        return None
    return raw


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


def _renewal_claimants_differ(registration: str | None, renewal: str | None) -> bool:
    """Return ``True`` when the registration's claimants disagree with the renewal's.

    The registration's claimants are stored as a ``" | "``-joined string
    (built by :func:`pd_groundtruth.build_queue._join`); the renewal's
    ``claimants`` come straight from the NYPL transcription. The comparison
    is whitespace-collapsed and lower-cased so a trivial formatting drift
    does not count as a difference. When either side is absent or empty the
    function returns ``False`` (no diff signal to surface).
    """
    if not registration or not renewal:
        return False
    return registration.strip().lower() != renewal.strip().lower()


def _has_renewal_details(row: ReviewPairRow) -> bool:
    """Return ``True`` when any persisted ``cce_renewal_*`` field is populated.

    Used by the template to decide whether to render the renewal-details
    sub-block. Older rows (pre-#42) carry ``None`` across all renewal fields
    and so collapse to ``False``, leaving the legacy "Renewed" badge as the
    sole renewal signal.
    """
    return any(
        value is not None
        for value in (
            row.cce_renewal_id,
            row.cce_renewal_oreg,
            row.cce_renewal_rdat,
            row.cce_renewal_author,
            row.cce_renewal_title,
            row.cce_renewal_claimants,
            row.cce_renewal_new_matter,
        )
    )


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
        marc_title_part_number=marc.title_part_number,
        marc_title_part_name=marc.title_part_name,
        marc_statement_of_responsibility=marc.statement_of_responsibility,
        marc_main_author=marc.main_author,
        marc_added_authors=marc.added_authors,
        marc_publisher=marc.publisher,
        marc_publication_place=marc.publication_place,
        marc_year=marc.publication_year,
        marc_publication_date_raw=_publication_date_raw_if_distinct(marc),
        marc_edition=marc.edition,
        marc_extent=marc.extent,
        marc_series_titles=marc.series_titles,
        marc_lccn=marc.lccn,
        marc_isbns=marc.isbns,
        marc_oclc=marc.oclc,
        marc_oclc_url=_oclc_url(marc.oclc),
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
        cce_lccn=row.cce_lccn,
        cce_lccn_url=_lccn_url(row.cce_lccn),
        cce_prev_regnums=_split_prev_regnums(row.cce_prev_regnums),
        predicted_status=row.cce_predicted_status,
        predicted_status_family=predicted_status_family(row.cce_predicted_status),
        cce_renewal_id=row.cce_renewal_id,
        cce_renewal_oreg=row.cce_renewal_oreg,
        cce_renewal_rdat=_parse_iso_date(row.cce_renewal_rdat),
        cce_renewal_author=row.cce_renewal_author,
        cce_renewal_title=row.cce_renewal_title,
        cce_renewal_claimants=row.cce_renewal_claimants,
        cce_renewal_new_matter=row.cce_renewal_new_matter,
        cce_renewal_claimants_differ=_renewal_claimants_differ(
            row.cce_claimants, row.cce_renewal_claimants
        ),
        cce_has_renewal_details=_has_renewal_details(row),
        evidence=_build_evidence(row.evidence_json, row.evidence_sources_json),
    )


def _build_evidence(evidence_json: str, evidence_sources_json: str) -> tuple[EvidenceBar, ...]:
    """Combine the decoded score map with the decoded source map into bars."""
    sources = parse_evidence_sources(evidence_sources_json)
    return tuple(
        EvidenceBar(scorer=bar.scorer, normalized=bar.normalized, source=sources.get(bar.scorer))
        for bar in parse_evidence(evidence_json)
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
    field_annotations: tuple[FieldAnnotation, ...]
    annotation_tags: tuple[str, ...]


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


def _annotation_tag(annotation: FieldAnnotation) -> str:
    """Render one annotation as a compact ``field:symbol`` tag for the labels table."""
    return f"{annotation.field}:{judgment_symbol(annotation.judgment)}"


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
        field_annotations=row.field_annotations,
        annotation_tags=tuple(_annotation_tag(ann) for ann in row.field_annotations),
    )


__all__ = [
    "CLAIMANT_LABEL",
    "PREDICTED_STATUS_FAMILY_IN_COPYRIGHT",
    "PREDICTED_STATUS_FAMILY_PD",
    "PREDICTED_STATUS_FAMILY_UNKNOWN",
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
    "parse_evidence_sources",
    "predicted_status_family",
    "render_renewal_label",
]
