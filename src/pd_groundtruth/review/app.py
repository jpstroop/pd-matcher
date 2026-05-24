"""FastAPI application for the local single-user review UI.

The app is a thin layer over :class:`pd_groundtruth.review_db.ReviewDb` and the
pure view model in :mod:`pd_groundtruth.review.view`. SQLite connections are
not safe to share across uvicorn's worker threads, so every request opens a
*fresh* :func:`ReviewDb.connect` against the path stashed in ``app.state`` at
startup and closes it via the context manager (committing on the label write).
The typed/business logic this layer touches — card projection, progress
counts, verdict handling, filter parsing — lives in tested pure modules; the
routes themselves are exercised under the deselected ``webui`` pytest marker.

Every accepted verdict is appended to the JSONL label vault
(:mod:`pd_groundtruth.label_vault`) immediately after the DB write. The vault
is the durable, git-tracked source of truth: ``review.db`` is regenerated each
time ``acquire`` / ``build-queue`` runs, but the vault survives. A vault-append
failure is logged but does not fail the HTTP request; the DB write already
succeeded and dropping the vault line is an integrity concern to surface, not
a user-facing 500.
"""

from datetime import UTC
from datetime import datetime
from logging import getLogger
from pathlib import Path

from fastapi import FastAPI
from fastapi import Form
from fastapi import Query
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from msgspec.json import decode as json_decode

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import append_entry
from pd_groundtruth.label_vault import extract_marc_identifiers
from pd_groundtruth.review.field_annotations import ALL_JUDGMENTS
from pd_groundtruth.review.field_annotations import ANNOTATABLE_FIELDS
from pd_groundtruth.review.field_annotations import FieldAnnotation
from pd_groundtruth.review.field_annotations import judgment_label
from pd_groundtruth.review.field_annotations import normalize_annotations
from pd_groundtruth.review.field_annotations import summarize_field_annotations
from pd_groundtruth.review.filters import ReviewFilters
from pd_groundtruth.review.filters import label_filters_active
from pd_groundtruth.review.filters import label_filters_query_string
from pd_groundtruth.review.filters import parse_filters
from pd_groundtruth.review.filters import parse_label_filters
from pd_groundtruth.review.reasons import NO_MATCH_REASONS
from pd_groundtruth.review.reasons import UNSURE_REASONS
from pd_groundtruth.review.reasons import ReasonCode
from pd_groundtruth.review.reasons import normalize_reasons
from pd_groundtruth.review.reasons import summarize_reasons
from pd_groundtruth.review.view import build_card
from pd_groundtruth.review.view import build_labeled_row
from pd_groundtruth.review_db import ReviewDb
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)

_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
_DB_PATH_ATTR: str = "review_db_path"
_VAULT_PATH_ATTR: str = "label_vault_path"
_LABELER: str = "jpstroop"
_REASON_FORM: list[str] = Form([])
_SKIP_QUERY: list[int] = Query([])
_REASON_CONTEXT: dict[str, tuple[ReasonCode, ...]] = {
    "no_match_reasons": NO_MATCH_REASONS,
    "unsure_reasons": UNSURE_REASONS,
}
_FIELD_ANNOTATION_CONTEXT: dict[str, tuple[str, ...]] = {
    "annotatable_fields": ANNOTATABLE_FIELDS,
    "field_judgments": ALL_JUDGMENTS,
}
_LANGUAGE_CHOICES: tuple[str, ...] = ("eng", "fre", "ger", "spa", "ita")
_VERDICT_CHOICES: tuple[str, ...] = ("match", "no_match", "unsure")
_LABELS_PAGE_SIZE: int = 100
_ALL_REASON_CODES: tuple[ReasonCode, ...] = NO_MATCH_REASONS + UNSURE_REASONS


def _db_path(request: Request) -> Path:
    """Return the configured review-db path from application state."""
    path: Path = getattr(request.app.state, _DB_PATH_ATTR)
    return path


def _vault_path(request: Request) -> Path:
    """Return the configured label-vault path from application state."""
    path: Path = getattr(request.app.state, _VAULT_PATH_ATTR)
    return path


def _redirect_to_next(filters: ReviewFilters) -> RedirectResponse:
    """Build a 303 redirect to ``/`` preserving the active filters."""
    query = filters.query_string()
    location = f"/?{query}" if query else "/"
    return RedirectResponse(url=location, status_code=303)


def create_app(db_path: Path | None = None, vault_path: Path | None = None) -> FastAPI:
    """Create the review FastAPI app, optionally binding paths now.

    Args:
        db_path: The review database path. May be left unset here and assigned
            later via :func:`set_db_path` (the CLI does this before launch).
        vault_path: The JSONL label-vault path. May be left unset here and
            assigned later via :func:`set_vault_path`.
    """
    app = FastAPI(title="pd-groundtruth review")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    if db_path is not None:
        set_db_path(app, db_path)
    if vault_path is not None:
        set_vault_path(app, vault_path)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        language: str | None = None,
        band: str | None = None,
        skip: list[int] = _SKIP_QUERY,
    ) -> HTMLResponse:
        filters = parse_filters(language, band, skip)
        with ReviewDb.connect(_db_path(request)) as db:
            row = db.next_unlabeled(
                language=filters.language,
                band=filters.band,
                exclude_pair_ids=filters.skip_ids,
            )
            counts = db.progress()
            back = db.previous_labeled(language=filters.language, band=filters.band)
        back_id = None if back is None else back.id
        if row is None:
            return templates.TemplateResponse(
                request,
                "empty.html",
                {"filters": filters, "counts": counts, "back_id": back_id},
            )
        return templates.TemplateResponse(
            request,
            "card.html",
            {
                "card": build_card(row),
                "filters": filters,
                "counts": counts,
                "back_id": back_id,
                **_REASON_CONTEXT,
                **_FIELD_ANNOTATION_CONTEXT,
            },
        )

    @app.get("/pair/{pair_id}", response_class=HTMLResponse)
    def pair(
        request: Request, pair_id: int, language: str | None = None, band: str | None = None
    ) -> HTMLResponse:
        filters = parse_filters(language, band)
        with ReviewDb.connect(_db_path(request)) as db:
            row = db.get_pair(pair_id)
            counts = db.progress()
            back = db.previous_labeled(before=pair_id, language=filters.language, band=filters.band)
        back_id = None if back is None else back.id
        if row is None:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"filters": filters, "counts": counts, "pair_id": pair_id},
                status_code=404,
            )
        return templates.TemplateResponse(
            request,
            "card.html",
            {
                "card": build_card(row),
                "filters": filters,
                "counts": counts,
                "back_id": back_id,
                **_REASON_CONTEXT,
                **_FIELD_ANNOTATION_CONTEXT,
            },
        )

    @app.post("/label")
    def label(
        request: Request,
        pair_id: int = Form(...),
        verdict: str = Form(...),
        reason: list[str] = _REASON_FORM,
        note: str | None = Form(None),
        language: str | None = Form(None),
        band: str | None = Form(None),
        annotation_title: str | None = Form(None),
        annotation_author: str | None = Form(None),
        annotation_publisher: str | None = Form(None),
        annotation_year: str | None = Form(None),
        annotation_edition: str | None = Form(None),
    ) -> RedirectResponse:
        filters = parse_filters(language, band)
        clean_note = note.strip() if note is not None and note.strip() else None
        clean_reasons = normalize_reasons(verdict, reason)
        clean_annotations = normalize_annotations(
            {
                "title": annotation_title or "",
                "author": annotation_author or "",
                "publisher": annotation_publisher or "",
                "year": annotation_year or "",
                "edition": annotation_edition or "",
            }
        )
        with ReviewDb.connect(_db_path(request)) as db:
            pair = db.get_pair(pair_id)
            result = db.add_label(
                pair_id,
                verdict,
                note=clean_note,
                reasons=clean_reasons,
                annotations=clean_annotations,
            )
        if pair is not None:
            _append_vault_entry(
                vault_path=_vault_path(request),
                marc_json=pair.marc_json,
                nypl_uuid=pair.nypl_uuid,
                verdict=verdict,
                reasons=clean_reasons,
                note=clean_note,
                labeled_at=result.labeled_at,
                field_annotations=clean_annotations,
            )
        return _redirect_to_next(filters)

    @app.get("/stats", response_class=HTMLResponse)
    def stats(
        request: Request, language: str | None = None, band: str | None = None
    ) -> HTMLResponse:
        filters = parse_filters(language, band)
        with ReviewDb.connect(_db_path(request)) as db:
            counts = db.progress()
            reason_summary = summarize_reasons(db.reason_counts())
            annotation_summary = summarize_field_annotations(db.field_annotation_counts())
        return templates.TemplateResponse(
            request,
            "stats.html",
            {
                "counts": counts,
                "filters": filters,
                "reason_summary": reason_summary,
                "annotation_summary": annotation_summary,
                "field_judgments": ALL_JUDGMENTS,
                "judgment_label": judgment_label,
            },
        )

    @app.get("/labels", response_class=HTMLResponse)
    def labels(
        request: Request,
        verdict: str | None = None,
        language: str | None = None,
        reason: str | None = None,
        q: str | None = None,
        page: int = 1,
    ) -> HTMLResponse:
        label_filters = parse_label_filters(verdict, language, reason, q)
        current_page = max(page, 1)
        with ReviewDb.connect(_db_path(request)) as db:
            counts = db.progress()
            total_labeled = counts.labeled
            filtered_total = db.count_labeled_pairs(label_filters)
            db_rows = db.iter_labeled_pairs(
                label_filters, page_size=_LABELS_PAGE_SIZE, page=current_page
            )
        now = datetime.now(UTC)
        rows = tuple(build_labeled_row(row, now) for row in db_rows)
        total_pages = max(1, (filtered_total + _LABELS_PAGE_SIZE - 1) // _LABELS_PAGE_SIZE)
        capped_page = min(current_page, total_pages)
        return templates.TemplateResponse(
            request,
            "labels.html",
            {
                "rows": rows,
                "counts": counts,
                "label_filters": label_filters,
                "filters_active": label_filters_active(label_filters),
                "filtered_total": filtered_total,
                "total_labeled": total_labeled,
                "page": capped_page,
                "total_pages": total_pages,
                "page_size": _LABELS_PAGE_SIZE,
                "language_choices": _LANGUAGE_CHOICES,
                "verdict_choices": _VERDICT_CHOICES,
                "reason_choices": _ALL_REASON_CODES,
                "query_string": label_filters_query_string(label_filters),
                "query_string_for": lambda drop=None: label_filters_query_string(
                    label_filters, drop=drop
                ),
            },
        )

    return app


def set_db_path(app: FastAPI, db_path: Path) -> None:
    """Bind the review database path into ``app.state`` for per-request use."""
    setattr(app.state, _DB_PATH_ATTR, db_path)


def set_vault_path(app: FastAPI, vault_path: Path) -> None:
    """Bind the label-vault path into ``app.state`` for per-request use."""
    setattr(app.state, _VAULT_PATH_ATTR, vault_path)


def _append_vault_entry(
    *,
    vault_path: Path,
    marc_json: str,
    nypl_uuid: str,
    verdict: str,
    reasons: tuple[str, ...],
    note: str | None,
    labeled_at: str,
    field_annotations: tuple[FieldAnnotation, ...] = (),
) -> None:
    """Append one verdict to the vault, swallowing and logging any I/O failure.

    The DB write has already succeeded by the time this is called; failing the
    HTTP request because of a vault problem would punish the reviewer for an
    operational hiccup, so we log loudly and move on. Repeat failures are easy
    to spot in the logs and the DB still carries the verdict.
    """
    try:
        marc = json_decode(marc_json.encode("utf-8"), type=MarcRecord)
        entry = VaultEntry(
            schema=SCHEMA_VERSION,
            marc_control_id=marc.control_id,
            nypl_uuid=nypl_uuid,
            verdict=verdict,
            reasons=reasons,
            note=note,
            labeled_at=labeled_at,
            labeler=_LABELER,
            marc_identifiers=extract_marc_identifiers(marc),
            field_annotations=field_annotations,
        )
        append_entry(vault_path, entry)
    except Exception:
        _LOGGER.exception(
            "label vault append failed for marc=%s nypl=%s",
            marc_json[:80],
            nypl_uuid,
        )


app: FastAPI = create_app()


__all__ = [
    "app",
    "create_app",
    "set_db_path",
    "set_vault_path",
]
