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
from pd_groundtruth.review import nav_history
from pd_groundtruth.review.filters import ReviewFilters
from pd_groundtruth.review.filters import label_filters_active
from pd_groundtruth.review.filters import label_filters_query_string
from pd_groundtruth.review.filters import parse_filters
from pd_groundtruth.review.filters import parse_label_filters
from pd_groundtruth.review.view import build_card
from pd_groundtruth.review.view import build_labeled_row
from pd_groundtruth.review_db import CurrentLabelRow
from pd_groundtruth.review_db import ProgressCounts
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.review_db import ReviewPairRow
from pd_groundtruth.sampling import BAND_70_80
from pd_groundtruth.sampling import BAND_80_90
from pd_groundtruth.sampling import BAND_BELOW
from pd_groundtruth.sampling import BAND_GE90
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)

_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
_DB_PATH_ATTR: str = "review_db_path"
_VAULT_PATH_ATTR: str = "label_vault_path"
_LABELER: str = "jpstroop"
_SKIP_QUERY: list[int] = Query([])
_LANGUAGE_CHOICES: tuple[str, ...] = ("eng", "fre", "ger", "spa", "ita")
_BAND_CHOICES: tuple[str, ...] = (BAND_GE90, BAND_80_90, BAND_70_80, BAND_BELOW)
_VERDICT_CHOICES: tuple[str, ...] = ("match", "no_match", "unsure")
_SORT_CHOICES: tuple[tuple[str, str], ...] = (("desc", "newest first"), ("asc", "oldest first"))
_LABELS_PAGE_SIZE: int = 100


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

    def _render_card(
        request: Request,
        row: ReviewPairRow,
        counts: ProgressCounts,
        filters: ReviewFilters,
        current_label: CurrentLabelRow | None,
    ) -> HTMLResponse:
        """Render one card, threading the cookie nav history through context."""
        history = nav_history.read_history(request)
        history = nav_history.advance(history, row.id)
        response = templates.TemplateResponse(
            request,
            "card.html",
            {
                "card": build_card(row, current_label=current_label),
                "filters": filters,
                "counts": counts,
                "back_id": nav_history.back_id(history),
                "forward_id": nav_history.forward_id(history),
                "language_choices": _LANGUAGE_CHOICES,
                "band_choices": _BAND_CHOICES,
            },
        )
        nav_history.write_history(response, history)
        return response

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
        if row is None:
            history = nav_history.read_history(request)
            response = templates.TemplateResponse(
                request,
                "empty.html",
                {
                    "filters": filters,
                    "counts": counts,
                    "back_id": nav_history.back_id(history),
                    "language_choices": _LANGUAGE_CHOICES,
                    "band_choices": _BAND_CHOICES,
                },
            )
            nav_history.write_history(response, history)
            return response
        return _render_card(request, row, counts, filters, current_label=None)

    @app.get("/pair/{pair_id}", response_class=HTMLResponse)
    def pair(
        request: Request, pair_id: int, language: str | None = None, band: str | None = None
    ) -> HTMLResponse:
        filters = parse_filters(language, band)
        with ReviewDb.connect(_db_path(request)) as db:
            row = db.get_pair(pair_id)
            counts = db.progress()
            current_label = db.get_current_label(pair_id) if row is not None else None
        if row is None:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"filters": filters, "counts": counts, "pair_id": pair_id},
                status_code=404,
            )
        return _render_card(request, row, counts, filters, current_label)

    @app.post("/label")
    def label(
        request: Request,
        pair_id: int = Form(...),
        verdict: str = Form(...),
        note: str | None = Form(None),
        language: str | None = Form(None),
        band: str | None = Form(None),
    ) -> RedirectResponse:
        filters = parse_filters(language, band)
        clean_note = note.strip() if note is not None and note.strip() else None
        with ReviewDb.connect(_db_path(request)) as db:
            pair = db.get_pair(pair_id)
            result = db.add_label(pair_id, verdict, note=clean_note)
        if pair is not None:
            _append_vault_entry(
                vault_path=_vault_path(request),
                marc_json=pair.marc_json,
                nypl_uuid=pair.nypl_uuid,
                verdict=verdict,
                note=clean_note,
                labeled_at=result.labeled_at,
                cce_regnum=pair.cce_regnum,
                cce_renewal_id=pair.cce_renewal_id,
                cce_renewal_oreg=pair.cce_renewal_oreg,
            )
        return _redirect_to_next(filters)

    @app.get("/stats", response_class=HTMLResponse)
    def stats(
        request: Request, language: str | None = None, band: str | None = None
    ) -> HTMLResponse:
        filters = parse_filters(language, band)
        with ReviewDb.connect(_db_path(request)) as db:
            counts = db.progress()
        return templates.TemplateResponse(
            request,
            "stats.html",
            {
                "counts": counts,
                "filters": filters,
            },
        )

    @app.get("/labels", response_class=HTMLResponse)
    def labels(
        request: Request,
        verdict: str | None = None,
        language: str | None = None,
        q: str | None = None,
        sort: str | None = None,
        page: int = 1,
    ) -> HTMLResponse:
        label_filters = parse_label_filters(verdict, language, q, sort)
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
                "sort_choices": _SORT_CHOICES,
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
    note: str | None,
    labeled_at: str,
    cce_regnum: str | None,
    cce_renewal_id: str | None,
    cce_renewal_oreg: str | None,
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
            note=note,
            labeled_at=labeled_at,
            labeler=_LABELER,
            marc_identifiers=extract_marc_identifiers(marc),
            cce_regnum=cce_regnum,
            cce_renewal_id=cce_renewal_id,
            cce_renewal_oreg=cce_renewal_oreg,
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
