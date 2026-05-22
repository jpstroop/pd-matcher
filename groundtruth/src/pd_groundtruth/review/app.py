"""FastAPI application for the local single-user review UI.

The app is a thin layer over :class:`pd_groundtruth.review_db.ReviewDb` and the
pure view model in :mod:`pd_groundtruth.review.view`. SQLite connections are
not safe to share across uvicorn's worker threads, so every request opens a
*fresh* :func:`ReviewDb.connect` against the path stashed in ``app.state`` at
startup and closes it via the context manager (committing on the label write).
The typed/business logic this layer touches — card projection, progress
counts, verdict handling, filter parsing — lives in tested pure modules; the
routes themselves are exercised under the deselected ``webui`` pytest marker.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi import Form
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from pd_groundtruth.review.filters import ReviewFilters
from pd_groundtruth.review.filters import parse_filters
from pd_groundtruth.review.reasons import NO_MATCH_REASONS
from pd_groundtruth.review.reasons import UNSURE_REASONS
from pd_groundtruth.review.reasons import ReasonCode
from pd_groundtruth.review.reasons import normalize_reasons
from pd_groundtruth.review.reasons import summarize_reasons
from pd_groundtruth.review.view import build_card
from pd_groundtruth.review_db import ReviewDb

_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
_DB_PATH_ATTR: str = "review_db_path"
_REASON_FORM: list[str] = Form([])
_REASON_CONTEXT: dict[str, tuple[ReasonCode, ...]] = {
    "no_match_reasons": NO_MATCH_REASONS,
    "unsure_reasons": UNSURE_REASONS,
}


def _db_path(request: Request) -> Path:
    """Return the configured review-db path from application state."""
    path: Path = getattr(request.app.state, _DB_PATH_ATTR)
    return path


def _redirect_to_next(filters: ReviewFilters) -> RedirectResponse:
    """Build a 303 redirect to ``/`` preserving the active filters."""
    query = filters.query_string()
    location = f"/?{query}" if query else "/"
    return RedirectResponse(url=location, status_code=303)


def create_app(db_path: Path | None = None) -> FastAPI:
    """Create the review FastAPI app, optionally binding ``db_path`` now.

    Args:
        db_path: The review database path. May be left unset here and assigned
            later via :func:`set_db_path` (the CLI does this before launch).
    """
    app = FastAPI(title="pd-groundtruth review")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    if db_path is not None:
        set_db_path(app, db_path)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request, language: str | None = None, band: str | None = None
    ) -> HTMLResponse:
        filters = parse_filters(language, band)
        with ReviewDb.connect(_db_path(request)) as db:
            row = db.next_unlabeled(language=filters.language, band=filters.band)
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
    ) -> RedirectResponse:
        filters = parse_filters(language, band)
        clean_note = note.strip() if note is not None and note.strip() else None
        clean_reasons = normalize_reasons(verdict, reason)
        with ReviewDb.connect(_db_path(request)) as db:
            db.add_label(pair_id, verdict, note=clean_note, reasons=clean_reasons)
        return _redirect_to_next(filters)

    @app.get("/stats", response_class=HTMLResponse)
    def stats(
        request: Request, language: str | None = None, band: str | None = None
    ) -> HTMLResponse:
        filters = parse_filters(language, band)
        with ReviewDb.connect(_db_path(request)) as db:
            counts = db.progress()
            reason_summary = summarize_reasons(db.reason_counts())
        return templates.TemplateResponse(
            request,
            "stats.html",
            {"counts": counts, "filters": filters, "reason_summary": reason_summary},
        )

    return app


def set_db_path(app: FastAPI, db_path: Path) -> None:
    """Bind the review database path into ``app.state`` for per-request use."""
    setattr(app.state, _DB_PATH_ATTR, db_path)


app: FastAPI = create_app()


__all__ = [
    "app",
    "create_app",
    "set_db_path",
]
