"""Active review-queue filters and their URL round-tripping.

The UI threads an optional ``(language, band)`` filter pair through every
request so a focused session (e.g. the English-first curriculum) stays on its
slice across labels and redirects. This module is the pure, unit-tested core
of that threading: it normalizes raw query/form values into a typed
:class:`ReviewFilters` and serializes them back into a query string for the
303 redirect after a label.

It also carries the session-local skip list. Skipping a pair adds its id to a
``skip`` query parameter that the next ``/`` request will pass through to
``next_unlabeled`` so the same pair is not returned again. State lives in the
URL (not the database) so a fresh tab — i.e. a fresh attention session —
starts with no skips.
"""

from urllib.parse import urlencode

from msgspec import Struct

from pd_groundtruth.review_db import LabelFilters


class ReviewFilters(Struct, frozen=True, forbid_unknown_fields=True):
    """The active language/band narrowing and skip list for a review session."""

    language: str | None = None
    band: str | None = None
    skip_ids: tuple[int, ...] = ()

    def query_string(self) -> str:
        """Render the active filters as a URL query string.

        Returns an empty string when no filter is set, otherwise a string of
        the form ``language=fre&band=ge90`` (only the set keys appear). The
        skip list is *not* included so callers that just need to preserve a
        focus filter across a redirect (e.g. after a label) do not also drag
        the session's skips along.
        """
        params: list[tuple[str, str]] = []
        if self.language is not None:
            params.append(("language", self.language))
        if self.band is not None:
            params.append(("band", self.band))
        return urlencode(params)

    def next_query_string(self, *, additional_skip_id: int | None = None) -> str:
        """Render filters plus skip list (and one extra id) as a query string.

        Used by the Skip button to build the URL it navigates to: keeps the
        active language/band, threads every already-skipped pair id through,
        and appends ``additional_skip_id`` so the freshly skipped pair joins
        the exclusion set on the next ``/`` request.
        """
        params: list[tuple[str, str]] = []
        if self.language is not None:
            params.append(("language", self.language))
        if self.band is not None:
            params.append(("band", self.band))
        for pair_id in self.skip_ids:
            params.append(("skip", str(pair_id)))
        if additional_skip_id is not None and additional_skip_id not in self.skip_ids:
            params.append(("skip", str(additional_skip_id)))
        return urlencode(params)


def _clean(value: str | None) -> str | None:
    """Strip a raw value and collapse blanks to ``None``."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _clean_skip_ids(raw: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    """Deduplicate a sequence of skip pair ids preserving first-seen order."""
    if not raw:
        return ()
    seen: set[int] = set()
    ordered: list[int] = []
    for pair_id in raw:
        if pair_id in seen:
            continue
        seen.add(pair_id)
        ordered.append(pair_id)
    return tuple(ordered)


def parse_filters(
    language: str | None,
    band: str | None,
    skip_ids: list[int] | tuple[int, ...] | None = None,
) -> ReviewFilters:
    """Normalize raw query/form values into typed :class:`ReviewFilters`.

    Whitespace is stripped and empty strings become ``None`` so that a form
    that submits ``language=`` does not over-narrow the queue. ``skip_ids`` is
    deduplicated while preserving first-seen order so the URL grows by at most
    one entry per Skip click.
    """
    return ReviewFilters(
        language=_clean(language),
        band=_clean(band),
        skip_ids=_clean_skip_ids(skip_ids),
    )


def parse_label_filters(
    verdict: str | None,
    language: str | None,
    q: str | None,
) -> LabelFilters:
    """Normalize raw ``/labels`` query values into typed :class:`LabelFilters`.

    Whitespace is stripped and empty strings become ``None`` so that a form
    that submits ``verdict=`` does not over-narrow the row set. ``q`` is kept
    in its raw case here; the DB layer lower-cases it for matching.
    """
    return LabelFilters(
        verdict=_clean(verdict),
        language=_clean(language),
        q=_clean(q),
    )


def label_filters_query_string(filters: LabelFilters, *, drop: str | None = None) -> str:
    """Render active label filters as a URL query string.

    Pass ``drop`` to omit one filter key from the rendered string — used by
    the per-filter "clear" links in the page header so each link removes only
    the filter it represents while preserving the rest.
    """
    params: list[tuple[str, str]] = []
    if filters.verdict is not None and drop != "verdict":
        params.append(("verdict", filters.verdict))
    if filters.language is not None and drop != "language":
        params.append(("language", filters.language))
    if filters.q is not None and drop != "q":
        params.append(("q", filters.q))
    return urlencode(params)


def label_filters_active(filters: LabelFilters) -> bool:
    """Return ``True`` when any of the label filters is set."""
    return any(value is not None for value in (filters.verdict, filters.language, filters.q))


__all__ = [
    "ReviewFilters",
    "label_filters_active",
    "label_filters_query_string",
    "parse_filters",
    "parse_label_filters",
]
