"""Active review-queue filters and their URL round-tripping.

The UI threads an optional ``(language, band)`` filter pair through every
request so a focused session (e.g. the English-first curriculum) stays on its
slice across labels and redirects. This module is the pure, unit-tested core
of that threading: it normalizes raw query/form values into a typed
:class:`ReviewFilters` and serializes them back into a query string for the
303 redirect after a label.
"""

from urllib.parse import urlencode

from msgspec import Struct


class ReviewFilters(Struct, frozen=True, forbid_unknown_fields=True):
    """The active language/band narrowing for a review session."""

    language: str | None = None
    band: str | None = None

    def query_string(self) -> str:
        """Render the active filters as a URL query string.

        Returns an empty string when no filter is set, otherwise a string of
        the form ``language=fre&band=ge90`` (only the set keys appear).
        """
        params: list[tuple[str, str]] = []
        if self.language is not None:
            params.append(("language", self.language))
        if self.band is not None:
            params.append(("band", self.band))
        return urlencode(params)


def _clean(value: str | None) -> str | None:
    """Strip a raw value and collapse blanks to ``None``."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_filters(language: str | None, band: str | None) -> ReviewFilters:
    """Normalize raw query/form values into typed :class:`ReviewFilters`.

    Whitespace is stripped and empty strings become ``None`` so that a form
    that submits ``language=`` does not over-narrow the queue.
    """
    return ReviewFilters(language=_clean(language), band=_clean(band))


__all__ = [
    "ReviewFilters",
    "parse_filters",
]
