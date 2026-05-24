"""Reliability bounds for absence-of-evidence rules.

The pub-year range over which registration / renewal evidence in the index
is reliable. Outside this range, absence-of-evidence rules (e.g. "no
registration -> PD") return
:attr:`~pd_matcher.copyright.status.CopyrightStatus.UNKNOWN_INSUFFICIENT_COVERAGE`
rather than misfiring on a corpus boundary.

Sources today (2026): NYPL's CCE transcription covers 1891-1977 (the entire
1909-Act regime; see Copyright Office Circular 23, available at
https://www.copyright.gov/circs/circ23.pdf). ``reg_max_year`` and
``ren_max_year`` are capped at :data:`HARD_REG_MAX_YEAR` by construction
because the 1909 Act ended 31 December 1977; post-1978 registrations live
in the 1976-Act-regime catalog at copyright.gov rather than in the CCE.
"""

from collections.abc import Iterable
from statistics import median

from msgspec import Struct

HARD_REG_MAX_YEAR: int = 1977
"""Upper bound on ``reg_max_year`` regardless of corpus content.

The 1909 Act ended 31 Dec 1977; post-1978 registrations are not part of
the CCE and live in a different catalog at copyright.gov. Treating any
year above this as "in coverage" would be a legal-regime error, not a
data-completeness one.
"""

HARD_REN_MAX_YEAR: int = 2005
"""Upper bound on ``ren_max_year`` regardless of corpus content.

Renewals under the 1909 Act regime were filed in the 28th year after
registration. A 1977 registration could be renewed as late as 2005 (1977
+ 28). Anything beyond this is outside the 1909-Act renewal window.
"""

_COVERAGE_RELIABILITY_RATIO: float = 0.10
"""Bucket-size ratio below which a year is treated as data-boundary noise.

A year is "reliable" when its bucket size is at least this fraction of
the median bucket size over the most recent
:data:`_COVERAGE_RELIABILITY_WINDOW` reliable years. The walk anchors at
the histogram peak (where the corpus is densest) and steps outward; the
first year whose bucket falls below ``ratio * median(recent reliable)``
ends the walk. ``0.10`` was chosen against the production CCE histograms:
the real renewal cliff is 1991 (23,254) → 1992 (21), a 99.9% drop that
clears any threshold up to ~99%, while the genuine within-corpus
year-to-year noise stays inside an order of magnitude.
"""

_COVERAGE_RELIABILITY_WINDOW: int = 5
"""Number of trailing reliable years used to compute the comparison median.

A trailing median (rather than the previous single year) absorbs
year-to-year jitter in the histogram so a single low year does not freeze
the walk inside the bulk of the data. Five years is small enough to track
real corpus trends (e.g. the steady ramp in 1949-1955 registrations) but
large enough to dampen single-year dips.
"""


class Coverage(Struct, frozen=True, forbid_unknown_fields=True):
    """The pub-year range over which registration / renewal evidence is reliable.

    Outside this range, absence-of-evidence rules return
    :attr:`~pd_matcher.copyright.status.CopyrightStatus.UNKNOWN_INSUFFICIENT_COVERAGE`
    rather than misfiring.

    Sources today (2026): NYPL CCE transcription covers 1891-1977 (the
    entire 1909-Act regime; see Copyright Office Circular 23).
    ``reg_max_year`` and ``ren_max_year`` are capped at
    :data:`HARD_REG_MAX_YEAR` / :data:`HARD_REN_MAX_YEAR` by construction
    because the 1909 Act ended 31 Dec 1977; post-1978 registrations live
    in the 1976-Act-regime catalog at copyright.gov.
    """

    reg_min_year: int
    reg_max_year: int
    ren_min_year: int
    ren_max_year: int


LEGACY_COVERAGE: Coverage = Coverage(
    reg_min_year=1891,
    reg_max_year=HARD_REG_MAX_YEAR,
    ren_min_year=1909,
    ren_max_year=1991,
)
"""Sensible fallback for callers without an index.

1891 is the start of registrations under the original 1891 Act predecessor
to the 1909 Act. 1977 is the legal upper bound (see
:data:`HARD_REG_MAX_YEAR`). 1991 is the latest renewal under the 1909-Act
regime that the NYPL transcription typically carries (28-year renewal of
1963 registrations, etc.). Production callers should derive a
:class:`Coverage` from the index meta via :func:`coverage_from_year_counts`
instead of using this default.
"""


def _walk_from_anchor(
    counts: dict[int, int],
    *,
    anchor: int,
    step: int,
    window: int,
    ratio: float,
) -> int:
    """Walk outward from ``anchor`` while bucket sizes stay reliable.

    The walk advances by ``step`` (``+1`` forward, ``-1`` backward).
    At each step the next year's count is compared against
    ``ratio * median(recent_reliable[-window:])``; the first year that
    fails ends the walk. Returns the last year that passed (the anchor
    itself when no neighbours qualify).

    Args:
        counts: Per-year bucket sizes (already cleaned of out-of-range
            garbage years).
        anchor: The year the walk starts from; always counted as reliable.
        step: ``+1`` to walk forward, ``-1`` to walk backward.
        window: Number of trailing reliable buckets included in the
            comparison median.
        ratio: A year is reliable when ``counts[year] >= ratio * median``.

    Returns:
        The outermost year that satisfied the reliability check.
    """
    reliable: list[int] = [counts[anchor]]
    last_good = anchor
    year = anchor + step
    while year in counts:
        reference = median(reliable[-window:])
        if counts[year] < ratio * reference:
            break
        reliable.append(counts[year])
        last_good = year
        year += step
    return last_good


def _bounds_from_counts(
    counts: dict[int, int],
    *,
    window: int = _COVERAGE_RELIABILITY_WINDOW,
    ratio: float = _COVERAGE_RELIABILITY_RATIO,
) -> tuple[int, int] | None:
    """Return ``(min_year, max_year)`` derived from a histogram via the anchored walk.

    The peak bucket is taken as the anchor (the densest, most reliable
    year by construction); the walk then expands outward in both
    directions, stopping at the first year whose count falls below
    ``ratio`` times the median of the recent reliable buckets. Returns
    ``None`` for an empty histogram so the caller can fall back to a
    legacy default.
    """
    if not counts:
        return None
    anchor = max(counts, key=lambda year: counts[year])
    min_year = _walk_from_anchor(counts, anchor=anchor, step=-1, window=window, ratio=ratio)
    max_year = _walk_from_anchor(counts, anchor=anchor, step=1, window=window, ratio=ratio)
    return min_year, max_year


def coverage_from_year_counts(
    *,
    reg_counts: dict[int, int],
    ren_counts: dict[int, int],
    hard_reg_max_year: int = HARD_REG_MAX_YEAR,
    hard_ren_max_year: int = HARD_REN_MAX_YEAR,
) -> Coverage:
    """Derive a :class:`Coverage` from per-year bucket histograms.

    Each bound is found by anchoring at the peak (largest-count) year and
    walking outward (forward for ``max``, backward for ``min``). A step
    succeeds while the next year's count is at least
    :data:`_COVERAGE_RELIABILITY_RATIO` times the median of the trailing
    :data:`_COVERAGE_RELIABILITY_WINDOW` reliable buckets; the walk stops
    at the first year that falls below that threshold. ``max`` is then
    capped at the legal-regime maximum. Falls back to
    :data:`LEGACY_COVERAGE` defaults when a histogram is empty so callers
    always receive a valid struct.

    Anchoring at the peak (rather than the first observed year) protects
    the bounds against the long tail of garbage / partial-data years a
    forward-only walk would freeze on. Single-year out-of-range entries
    in the histogram cannot become the anchor (their count is one) and
    cannot keep the backward walk from terminating (the trailing-median
    comparison rejects them as soon as the bulk is exhausted).

    Args:
        reg_counts: ``{year: count}`` for registrations.
        ren_counts: ``{year: count}`` for renewals.
        hard_reg_max_year: Upper cap on ``reg_max_year`` regardless of
            the histogram (defaults to :data:`HARD_REG_MAX_YEAR`).
        hard_ren_max_year: Upper cap on ``ren_max_year`` regardless of
            the histogram (defaults to :data:`HARD_REN_MAX_YEAR`).

    Returns:
        A frozen :class:`Coverage` capping ``reg_max_year`` at
        ``hard_reg_max_year`` and ``ren_max_year`` at
        ``hard_ren_max_year``.
    """
    reg_bounds = _bounds_from_counts(reg_counts)
    ren_bounds = _bounds_from_counts(ren_counts)
    reg_min, reg_uncapped = (
        reg_bounds
        if reg_bounds is not None
        else (LEGACY_COVERAGE.reg_min_year, LEGACY_COVERAGE.reg_max_year)
    )
    ren_min, ren_uncapped = (
        ren_bounds
        if ren_bounds is not None
        else (LEGACY_COVERAGE.ren_min_year, LEGACY_COVERAGE.ren_max_year)
    )
    reg_max = min(reg_uncapped, hard_reg_max_year)
    ren_max = min(ren_uncapped, hard_ren_max_year)
    return Coverage(
        reg_min_year=reg_min,
        reg_max_year=reg_max,
        ren_min_year=ren_min,
        ren_max_year=ren_max,
    )


def coverage_from_pairs(
    *,
    reg_year_counts: Iterable[tuple[int, int]],
    ren_year_counts: Iterable[tuple[int, int]],
) -> Coverage:
    """Convenience wrapper accepting iterables of ``(year, count)`` pairs."""
    return coverage_from_year_counts(
        reg_counts=dict(reg_year_counts),
        ren_counts=dict(ren_year_counts),
    )


__all__ = [
    "HARD_REG_MAX_YEAR",
    "HARD_REN_MAX_YEAR",
    "LEGACY_COVERAGE",
    "Coverage",
    "coverage_from_pairs",
    "coverage_from_year_counts",
]
