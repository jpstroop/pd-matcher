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
from itertools import pairwise

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

_COVERAGE_PARTIALNESS_THRESHOLD: float = 0.50
"""Bucket-size ratio below which a year is treated as partial-data.

The "last reliable year" is the highest year ``y`` such that
``bucket(y) >= _COVERAGE_PARTIALNESS_THRESHOLD * bucket(y-1)``. A bucket
that drops below half the prior year's size is treated as evidence that
the corpus is truncated at that boundary rather than that the year was
genuinely quieter.
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


def _last_reliable_year(
    counts: dict[int, int],
    *,
    threshold: float = _COVERAGE_PARTIALNESS_THRESHOLD,
) -> int | None:
    """Return the highest year whose bucket is not partial-truncation evidence.

    Walks years in ascending order; the result is the last year ``y`` such
    that ``counts[y] >= threshold * counts[y-1]``. A bucket that falls
    below the threshold is treated as data-boundary truncation, so the
    result is the year *before* that drop.

    Args:
        counts: Per-year bucket sizes.
        threshold: The partial-data ratio. Years whose bucket size is
            below this fraction of the previous year's bucket are treated
            as truncated.

    Returns:
        The highest reliable year, or ``None`` when ``counts`` is empty.
    """
    if not counts:
        return None
    years = sorted(counts)
    last_good = years[0]
    for prev, curr in pairwise(years):
        prev_count = counts[prev]
        curr_count = counts[curr]
        if prev_count > 0 and curr_count < threshold * prev_count:
            break
        last_good = curr
    return last_good


def _first_year(counts: dict[int, int]) -> int | None:
    """Return the lowest year in ``counts`` (``None`` when empty)."""
    if not counts:
        return None
    return min(counts)


def coverage_from_year_counts(
    *,
    reg_counts: dict[int, int],
    ren_counts: dict[int, int],
    hard_reg_max_year: int = HARD_REG_MAX_YEAR,
    hard_ren_max_year: int = HARD_REN_MAX_YEAR,
) -> Coverage:
    """Derive a :class:`Coverage` from per-year bucket histograms.

    The "last reliable year" per source is the highest year whose bucket
    size is at least :data:`_COVERAGE_PARTIALNESS_THRESHOLD` of the
    previous year's bucket size, capped at the legal-regime maximum.
    Falls back to :data:`LEGACY_COVERAGE` defaults when a histogram is
    empty so callers always receive a valid struct.

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
    reg_min = _first_year(reg_counts) or LEGACY_COVERAGE.reg_min_year
    ren_min = _first_year(ren_counts) or LEGACY_COVERAGE.ren_min_year
    reg_last = _last_reliable_year(reg_counts)
    ren_last = _last_reliable_year(ren_counts)
    reg_uncapped = reg_last if reg_last is not None else LEGACY_COVERAGE.reg_max_year
    ren_uncapped = ren_last if ren_last is not None else LEGACY_COVERAGE.ren_max_year
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
