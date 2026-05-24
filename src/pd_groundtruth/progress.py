"""Groundtruth-specific progress detail: the kept-per-stratum suffix.

The generic progress readout (rate, ETA, percent, ``mm:ss`` formatting and
the cadence-gated reporter) lives in :mod:`pd_matcher.progress` and is shared
with the production matcher. This module supplies only the domain-specific
``kept: …`` suffix that :mod:`pd_groundtruth.build_queue` appends to each
emitted line via the reporter's ``detail`` hook.
"""

from pd_groundtruth.sampling import BAND_BELOW
from pd_groundtruth.sampling import BudgetModel
from pd_groundtruth.sampling import iter_capped_bands

_ALL_BANDS: tuple[str, ...] = (*iter_capped_bands(), BAND_BELOW)


def render_kept_suffix(
    budget: BudgetModel,
    kept_by_stratum: dict[tuple[str, str], int],
) -> str:
    """Render kept-per-stratum counts as a ``kept: …`` suffix.

    The first configured language gets a per-band breakdown with caps
    (``eng[ge90 412/500 ...]``); every other language collapses to a single
    running total (``fre 140``). An empty budget yields the bare ``kept: ``
    prefix.
    """
    languages = budget.languages()
    if not languages:
        return "kept: "
    lead = languages[0]
    bands = " ".join(
        f"{band} {kept_by_stratum.get((lead, band), 0)}/{budget.cap_for(lead, band)}"
        for band in _ALL_BANDS
    )
    parts = [f"{lead}[{bands}]"]
    for language in languages[1:]:
        total = sum(kept_by_stratum.get((language, band), 0) for band in _ALL_BANDS)
        parts.append(f"{language} {total}")
    return f"kept: {' '.join(parts)}"


__all__ = [
    "render_kept_suffix",
]
