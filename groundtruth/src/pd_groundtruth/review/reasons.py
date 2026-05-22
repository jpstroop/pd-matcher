"""Controlled reason vocabulary for ``no_match`` / ``unsure`` verdicts.

Each verdict that expresses *doubt* carries an optional structured reason code
so failure modes can be aggregated (feeding the scoring/pairing investigations)
rather than buried in free text. The codes here are the single source of truth
shared by the route validation and the card template; a free-text note rides
alongside the code for anything the vocabulary does not anticipate.
"""

from msgspec import Struct

from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.review_db import VERDICT_UNSURE


class ReasonCode(Struct, frozen=True, forbid_unknown_fields=True):
    """One controlled reason: a stored ``code`` and its human-readable label."""

    code: str
    label: str


NO_MATCH_REASONS: tuple[ReasonCode, ...] = (
    ReasonCode(code="diff_work", label="Different work / title collision"),
    ReasonCode(code="diff_author", label="Same title, different author"),
    ReasonCode(code="wrong_year_edition", label="Wrong year or edition"),
    ReasonCode(code="translation", label="Translation / different language"),
    ReasonCode(code="garbled", label="Garbled transcription"),
    ReasonCode(code="publisher_only", label="Publisher-only overlap"),
)

UNSURE_REASONS: tuple[ReasonCode, ...] = (
    ReasonCode(code="insufficient_data", label="Insufficient data on one side"),
    ReasonCode(code="plausible_unverified", label="Plausible but unverified"),
    ReasonCode(code="edition_unsure", label="Unsure about edition"),
)

_BY_VERDICT: dict[str, tuple[ReasonCode, ...]] = {
    VERDICT_NO_MATCH: NO_MATCH_REASONS,
    VERDICT_UNSURE: UNSURE_REASONS,
}


def reasons_for(verdict: str) -> tuple[ReasonCode, ...]:
    """Return the controlled reason codes offered for ``verdict``.

    Verdicts without a doubt vocabulary (e.g. ``match``) return an empty tuple.
    """
    return _BY_VERDICT.get(verdict, ())


def is_valid_reason(verdict: str, code: str) -> bool:
    """Return ``True`` when ``code`` is an allowed reason for ``verdict``."""
    return any(reason.code == code for reason in reasons_for(verdict))


def normalize_reasons(verdict: str, codes: list[str]) -> tuple[str, ...]:
    """Return the subset of ``codes`` valid for ``verdict``, de-duplicated.

    Codes not in the verdict's vocabulary are dropped, duplicates are
    collapsed, and the result follows the vocabulary's declared order rather
    than the caller's submission order, so a label's stored reasons are stable
    regardless of which chips the reviewer clicked first.
    """
    submitted = set(codes)
    return tuple(reason.code for reason in reasons_for(verdict) if reason.code in submitted)


class ReasonSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """One render-ready reason tally row for the stats page."""

    verdict: str
    code: str
    label: str
    count: int


def summarize_reasons(counts: dict[tuple[str, str], int]) -> tuple[ReasonSummary, ...]:
    """Project ``(verdict, code) -> count`` into ordered, labeled summary rows.

    Rows follow the vocabulary's declared order (no_match codes, then unsure
    codes) and omit codes with a zero count, so the stats table shows only
    reasons actually used.
    """
    rows: list[ReasonSummary] = []
    for verdict in (VERDICT_NO_MATCH, VERDICT_UNSURE):
        for reason in reasons_for(verdict):
            count = counts.get((verdict, reason.code), 0)
            if count:
                rows.append(
                    ReasonSummary(
                        verdict=verdict,
                        code=reason.code,
                        label=reason.label,
                        count=count,
                    )
                )
    return tuple(rows)


__all__ = [
    "NO_MATCH_REASONS",
    "UNSURE_REASONS",
    "ReasonCode",
    "ReasonSummary",
    "is_valid_reason",
    "normalize_reasons",
    "reasons_for",
    "summarize_reasons",
]
