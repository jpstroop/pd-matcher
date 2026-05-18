"""Pragmatic-assumption wrappers around :mod:`predicates`.

The Cornell matrix conditions on facts we cannot directly observe from
MARC + the CCE registration corpus (notice present? US-gov work? in
foreign-country PD by 1996?). We do not attempt strict three-valued
logic; instead each inference here encodes one *documented* assumption
and returns the assumption string so the rule engine can surface it in
the final assessment. A human reviewer can therefore see exactly which
leaps the engine took.

Inference functions return ``(value, assumption | None)``. ``value`` is
the boolean the predicate slot expects; ``assumption`` is ``None`` when
the underlying signal was directly observable (no leap taken) or a short
explanation string when an assumption fired.
"""

from re import IGNORECASE
from re import Pattern
from re import compile as compile_pattern

from pd_matcher.copyright.facts import Facts
from pd_matcher.copyright.predicates import country_is_foreign

_US_GOV_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "u.s. government printing office",
        compile_pattern(r"u\.?\s*s\.?\s*government\s+printing\s+office", IGNORECASE),
    ),
    ("g.p.o.", compile_pattern(r"\bg\.?\s*p\.?\s*o\.?\b", IGNORECASE)),
    (
        "government printing office",
        compile_pattern(r"\bgovernment\s+printing\s+office\b", IGNORECASE),
    ),
    (
        "u.s. government publishing office",
        compile_pattern(r"u\.?\s*s\.?\s*government\s+publishing\s+office", IGNORECASE),
    ),
    ("department of", compile_pattern(r"\bdepartment\s+of\b", IGNORECASE)),
    ("bureau of", compile_pattern(r"\bbureau\s+of\b", IGNORECASE)),
    (
        "national park service",
        compile_pattern(r"\bnational\s+park\s+service\b", IGNORECASE),
    ),
    (
        "smithsonian institution",
        compile_pattern(r"\bsmithsonian\s+institution\b", IGNORECASE),
    ),
    (
        "library of congress",
        compile_pattern(r"\blibrary\s+of\s+congress\b", IGNORECASE),
    ),
    (
        "national archives",
        compile_pattern(r"\bnational\s+archives\b", IGNORECASE),
    ),
)


def has_us_notice(facts: Facts) -> tuple[bool, str | None]:
    """Return whether the work bore a US copyright notice (assumed from registration).

    Catalogers attached the copyright symbol to registered works as a
    matter of standard practice; a surviving CCE registration is itself
    strong evidence of notice. We therefore treat ``was_registered=True``
    as sufficient and surface the assumption to the assessment.
    """
    if facts.was_registered:
        return True, "Assumed notice: registration implies notice was affixed"
    return False, None


def is_us_government_work(facts: Facts) -> tuple[bool, str | None]:
    """Return whether the publisher text matches a US-government pattern.

    Detection is case-insensitive regex over the joined publisher string
    on :class:`Facts`. The function returns the matched pattern in the
    assumption so a reviewer can see *why* the rule fired.
    """
    if facts.publisher_text is None:
        return False, None
    for label, pattern in _US_GOV_PATTERNS:
        if pattern.search(facts.publisher_text):
            return True, f"Assumed US-government work: publisher matches {label!r}"
    return False, None


def foreign_in_pd_home_country_1996(facts: Facts) -> tuple[bool, str | None]:
    """Return whether a pre-1923 foreign work was likely PD in its source country by 1996.

    Most foreign jurisdictions used life+50 (or shorter) before Berne;
    a 1923-published foreign work's author was almost certainly dead by
    1946, fifty years before the URAA's 1 January 1996 baseline. We
    require ``country_is_foreign`` and ``pub_year < 1923`` and surface
    the assumption.
    """
    if not country_is_foreign(facts):
        return False, None
    if facts.pub_year is None or facts.pub_year >= 1923:
        return False, None
    return (
        True,
        "Assumed foreign-PD-by-1996: foreign work published before 1923",
    )


__all__ = [
    "foreign_in_pd_home_country_1996",
    "has_us_notice",
    "is_us_government_work",
]
