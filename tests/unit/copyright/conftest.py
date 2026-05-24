"""Shared fixtures for the copyright rule-engine test suite."""

from pytest import fixture

from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.copyright import default_ruleset
from pd_matcher.copyright.coverage import Coverage
from pd_matcher.copyright.facts import Facts

AS_OF_YEAR: int = 2026

WIDE_COVERAGE: Coverage = Coverage(
    reg_min_year=1800,
    ren_min_year=1800,
    reg_max_year=2100,
    ren_max_year=2100,
)
"""Coverage struct wide enough to neutralize coverage-guard short-circuits.

Test cases that exist to exercise a specific rule (not the coverage
mechanism) pass this so absence-of-evidence rules fire regardless of
pub-year. Production callers should never construct a coverage this
wide; use
:func:`~pd_matcher.copyright.coverage.coverage_from_year_counts` instead,
which respects the
:data:`~pd_matcher.copyright.coverage.HARD_REG_MAX_YEAR` legal cap.
"""


def make_facts(
    *,
    pub_year: int | None = None,
    pub_country_code: str | None = None,
    language_code: str | None = None,
    publisher_text: str | None = None,
    was_registered: bool = False,
    was_renewed: bool = False,
    match_confidence: float = 0.0,
    as_of_year: int = AS_OF_YEAR,
) -> Facts:
    """Return a :class:`Facts` populated with the supplied overrides."""
    return Facts(
        pub_year=pub_year,
        pub_country_code=pub_country_code,
        language_code=language_code,
        publisher_text=publisher_text,
        was_registered=was_registered,
        was_renewed=was_renewed,
        match_confidence=match_confidence,
        as_of_year=as_of_year,
    )


@fixture
def as_of_year() -> int:
    """Pinned reference year used by every test in the suite."""
    return AS_OF_YEAR


@fixture
def ruleset() -> CopyrightRuleSet:
    """The shipped Cornell ruleset."""
    return default_ruleset()


@fixture
def wide_coverage() -> Coverage:
    """Return :data:`WIDE_COVERAGE` for tests that bypass the coverage guard."""
    return WIDE_COVERAGE
