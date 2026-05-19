"""Shared fixtures for the copyright rule-engine test suite."""

from pytest import fixture

from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.copyright import default_ruleset
from pd_matcher.copyright.facts import Facts

AS_OF_YEAR: int = 2026


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
