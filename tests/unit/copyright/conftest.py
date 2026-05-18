"""Shared fixtures for the copyright rule-engine test suite."""

from datetime import date

from pytest import fixture

from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.copyright import default_ruleset
from pd_matcher.copyright.facts import Facts

TODAY: date = date(2026, 5, 18)


def make_facts(
    *,
    pub_year: int | None = None,
    pub_country_code: str | None = None,
    language_code: str | None = None,
    publisher_text: str | None = None,
    was_registered: bool = False,
    was_renewed: bool = False,
    match_confidence: float = 0.0,
    today: date = TODAY,
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
        today=today,
    )


@fixture
def today() -> date:
    """Pinned reference date used by every test in the suite."""
    return TODAY


@fixture
def ruleset() -> CopyrightRuleSet:
    """The shipped Cornell ruleset."""
    return default_ruleset()
