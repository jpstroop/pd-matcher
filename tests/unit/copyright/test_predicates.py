"""Tests for :mod:`pd_matcher.copyright.predicates`."""

from datetime import date

from pd_matcher.copyright.predicates import country_delayed_uraa
from pd_matcher.copyright.predicates import country_is_foreign
from pd_matcher.copyright.predicates import country_is_us
from pd_matcher.copyright.predicates import country_no_treaty
from pd_matcher.copyright.predicates import in_pd_by_age
from pd_matcher.copyright.predicates import match_confidence_at_least
from pd_matcher.copyright.predicates import published_before
from pd_matcher.copyright.predicates import published_between
from pd_matcher.copyright.predicates import published_on_or_after
from pd_matcher.copyright.predicates import was_registered
from pd_matcher.copyright.predicates import was_renewed
from tests.unit.copyright.conftest import TODAY
from tests.unit.copyright.conftest import make_facts


def test_in_pd_by_age_handles_missing_year() -> None:
    """Absent ``pub_year`` cannot be in PD by age."""
    assert in_pd_by_age(make_facts(pub_year=None)) is False


def test_in_pd_by_age_boundaries_around_moving_wall() -> None:
    """Boundary years around ``today.year - 95`` behave correctly."""
    cutoff = TODAY.year - 95
    assert in_pd_by_age(make_facts(pub_year=cutoff - 1)) is True
    assert in_pd_by_age(make_facts(pub_year=cutoff)) is False
    assert in_pd_by_age(make_facts(pub_year=cutoff + 1)) is False


def test_in_pd_by_age_advances_with_today() -> None:
    """A later ``today`` shifts the wall forward."""
    facts = make_facts(pub_year=1931, today=date(2027, 1, 1))
    assert in_pd_by_age(facts) is True


def test_published_between_inclusive_both_ends() -> None:
    """``published_between`` is inclusive at both endpoints."""
    assert published_between(make_facts(pub_year=1931), 1931, 1977) is True
    assert published_between(make_facts(pub_year=1977), 1931, 1977) is True
    assert published_between(make_facts(pub_year=1930), 1931, 1977) is False
    assert published_between(make_facts(pub_year=1978), 1931, 1977) is False
    assert published_between(make_facts(pub_year=None), 1931, 1977) is False


def test_published_before_and_on_or_after() -> None:
    """The strict / inclusive year comparators handle missing year."""
    assert published_before(make_facts(pub_year=1900), 1923) is True
    assert published_before(make_facts(pub_year=1923), 1923) is False
    assert published_before(make_facts(pub_year=None), 1923) is False
    assert published_on_or_after(make_facts(pub_year=1989), 1989) is True
    assert published_on_or_after(make_facts(pub_year=1988), 1989) is False
    assert published_on_or_after(make_facts(pub_year=None), 1989) is False


def test_country_predicates_resolve_known_codes() -> None:
    """The country predicates recognize US, foreign, no-treaty, delayed-URAA codes."""
    assert country_is_us(make_facts(pub_country_code="nyu")) is True
    assert country_is_us(make_facts(pub_country_code="fr")) is False
    assert country_is_us(make_facts(pub_country_code=None)) is False
    assert country_is_foreign(make_facts(pub_country_code="fr")) is True
    assert country_is_foreign(make_facts(pub_country_code="cau")) is False
    assert country_is_foreign(make_facts(pub_country_code=None)) is False
    assert country_no_treaty(make_facts(pub_country_code="er")) is True
    assert country_no_treaty(make_facts(pub_country_code="fr")) is False
    assert country_no_treaty(make_facts(pub_country_code=None)) is False
    assert country_delayed_uraa(make_facts(pub_country_code="af")) is True
    assert country_delayed_uraa(make_facts(pub_country_code="fr")) is False
    assert country_delayed_uraa(make_facts(pub_country_code=None)) is False


def test_country_predicates_are_case_insensitive() -> None:
    """MARC codes occasionally arrive uppercased."""
    assert country_is_us(make_facts(pub_country_code="NYU")) is True


def test_was_registered_and_was_renewed() -> None:
    """Pass-through predicates expose their underlying Facts attributes."""
    assert was_registered(make_facts(was_registered=True)) is True
    assert was_registered(make_facts(was_registered=False)) is False
    assert was_renewed(make_facts(was_renewed=True)) is True
    assert was_renewed(make_facts(was_renewed=False)) is False


def test_match_confidence_at_least_threshold() -> None:
    """``match_confidence_at_least`` is a `>=` comparison."""
    facts = make_facts(match_confidence=0.95)
    assert match_confidence_at_least(facts, 0.95) is True
    assert match_confidence_at_least(facts, 0.90) is True
    assert match_confidence_at_least(facts, 0.96) is False
