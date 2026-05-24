"""Tests for :mod:`pd_matcher.copyright.predicates`."""

from pd_matcher.copyright.coverage import Coverage
from pd_matcher.copyright.predicates import country_delayed_uraa
from pd_matcher.copyright.predicates import country_is_foreign
from pd_matcher.copyright.predicates import country_is_us
from pd_matcher.copyright.predicates import country_no_treaty
from pd_matcher.copyright.predicates import in_pd_by_age
from pd_matcher.copyright.predicates import match_confidence_at_least
from pd_matcher.copyright.predicates import pub_year_in_reg_coverage
from pd_matcher.copyright.predicates import pub_year_in_ren_coverage
from pd_matcher.copyright.predicates import published_before
from pd_matcher.copyright.predicates import published_between
from pd_matcher.copyright.predicates import published_on_or_after
from pd_matcher.copyright.predicates import was_registered
from pd_matcher.copyright.predicates import was_renewed
from tests.unit.copyright.conftest import AS_OF_YEAR
from tests.unit.copyright.conftest import make_facts


def test_in_pd_by_age_handles_missing_year() -> None:
    """Absent ``pub_year`` cannot be in PD by age."""
    assert in_pd_by_age(make_facts(pub_year=None)) is False


def test_in_pd_by_age_boundaries_around_moving_wall() -> None:
    """Boundary years around ``as_of_year - 95`` behave correctly."""
    cutoff = AS_OF_YEAR - 95
    assert in_pd_by_age(make_facts(pub_year=cutoff - 1)) is True
    assert in_pd_by_age(make_facts(pub_year=cutoff)) is False
    assert in_pd_by_age(make_facts(pub_year=cutoff + 1)) is False


def test_in_pd_by_age_advances_with_as_of_year() -> None:
    """A later ``as_of_year`` shifts the wall forward."""
    facts = make_facts(pub_year=1931, as_of_year=2027)
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


_NARROW_COVERAGE: Coverage = Coverage(
    reg_min_year=1931,
    reg_max_year=1977,
    ren_min_year=1959,
    ren_max_year=2005,
)


def test_pub_year_in_reg_coverage_handles_missing_year() -> None:
    """A missing ``pub_year`` is treated as out of coverage."""
    assert pub_year_in_reg_coverage(make_facts(pub_year=None), _NARROW_COVERAGE) is False


def test_pub_year_in_reg_coverage_boundaries() -> None:
    """Boundaries are inclusive."""
    assert pub_year_in_reg_coverage(make_facts(pub_year=1931), _NARROW_COVERAGE) is True
    assert pub_year_in_reg_coverage(make_facts(pub_year=1977), _NARROW_COVERAGE) is True
    assert pub_year_in_reg_coverage(make_facts(pub_year=1930), _NARROW_COVERAGE) is False
    assert pub_year_in_reg_coverage(make_facts(pub_year=1978), _NARROW_COVERAGE) is False


def test_pub_year_in_ren_coverage_handles_missing_year() -> None:
    """A missing ``pub_year`` is treated as out of coverage."""
    assert pub_year_in_ren_coverage(make_facts(pub_year=None), _NARROW_COVERAGE) is False


def test_pub_year_in_ren_coverage_requires_both_27_and_28() -> None:
    """``ren_coverage`` requires both ``pub_year + 27`` and ``+ 28`` in range.

    The 1909 Act §24 renewal window straddled the 28th calendar year
    after registration: a registration in early year ``y`` renews in
    ``y + 27``; a late-year registration renews in ``y + 28``. Both
    candidate years must be inside coverage for the cohort to be fully
    observable.
    """
    assert pub_year_in_ren_coverage(make_facts(pub_year=1932), _NARROW_COVERAGE) is True
    assert pub_year_in_ren_coverage(make_facts(pub_year=1977), _NARROW_COVERAGE) is True
    assert pub_year_in_ren_coverage(make_facts(pub_year=1931), _NARROW_COVERAGE) is False
    assert pub_year_in_ren_coverage(make_facts(pub_year=1978), _NARROW_COVERAGE) is False


def test_pub_year_in_ren_coverage_requires_both_at_upper_boundary() -> None:
    """``pub_year + 28`` must fit, not just ``pub_year + 27``."""
    boundary = Coverage(reg_min_year=1923, reg_max_year=1977, ren_min_year=1950, ren_max_year=1990)
    assert pub_year_in_ren_coverage(make_facts(pub_year=1963), boundary) is False
    assert pub_year_in_ren_coverage(make_facts(pub_year=1962), boundary) is True


def test_pub_year_in_ren_coverage_accepts_when_both_in_range() -> None:
    """A widened upper bound lets ``pub_year + 28`` slip inside, satisfying both."""
    widened = Coverage(reg_min_year=1923, reg_max_year=1977, ren_min_year=1950, ren_max_year=1991)
    assert pub_year_in_ren_coverage(make_facts(pub_year=1963), widened) is True
