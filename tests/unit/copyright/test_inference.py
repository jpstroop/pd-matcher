"""Tests for :mod:`pd_matcher.copyright.inference`."""

from pd_matcher.copyright.inference import foreign_in_pd_home_country_1996
from pd_matcher.copyright.inference import has_us_notice
from pd_matcher.copyright.inference import is_us_government_work
from tests.unit.copyright.conftest import make_facts


def test_has_us_notice_fires_when_registered() -> None:
    """Registration implies notice and surfaces an assumption."""
    value, assumption = has_us_notice(make_facts(was_registered=True))
    assert value is True
    assert assumption is not None
    assert "registration" in assumption.lower()


def test_has_us_notice_silent_when_not_registered() -> None:
    """No registration -> no notice claim, no assumption."""
    value, assumption = has_us_notice(make_facts(was_registered=False))
    assert value is False
    assert assumption is None


def test_is_us_government_work_matches_known_patterns() -> None:
    """Each documented government-publisher pattern fires."""
    patterns = [
        "U.S. Government Printing Office",
        "GPO",
        "Government Printing Office",
        "U.S. Government Publishing Office",
        "Department of the Interior",
        "Bureau of Indian Affairs",
        "National Park Service",
        "Smithsonian Institution",
        "Library of Congress",
        "National Archives",
    ]
    for text in patterns:
        value, assumption = is_us_government_work(make_facts(publisher_text=text))
        assert value is True, f"pattern did not fire for {text!r}"
        assert assumption is not None
        assert "publisher matches" in assumption


def test_is_us_government_work_silent_when_no_publisher() -> None:
    """Missing publisher -> not a government work, no assumption."""
    value, assumption = is_us_government_work(make_facts(publisher_text=None))
    assert value is False
    assert assumption is None


def test_is_us_government_work_silent_on_commercial_publisher() -> None:
    """A commercial publisher should not fire."""
    facts = make_facts(publisher_text="Penguin Random House")
    value, assumption = is_us_government_work(facts)
    assert value is False
    assert assumption is None


def test_foreign_in_pd_home_country_1996_fires_only_on_pre_1923_foreign() -> None:
    """The inference requires foreign country AND pub_year < 1923."""
    fr_pre_1923 = make_facts(pub_country_code="fr", pub_year=1920)
    value, assumption = foreign_in_pd_home_country_1996(fr_pre_1923)
    assert value is True
    assert assumption is not None
    assert "foreign-PD-by-1996" in assumption


def test_foreign_in_pd_home_country_1996_silent_when_us() -> None:
    """A US-published work is not foreign."""
    us_pre_1923 = make_facts(pub_country_code="nyu", pub_year=1920)
    value, assumption = foreign_in_pd_home_country_1996(us_pre_1923)
    assert value is False
    assert assumption is None


def test_foreign_in_pd_home_country_1996_silent_when_post_1923() -> None:
    """Foreign-published works from 1923+ require URAA analysis instead."""
    fr_1925 = make_facts(pub_country_code="fr", pub_year=1925)
    value, assumption = foreign_in_pd_home_country_1996(fr_1925)
    assert value is False
    assert assumption is None


def test_foreign_in_pd_home_country_1996_silent_when_year_missing() -> None:
    """No pub_year -> cannot apply the pre-1923 cutoff."""
    fr_unknown = make_facts(pub_country_code="fr", pub_year=None)
    value, assumption = foreign_in_pd_home_country_1996(fr_unknown)
    assert value is False
    assert assumption is None
