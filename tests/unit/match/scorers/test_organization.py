"""Tests for :mod:`pd_matcher.match.scorers.organization`."""

from pytest import mark

from pd_matcher.match.scorers.organization import looks_like_organization

_ORGANIZATIONS = (
    "Judson Press",
    "A & A Mfg. Co., Inc.",
    "Davis Publishing Company, Inc.",
    "Prentice-Hall, Inc.",
    "American Bar Assn.",
    "Child Welfare League of America, Inc.",
    "Order of St. Benedict, inc.",
    "Cornell University",
    "Knopf and Sons",
    "Editions Denoel",
    "University Microfilms Library Services",
    "Random House Publishers",
    "The Macmillan Company",
    "Oxford University Press",
    "National Geographic Society",
    "United States Government Printing Office",
    "Doubleday, Doran & Co.",
)

_PERSONS = (
    "BANNISTER, FRANK THEODORE, JR.",
    "Azevedo, Aluizio",
    "Gerald S. Snyder",
    "Robert Dayton",
    "Howland, Arthur Hoag",
    "Maxwell Geismar",
    "Wickham Skinner",
    "Britcher, Phyllis I.",
    "Knopf, Alfred A.",
    "Coppel, Alfred",
    "Wolf, Leonard",
    "Smith, John",
    "Jane Albuquerque",
)


@mark.parametrize("name", _ORGANIZATIONS)
def test_organization_markers_detected(name: str) -> None:
    """Names carrying a corporate / institutional marker are organizations."""
    assert looks_like_organization(name) is True


@mark.parametrize("name", _PERSONS)
def test_personal_names_not_organizations(name: str) -> None:
    """Inverted and natural personal names carry no marker."""
    assert looks_like_organization(name) is False


def test_none_is_not_an_organization() -> None:
    assert looks_like_organization(None) is False


def test_empty_string_is_not_an_organization() -> None:
    assert looks_like_organization("") is False


def test_punctuation_only_is_not_an_organization() -> None:
    assert looks_like_organization(",.-") is False
