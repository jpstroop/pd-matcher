"""Unit tests for review-filter parsing and URL round-tripping."""

from pd_groundtruth.review.filters import parse_filters


def test_parse_filters_passes_through_set_values() -> None:
    filters = parse_filters("fre", "ge90")
    assert filters.language == "fre"
    assert filters.band == "ge90"


def test_parse_filters_blanks_become_none() -> None:
    filters = parse_filters("  ", "")
    assert filters.language is None
    assert filters.band is None


def test_parse_filters_strips_whitespace() -> None:
    filters = parse_filters("  eng ", " below ")
    assert filters.language == "eng"
    assert filters.band == "below"


def test_query_string_empty_when_no_filters() -> None:
    assert parse_filters(None, None).query_string() == ""


def test_query_string_includes_only_set_keys() -> None:
    assert parse_filters("fre", None).query_string() == "language=fre"
    assert parse_filters(None, "ge90").query_string() == "band=ge90"
    assert parse_filters("fre", "ge90").query_string() == "language=fre&band=ge90"
