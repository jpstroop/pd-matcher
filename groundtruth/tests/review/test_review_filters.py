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


def test_query_string_excludes_skip_ids() -> None:
    filters = parse_filters("fre", None, [1, 2])
    assert filters.query_string() == "language=fre"


def test_parse_filters_dedupes_skip_ids_preserving_order() -> None:
    filters = parse_filters(None, None, [3, 1, 3, 2, 1])
    assert filters.skip_ids == (3, 1, 2)


def test_parse_filters_empty_skip_ids_is_empty_tuple() -> None:
    assert parse_filters(None, None, None).skip_ids == ()
    assert parse_filters(None, None, []).skip_ids == ()


def test_next_query_string_appends_additional_skip_id() -> None:
    filters = parse_filters(None, None, [1, 2])
    assert filters.next_query_string(additional_skip_id=3) == "skip=1&skip=2&skip=3"


def test_next_query_string_does_not_duplicate_additional_id() -> None:
    filters = parse_filters(None, None, [1, 2])
    assert filters.next_query_string(additional_skip_id=2) == "skip=1&skip=2"


def test_next_query_string_threads_filters_and_skip_ids() -> None:
    filters = parse_filters("fre", "ge90", [1])
    assert filters.next_query_string(additional_skip_id=2) == (
        "language=fre&band=ge90&skip=1&skip=2"
    )


def test_next_query_string_without_additional_id_includes_existing_skips() -> None:
    filters = parse_filters(None, None, [4, 5])
    assert filters.next_query_string() == "skip=4&skip=5"
