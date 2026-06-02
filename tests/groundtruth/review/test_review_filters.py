"""Unit tests for review-filter parsing and URL round-tripping."""

from pd_groundtruth.review.filters import label_filters_active
from pd_groundtruth.review.filters import label_filters_query_string
from pd_groundtruth.review.filters import parse_filters
from pd_groundtruth.review.filters import parse_label_filters
from pd_groundtruth.review_db import SORT_ASC
from pd_groundtruth.review_db import SORT_DESC


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


def test_parse_label_filters_passes_through_set_values() -> None:
    filters = parse_label_filters("match", "eng", "acme")
    assert filters.verdict == "match"
    assert filters.language == "eng"
    assert filters.q == "acme"


def test_parse_label_filters_blanks_become_none() -> None:
    filters = parse_label_filters("  ", "", "")
    assert filters.verdict is None
    assert filters.language is None
    assert filters.q is None


def test_label_filters_active_detects_any_set_filter() -> None:
    assert not label_filters_active(parse_label_filters(None, None, None))
    assert label_filters_active(parse_label_filters("match", None, None))
    assert label_filters_active(parse_label_filters(None, "eng", None))
    assert label_filters_active(parse_label_filters(None, None, "acme"))


def test_label_filters_query_string_renders_all_set_keys() -> None:
    filters = parse_label_filters("match", "eng", "acme")
    qs = label_filters_query_string(filters)
    assert "verdict=match" in qs
    assert "language=eng" in qs
    assert "q=acme" in qs


def test_label_filters_query_string_empty_when_no_filters() -> None:
    assert label_filters_query_string(parse_label_filters(None, None, None)) == ""


def test_label_filters_query_string_drop_excludes_one_key() -> None:
    filters = parse_label_filters("match", "eng", None)
    assert label_filters_query_string(filters, drop="verdict") == "language=eng"
    assert label_filters_query_string(filters, drop="language") == "verdict=match"


def test_parse_label_filters_defaults_sort_to_desc() -> None:
    assert parse_label_filters(None, None, None).sort == SORT_DESC
    assert parse_label_filters(None, None, None, None).sort == SORT_DESC


def test_parse_label_filters_accepts_asc() -> None:
    assert parse_label_filters(None, None, None, "asc").sort == SORT_ASC


def test_parse_label_filters_accepts_explicit_desc() -> None:
    assert parse_label_filters(None, None, None, "desc").sort == SORT_DESC


def test_parse_label_filters_strips_whitespace_around_sort() -> None:
    assert parse_label_filters(None, None, None, "  asc ").sort == SORT_ASC


def test_parse_label_filters_garbage_sort_falls_back_to_default() -> None:
    assert parse_label_filters(None, None, None, "garbage").sort == SORT_DESC
    assert parse_label_filters(None, None, None, "").sort == SORT_DESC
    assert parse_label_filters(None, None, None, "   ").sort == SORT_DESC


def test_label_filters_query_string_omits_sort_when_default() -> None:
    filters = parse_label_filters(None, None, None, "desc")
    assert label_filters_query_string(filters) == ""


def test_label_filters_query_string_includes_sort_when_non_default() -> None:
    filters = parse_label_filters(None, None, None, "asc")
    assert label_filters_query_string(filters) == "sort=asc"


def test_label_filters_query_string_appends_sort_after_filters() -> None:
    filters = parse_label_filters("match", "eng", "acme", "asc")
    qs = label_filters_query_string(filters)
    assert "verdict=match" in qs
    assert "language=eng" in qs
    assert "q=acme" in qs
    assert "sort=asc" in qs


def test_label_filters_query_string_preserves_sort_when_dropping_a_filter() -> None:
    filters = parse_label_filters("match", "eng", None, "asc")
    assert "sort=asc" in label_filters_query_string(filters, drop="verdict")
    assert "sort=asc" in label_filters_query_string(filters, drop="language")


def test_label_filters_active_ignores_sort() -> None:
    assert not label_filters_active(parse_label_filters(None, None, None, "asc"))
    assert not label_filters_active(parse_label_filters(None, None, None, "desc"))


def test_parse_label_filters_accepts_categories_list() -> None:
    filters = parse_label_filters(None, None, None, None, ["translation", "ocr_confusion"])
    assert filters.categories == ("translation", "ocr_confusion")


def test_parse_label_filters_drops_unknown_category_keys() -> None:
    filters = parse_label_filters(None, None, None, None, ["translation", "bogus", "generic_title"])
    assert filters.categories == ("translation", "generic_title")


def test_parse_label_filters_empty_categories_collapse_to_empty_tuple() -> None:
    assert parse_label_filters(None, None, None, None, None).categories == ()
    assert parse_label_filters(None, None, None, None, []).categories == ()


def test_label_filters_active_detects_categories() -> None:
    assert label_filters_active(parse_label_filters(None, None, None, None, ["translation"]))


def test_label_filters_query_string_renders_each_category_as_own_pair() -> None:
    filters = parse_label_filters(None, None, None, None, ["translation", "ocr_confusion"])
    qs = label_filters_query_string(filters)
    assert qs == "categories=translation&categories=ocr_confusion"


def test_label_filters_query_string_drops_categories_when_requested() -> None:
    filters = parse_label_filters("match", None, None, None, ["translation"])
    qs = label_filters_query_string(filters, drop="categories")
    assert "categories" not in qs
    assert "verdict=match" in qs


def test_label_filters_query_string_preserves_categories_when_dropping_other_filter() -> None:
    filters = parse_label_filters("match", None, None, None, ["translation"])
    qs = label_filters_query_string(filters, drop="verdict")
    assert "categories=translation" in qs
    assert "verdict" not in qs
