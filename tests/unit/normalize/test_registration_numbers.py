"""Tests for :mod:`pd_matcher.normalize.registration_numbers`."""

from hypothesis import given
from hypothesis import strategies as st
from pytest import mark

from pd_matcher.normalize.registration_numbers import is_multi_regnum
from pd_matcher.normalize.registration_numbers import normalize_regnum
from pd_matcher.normalize.registration_numbers import reg_class
from pd_matcher.normalize.registration_numbers import reg_format


def test_already_canonical_passes_through() -> None:
    assert normalize_regnum("A111111") == "A111111"


def test_strips_interior_space() -> None:
    assert normalize_regnum("A 963122") == "A963122"


def test_lowercases_to_uppercase() -> None:
    assert normalize_regnum("a963122") == "A963122"


def test_strips_surrounding_whitespace() -> None:
    assert normalize_regnum("  A963122  ") == "A963122"


def test_drops_hyphen() -> None:
    assert normalize_regnum("AI-9217") == "AI9217"


def test_drops_hyphen_inside_lettered_class() -> None:
    assert normalize_regnum("B5-73742") == "B573742"


def test_drops_period_and_comma() -> None:
    assert normalize_regnum("A.963,122") == "A963122"


def test_collapses_foreign_em_dash_phrase() -> None:
    assert normalize_regnum("A—Foreign 32851") == "AF32851"


def test_collapses_foreign_double_hyphen_phrase() -> None:
    assert normalize_regnum("A--Foreign 32851") == "AF32851"


def test_collapses_foreign_abbreviation_phrase() -> None:
    assert normalize_regnum("A for. 48359") == "AF48359"


def test_collapses_interim_ad_int_phrase() -> None:
    assert normalize_regnum("A ad int. 8956") == "AI8956"


def test_collapses_interim_abbreviation_phrase() -> None:
    assert normalize_regnum("A int. 241") == "AI241"


def test_leaves_international_serial_alone() -> None:
    assert normalize_regnum("A INTERNATIONAL") == "AINTERNATIONAL"


def test_preserves_letter_o_and_digit_zero_as_distinct() -> None:
    assert normalize_regnum("AO0123") == "AO0123"


@mark.parametrize(
    ("raw", "expected"),
    [
        ("AIO-4671", "AI4671"),
        ("AI0-4671", "AI4671"),
        ("AIO4671", "AI4671"),
        ("AI04671", "AI4671"),
        ("AF0-76081", "AF76081"),
        ("AFO-76081", "AF76081"),
        ("AFO76081", "AF76081"),
        ("AF076081", "AF76081"),
    ],
)
def test_folds_interim_foreign_class_token_variant(raw: str, expected: str) -> None:
    assert normalize_regnum(raw) == expected


@mark.parametrize(
    ("raw", "expected"),
    [
        ("A193774", "A193774"),
        ("A0193774", "A0193774"),
    ],
)
def test_class_token_fold_leaves_plain_a_serial_zeros_untouched(raw: str, expected: str) -> None:
    assert normalize_regnum(raw) == expected


def test_class_token_fold_requires_trailing_digits() -> None:
    assert normalize_regnum("AIO") == "AIO"
    assert normalize_regnum("AFO") == "AFO"


def test_class_token_fold_variants_canonicalise_to_same_token() -> None:
    assert normalize_regnum("AIO-4671") == normalize_regnum("AI-4671")
    assert normalize_regnum("AF0-76081") == normalize_regnum("AF76081")


def test_returns_empty_when_no_alphanumerics() -> None:
    assert normalize_regnum(" -.— ") == ""


def test_variant_pair_canonicalises_to_same_token() -> None:
    assert normalize_regnum("A 963122") == normalize_regnum("A963122")
    assert normalize_regnum("AI-9217") == normalize_regnum("AI9217")
    assert normalize_regnum("A—Foreign 32851") == normalize_regnum("AF32851")
    assert normalize_regnum("A ad int. 8956") == normalize_regnum("AI8956")


@mark.parametrize(
    ("raw", "expected"),
    [
        ("A160078", "A"),
        ("AA12345", "AA"),
        ("AF32851", "AF"),
        ("AI9217", "AI"),
        ("AFO123", "AF"),
        ("AIO456", "AI"),
        ("BB21524", "BB"),
        ("B573742", "B"),
        ("DP123", "DP"),
        ("TX7654321", "TX"),
        ("F12345", "F"),
        ("UCCWORK123", "UCCWORK"),
        ("A--Foreign 32851", "AF"),
        ("A ad int. 8956", "AI"),
        ("a163122", "A"),
    ],
)
def test_reg_class_reads_leading_alpha_class(raw: str, expected: str) -> None:
    assert reg_class(raw) == expected


@mark.parametrize("raw", ["963122", " -.— ", "", "0123"])
def test_reg_class_returns_sentinel_for_unparseable(raw: str) -> None:
    assert reg_class(raw) == ""


def test_reg_class_returns_sentinel_for_none() -> None:
    assert reg_class(None) == ""


@mark.parametrize(
    ("raw", "expected"),
    [
        ("A123", "Book"),
        ("AA12345", "Book (pamphlet)"),
        ("AF32851", "Book (foreign)"),
        ("AI9217", "Book (ad interim)"),
        ("BB99", "Periodical contribution"),
        ("B573742", "Periodical"),
        ("DP123", "Drama"),
        ("E4567", "Music"),
        ("F12345", "Map"),
        ("TX7654321", "Nondramatic literary (post-1978)"),
    ],
)
def test_reg_format_maps_class_to_label(raw: str, expected: str) -> None:
    assert reg_format(raw) == expected


@mark.parametrize("raw", ["963122", "UCCWORK123", "", " -.— "])
def test_reg_format_unknown_for_unmapped_or_unparseable(raw: str) -> None:
    assert reg_format(raw) == "Unknown"


def test_reg_format_unknown_for_none() -> None:
    assert reg_format(None) == "Unknown"


@given(value=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=40))
def test_is_idempotent(value: str) -> None:
    once = normalize_regnum(value)
    assert normalize_regnum(once) == once


@mark.parametrize(
    "raw",
    [
        "A692774 A692775",
        "a692774 a692775",
        "A160078 A160079 A160080",
        "692774 692775",
        "A692774  A692775",
    ],
)
def test_is_multi_regnum_true_for_space_separated_number_lists(raw: str) -> None:
    assert is_multi_regnum(raw) is True


@mark.parametrize(
    "raw",
    [
        "A692774",
        "A 963122",
        "A ad int. 8956",
        "A int. 241",
        "A INTERNATIONAL",
        "AI-9217",
        "",
        "   ",
        "A692774 ",
    ],
)
def test_is_multi_regnum_false_for_singles_and_verbose_phrases(raw: str) -> None:
    assert is_multi_regnum(raw) is False
