"""Tests for :mod:`pd_matcher.normalize.registration_numbers`."""

from hypothesis import given
from hypothesis import strategies as st

from pd_matcher.normalize.registration_numbers import normalize_regnum


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


def test_returns_empty_when_no_alphanumerics() -> None:
    assert normalize_regnum(" -.— ") == ""


def test_variant_pair_canonicalises_to_same_token() -> None:
    assert normalize_regnum("A 963122") == normalize_regnum("A963122")
    assert normalize_regnum("AI-9217") == normalize_regnum("AI9217")
    assert normalize_regnum("A—Foreign 32851") == normalize_regnum("AF32851")
    assert normalize_regnum("A ad int. 8956") == normalize_regnum("AI8956")


@given(value=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=40))
def test_is_idempotent(value: str) -> None:
    once = normalize_regnum(value)
    assert normalize_regnum(once) == once
