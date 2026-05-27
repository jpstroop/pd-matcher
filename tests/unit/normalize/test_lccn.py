"""Tests for :mod:`pd_matcher.normalize.lccn`."""

from hypothesis import given
from hypothesis import strategies as st

from pd_matcher.normalize.lccn import canonical


def test_canonical_returns_none_for_none() -> None:
    assert canonical(None) is None


def test_canonical_returns_none_for_empty_string() -> None:
    assert canonical("") is None


def test_canonical_returns_none_for_whitespace_only() -> None:
    assert canonical("   ") is None


def test_canonical_passes_through_already_normalised_8_digit_form() -> None:
    assert canonical("37013688") == "37013688"


def test_canonical_normalises_hyphenated_cce_form() -> None:
    assert canonical("37-13688") == "37013688"


def test_canonical_strips_surrounding_whitespace() -> None:
    assert canonical(" 37-13688 ") == "37013688"


def test_canonical_drops_suffix_after_slash() -> None:
    assert canonical("75-425165/M/r842") == "75425165"


def test_canonical_returns_none_when_only_slash_suffix_present() -> None:
    assert canonical("/abc") is None


def test_canonical_handles_alphabetic_prefix_without_hyphen() -> None:
    assert canonical("n79018774") == "n79018774"


def test_canonical_pads_short_suffix_under_alphabetic_prefix() -> None:
    assert canonical("n81-021") == "n81000021"


def test_canonical_preserves_long_right_substring() -> None:
    assert canonical("12-1234567") == "121234567"


def test_canonical_passes_through_multi_hyphen_input_unchanged_after_blanks() -> None:
    assert canonical("12-34-56") == "12-34-56"


@given(value=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=40))
def test_canonical_is_idempotent(value: str) -> None:
    once = canonical(value)
    twice = canonical(once)
    assert twice == once
