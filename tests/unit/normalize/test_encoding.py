"""Tests for :mod:`pd_matcher.normalize.encoding`."""

from pytest import mark

from pd_matcher.normalize.encoding import CleanedText
from pd_matcher.normalize.encoding import clean_text


def test_clean_text_passes_pure_ascii_unchanged() -> None:
    result = clean_text("Hello, world!")
    assert result == CleanedText(text="Hello, world!", mojibake_fixed=False)


def test_clean_text_repairs_classic_mojibake() -> None:
    result = clean_text("cafÃ©")
    assert result.text == "café"
    assert result.mojibake_fixed is True


def test_clean_text_repairs_copyright_mojibake() -> None:
    result = clean_text("Â© 2020 Acme")
    assert result.text == "© 2020 Acme"
    assert result.mojibake_fixed is True


def test_clean_text_strips_inline_bom() -> None:
    bom = "﻿"
    result = clean_text(f"{bom}hello")
    assert result.text == "hello"
    assert result.mojibake_fixed is True


def test_clean_text_strips_right_to_left_mark() -> None:
    rlm = "‏"
    result = clean_text(f"he{rlm}llo")
    assert result.text == "hello"
    assert result.mojibake_fixed is True


@mark.parametrize(
    "codepoint",
    [
        "‎",
        "‏",
        "‪",
        "‫",
        "‬",
        "‭",
        "‮",
    ],
)
def test_clean_text_strips_every_bidi_formatting_mark(codepoint: str) -> None:
    result = clean_text(f"foo{codepoint}bar")
    assert result.text == "foobar"
    assert result.mojibake_fixed is True


def test_clean_text_empty_string_short_circuits() -> None:
    assert clean_text("") == CleanedText(text="", mojibake_fixed=False)


@mark.parametrize(
    "sample",
    [
        "plain ascii",
        "café",
        "cafÃ©",
        "Â© 2020 Acme",
        "﻿hello",
        "he‏llo",
        "Le petit livre",
    ],
)
def test_clean_text_is_idempotent(sample: str) -> None:
    once = clean_text(sample).text
    twice = clean_text(once).text
    assert once == twice
