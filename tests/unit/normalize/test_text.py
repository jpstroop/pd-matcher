"""Tests for :mod:`pd_matcher.normalize.text`."""

from hypothesis import given
from hypothesis.strategies import text

from pd_matcher.normalize.text import normalize_text
from pd_matcher.normalize.text import tokenize


def test_normalize_text_lowercases_and_strips_diacritics() -> None:
    assert normalize_text("Café CRÈME") == "cafe creme"


def test_normalize_text_collapses_punctuation_and_whitespace() -> None:
    assert normalize_text("Hello,  world!!!") == "hello world"


def test_normalize_text_keeps_alphanumerics() -> None:
    assert normalize_text("Vol. 12") == "vol 12"


def test_normalize_text_empty_returns_empty() -> None:
    assert normalize_text("") == ""


def test_normalize_text_pure_punctuation_returns_empty() -> None:
    assert normalize_text("!!!---***") == ""


def test_normalize_text_folds_oe_ligature() -> None:
    assert normalize_text("œuvre") == "oeuvre"


def test_normalize_text_folds_ae_ligature() -> None:
    assert normalize_text("æsop") == "aesop"


def test_normalize_text_folds_uppercase_ligatures() -> None:
    assert normalize_text("ŒUVRE") == "oeuvre"
    assert normalize_text("ÆSOP") == "aesop"


def test_normalize_text_folds_latin_typographic_ligatures() -> None:
    assert normalize_text("ﬁnal ﬂag") == "final flag"


def test_normalize_text_diacritic_folding_unchanged() -> None:
    assert normalize_text("José café Sénèque") == "jose cafe seneque"


def test_tokenize_folds_oe_ligature() -> None:
    assert tokenize("œuvre") == ("oeuvre",)


def test_tokenize_folds_ae_ligature() -> None:
    assert tokenize("æsop") == ("aesop",)


def test_tokenize_splits_normalized_string() -> None:
    assert tokenize("Café CRÈME, fresh!") == ("cafe", "creme", "fresh")


def test_tokenize_empty_input_returns_empty_tuple() -> None:
    assert tokenize("") == ()


def test_tokenize_pure_punctuation_returns_empty_tuple() -> None:
    assert tokenize("!!!") == ()


@given(text())
def test_normalize_text_is_idempotent(value: str) -> None:
    once = normalize_text(value)
    twice = normalize_text(once)
    assert once == twice


@given(text())
def test_tokenize_never_returns_empty_tokens(value: str) -> None:
    for token in tokenize(value):
        assert token
        assert " " not in token
