"""Tests for :mod:`pd_matcher.normalize.numbers`."""

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import text

from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.numbers import ordinal_word_to_int
from pd_matcher.normalize.numbers import roman_to_arabic
from pd_matcher.normalize.numbers import word_to_int


def test_roman_to_arabic_handles_standard_cases() -> None:
    assert roman_to_arabic("I") == 1
    assert roman_to_arabic("iv") == 4
    assert roman_to_arabic("XIV") == 14
    assert roman_to_arabic("MCMXLIV") == 1944


def test_roman_to_arabic_rejects_empty_and_invalid() -> None:
    assert roman_to_arabic("") is None
    assert roman_to_arabic("abc") is None
    assert roman_to_arabic("xq") is None


def test_roman_to_arabic_returns_none_for_case_expanding_unicode() -> None:
    """The Turkish dotted capital lowercases into a combining mark, not a numeral."""
    assert roman_to_arabic("İ") is None


def test_roman_to_arabic_returns_none_for_multichar_case_expanding_token() -> None:
    """A token whose lowering expands a char must reject, never raise KeyError."""
    assert roman_to_arabic("İV") is None


def test_roman_to_arabic_still_parses_normal_numerals() -> None:
    assert roman_to_arabic("XIV") == 14
    assert roman_to_arabic("mcm") == 1900
    assert roman_to_arabic("MiX") == 1009


def test_normalize_numbers_does_not_crash_on_case_expanding_token() -> None:
    """The composite path (the production caller) tolerates the crashy token."""
    assert normalize_numbers("İ widgets", "eng") == "İ widgets"


@given(text(max_size=8))
def test_roman_to_arabic_never_raises(value: str) -> None:
    """Across arbitrary Unicode input the parser returns ``int | None``, never raises."""
    result = roman_to_arabic(value)
    assert result is None or isinstance(result, int)


def _int_to_roman(value: int) -> str:
    table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out: list[str] = []
    remaining = value
    for amount, symbol in table:
        while remaining >= amount:
            out.append(symbol)
            remaining -= amount
    return "".join(out)


@given(integers(min_value=1, max_value=3999))
def test_roman_to_arabic_round_trips_canonical_form(value: int) -> None:
    assert roman_to_arabic(_int_to_roman(value)) == value


def test_word_to_int_supported_languages() -> None:
    assert word_to_int("Three", "eng") == 3
    assert word_to_int("trois", "fre") == 3
    assert word_to_int("drei", "ger") == 3
    assert word_to_int("tres", "spa") == 3
    assert word_to_int("tre", "ita") == 3


def test_word_to_int_falls_back_to_english_for_unknown_language() -> None:
    assert word_to_int("seven", "lat") == 7


def test_word_to_int_returns_none_for_unknown_word() -> None:
    assert word_to_int("flarp", "eng") is None
    assert word_to_int("", "eng") is None


def test_ordinal_word_to_int_supported_languages() -> None:
    assert ordinal_word_to_int("First", "eng") == 1
    assert ordinal_word_to_int("premier", "fre") == 1
    assert ordinal_word_to_int("erste", "ger") == 1
    assert ordinal_word_to_int("primero", "spa") == 1
    assert ordinal_word_to_int("primo", "ita") == 1


def test_ordinal_word_to_int_returns_none_for_unknown_word() -> None:
    assert ordinal_word_to_int("zeroth", "eng") is None
    assert ordinal_word_to_int("", "eng") is None


def test_ordinal_word_to_int_falls_back_to_english() -> None:
    assert ordinal_word_to_int("second", "lat") == 2


def test_normalize_numbers_expands_abbreviations_and_words() -> None:
    out = normalize_numbers("Vol. III, ed. first", "eng")
    assert out == "volume 3 edition 1"


def test_normalize_numbers_passes_unknown_tokens_through() -> None:
    out = normalize_numbers("a study of widgets", "eng")
    assert out == "a study of widgets"


def test_normalize_numbers_converts_number_words_in_situ() -> None:
    out = normalize_numbers("three little widgets", "eng")
    assert out == "3 little widgets"


def test_normalize_numbers_handles_french_ordinals() -> None:
    out = normalize_numbers("premier livre", "fre")
    assert out == "1 livre"


def test_normalize_numbers_collapses_digit_ordinals_to_cardinal() -> None:
    """Digit ordinals match the Roman/word forms ('II' and 'second' -> '2')."""
    assert normalize_numbers("2nd", "eng") == "2"
    assert normalize_numbers("1st", "eng") == "1"
    assert normalize_numbers("3rd", "eng") == "3"
    assert normalize_numbers("21st", "eng") == "21"
    assert normalize_numbers("13th", "eng") == "13"


def test_normalize_numbers_digit_ordinal_matches_roman_in_title() -> None:
    """'... Gheyn II' and '... Gheyn, 2nd' normalize to the same digit (pair 367)."""
    marc = normalize_numbers("Jacob de Gheyn II", "eng")
    cce = normalize_numbers("Jacob DeGheyn, 2nd", "eng")
    assert marc.endswith("2")
    assert cce.endswith("2")


def test_normalize_numbers_empty_string() -> None:
    assert normalize_numbers("", "eng") == ""


def test_normalize_numbers_handles_multiple_abbreviations() -> None:
    out = normalize_numbers("no. 3 pt. ii bk. iv", "eng")
    assert out == "number 3 part 2 book 4"


def test_normalize_numbers_treats_pure_punctuation_token_as_passthrough() -> None:
    out = normalize_numbers("3 , ii", "eng")
    assert out == "3 , 2"


def test_normalize_numbers_expands_corporate_suffixes() -> None:
    out = normalize_numbers("Carrick & Evans, inc. and Sons & Co. and Dennis Corp.", "eng")
    assert "incorporated" in out
    assert "company" in out
    assert "corporation" in out


def test_normalize_numbers_expands_publishing_abbreviations() -> None:
    assert normalize_numbers("State Art Pub.", "eng") == "State Art publishing"
    assert normalize_numbers("Hebrew Publ.", "eng") == "Hebrew publishing"
    assert normalize_numbers("Tiny Pubs.", "eng") == "Tiny publishing"


def test_normalize_numbers_expands_society_and_association_abbreviations() -> None:
    assert normalize_numbers("American Insurance Assn.", "eng") == (
        "American Insurance association"
    )
    assert normalize_numbers("American Insurance Assoc.", "eng") == (
        "American Insurance association"
    )
    assert normalize_numbers("Royal Soc.", "eng") == "Royal society"


def test_normalize_numbers_expands_brothers_and_limited() -> None:
    assert normalize_numbers("Smith Bros.", "eng") == "Smith brothers"
    assert normalize_numbers("Acme Ltd.", "eng") == "Acme limited"


def test_normalize_numbers_expands_institutional_abbreviations() -> None:
    assert normalize_numbers("Univ. of California Press", "eng") == (
        "university of California Press"
    )
    assert normalize_numbers("Regents of the Univ. of Calif.", "eng") == (
        "Regents of the university of california"
    )


def test_normalize_numbers_leaves_unabbreviated_institutional_words_unchanged() -> None:
    assert normalize_numbers("university", "eng") == "university"
    assert normalize_numbers("university press", "eng") == "university press"
    assert normalize_numbers("california", "eng") == "california"
