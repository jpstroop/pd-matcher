"""Tests for :mod:`pd_matcher.normalize.script`."""

from pd_matcher.normalize.script import dominant_script


def test_latin_string_is_latin() -> None:
    """A plain Latin sentence reports ``LATIN``."""
    assert dominant_script("Hello world") == "LATIN"


def test_romanized_text_reads_as_latin() -> None:
    """Romanized non-Western text is still Latin by character class."""
    assert dominant_script("Bereshit bara Elohim") == "LATIN"


def test_hebrew_string_is_hebrew() -> None:
    """A Hebrew-only string reports ``HEBREW``."""
    assert dominant_script("בראשית ברא אלהים") == "HEBREW"


def test_cyrillic_string_is_cyrillic() -> None:
    """A Cyrillic string reports ``CYRILLIC``."""
    assert dominant_script("Война и мир") == "CYRILLIC"


def test_greek_string_is_greek() -> None:
    """A Greek string reports ``GREEK``."""
    assert dominant_script("Ἰλιάς") == "GREEK"


def test_arabic_string_is_arabic() -> None:
    """An Arabic string reports ``ARABIC``."""
    assert dominant_script("كتاب") == "ARABIC"


def test_cjk_string_is_cjk() -> None:
    """A CJK string reports ``CJK``."""
    assert dominant_script("中文書名") == "CJK"


def test_hiragana_string_is_hiragana() -> None:
    """A Hiragana-only string reports ``HIRAGANA``."""
    assert dominant_script("さくらんぼ") == "HIRAGANA"


def test_katakana_string_is_katakana() -> None:
    """A Katakana-only string reports ``KATAKANA``."""
    assert dominant_script("カタカナ") == "KATAKANA"


def test_hangul_string_is_hangul() -> None:
    """A Hangul string reports ``HANGUL``."""
    assert dominant_script("한국어") == "HANGUL"


def test_devanagari_string_is_devanagari() -> None:
    """A Devanagari string reports ``DEVANAGARI``."""
    assert dominant_script("नमस्ते") == "DEVANAGARI"


def test_majority_wins_in_mixed_input() -> None:
    """When two scripts coexist, the more frequent one wins."""
    assert dominant_script("Hello בראשית") == "HEBREW"
    assert dominant_script("Cold mountain בא") == "LATIN"


def test_empty_string_returns_none() -> None:
    """An empty input returns ``None``."""
    assert dominant_script("") is None


def test_symbol_only_string_returns_none() -> None:
    """A symbol-only input has no alphabetic characters; returns ``None``."""
    assert dominant_script("---!!!") is None


def test_digit_only_string_returns_none() -> None:
    """A digit-only input has no alphabetic characters; returns ``None``."""
    assert dominant_script("12345") is None


def test_whitespace_only_string_returns_none() -> None:
    """Whitespace alone yields no detectable script."""
    assert dominant_script("   ") is None


def test_single_latin_character_is_latin() -> None:
    """A single Latin character is enough to fire ``LATIN``."""
    assert dominant_script("A") == "LATIN"


def test_single_hebrew_character_is_hebrew() -> None:
    """A single Hebrew character is enough to fire ``HEBREW``."""
    assert dominant_script("ב") == "HEBREW"


def test_unnamed_alphabetic_character_is_skipped() -> None:
    """An alphabetic character with no Unicode name is ignored.

    Tangut characters in the U+17000 block are alphabetic per
    :meth:`str.isalpha` but :func:`unicodedata.name` raises
    :class:`ValueError` on them in this Python build. The function must
    not crash and must skip such characters when accumulating script
    counts.
    """
    assert dominant_script(chr(0x17000)) is None
    assert dominant_script("Hello " + chr(0x17000)) == "LATIN"


def test_untracked_script_does_not_block_tracked_winner() -> None:
    """A character whose script prefix is not tracked is silently skipped.

    The Thai letter ``ก`` is alphabetic and named (so it is not caught by
    the ``ValueError`` branch) but ``THAI`` is not in
    :data:`_SCRIPT_PREFIXES`. The inner loop must exhaust without
    recording a hit for it, leaving the Hebrew letter as the sole vote.
    """
    assert dominant_script("ก") is None
    assert dominant_script("ק" + "ก") == "HEBREW"
