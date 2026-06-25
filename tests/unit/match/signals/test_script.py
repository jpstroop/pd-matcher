"""Tests for :mod:`pd_matcher.match.signals.script`."""

from pd_matcher.match.signals.script import is_script_mismatch
from pd_matcher.match.signals.script import scripts_mismatch


def test_same_script_same_text_is_not_mismatch() -> None:
    """Identical inputs share a script and do not mismatch."""
    assert is_script_mismatch("Cold mountain", "Cold mountain") is False


def test_same_script_different_text_is_not_mismatch() -> None:
    """Same dominant script on both sides is not a mismatch even when text differs."""
    assert is_script_mismatch("Cold mountain", "Bleak harbor") is False


def test_different_script_is_mismatch() -> None:
    """Latin vs. Hebrew titles mismatch."""
    assert is_script_mismatch("Bereshit bara Elohim", "בראשית ברא אלהים") is True


def test_empty_marc_is_not_mismatch() -> None:
    """An empty MARC side has no detectable script; no mismatch fires."""
    assert is_script_mismatch("", "Cold mountain") is False


def test_empty_cce_is_not_mismatch() -> None:
    """An empty CCE side has no detectable script; no mismatch fires."""
    assert is_script_mismatch("Cold mountain", "") is False


def test_both_empty_is_not_mismatch() -> None:
    """Two empty inputs do not mismatch."""
    assert is_script_mismatch("", "") is False


def test_symbol_only_marc_is_not_mismatch() -> None:
    """MARC text without alphabetic characters has no script; no mismatch."""
    assert is_script_mismatch("---!!!", "Cold mountain") is False


def test_symbol_only_cce_is_not_mismatch() -> None:
    """CCE text without alphabetic characters has no script; no mismatch."""
    assert is_script_mismatch("Cold mountain", "12345") is False


def test_latin_vs_cyrillic_is_mismatch() -> None:
    """Latin vs. Cyrillic titles mismatch."""
    assert is_script_mismatch("Voyna i mir", "Война и мир") is True


def test_latin_vs_cjk_is_mismatch() -> None:
    """Latin vs. CJK titles mismatch."""
    assert is_script_mismatch("Dream of the red chamber", "紅樓夢") is True


def test_scripts_mismatch_differing_scripts_is_true() -> None:
    """Two distinct resolved scripts mismatch."""
    assert scripts_mismatch("LATIN", "HEBREW") is True


def test_scripts_mismatch_equal_scripts_is_false() -> None:
    """Two equal resolved scripts do not mismatch."""
    assert scripts_mismatch("LATIN", "LATIN") is False


def test_scripts_mismatch_none_marc_is_false() -> None:
    """A ``None`` MARC script never mismatches."""
    assert scripts_mismatch(None, "HEBREW") is False


def test_scripts_mismatch_none_cce_is_false() -> None:
    """A ``None`` CCE script never mismatches."""
    assert scripts_mismatch("LATIN", None) is False


def test_scripts_mismatch_both_none_is_false() -> None:
    """Two ``None`` scripts do not mismatch."""
    assert scripts_mismatch(None, None) is False
