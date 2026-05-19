"""Tests for :mod:`pd_matcher.normalize.cp1255_fallback`."""

from pd_matcher.normalize.cp1255_fallback import DecodedSubfield
from pd_matcher.normalize.cp1255_fallback import decode_subfield


def test_decode_subfield_clean_ascii_uses_utf8() -> None:
    result = decode_subfield(b"hello")
    assert result == DecodedSubfield(text="hello", encoding_used="utf-8")


def test_decode_subfield_valid_utf8_with_hebrew_stays_utf8() -> None:
    raw = "שלום".encode()
    result = decode_subfield(raw)
    assert result == DecodedSubfield(text="שלום", encoding_used="utf-8")


def test_decode_subfield_cp1255_hebrew_falls_back_to_windows_1255() -> None:
    raw = "שלום".encode("cp1255")
    result = decode_subfield(raw)
    assert result == DecodedSubfield(text="שלום", encoding_used="windows-1255")


def test_decode_subfield_cp1255_succeeds_without_hebrew_falls_through_to_replace() -> None:
    raw = b"\x80\x99"
    result = decode_subfield(raw)
    assert result.encoding_used == "utf-8-replace"
    assert "€" not in result.text
    assert "�" in result.text


def test_decode_subfield_both_strict_decoders_fail_uses_utf8_replace() -> None:
    raw = b"\xff"
    result = decode_subfield(raw)
    assert result == DecodedSubfield(text="�", encoding_used="utf-8-replace")
