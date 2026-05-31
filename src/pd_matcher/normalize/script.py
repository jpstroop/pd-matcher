"""Detect the dominant Unicode script in a text string.

The matcher pairs a MARC title against a CCE title even when the two
records describe the same work in incompatibly transcribed scripts —
e.g. a Hebrew CCE entry whose title is Hebrew letters versus a MARC
record carrying the romanized title. The fuzzy token-set scorer cannot
recover that pair (the character sets are disjoint), so the safer
behavior is to recognize the script mismatch and let the title scorer
emit a zero so the pair falls out of consideration on title weight
rather than coincidentally agreeing on a stray Latin token.

Detection is by majority of alphabetic Unicode characters: each char's
name is consulted via :func:`unicodedata.name` and bucketed under the
first matching script prefix. Non-alphabetic characters (digits,
punctuation, whitespace) and characters with no Unicode name are
ignored; empty inputs, symbol-only inputs and digit-only inputs all
return ``None``.
"""

from unicodedata import name as unicode_name

_SCRIPT_PREFIXES: tuple[str, ...] = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "HEBREW",
    "ARABIC",
    "CJK",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "DEVANAGARI",
)


def dominant_script(text: str) -> str | None:
    """Return the most-frequent Unicode script prefix in ``text``.

    Args:
        text: The string to inspect.

    Returns:
        The matching prefix from :data:`_SCRIPT_PREFIXES` (e.g. ``"LATIN"``,
        ``"HEBREW"``) with the most alphabetic-character hits. ``None`` when
        ``text`` is empty, has no alphabetic characters, or has no
        characters in any tracked script.
    """
    if not text:
        return None
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            ch_name = unicode_name(ch)
        except ValueError:
            continue
        for prefix in _SCRIPT_PREFIXES:
            if ch_name.startswith(prefix):
                counts[prefix] = counts.get(prefix, 0) + 1
                break
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


__all__ = [
    "dominant_script",
]
