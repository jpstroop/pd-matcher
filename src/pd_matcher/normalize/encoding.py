"""Subfield-level encoding hygiene applied at parse time.

Raw bibliographic data from MARCXML and the NYPL transcriptions of the CCE
arrives with a long tail of encoding accidents: mojibake from earlier
double-encoding (``Ã©`` for ``é``, ``Â©`` for ``©``), stray byte-order marks
embedded mid-string (``U+FEFF``), and bidirectional control codepoints
(``U+200E``, ``U+200F``, ``U+202A`` through ``U+202E``) that pollute tokens
without rendering visibly. Left untouched, every one of these surfaces in
the downstream scorers as a phantom token mismatch.

We route every finalized subfield value through :func:`clean_text` at parse
time. The heavy lifting is delegated to :func:`ftfy.fix_text`, which
performs mojibake repair, NFC normalization, lossy-sequence replacement,
and stray-BOM removal in one pass. A tiny ``str.translate`` postpass
additionally drops the bidirectional formatting marks ftfy preserves by
design (LRM, RLM, the LRE/RLE/PDF/LRO/RLO embedding/override family).
Those marks are semantically meaningful inside bidi-aware renderers but
appear in our data only as transcription artifacts; left in tokens they
would split words for downstream scorers.

The returned :class:`CleanedText` carries a boolean indicating whether
the input was altered. Parsers use that flag to bump a per-corpus counter
so dataset quality can be surfaced in run statistics without re-walking
the source files.
"""

from ftfy import fix_text
from msgspec import Struct

_BIDI_MARKS: tuple[str, ...] = (
    "‎",
    "‏",
    "‪",
    "‫",
    "‬",
    "‭",
    "‮",
)
_BIDI_TRANSLATION: dict[int, None] = dict.fromkeys(ord(c) for c in _BIDI_MARKS)


class CleanedText(Struct, frozen=True, forbid_unknown_fields=True):
    """Result of a single :func:`clean_text` invocation."""

    text: str
    mojibake_fixed: bool


def clean_text(value: str) -> CleanedText:
    """Repair mojibake and strip stray bidi/BOM characters from ``value``.

    Args:
        value: Raw subfield text. The empty string is treated as a no-op
            and does not invoke ftfy.

    Returns:
        A :class:`CleanedText` whose ``text`` is the repaired string and
        whose ``mojibake_fixed`` flag is ``True`` whenever the input was
        altered by either ftfy (mojibake repair, BOM removal, NFC
        normalization, etc.) or the bidi-mark postpass.
    """
    if not value:
        return CleanedText(text="", mojibake_fixed=False)
    fixed = fix_text(value).translate(_BIDI_TRANSLATION)
    return CleanedText(text=fixed, mojibake_fixed=fixed != value)


__all__ = [
    "CleanedText",
    "clean_text",
]
