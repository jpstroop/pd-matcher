"""Defensive byte-level decoder for renewal TSV cells that fail UTF-8.

The NYPL CCE renewal corpus has historically been delivered as UTF-8 and
the current submodule snapshot decodes cleanly. The supplied ingest
pipeline therefore never reaches this fallback. We carry it anyway: a
future ingest of a separately-encoded slice (Hebrew titles published in
Israel were occasionally transcribed in Windows-1255 before the corpus
was consolidated) would otherwise lose every affected row to a
``UnicodeDecodeError`` at parse time.

Decode order:

1. UTF-8 strict. The overwhelmingly common case; returned with
   ``encoding_used="utf-8"``.
2. Windows-1255 strict. Accepted *only* if the decoded string contains
   at least one character in the Hebrew block (U+0590..U+05FF). This
   guard rejects cp1255 decodings that happen to succeed but produce
   garbage (cp1255 maps every byte, so success alone is meaningless).
3. UTF-8 with ``errors="replace"``. Last-resort fallback that preserves
   the rest of the row's content and surfaces the damaged bytes as
   U+FFFD replacement characters.

The renewal parser probes the file once at open time. If the whole-file
decode succeeds, the cell-level fallback is never invoked. If it fails,
the parser switches to bytes-level reads and routes every cell through
:func:`decode_subfield`.
"""

from msgspec import Struct

_HEBREW_BLOCK_START = 0x0590
_HEBREW_BLOCK_END = 0x05FF


class DecodedSubfield(Struct, frozen=True, forbid_unknown_fields=True):
    """Result of :func:`decode_subfield` for one raw byte cell."""

    text: str
    encoding_used: str


def _contains_hebrew(value: str) -> bool:
    """Return ``True`` if ``value`` contains any Hebrew-block codepoint."""
    return any(_HEBREW_BLOCK_START <= ord(ch) <= _HEBREW_BLOCK_END for ch in value)


def decode_subfield(raw: bytes) -> DecodedSubfield:
    """Decode ``raw`` using the defensive UTF-8 / cp1255 / replace ladder.

    Args:
        raw: Bytes from a single TSV cell.

    Returns:
        A :class:`DecodedSubfield` whose ``encoding_used`` records which
        ladder rung produced the result: ``"utf-8"``, ``"windows-1255"``,
        or ``"utf-8-replace"``.
    """
    try:
        return DecodedSubfield(text=raw.decode("utf-8"), encoding_used="utf-8")
    except UnicodeDecodeError:
        pass
    try:
        candidate = raw.decode("windows-1255")
    except UnicodeDecodeError:
        candidate = None
    if candidate is not None and _contains_hebrew(candidate):
        return DecodedSubfield(text=candidate, encoding_used="windows-1255")
    return DecodedSubfield(
        text=raw.decode("utf-8", errors="replace"),
        encoding_used="utf-8-replace",
    )


__all__ = [
    "DecodedSubfield",
    "decode_subfield",
]
