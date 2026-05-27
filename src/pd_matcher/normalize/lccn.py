"""LCCN canonicalisation following the LoC namespace algorithm.

The Library of Congress Control Number is the same identifier that MARC
field 010$a and the NYPL CCE ``<lccn>`` element carry, but the two sides
encode it differently in practice — MARC tends to ship the 8-digit
normalized form, while the CCE transcription mirrors the human
``NN-NNNNNN`` form printed in the bound volumes. Bringing both onto the
same canonical form is the only meaningful prerequisite for equality
comparison and for displaying the value consistently in the review UI.

The algorithm (https://www.loc.gov/marc/lccn-namespace.html):

1. Remove all blanks (whitespace).
2. If a forward slash is present, drop it and everything to the right.
3. If a hyphen is present, drop it; left-pad the substring to the right
   of the (removed) hyphen with leading zeros until it is exactly six
   digits.

Inputs whose right-of-hyphen substring exceeds six digits are malformed
under the spec but are kept as-is rather than truncated: the spec says
the substring "should be 6 digits or less", and truncating would
silently merge distinct identifiers. Inputs with more than one hyphen
are also outside the spec; they are returned with whitespace removed
but otherwise unchanged. Either way the canonical form simply fails to
equal any well-formed LCCN.
"""

_SUFFIX_WIDTH: int = 6


def canonical(value: str | None) -> str | None:
    """Apply the LoC LCCN canonicalisation algorithm.

    Args:
        value: Candidate LCCN in any of the documented forms — the
            8-digit normalized form, the hyphenated ``NN-NNNNNN`` form,
            or a value carrying the optional ``/`` suffix.

    Returns:
        The canonical 8-digit (or alphabetic-prefix + 6-digit) form, or
        ``None`` for ``None`` or whitespace-only input.
    """
    if value is None:
        return None
    no_blanks = "".join(value.split())
    if not no_blanks:
        return None
    slash_index = no_blanks.find("/")
    if slash_index != -1:
        no_blanks = no_blanks[:slash_index]
        if not no_blanks:
            return None
    if no_blanks.count("-") != 1:
        return no_blanks
    hyphen_index = no_blanks.index("-")
    left = no_blanks[:hyphen_index]
    right = no_blanks[hyphen_index + 1 :]
    if len(right) < _SUFFIX_WIDTH:
        right = right.rjust(_SUFFIX_WIDTH, "0")
    return left + right


__all__ = [
    "canonical",
]
