"""Canonicalise CCE copyright registration numbers for the renewal join.

A renewal row records the registration it renews as ``oreg``; a registration
record carries the same identifier as ``regnum``. The two transcriptions
disagree on surface format far more often than on substance: ``A 963122`` vs
``A963122`` (interior space), ``AI-9217`` vs ``AI9217`` (hyphen), ``A-Foreign
32851`` vs ``AF32851`` (verbose class phrase), ``A ad int. 8956`` vs ``AI8956``
(verbose interim phrase). Concatenating the raw value into a join key lets all
of these silently fail to join, dropping otherwise-correct renewal links.

:func:`normalize_regnum` collapses the documented format variance to a single
canonical alphanumeric token so both sides of the join land on the same key.
The transform is deterministic and applied identically to ``regnum`` and
``oreg``, so any variant that maps to the canonical token recovers the join.
Steps, in order:

1. Uppercase and strip surrounding whitespace.
2. Collapse the verbose foreign/interim class phrases the registration guide
   and renewal README enumerate (``A--Foreign 32851`` / ``A for. 48359`` ->
   ``AF``; ``A ad int. 8956`` / ``A int. 241`` -> ``AI``) using leading,
   token-anchored matches so a serial like ``A INTERNATIONAL`` is left alone.
3. Drop every remaining non-alphanumeric byte (interior spaces, hyphens,
   periods, em/en dashes, commas).

Letter ``O`` and digit ``0`` are intentionally preserved as distinct: the
source numbering scheme treats them as different symbols and conflating them
would merge unrelated registrations.
"""

from re import Pattern
from re import compile as re_compile

_NON_ALNUM: Pattern[str] = re_compile(r"[^A-Z0-9]")

_SEP = "[\\s\\u2014\\u2013\\-]*"

_CLASS_PREFIX_EXPANSIONS: tuple[tuple[Pattern[str], str], ...] = (
    (re_compile(f"^A{_SEP}(?:FOREIGN|FOR)\\.?(?=\\s|$)"), "AF"),
    (re_compile(f"^A{_SEP}(?:AD{_SEP}INT|INT)\\.?(?=\\s|$)"), "AI"),
)


def normalize_regnum(raw: str) -> str:
    """Collapse documented registration-number format variance to a canon.

    Args:
        raw: The registration number as transcribed (``regnum`` or ``oreg``).

    Returns:
        The canonical alphanumeric registration-number token, possibly empty
        when ``raw`` carried no alphanumeric characters.
    """
    upper = raw.upper().strip()
    for pattern, replacement in _CLASS_PREFIX_EXPANSIONS:
        upper = pattern.sub(replacement, upper)
    return _NON_ALNUM.sub("", upper)


__all__ = [
    "normalize_regnum",
]
