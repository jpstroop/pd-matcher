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

_INTERNAL_WS: Pattern[str] = re_compile(r"\S+\s+\S+")
_RANGE_TOKEN: Pattern[str] = re_compile(r"^[A-Z]*[0-9]+$")
_CLASS_PREFIX: Pattern[str] = re_compile(r"^[A-Z]+")

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


def reg_class(raw: str | None) -> str:
    """Return the leading alpha class prefix of a normalized registration number.

    A CCE registration number opens with an alphabetic class token (``A`` book,
    ``BB`` periodical, ``DP`` dramatic composition published, ``E`` music, ``F``
    map, ``TX`` post-1978, ...) followed by its serial digits. The class is what
    determines whether NYPL ever transcribed the registration at all, so callers
    scope on it. :func:`normalize_regnum` is applied first so verbose class
    phrases (``A--Foreign`` -> ``AF``) and surface noise collapse before the
    prefix is read.

    Args:
        raw: The registration number as transcribed (``regnum`` or ``oreg``), or
            ``None`` when the record carries none.

    Returns:
        The uppercase leading alpha class token (``"A"``, ``"BB"``, ``"TX"``,
        ``"UCCWORK"``), or ``""`` when ``raw`` is ``None`` or normalizes to a
        value with no leading alphabetic character.
    """
    if raw is None:
        return ""
    match = _CLASS_PREFIX.match(normalize_regnum(raw))
    return match.group(0) if match is not None else ""


def is_multi_regnum(raw: str) -> bool:
    """Report whether ``raw`` is a space-separated multi-number regnum range.

    A subset of CCE registrations record several numbers in the single
    ``regnum`` attribute (``"A692774 A692775"``) because they register a
    multi-volume whole. :func:`normalize_regnum` strips internal whitespace,
    so it would concatenate such a value into one unmatchable token; callers
    that need per-number join keys use this predicate to fan the value out
    instead.

    The verbose foreign/interim class phrases (``"A ad int. 8956"``) and
    interior-space singles (``"A 963122"``, whose ``A`` token carries no
    digit) also have internal whitespace but are NOT ranges, so every
    whitespace token must independently look like a registration number
    (optional letter class prefix followed by at least one digit).

    Args:
        raw: The registration number as transcribed (``regnum`` or ``oreg``).

    Returns:
        ``True`` when ``raw`` has internal whitespace and every whitespace
        token matches ``^[A-Z]*[0-9]+$``; ``False`` otherwise.
    """
    if _INTERNAL_WS.search(raw) is None:
        return False
    tokens = raw.upper().split()
    return len(tokens) > 1 and all(_RANGE_TOKEN.match(token) is not None for token in tokens)


__all__ = [
    "is_multi_regnum",
    "normalize_regnum",
    "reg_class",
]
