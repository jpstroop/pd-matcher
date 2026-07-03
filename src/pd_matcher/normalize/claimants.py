"""Parse and classify the statutory claimant codes on a CCE renewal record.

A renewal names its claimants in :attr:`pd_matcher.models.NyplRenRecord.claimants`
as a ``Name|CODE`` list joined with ``||``; a minority of transcriptions use
``;`` separators or a trailing ``Name (CODE)`` parenthetical. The trailing code
is the statutory renewal-right class the Copyright Office recorded:

* ``A`` — the author renewing in person;
* ``W`` widow/widower, ``C`` child, ``E`` executor, ``NK`` next-of-kin — the
  author's *estate* renewing on their behalf;
* ``PWH`` proprietor of a work made for hire, ``PPW`` / ``PCW`` proprietor of a
  posthumous / composite work — a *proprietor* renewal.

The class governs whether the claimant is expected to share a name with the
MARC author. :func:`parse_claimants` recovers the ``(name, code)`` pairs and
:func:`claimant_renewal_label` renders the statutory renewal-right class as a
human-readable label for the review UI.
"""

from re import Pattern
from re import compile as re_compile

_SEPARATOR: Pattern[str] = re_compile(r"\|\||;")


def parse_claimants(claimants: str | None) -> tuple[tuple[str, str], ...]:
    """Return the ``(name, statutory_code)`` pairs parsed from a claimants string.

    Each ``||``- or ``;``-separated part is read as ``Name|CODE``; a trailing
    ``Name (CODE)`` parenthetical is accepted as a fallback. A part whose
    trailing token is not an all-uppercase alphabetic code yields that part as
    the name with an empty code, so no claimant is silently dropped.

    Args:
        claimants: The renewal's raw claimants transcription, or ``None``.

    Returns:
        A tuple of ``(name, code)`` pairs in source order; empty when
        ``claimants`` is ``None`` or blank.
    """
    if not claimants:
        return ()
    pairs: list[tuple[str, str]] = []
    for raw_part in _SEPARATOR.split(claimants):
        part = raw_part.strip()
        if not part:
            continue
        head, _, tail = part.rpartition("|")
        code = tail.strip()
        if head and code.isalpha() and code.isupper():
            pairs.append((head.strip(), code))
            continue
        if part.endswith(")") and "(" in part:
            name, _, bracket = part.rpartition("(")
            inner = bracket[:-1].strip()
            if inner.isalpha() and inner.isupper():
                pairs.append((name.strip(), inner))
                continue
        pairs.append((part, ""))
    return tuple(pairs)


_CLAIMANT_RENEWAL_LABELS: dict[str, str] = {
    "A": "author",
    "W": "widow/widower (estate)",
    "C": "child (estate)",
    "E": "executor (estate)",
    "NK": "next of kin (estate)",
    "PWH": "proprietor (work for hire)",
    "PPW": "proprietor (posthumous work)",
    "PCW": "proprietor (composite work)",
}

_CLAIMANT_RENEWAL_UNKNOWN: str = "Unknown"


def claimant_renewal_label(claimants: str | None) -> str:
    """Describe who exercised the statutory renewal right, in human terms.

    Maps each claimant's statutory code to the relationship it encodes — the
    author renewing in person, an estate class (widow/widower, child, executor,
    next of kin), or a proprietor class (work for hire, posthumous, composite) —
    so a labeler sees who renewed without decoding the raw codes. When a renewal
    carries more than one distinct relationship the labels are joined with
    ``"; "`` in source order, de-duplicated.

    Args:
        claimants: The renewal's raw claimants transcription, or ``None``.

    Returns:
        A concise renewal-right label; :data:`_CLAIMANT_RENEWAL_UNKNOWN`
        (``"Unknown"``) when no recognized code is present.
    """
    labels: list[str] = []
    for _name, code in parse_claimants(claimants):
        label = _CLAIMANT_RENEWAL_LABELS.get(code)
        if label is not None and label not in labels:
            labels.append(label)
    if not labels:
        return _CLAIMANT_RENEWAL_UNKNOWN
    return "; ".join(labels)


__all__ = [
    "claimant_renewal_label",
    "parse_claimants",
]
