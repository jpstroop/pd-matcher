"""Output container produced by :mod:`pd_matcher.copyright.rules`.

A :class:`CopyrightAssessment` is the rule engine's verdict for one
record. It carries the terminal :class:`~pd_matcher.copyright.status.CopyrightStatus`,
the name of the rule that fired, a human-readable explanation, and the
list of pragmatic assumptions that contributed to the outcome (e.g.
"Assumed notice: registered work"). Surfacing the assumptions is what
lets a human reviewer audit each row of the output CSV.
"""

from msgspec import Struct

from pd_matcher.copyright.status import CopyrightStatus


class CopyrightAssessment(Struct, frozen=True, forbid_unknown_fields=True):
    """The rule engine's verdict for one (MARC, match) pair."""

    status: CopyrightStatus
    matched_rule_name: str | None
    explanation: str
    assumptions: tuple[str, ...]


__all__ = [
    "CopyrightAssessment",
]
