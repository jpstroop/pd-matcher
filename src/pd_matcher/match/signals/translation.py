"""Detect translation registrations in CCE entries.

The CCE corpus records translation status as free-text annotations on
the ``<desc>``, ``<notes>``, ``<new-matter-claimed>``, and
``renewal_new_matter`` fields. The labeler flags translations as a
distinct category because the original-author / translator difference
makes the author scorer's penalty against a strong title+extent match
spurious. This module exposes one boolean predicate the pipeline can
consult to downweight the author scorer on those pairings; it
deliberately does *not* emit Evidence on its own — the title, year,
and extent scorers already carry the actual match strength.

Patterns (all case-insensitive):

* ``\\btr(?:ans(?:lated|lation)?)?\\.?\\b`` — covers ``"tr."``,
  ``"trans."``, ``"translated"``, ``"translation"``. The optional
  trailing period is matched explicitly so MARC/CCE abbreviation
  conventions are picked up.
* ``\\bversion\\b(?!.*program)`` — bare ``"version"`` is a common
  translation cue ("English version", "abridged version"); the
  negative lookahead suppresses computer-software false positives
  ("version 2.0 program"). Python's ``re`` does not allow
  variable-width lookbehinds, so the lookahead spans the rest of the
  string.
* ``\\bfrom the (?:French|German|Spanish|Italian|Russian|Latin|Greek|
  Hebrew|Japanese|Chinese|Arabic|Portuguese|Polish|Dutch|Swedish|
  Norwegian|Danish|Czech|Hungarian|Yiddish)\\b`` — the explicit
  "translated from the X" phrasing for major languages in the
  pre-1978 catalog.

Coverage from the labeled corpus: 14/500 pairs flag positive, all 14
labeled as matches. Adding the explicit "from the X" languages picks
up the registrations that simply note the source language without the
``"tr."`` token.
"""

from re import IGNORECASE
from re import compile as re_compile

from pd_matcher.models import IndexedNyplRegRecord

_TR_TOKEN_RE = re_compile(r"\btr(?:ans(?:lated|lation)?)?\.?\b", IGNORECASE)
_VERSION_RE = re_compile(r"\bversion\b(?!.*program)", IGNORECASE)
_FROM_THE_LANGUAGE_RE = re_compile(
    r"\bfrom the (?:French|German|Spanish|Italian|Russian|Latin|Greek|"
    r"Hebrew|Japanese|Chinese|Arabic|Portuguese|Polish|Dutch|Swedish|"
    r"Norwegian|Danish|Czech|Hungarian|Yiddish)\b",
    IGNORECASE,
)


def _value_matches(value: str | None) -> bool:
    if not value:
        return False
    if _TR_TOKEN_RE.search(value) is not None:
        return True
    if _VERSION_RE.search(value) is not None:
        return True
    return _FROM_THE_LANGUAGE_RE.search(value) is not None


def any_value_matches(*values: str | None) -> bool:
    """Return ``True`` when any of ``values`` contains a translation cue.

    A small generic helper so the UI projection (which reads pre-flattened
    CCE strings off the review DB) can apply the same patterns as the
    matcher without reconstructing an :class:`IndexedNyplRegRecord`.
    """
    return any(_value_matches(value) for value in values)


def is_translation_signal(cce: IndexedNyplRegRecord) -> bool:
    """Return ``True`` when any translation-cue regex hits a CCE text field."""
    joined_notes = " ".join(cce.notes) if cce.notes else None
    return any_value_matches(
        cce.desc,
        joined_notes,
        cce.new_matter_claimed,
        cce.renewal_new_matter,
    )


__all__ = [
    "any_value_matches",
    "is_translation_signal",
]
