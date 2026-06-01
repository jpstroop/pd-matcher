"""Detect pair-level evidence isolation (no non-title scorer corroborates).

The rule-based weighted-mean combiner over-trusts a title-only match when
every other scorer disagrees: even one strong scorer's score can dominate
the average when the other scorers contribute zero. The 2026-05-31
diagnostic surfaced six concrete cases in the top-30 disagreement list
(Cold war to détente / From the cold war to detente, Selected poems / A
selection of poems., etc.) where the title agreed but author / year /
publisher all contradicted — and the combined score still landed around
0.7, well above the 0.5 decision threshold.

The predicate exposed here is the test the pipeline runs after all
scorers have fired: "did any non-title scorer contribute a meaningful
signal?" When the answer is no, the pipeline applies a graduated
``weight_multiplier`` to the title scorer's Evidence to reflect the
absence of corroboration. The mechanism is the same as the existing
translation-signal author downweight; only the trigger differs.
"""

from collections.abc import Iterable

from pd_matcher.match.evidence import Evidence


def has_no_corroboration(other_evidences: Iterable[Evidence], threshold: float) -> bool:
    """Return ``True`` when no Evidence in ``other_evidences`` corroborates.

    An Evidence corroborates when it is not ``skipped`` and its ``score``
    reaches ``threshold``. Score is on the scorer's own 0-100 scale; a
    reasonable threshold is half-max (``50.0``), which excludes the
    graduated penalty band most scorers use for "weakly agrees" without
    catching anything below "actively agrees."
    """
    return not any(
        not evidence.skipped and evidence.score >= threshold for evidence in other_evidences
    )


__all__ = [
    "has_no_corroboration",
]
