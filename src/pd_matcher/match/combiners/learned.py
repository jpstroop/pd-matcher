"""Phase 9 placeholder: lightgbm-based learned combiner.

This module exists so the :class:`pd_matcher.match.combiners.base.Combiner`
Protocol has a second concrete implementation referenced by name during
Phase 4. The actual lightgbm model integration lands in Phase 9; until
then any attempt to invoke :meth:`LearnedCombiner.combine` raises
``NotImplementedError``.
"""

from collections.abc import Sequence

from msgspec import Struct

from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence


class LearnedCombiner(Struct, frozen=True, forbid_unknown_fields=True):
    """Stub combiner that will host the lightgbm model in Phase 9."""

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:
        """Raise — the learned combiner is not implemented in Phase 4."""
        del evidence
        raise NotImplementedError("Phase 9 not yet implemented")


__all__ = [
    "LearnedCombiner",
]
