"""Combiner Protocol and shared :class:`CombinedScore` definition.

A combiner takes a sequence of per-scorer :class:`Evidence` and produces a
single :class:`CombinedScore`. Two concrete combiners live in this package:
:class:`pd_matcher.match.combiners.weighted_mean.WeightedMeanCombiner`
(Phase 4 default) and
:class:`pd_matcher.match.combiners.learned.LearnedCombiner` (Phase 9
placeholder). Both honor the same :class:`Combiner` Protocol so the
pipeline does not care which is wired in.
"""

from collections.abc import Sequence
from typing import Protocol

from msgspec import Struct

from pd_matcher.match.evidence import Evidence


class CombinedScore(Struct, frozen=True, forbid_unknown_fields=True):
    """Final combined verdict for one candidate pair.

    Attributes:
        raw: Weighted mean of normalized non-skipped Evidence in
            ``[0, 100]``.
        calibrated: Platt-scaled probability of a true match in
            ``[0, 1]``. The pipeline fills this in when a calibrator is
            available; otherwise it equals ``raw / 100``.
    """

    raw: float
    calibrated: float


class Combiner(Protocol):
    """Pure Protocol for evidence-combining objects."""

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:  # pragma: no cover
        """Return a :class:`CombinedScore` from the supplied Evidence."""
        ...


__all__ = [
    "CombinedScore",
    "Combiner",
]
