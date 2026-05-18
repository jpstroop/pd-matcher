"""Per-scorer structured output produced by every Phase 4 scorer.

:class:`Evidence` is the universal currency of the matching pipeline. Each
scorer returns an ``Evidence`` instance describing what it observed; the
combiner consumes a sequence of ``Evidence`` and produces the final
:class:`pd_matcher.match.combiners.base.CombinedScore`. Keeping the
representation rich (sub-features, ``skipped`` flag, ``decisive`` flag) is
what lets us audit ambiguous matches and lets Phase 9's learned combiner
consume the same Evidence stream as the weighted-mean default.
"""

from msgspec import Struct


class Evidence(Struct, frozen=True, forbid_unknown_fields=True):
    """One scorer's verdict on a (marc, nypl) field pairing.

    Attributes:
        scorer: Dotted scorer identifier (e.g. ``"title.token_set"``).
        score: Raw score this scorer produced, in ``[0, max]``.
        max: The score a perfect input pairing would have produced.
        skipped: ``True`` when either input was absent or unusable; the
            combiner excludes skipped Evidence from its weighted mean.
        decisive: Set to ``True`` by scorers that compared exact standard
            identifiers (LCCN, ISBN, regnum). Useful for audit and for
            ML feature inspection (it tells a human or a learned model
            "this scorer found an exact identifier hit") but **NOT** used
            to short-circuit the combiner: in this corpus identifier
            matches have a non-trivial false-positive rate (>5%) from
            transcription/OCR errors, so the calibrator owns the actual
            ``P(true match)``.
        features: Named numeric sub-features useful for debugging and for
            Phase 9's learned combiner.
    """

    scorer: str
    score: float
    max: float
    skipped: bool
    decisive: bool
    features: tuple[tuple[str, float], ...]

    @property
    def normalized(self) -> float:
        """Return :attr:`score` divided by :attr:`max`, or ``0.0`` if skipped."""
        if self.skipped or self.max <= 0.0:
            return 0.0
        return self.score / self.max


__all__ = [
    "Evidence",
]
