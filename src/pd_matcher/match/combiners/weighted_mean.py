"""Phase 4 default combiner: a plain weighted mean over present Evidence.

Each scorer's :class:`Evidence` is mapped to a per-scorer weight via the
``scorer`` field. Skipped Evidence contributes neither to the numerator
nor to the denominator (i.e. the weighted mean is over the *present*
Evidence). Identifier scorers (LCCN, ISBN) participate as ordinary
heavily-weighted scorers rather than short-circuiting the combiner: in
this corpus the >5% transcription/OCR error rate on standard identifiers
makes a hard "decisive" override unsafe. The Platt calibrator learns the
empirical ``P(true match)`` for the raw scores this combiner emits.
"""

from collections.abc import Mapping
from collections.abc import Sequence

from msgspec import Struct

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence

_RAW_MAX: float = 100.0


class WeightedMeanCombiner(Struct, frozen=True, forbid_unknown_fields=True):
    """Weighted-mean combiner parameterised by a :class:`MatchingConfig`."""

    config: MatchingConfig

    def _weights(self) -> Mapping[str, float]:
        cfg = self.config
        return {
            "title.token_set": cfg.title_weight,
            "name.author": cfg.author_weight,
            "name.publisher": cfg.publisher_weight,
            "year.delta": cfg.year_weight,
            "edition.compat": cfg.edition_weight,
            "lccn.exact": cfg.lccn_weight,
            "isbn.exact": cfg.isbn_weight,
            "extent.page_count": cfg.extent_weight,
            "volume.compat": cfg.volume_weight,
        }

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:
        """Combine ``evidence`` into a :class:`CombinedScore`.

        Each Evidence's effective weight in the mean is the configured
        scorer weight multiplied by :attr:`Evidence.weight_multiplier`
        (default ``1.0``). The multiplier is applied symmetrically to
        the numerator and denominator so that downweighting one scorer
        on a specific pairing does not deflate the mean — it just
        reduces that scorer's share of it.
        """
        weights = self._weights()
        numerator = 0.0
        denominator = 0.0
        for item in evidence:
            if item.skipped:
                continue
            weight = weights.get(item.scorer)
            if weight is None or weight <= 0.0:
                continue
            effective_weight = weight * item.weight_multiplier
            if effective_weight <= 0.0:
                continue
            numerator += effective_weight * item.normalized
            denominator += effective_weight
        raw = (numerator / denominator) * _RAW_MAX if denominator > 0.0 else 0.0
        return CombinedScore(raw=raw, calibrated=raw / _RAW_MAX)


__all__ = [
    "WeightedMeanCombiner",
]
