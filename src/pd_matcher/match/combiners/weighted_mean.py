"""Phase 4 default combiner: a plain weighted mean over present Evidence.

Each scorer's :class:`Evidence` is mapped to a per-scorer weight via the
``scorer`` field. Skipped Evidence contributes neither to the numerator
nor to the denominator (i.e. the weighted mean is over the *present*
Evidence). Identifier scorers (LCCN, ISBN) participate as ordinary
heavily-weighted scorers rather than short-circuiting the combiner: in
this corpus the >5% transcription/OCR error rate on standard identifiers
makes a hard "decisive" override unsafe. The Platt calibrator learns the
empirical ``P(true match)`` for the raw scores this combiner emits.

One decisive exception (issue #82): an uncorroborated whole/part volume
incompatibility (``volume.compat`` normalized ``0.0``, not skipped, and NOT
vetoed by an exact LCCN/ISBN hit) caps the combined score at
:data:`_WHOLE_PART_PENALTY_CAP`. Such a pair describes different
bibliographic units (a multi-volume whole vs a single registered part) yet
agrees on title/author/year by nature, so averaging the lone ``0.0`` leaves
it falsely high; the cap pushes it down for threshold triage. The veto built
into the signal spares identifier-corroborated true matches.
"""

from collections.abc import Mapping
from collections.abc import Sequence

from msgspec import Struct

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.features import volume_incompatible_uncorroborated
from pd_matcher.match.evidence import Evidence

_RAW_MAX: float = 100.0

# Decisive whole/part penalty (issue #82). When the cross-scorer
# ``volume.incompatible_uncorroborated`` signal fires — ``volume.compat``
# scored a whole-vs-part incompatibility (normalized 0.0, not skipped) and no
# exact LCCN/ISBN vetoes it — the combined calibrated score is CAPPED at this
# ceiling rather than letting one 0.0 be averaged away among several high
# signals. The cap, not a subtraction, is what makes the signal decisive: a
# whole/part no_match cannot score above the cap no matter how strongly title /
# author / year agree (they agree by nature on whole/part pairs). The veto
# inside the signal protects identifier-corroborated true matches, so the cap
# never touches an LCCN-confirmed pair. Tunable: a lower cap rejects harder; a
# higher cap is gentler.
_WHOLE_PART_PENALTY_CAP: float = 0.30


class WeightedMeanCombiner(Struct, frozen=True, forbid_unknown_fields=True):
    """Weighted-mean combiner parameterised by a :class:`MatchingConfig`."""

    config: MatchingConfig

    def _weights(self) -> Mapping[str, float]:
        cfg = self.config
        return {
            "title.token_set": cfg.title_weight,
            "name.author": cfg.author_weight,
            "name.publisher": cfg.publisher_weight,
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
        calibrated = raw / _RAW_MAX
        if (
            volume_incompatible_uncorroborated(evidence) == 1.0
            and calibrated > _WHOLE_PART_PENALTY_CAP
        ):
            calibrated = _WHOLE_PART_PENALTY_CAP
            raw = calibrated * _RAW_MAX
        return CombinedScore(raw=raw, calibrated=calibrated)


__all__ = [
    "WeightedMeanCombiner",
]
