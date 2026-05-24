"""Stratified sampling primitives for the review-queue builder.

The full-MARC candidate pool (~339k records) yields far more high-scoring
matcher outcomes than can ever be hand-labeled, and the score distribution
is wildly uneven across languages (English is match-rich, non-English
sparse). To build a review queue that is both affordable to label and
useful for measuring recall, we sample within ``(language, score-band)``
caps toward a fixed budget, English-weighted, PLUS a random below-0.7
sample so the eventual learned scorer sees the matcher's blind spots.

Everything in this module is pure and deterministic given a seed; the
parallel matching orchestration lives in :mod:`pd_groundtruth.build_queue`.
"""

from collections.abc import Iterable
from collections.abc import Iterator
from random import Random

from msgspec import Struct

BAND_GE90: str = "ge90"
BAND_80_90: str = "b80_90"
BAND_70_80: str = "b70_80"
BAND_BELOW: str = "below"

SOURCE_BANDED: str = "banded"
SOURCE_BELOW_SAMPLE: str = "below_sample"

_THRESHOLD_GE90: float = 0.9
_THRESHOLD_80: float = 0.8
_THRESHOLD_70: float = 0.7

_CAPPED_BANDS: tuple[str, ...] = (BAND_GE90, BAND_80_90, BAND_70_80)


def band_of(score: float) -> str:
    """Return the score-band label for ``score``.

    Bands are half-open from above: ``>=0.9`` is :data:`BAND_GE90`,
    ``[0.8, 0.9)`` is :data:`BAND_80_90`, ``[0.7, 0.8)`` is
    :data:`BAND_70_80`, and anything ``<0.7`` is :data:`BAND_BELOW`
    (eligible for the random :data:`SOURCE_BELOW_SAMPLE` bucket).
    """
    if score >= _THRESHOLD_GE90:
        return BAND_GE90
    if score >= _THRESHOLD_80:
        return BAND_80_90
    if score >= _THRESHOLD_70:
        return BAND_70_80
    return BAND_BELOW


class BudgetModel(Struct, frozen=True, forbid_unknown_fields=True):
    """Per-(language, band) caps for the review queue.

    ``caps`` maps a ``(language, band)`` key to the maximum number of pairs
    to persist for that stratum. The banded caps (:data:`BAND_GE90`,
    :data:`BAND_80_90`, :data:`BAND_70_80`) draw the first accepted
    outcomes in stream order; the :data:`BAND_BELOW` cap feeds the random
    :data:`SOURCE_BELOW_SAMPLE` reservoir so recall stays measurable.
    """

    caps: dict[tuple[str, str], int]

    def cap_for(self, language: str, band: str) -> int:
        """Return the cap for ``(language, band)`` (``0`` when unconfigured)."""
        return self.caps.get((language, band), 0)

    def total(self) -> int:
        """Return the sum of every configured cap (the nominal budget)."""
        return sum(self.caps.values())

    def languages(self) -> tuple[str, ...]:
        """Return the configured languages in first-seen order."""
        seen: list[str] = []
        for language, _band in self.caps:
            if language not in seen:
                seen.append(language)
        return tuple(seen)


_DEFAULT_CAPS: dict[tuple[str, str], int] = {
    ("eng", BAND_GE90): 500,
    ("eng", BAND_80_90): 200,
    ("eng", BAND_70_80): 200,
    ("eng", BAND_BELOW): 300,
    ("fre", BAND_GE90): 60,
    ("fre", BAND_80_90): 30,
    ("fre", BAND_70_80): 30,
    ("fre", BAND_BELOW): 80,
    ("ger", BAND_GE90): 60,
    ("ger", BAND_80_90): 30,
    ("ger", BAND_70_80): 30,
    ("ger", BAND_BELOW): 80,
    ("spa", BAND_GE90): 60,
    ("spa", BAND_80_90): 30,
    ("spa", BAND_70_80): 30,
    ("spa", BAND_BELOW): 80,
    ("ita", BAND_GE90): 60,
    ("ita", BAND_80_90): 30,
    ("ita", BAND_70_80): 30,
    ("ita", BAND_BELOW): 80,
}

_DEFAULT_TOTAL: int = sum(_DEFAULT_CAPS.values())


def default_budget() -> BudgetModel:
    """Return the documented default budget (~2,000 pairs, English-weighted).

    English: ge90=500, b80_90=200, b70_80=200, below=300. Each of fre/ger/
    spa/ita: ge90=60, b80_90=30, b70_80=30, below=80. The exact total is
    :data:`_DEFAULT_TOTAL`.
    """
    return BudgetModel(caps=dict(_DEFAULT_CAPS))


def scale_budget(budget: BudgetModel, target_total: int) -> BudgetModel:
    """Return ``budget`` rescaled proportionally to ``target_total``.

    Each cap is multiplied by ``target_total / budget.total()`` and rounded
    to the nearest integer (minimum ``1`` for any originally non-zero cap so
    no stratum is silently dropped by rounding). A ``target_total`` equal to
    the current total returns an identical budget.

    Raises:
        ValueError: If ``target_total`` is not positive or the source
            budget has a zero total.
    """
    if target_total <= 0:
        raise ValueError(f"target_total must be positive (got {target_total!r})")
    current = budget.total()
    if current <= 0:
        raise ValueError("cannot scale a budget whose total is zero")
    factor = target_total / current
    scaled: dict[tuple[str, str], int] = {}
    for key, cap in budget.caps.items():
        if cap <= 0:
            scaled[key] = 0
            continue
        scaled[key] = max(1, round(cap * factor))
    return BudgetModel(caps=scaled)


def reservoir_sample[T](iterable: Iterable[T], k: int, seed: int) -> list[T]:
    """Return up to ``k`` items drawn uniformly from ``iterable``.

    Implements Algorithm R: a single streaming pass with O(k) memory,
    deterministic for a fixed ``seed`` and a fixed input order. When the
    stream has at most ``k`` items every item is returned (in order); when
    it is longer the result is a uniform sample whose order reflects the
    reservoir's final state.

    Args:
        iterable: Source stream; consumed exactly once.
        k: Desired sample size. ``k <= 0`` returns an empty list.
        seed: Seed for the internal :class:`random.Random`.
    """
    if k <= 0:
        return []
    rng = Random(seed)
    reservoir: list[T] = []
    for index, item in enumerate(iterable):
        if index < k:
            reservoir.append(item)
            continue
        j = rng.randint(0, index)
        if j < k:
            reservoir[j] = item
    return reservoir


class StratumOutcome(Struct, frozen=True, forbid_unknown_fields=True):
    """One matched outcome offered to the :class:`Stratifier`.

    ``key`` is an opaque caller-side identifier (e.g. the MARC control id or
    a positional index) used only to report which outcomes were accepted.
    """

    key: str
    language: str
    score: float


class AcceptedPair(Struct, frozen=True, forbid_unknown_fields=True):
    """A persisted decision emitted by the :class:`Stratifier`."""

    key: str
    language: str
    band: str
    source: str
    score: float


class Stratifier:
    """Assign matched outcomes to strata and accept up to each cap.

    Banded outcomes (``>=0.7``) are accepted greedily in stream order until
    their ``(language, band)`` cap fills. Below-0.7 outcomes are buffered
    and, on :meth:`finalize`, a deterministic seeded reservoir draws up to
    the ``below`` cap per language for the :data:`SOURCE_BELOW_SAMPLE`
    bucket. ``finalize`` returns every accepted pair (banded then sampled).
    """

    __slots__ = ("_accepted", "_below_buffer", "_budget", "_counts", "_seed")

    def __init__(self, budget: BudgetModel, *, seed: int) -> None:
        self._budget = budget
        self._seed = seed
        self._counts: dict[tuple[str, str], int] = {}
        self._accepted: list[AcceptedPair] = []
        self._below_buffer: dict[str, list[StratumOutcome]] = {}

    def offer(self, outcome: StratumOutcome) -> AcceptedPair | None:
        """Offer one outcome; return an :class:`AcceptedPair` if accepted now.

        Banded outcomes are decided immediately. Below-0.7 outcomes are
        buffered for the reservoir draw in :meth:`finalize` and always
        return ``None`` here.
        """
        band = band_of(outcome.score)
        if band == BAND_BELOW:
            self._below_buffer.setdefault(outcome.language, []).append(outcome)
            return None
        key = (outcome.language, band)
        if self._counts.get(key, 0) >= self._budget.cap_for(*key):
            return None
        self._counts[key] = self._counts.get(key, 0) + 1
        pair = AcceptedPair(
            key=outcome.key,
            language=outcome.language,
            band=band,
            source=SOURCE_BANDED,
            score=outcome.score,
        )
        self._accepted.append(pair)
        return pair

    def finalize(self) -> list[AcceptedPair]:
        """Draw the below-0.7 reservoir and return all accepted pairs."""
        for language, outcomes in self._below_buffer.items():
            cap = self._budget.cap_for(language, BAND_BELOW)
            language_seed = self._seed ^ hash(language)
            for outcome in reservoir_sample(outcomes, cap, language_seed):
                key = (language, BAND_BELOW)
                self._counts[key] = self._counts.get(key, 0) + 1
                self._accepted.append(
                    AcceptedPair(
                        key=outcome.key,
                        language=language,
                        band=BAND_BELOW,
                        source=SOURCE_BELOW_SAMPLE,
                        score=outcome.score,
                    )
                )
        return list(self._accepted)

    def counts(self) -> dict[tuple[str, str], int]:
        """Return a snapshot of accepted counts per ``(language, band)``."""
        return dict(self._counts)


def iter_capped_bands() -> Iterator[str]:
    """Yield the band labels that fill greedily in stream order."""
    yield from _CAPPED_BANDS


__all__ = [
    "BAND_70_80",
    "BAND_80_90",
    "BAND_BELOW",
    "BAND_GE90",
    "SOURCE_BANDED",
    "SOURCE_BELOW_SAMPLE",
    "AcceptedPair",
    "BudgetModel",
    "Stratifier",
    "StratumOutcome",
    "band_of",
    "default_budget",
    "iter_capped_bands",
    "reservoir_sample",
    "scale_budget",
]
