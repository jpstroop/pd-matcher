"""Unit tests for the stratified sampling primitives."""

from pytest import approx
from pytest import raises

from pd_groundtruth.sampling import BAND_70_80
from pd_groundtruth.sampling import BAND_80_90
from pd_groundtruth.sampling import BAND_BELOW
from pd_groundtruth.sampling import BAND_GE90
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import SOURCE_BELOW_SAMPLE
from pd_groundtruth.sampling import BudgetModel
from pd_groundtruth.sampling import Stratifier
from pd_groundtruth.sampling import StratumOutcome
from pd_groundtruth.sampling import band_of
from pd_groundtruth.sampling import default_budget
from pd_groundtruth.sampling import iter_capped_bands
from pd_groundtruth.sampling import reservoir_sample
from pd_groundtruth.sampling import scale_budget


def test_band_of_boundaries_are_half_open_from_above() -> None:
    assert band_of(1.0) == BAND_GE90
    assert band_of(0.9) == BAND_GE90
    assert band_of(0.8999) == BAND_80_90
    assert band_of(0.8) == BAND_80_90
    assert band_of(0.7999) == BAND_70_80
    assert band_of(0.7) == BAND_70_80
    assert band_of(0.6999) == BAND_BELOW
    assert band_of(0.0) == BAND_BELOW


def test_default_budget_matches_documented_caps() -> None:
    budget = default_budget()
    assert budget.cap_for("eng", BAND_GE90) == 500
    assert budget.cap_for("eng", BAND_80_90) == 200
    assert budget.cap_for("eng", BAND_70_80) == 200
    assert budget.cap_for("eng", BAND_BELOW) == 300
    for language in ("fre", "ger", "spa", "ita"):
        assert budget.cap_for(language, BAND_GE90) == 60
        assert budget.cap_for(language, BAND_80_90) == 30
        assert budget.cap_for(language, BAND_70_80) == 30
        assert budget.cap_for(language, BAND_BELOW) == 80
    assert budget.total() == 2000
    assert budget.languages() == ("eng", "fre", "ger", "spa", "ita")


def test_cap_for_unconfigured_stratum_is_zero() -> None:
    budget = default_budget()
    assert budget.cap_for("rus", BAND_GE90) == 0
    assert budget.cap_for("eng", "nonsense") == 0


def test_scale_budget_doubles_proportionally() -> None:
    budget = default_budget()
    scaled = scale_budget(budget, budget.total() * 2)
    assert scaled.cap_for("eng", BAND_GE90) == 1000
    assert scaled.cap_for("fre", BAND_BELOW) == 160


def test_scale_budget_identity_when_target_equals_total() -> None:
    budget = default_budget()
    scaled = scale_budget(budget, budget.total())
    assert scaled.caps == budget.caps


def test_scale_budget_keeps_nonzero_caps_at_least_one() -> None:
    budget = default_budget()
    scaled = scale_budget(budget, 5)
    assert all(cap >= 1 for cap in scaled.caps.values())


def test_scale_budget_preserves_zero_caps() -> None:
    budget = BudgetModel(caps={("eng", BAND_GE90): 100, ("eng", BAND_BELOW): 0})
    scaled = scale_budget(budget, 50)
    assert scaled.cap_for("eng", BAND_BELOW) == 0
    assert scaled.cap_for("eng", BAND_GE90) == 50


def test_scale_budget_rejects_nonpositive_target() -> None:
    with raises(ValueError, match="target_total must be positive"):
        scale_budget(default_budget(), 0)


def test_scale_budget_rejects_zero_total_source() -> None:
    with raises(ValueError, match="total is zero"):
        scale_budget(BudgetModel(caps={}), 100)


def test_reservoir_sample_returns_all_when_stream_smaller_than_k() -> None:
    assert reservoir_sample(iter(range(3)), 5, seed=1) == [0, 1, 2]


def test_reservoir_sample_empty_for_nonpositive_k() -> None:
    assert reservoir_sample(iter(range(10)), 0, seed=1) == []
    assert reservoir_sample(iter(range(10)), -1, seed=1) == []


def test_reservoir_sample_is_deterministic_for_fixed_seed() -> None:
    first = reservoir_sample(iter(range(100)), 5, seed=42)
    second = reservoir_sample(iter(range(100)), 5, seed=42)
    assert first == second
    assert len(first) == 5
    assert all(0 <= value < 100 for value in first)


def test_reservoir_sample_differs_across_seeds() -> None:
    first = reservoir_sample(iter(range(1000)), 10, seed=1)
    second = reservoir_sample(iter(range(1000)), 10, seed=2)
    assert first != second


def test_iter_capped_bands_excludes_below() -> None:
    bands = tuple(iter_capped_bands())
    assert bands == (BAND_GE90, BAND_80_90, BAND_70_80)
    assert BAND_BELOW not in bands


def _outcome(key: str, language: str, score: float) -> StratumOutcome:
    return StratumOutcome(key=key, language=language, score=score)


def test_stratifier_accepts_banded_until_cap_then_rejects() -> None:
    budget = BudgetModel(caps={("eng", BAND_GE90): 2})
    stratifier = Stratifier(budget, seed=1)
    first = stratifier.offer(_outcome("a", "eng", 0.95))
    second = stratifier.offer(_outcome("b", "eng", 0.91))
    third = stratifier.offer(_outcome("c", "eng", 0.99))
    assert first is not None
    assert first.band == BAND_GE90
    assert first.source == SOURCE_BANDED
    assert second is not None
    assert third is None
    assert stratifier.counts()[("eng", BAND_GE90)] == 2


def test_stratifier_below_outcomes_buffered_until_finalize() -> None:
    budget = BudgetModel(caps={("eng", BAND_BELOW): 2})
    stratifier = Stratifier(budget, seed=7)
    for index in range(10):
        immediate = stratifier.offer(_outcome(f"k{index}", "eng", 0.1))
        assert immediate is None
    accepted = stratifier.finalize()
    below = [pair for pair in accepted if pair.source == SOURCE_BELOW_SAMPLE]
    assert len(below) == 2
    assert all(pair.band == BAND_BELOW for pair in below)
    assert stratifier.counts()[("eng", BAND_BELOW)] == 2


def test_stratifier_below_sample_is_deterministic() -> None:
    budget = BudgetModel(caps={("eng", BAND_BELOW): 3})
    keys = [f"k{index}" for index in range(20)]

    def _run() -> list[str]:
        stratifier = Stratifier(budget, seed=99)
        for key in keys:
            stratifier.offer(_outcome(key, "eng", 0.2))
        return [pair.key for pair in stratifier.finalize()]

    assert _run() == _run()


def test_stratifier_finalize_combines_banded_and_below() -> None:
    budget = BudgetModel(
        caps={("eng", BAND_GE90): 1, ("eng", BAND_BELOW): 1, ("fre", BAND_70_80): 1}
    )
    stratifier = Stratifier(budget, seed=3)
    stratifier.offer(_outcome("high", "eng", 0.95))
    stratifier.offer(_outcome("low", "eng", 0.3))
    stratifier.offer(_outcome("fre", "fre", 0.75))
    accepted = stratifier.finalize()
    sources = {pair.key: (pair.band, pair.source) for pair in accepted}
    assert sources["high"] == (BAND_GE90, SOURCE_BANDED)
    assert sources["fre"] == (BAND_70_80, SOURCE_BANDED)
    assert sources["low"] == (BAND_BELOW, SOURCE_BELOW_SAMPLE)


def test_stratifier_below_cap_zero_draws_nothing() -> None:
    budget = BudgetModel(caps={("eng", BAND_BELOW): 0})
    stratifier = Stratifier(budget, seed=1)
    stratifier.offer(_outcome("a", "eng", 0.1))
    assert stratifier.finalize() == []


def test_accepted_pair_preserves_score() -> None:
    budget = BudgetModel(caps={("eng", BAND_80_90): 1})
    stratifier = Stratifier(budget, seed=1)
    pair = stratifier.offer(_outcome("a", "eng", 0.85))
    assert pair is not None
    assert pair.score == approx(0.85)
