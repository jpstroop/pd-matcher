"""Unit tests for the groundtruth kept-per-stratum suffix renderer.

The generic rate/ETA/percent math and the cadence-gated reporter live in
:mod:`pd_matcher.progress` and are tested in the main app; here only the
domain-specific ``kept: …`` suffix is exercised.
"""

from pd_groundtruth.progress import render_kept_suffix
from pd_groundtruth.sampling import BudgetModel


def _budget() -> BudgetModel:
    return BudgetModel(
        caps={
            ("eng", "ge90"): 500,
            ("eng", "b80_90"): 200,
            ("eng", "b70_80"): 200,
            ("eng", "b60_70"): 200,
            ("eng", "below"): 300,
            ("fre", "ge90"): 60,
            ("fre", "below"): 80,
        }
    )


def test_render_kept_suffix_breaks_out_lead_language() -> None:
    kept = {
        ("eng", "ge90"): 412,
        ("eng", "b80_90"): 90,
        ("eng", "b70_80"): 88,
        ("eng", "b60_70"): 50,
        ("eng", "below"): 300,
        ("fre", "ge90"): 100,
        ("fre", "below"): 40,
    }
    suffix = render_kept_suffix(_budget(), kept)
    assert suffix.startswith("kept: ")
    assert "eng[ge90 412/500 b80_90 90/200 b70_80 88/200 b60_70 50/200 below 300/300]" in suffix
    assert "fre 140" in suffix


def test_render_kept_suffix_empty_budget() -> None:
    assert render_kept_suffix(BudgetModel(caps={}), {}) == "kept: "


def test_render_kept_suffix_missing_counts_default_to_zero() -> None:
    suffix = render_kept_suffix(_budget(), {})
    assert "eng[ge90 0/500 b80_90 0/200 b70_80 0/200 b60_70 0/200 below 0/300]" in suffix
    assert "fre 0" in suffix
