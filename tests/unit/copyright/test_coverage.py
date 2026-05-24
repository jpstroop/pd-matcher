"""Tests for :mod:`pd_matcher.copyright.coverage`."""

from pd_matcher.copyright.coverage import HARD_REG_MAX_YEAR
from pd_matcher.copyright.coverage import HARD_REN_MAX_YEAR
from pd_matcher.copyright.coverage import LEGACY_COVERAGE
from pd_matcher.copyright.coverage import Coverage
from pd_matcher.copyright.coverage import coverage_from_pairs
from pd_matcher.copyright.coverage import coverage_from_year_counts


def test_legacy_coverage_defaults_are_sensible() -> None:
    """The shipped legacy coverage covers the entire 1909-Act regime."""
    assert LEGACY_COVERAGE.reg_min_year == 1891
    assert LEGACY_COVERAGE.reg_max_year == 1977
    assert LEGACY_COVERAGE.ren_min_year == 1909
    assert LEGACY_COVERAGE.ren_max_year == 1991


def test_hard_caps_match_legal_regime() -> None:
    """The legal-regime caps reflect the 1909-Act end and a 28-year tail."""
    assert HARD_REG_MAX_YEAR == 1977
    assert HARD_REN_MAX_YEAR == 2005


def test_coverage_struct_is_frozen() -> None:
    """Coverage is a frozen :class:`msgspec.Struct`."""
    coverage = Coverage(reg_min_year=1891, reg_max_year=1977, ren_min_year=1909, ren_max_year=1991)
    assert coverage.reg_min_year == 1891
    assert coverage.reg_max_year == 1977
    assert coverage.ren_min_year == 1909
    assert coverage.ren_max_year == 1991


def test_coverage_from_year_counts_with_smooth_histogram() -> None:
    """A smooth histogram yields min and max years matching the data."""
    reg_counts = {1931: 100, 1932: 105, 1933: 95, 1934: 110}
    ren_counts = {1960: 50, 1961: 48, 1962: 55}
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts=ren_counts)
    assert coverage.reg_min_year == 1931
    assert coverage.reg_max_year == 1934
    assert coverage.ren_min_year == 1960
    assert coverage.ren_max_year == 1962


def test_coverage_from_year_counts_detects_partial_truncation() -> None:
    """A bucket below 50% of the prior year is treated as boundary truncation."""
    reg_counts = {1975: 100, 1976: 110, 1977: 30}
    ren_counts = {1985: 50, 1986: 55, 1987: 60}
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts=ren_counts)
    assert coverage.reg_max_year == 1976


def test_coverage_from_year_counts_caps_at_legal_regime() -> None:
    """Even a smooth histogram extending beyond 1977 is capped at the legal max."""
    reg_counts = {1975: 100, 1976: 105, 1977: 110, 1978: 115, 1979: 120}
    ren_counts = {1980: 50}
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts=ren_counts)
    assert coverage.reg_max_year == HARD_REG_MAX_YEAR


def test_coverage_from_year_counts_caps_renewals_at_legal_regime() -> None:
    """Renewal-year max is capped at :data:`HARD_REN_MAX_YEAR`."""
    reg_counts = {1970: 100}
    ren_counts = dict.fromkeys(range(1998, 2020), 50)
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts=ren_counts)
    assert coverage.ren_max_year == HARD_REN_MAX_YEAR


def test_coverage_from_year_counts_handles_empty_reg_histogram() -> None:
    """An empty reg histogram falls back to LEGACY_COVERAGE bounds."""
    coverage = coverage_from_year_counts(reg_counts={}, ren_counts={1960: 50, 1961: 55})
    assert coverage.reg_min_year == LEGACY_COVERAGE.reg_min_year
    assert coverage.reg_max_year == LEGACY_COVERAGE.reg_max_year


def test_coverage_from_year_counts_handles_empty_ren_histogram() -> None:
    """An empty ren histogram falls back to LEGACY_COVERAGE bounds."""
    coverage = coverage_from_year_counts(
        reg_counts={1950: 100, 1951: 105},
        ren_counts={},
    )
    assert coverage.ren_min_year == LEGACY_COVERAGE.ren_min_year
    assert coverage.ren_max_year == LEGACY_COVERAGE.ren_max_year


def test_coverage_from_year_counts_handles_both_empty() -> None:
    """Two empty histograms produce LEGACY_COVERAGE."""
    coverage = coverage_from_year_counts(reg_counts={}, ren_counts={})
    assert coverage == LEGACY_COVERAGE


def test_coverage_from_year_counts_with_single_year_bucket() -> None:
    """A one-year histogram yields a one-year coverage window."""
    coverage = coverage_from_year_counts(reg_counts={1950: 100}, ren_counts={1978: 50})
    assert coverage.reg_min_year == 1950
    assert coverage.reg_max_year == 1950
    assert coverage.ren_min_year == 1978
    assert coverage.ren_max_year == 1978


def test_coverage_from_year_counts_with_zero_bucket_does_not_divide() -> None:
    """A bucket whose previous value is zero is not a partial-truncation event."""
    reg_counts = {1950: 0, 1951: 100, 1952: 110}
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts={1960: 50})
    assert coverage.reg_max_year == 1952


def test_coverage_from_pairs_round_trips_through_dicts() -> None:
    """The pairs-form convenience yields the same result as the dict form."""
    pairs_reg = [(1931, 100), (1932, 110), (1933, 105)]
    pairs_ren = [(1960, 50), (1961, 55)]
    via_pairs = coverage_from_pairs(reg_year_counts=pairs_reg, ren_year_counts=pairs_ren)
    via_dicts = coverage_from_year_counts(
        reg_counts=dict(pairs_reg),
        ren_counts=dict(pairs_ren),
    )
    assert via_pairs == via_dicts
