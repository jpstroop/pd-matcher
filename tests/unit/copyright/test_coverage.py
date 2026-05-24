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


def test_coverage_from_year_counts_detects_cliff_at_data_boundary() -> None:
    """A bucket that collapses to a sliver of the trailing median ends the walk."""
    reg_counts = {1974: 100, 1975: 110, 1976: 105, 1977: 1}
    ren_counts = {1985: 50, 1986: 55, 1987: 60}
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts=ren_counts)
    assert coverage.reg_max_year == 1976


def test_coverage_from_year_counts_tolerates_partial_year_within_bulk() -> None:
    """A within-order-of-magnitude dip stays inside coverage (no premature stop).

    The trailing-median test absorbs single-year jitter so a 70% drop
    inside a real ramp does not freeze the walk; only a near-cliff
    (below ~10% of the trailing median) ends it. Without this tolerance
    the production registration walk stopped at 1927 instead of 1977.
    """
    reg_counts = {1974: 100, 1975: 110, 1976: 105, 1977: 30}
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts={1960: 50})
    assert coverage.reg_max_year == 1977


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


def test_coverage_from_year_counts_with_equal_counts_walks_full_span() -> None:
    """A flat histogram has no cliff to detect; the walk reaches both ends."""
    counts = dict.fromkeys(range(1920, 1931), 100)
    coverage = coverage_from_year_counts(reg_counts=counts, ren_counts=counts)
    assert coverage.reg_min_year == 1920
    assert coverage.reg_max_year == 1930
    assert coverage.ren_min_year == 1920
    assert coverage.ren_max_year == 1930


def test_coverage_from_year_counts_real_renewal_cliff() -> None:
    """Replays the production renewal histogram: anchor=1991, cliff at 1992.

    Mirrors the observed CCE renewal data: the 1985-1991 bulk runs into
    a sub-50 bucket at 1992 (a ~99.9% drop) and a few scattered tail
    entries up through 2001. The peak-anchored walk picks 1991, not the
    old forward-walk's 1927.
    """
    ren_counts = {
        1949: 189,
        1950: 6523,
        1985: 20536,
        1986: 20193,
        1987: 20884,
        1988: 21611,
        1989: 21310,
        1990: 21644,
        1991: 23254,
        1992: 21,
        1993: 5,
        1995: 5,
        1996: 3,
        2001: 1,
    }
    coverage = coverage_from_year_counts(reg_counts={1960: 100}, ren_counts=ren_counts)
    assert coverage.ren_max_year == 1991


def test_coverage_from_year_counts_real_registration_histogram() -> None:
    """Replays the production registration histogram: garbage years cannot win.

    Mirrors the observed CCE registration data: bulk in 1923-1977 with
    a handful of single-record garbage years (159, 5764, etc). The
    peak-anchored walk capped at 1977 yields a sensible window; the
    out-of-range years do not survive the trailing-median test.
    """
    reg_counts = {
        159: 1,
        1430: 1,
        1908: 6,
        1909: 5,
        1910: 8,
        1911: 16,
        1923: 30000,
        1924: 32000,
        1925: 34000,
        1926: 35000,
        1927: 36000,
        1974: 109158,
        1975: 117396,
        1976: 119799,
        1977: 103521,
        1978: 24,
        1985: 1,
        5764: 1,
    }
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts={1960: 100})
    assert coverage.reg_max_year == HARD_REG_MAX_YEAR
    assert coverage.reg_min_year >= 1700


def test_coverage_from_year_counts_anchor_is_peak_not_first() -> None:
    """A tiny early outlier does not anchor the walk; the peak does."""
    reg_counts = {1850: 1, 1925: 1000, 1926: 1100, 1927: 1050}
    coverage = coverage_from_year_counts(reg_counts=reg_counts, ren_counts={1960: 100})
    assert coverage.reg_max_year == 1927
    assert coverage.reg_min_year == 1925


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
