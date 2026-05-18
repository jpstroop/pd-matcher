"""Optional end-to-end index build against the real submodule data.

The submodules contain NYPL's XML/TSV transcription of the U.S. Copyright
Office's Catalog of Copyright Entries (CCE), which is published by the
Library of Congress. This test is skipped when those submodules are not
checked out. When the submodules are present it builds the full LMDB
index into ``tmp_path`` and asserts the produced counts are within
order-of-magnitude bands. Measured against the corpus as of project init:
~2.17M registrations, ~444k renewals, ~160k renewal joins, ~95 year
buckets, full build in ~37s. The corpus may evolve over time, so the
assertions deliberately use generous bands instead of exact counts.
"""

from pathlib import Path

from pytest import mark

from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REG_DIR = _REPO_ROOT / "data" / "nypl-reg" / "xml"
_REN_DIR = _REPO_ROOT / "data" / "nypl-ren" / "data"


@mark.slow
@mark.skipif(
    not (_REG_DIR.exists() and _REN_DIR.exists()),
    reason="NYPL submodules not checked out",
)
def test_full_index_build_against_real_sources(tmp_path: Path) -> None:
    out_path = tmp_path / "nypl.lmdb"
    report = build_index(reg_dir=_REG_DIR, ren_dir=_REN_DIR, out_path=out_path)

    assert report.skipped is False
    # Measured ~2.17M registrations at project init; allow a wide band so the
    # test does not flake when the upstream submodule grows or trims a year.
    assert 1_500_000 <= report.registrations_written <= 5_000_000
    # Renewals: measured ~444k at project init.
    assert 200_000 <= report.renewals_written <= 1_000_000
    # Year buckets: measured 95 at project init (corpus spans roughly 1908-2002).
    assert 50 <= report.year_buckets <= 150
    # Renewal joins are bounded by registrations and renewals.
    assert 0 < report.renewal_joins <= report.registrations_written

    with NyplIndexLookup(out_path) as lookup:
        stats = lookup.stats()
        assert stats.registrations_written == report.registrations_written
        assert stats.renewals_written == report.renewals_written

        # Spot-check that 1940 has a non-trivial number of candidates.
        candidates = list(lookup.candidates_for_year(1940, window=1))
        assert len(candidates) > 100
