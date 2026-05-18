"""Optional end-to-end index build against the real submodule data.

Skipped when the NYPL submodules are not checked out. When the submodules
are present this test builds the full LMDB index into ``tmp_path`` and
asserts the produced counts are within the order of magnitude documented in
the project plan (~642k registrations, ~107k renewals). The corpus may
evolve over time, so the assertions deliberately use generous bands instead
of exact counts.
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
    # Plan documents roughly 642k registrations; allow a wide band so the
    # test does not flake when the upstream submodule grows or trims a year.
    assert 400_000 <= report.registrations_written <= 1_000_000
    # Renewals: roughly 100k expected.
    assert 50_000 <= report.renewals_written <= 300_000
    # Year buckets: 1923-1977 plus margins.
    assert 50 <= report.year_buckets <= 80
    # Renewal joins are bounded by registrations and renewals.
    assert 0 < report.renewal_joins <= report.registrations_written

    with NyplIndexLookup(out_path) as lookup:
        stats = lookup.stats()
        assert stats.registrations_written == report.registrations_written
        assert stats.renewals_written == report.renewals_written

        # Spot-check that 1940 has a non-trivial number of candidates.
        candidates = list(lookup.candidates_for_year(1940, window=1))
        assert len(candidates) > 100
