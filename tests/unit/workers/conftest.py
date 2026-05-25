"""Shared fixtures for Phase 6 worker tests.

Most tests need: a tiny LMDB index built from the project's tiny
fixtures, a minimal :class:`MatchingConfig`, an :class:`IdfTable` built
from the index, and a compiled pairings struct. The fixtures here
construct each once per test so the worker / writer / pool suites can
exercise the real pipeline without standing up the full package on their
own.
"""

from pathlib import Path

from pytest import fixture

from pd_matcher.config.loader import load_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_DEFAULTS_DIR = Path(__file__).resolve().parents[3] / "src" / "pd_matcher" / "config" / "defaults"
_PAIRINGS = _DEFAULTS_DIR / "field_pairings.yaml"


@fixture
def tiny_index_path(tmp_path: Path) -> Path:
    """Build a tiny LMDB index from the shared fixtures and return its path."""
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


@fixture
def tiny_idf(tiny_index_path: Path) -> IdfTable:
    """Return an :class:`IdfTable` built from the tiny index."""
    with NyplIndexLookup(tiny_index_path) as lookup:
        return build_idf_table(lookup)


@fixture
def matching_config() -> MatchingConfig:
    """Return the project-default :class:`MatchingConfig`."""
    return MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        year_window=2,
        min_combined_score=30.0,
        scorer="weighted_mean",
    )


@fixture
def pairing_config() -> PairingConfig:
    """Return the shipped default field-pairing configuration."""
    return load_pairing_config(_PAIRINGS)


@fixture
def compiled_pairings(pairing_config: PairingConfig) -> CompiledPairings:
    """Return the shipped default pairings compiled for the pipeline."""
    return compile_pairings(pairing_config)
