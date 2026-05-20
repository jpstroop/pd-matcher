"""Shared fixtures for Phase 4 tests.

The fixtures here construct a minimal :class:`ScorerContext` with a tiny
synthetic IDF table so that scorer unit tests do not need to stand up an
LMDB env. The context uses the English stopword set, the English Snowball
stemmer, and the project default :class:`MatchingConfig`.
"""

from collections.abc import Callable
from pathlib import Path

from pytest import fixture

from pd_matcher.config.loader import load_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.normalize.stemming import stemmer_for
from pd_matcher.normalize.stopwords import load_stopwords

_PAIRINGS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "pd_matcher"
    / "config"
    / "defaults"
    / "field_pairings.yaml"
)


@fixture
def compiled_pairings() -> CompiledPairings:
    """Return the shipped default pairings compiled for the pipeline."""
    return compile_pairings(load_pairing_config(_PAIRINGS))


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
        min_combined_score=70.0,
        scorer="weighted_mean",
    )


@fixture
def idf_table() -> IdfTable:
    """Return a tiny IDF table with stemmed English tokens."""
    return IdfTable(
        document_count=10,
        default_idf=2.0,
        source_hash="test-hash",
        language="eng",
        idf={
            "widget": 3.0,
            "studi": 2.5,
            "small": 1.5,
            "part": 1.5,
            "machin": 3.0,
            "albuquerqu": 5.0,
            "american": 1.0,
        },
    )


@fixture
def english_stemmer() -> Callable[[str], str]:
    """Return the English Snowball stemmer callable."""
    return stemmer_for("eng")


@fixture
def scorer_context(matching_config: MatchingConfig, idf_table: IdfTable) -> ScorerContext:
    """Return an English-language :class:`ScorerContext`."""
    return ScorerContext(
        language="eng",
        stopwords=load_stopwords("eng"),
        stemmer=stemmer_for("eng"),
        idf=idf_table,
        config=matching_config,
    )


@fixture
def french_scorer_context(matching_config: MatchingConfig, idf_table: IdfTable) -> ScorerContext:
    """Return a French-language :class:`ScorerContext`."""
    return ScorerContext(
        language="fre",
        stopwords=load_stopwords("fre"),
        stemmer=stemmer_for("fre"),
        idf=idf_table,
        config=matching_config,
    )
