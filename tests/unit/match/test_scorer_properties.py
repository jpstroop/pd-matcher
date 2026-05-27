"""Property-based invariants every scorer must obey."""

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import text

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.edition import score_edition
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.match.scorers.title import score_title
from pd_matcher.match.scorers.year import score_year
from pd_matcher.normalize.stemming import stemmer_for
from pd_matcher.normalize.stopwords import load_stopwords

_TEXT = text(min_size=1, max_size=40, alphabet="abcdefghijklmnopqrstuvwxyz ")


def _ctx() -> ScorerContext:
    return ScorerContext(
        language="eng",
        stopwords=load_stopwords("eng"),
        stemmer=stemmer_for("eng"),
        idf=IdfTable(
            document_count=10,
            default_idf=2.0,
            source_hash="h",
            language="eng",
            idf={},
        ),
        config=MatchingConfig(
            title_weight=0.40,
            author_weight=0.20,
            publisher_weight=0.10,
            year_weight=0.10,
            edition_weight=0.05,
            lccn_weight=0.10,
            isbn_weight=0.05,
            extent_weight=0.0,
            volume_weight=0.0,
            year_window=2,
            min_combined_score=70.0,
        ),
    )


@given(_TEXT)
def test_title_identity_at_max(value: str) -> None:
    """The title scorer's identity must be at most its max score."""
    ev = score_title(value, value, _ctx())
    assert ev.score <= ev.max
    if not ev.skipped:
        assert ev.score == ev.max


@given(_TEXT)
def test_author_identity_at_max(value: str) -> None:
    """The author scorer's identity invariant."""
    ev = score_author(value, value, _ctx())
    assert ev.score <= ev.max
    if not ev.skipped:
        assert ev.score == ev.max


@given(_TEXT)
def test_publisher_identity_at_max(value: str) -> None:
    """The publisher scorer's identity invariant."""
    ev = score_publisher(value, value, _ctx())
    assert ev.score <= ev.max
    if not ev.skipped:
        assert ev.score == ev.max


@given(integers(min_value=1450, max_value=2050))
def test_year_identity_at_max(year: int) -> None:
    """Year identity always scores max."""
    ev = score_year(year, year, _ctx())
    assert ev.score == ev.max == 100.0


@given(integers(min_value=1450, max_value=2050), integers(min_value=0, max_value=50))
def test_year_score_within_bounds(year: int, delta: int) -> None:
    """Year score always lies in ``[0, max]`` regardless of delta."""
    ev = score_year(year, year + delta, _ctx())
    assert 0.0 <= ev.score <= ev.max


@given(_TEXT)
def test_edition_identity_at_max(value: str) -> None:
    """The edition scorer's identity invariant."""
    ev = score_edition(value, value, _ctx())
    if not ev.skipped:
        assert ev.score == ev.max
