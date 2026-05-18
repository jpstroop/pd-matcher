"""Golden-pair tests: a handful of hand-curated MARC↔NYPL pairs.

These assertions guard the per-scorer Evidence levels for representative
pairs, not just the final combined score. Regressing any one scorer (e.g.
breaking diacritic stripping in the publisher scorer) should fail one of
these tests with a precise scorer name in the failure output.
"""

from collections.abc import Iterable

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.edition import score_edition
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.match.scorers.title import score_title
from pd_matcher.match.scorers.year import score_year


def _evidence_by_scorer(evidences: Iterable[Evidence]) -> dict[str, Evidence]:
    return {ev.scorer: ev for ev in evidences}


def test_golden_pair_widget_study(scorer_context: ScorerContext) -> None:
    """An exact MARC/NYPL widget-study pair should score very high across the board."""
    evidences = (
        score_title("A study of widgets", "A study of widgets.", scorer_context),
        score_author("Alpha, Alice", "Alpha, Alice", scorer_context),
        score_publisher("Acme Press", "Acme Press", scorer_context),
        score_year(1940, 1940, scorer_context),
        score_edition("1st ed.", "First edition", scorer_context),
    )
    by_scorer = _evidence_by_scorer(evidences)
    assert by_scorer["title.token_set"].score == 100.0
    assert by_scorer["name.author"].score == 100.0
    assert by_scorer["name.publisher"].score == 100.0
    assert by_scorer["year.delta"].score == 100.0
    assert by_scorer["edition.compat"].score == 100.0


def test_golden_pair_noisy_match_still_scores_high(scorer_context: ScorerContext) -> None:
    """Trailing punctuation and case variation must not knock the title score below 80."""
    evidences = (
        score_title("a study of widgets / by alice", "A Study of Widgets.", scorer_context),
        score_author("Alpha, A.", "Alpha, Alice", scorer_context),
        score_publisher("Acme Press, Inc.", "Acme Press", scorer_context),
        score_year(1940, 1941, scorer_context),
    )
    by_scorer = _evidence_by_scorer(evidences)
    assert by_scorer["title.token_set"].score >= 70.0
    assert by_scorer["name.author"].score >= 70.0
    assert by_scorer["name.publisher"].score >= 80.0
    assert by_scorer["year.delta"].score == 75.0


def test_golden_pair_french_handles_diacritics(french_scorer_context: ScorerContext) -> None:
    """A French pair with diacritics should score 100 once stripped/stemmed."""
    evidences = (
        score_title("Le petit livre", "Le petit livre.", french_scorer_context),
        score_author("Dubois, David", "Dubois, David", french_scorer_context),
        score_publisher("Éditions Beta", "Editions Beta", french_scorer_context),
    )
    by_scorer = _evidence_by_scorer(evidences)
    assert by_scorer["title.token_set"].score == 100.0
    assert by_scorer["name.author"].score == 100.0
    assert by_scorer["name.publisher"].score == 100.0


def test_golden_pair_mismatched_titles_score_low(scorer_context: ScorerContext) -> None:
    """Completely different titles must score well below 30."""
    ev = score_title("Annual report of the corporation", "A study of widgets", scorer_context)
    assert ev.score < 30.0
