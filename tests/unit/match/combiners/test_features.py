"""Tests for :mod:`pd_matcher.match.combiners.features`."""

from pd_matcher.match.combiners.features import SCORER_ORDER
from pd_matcher.match.combiners.features import feature_names
from pd_matcher.match.combiners.features import feature_row
from pd_matcher.match.evidence import Evidence

_EXPECTED_FEATURE_COUNT: int = 49


def _evidence(
    scorer: str,
    *,
    score: float = 0.0,
    max_score: float = 100.0,
    skipped: bool = False,
    features: tuple[tuple[str, float], ...] = (),
) -> Evidence:
    """Build one Evidence with sensible defaults for the projection tests."""
    return Evidence(
        scorer=scorer,
        score=score,
        max=max_score,
        skipped=skipped,
        decisive=False,
        features=features,
    )


def test_feature_names_length_matches_expected_count() -> None:
    """The canonical builder yields exactly 49 columns."""
    assert len(feature_names()) == _EXPECTED_FEATURE_COUNT


def test_year_is_not_a_combiner_feature() -> None:
    """Year was dropped as a scoring feature in issue #88.

    Exact-year retrieval bucketing makes ``year.delta`` a constant, so it is
    no longer a scorer column nor a named sub-feature. No feature name may
    reference year, and a year Evidence passed to ``feature_row`` is ignored.
    """
    names = feature_names()
    assert "year.delta" not in SCORER_ORDER
    assert not any("year" in name for name in names)
    with_year = feature_row((_evidence("year.delta", score=100.0),))
    assert set(with_year) == {0.0}


def test_feature_names_are_unique() -> None:
    """No duplicate column names (the scorer-name prefix disambiguates)."""
    names = feature_names()
    assert len(set(names)) == len(names)


def test_feature_names_deterministic_order() -> None:
    """Two calls return the identical ordered tuple."""
    assert feature_names() == feature_names()


def test_feature_names_canonical_layout() -> None:
    """Scores first, then skipped flags, then sub-features, then the pair ratio."""
    names = feature_names()
    n_scorers = len(SCORER_ORDER)
    assert names[:n_scorers] == SCORER_ORDER
    assert names[n_scorers : 2 * n_scorers] == tuple(
        f"{scorer}__skipped" for scorer in SCORER_ORDER
    )
    assert names[-1] == "pair.title_len_ratio"
    assert "name.author.token_overlap" in names
    assert "name.publisher.token_overlap" in names


def test_feature_row_length_agrees_with_names() -> None:
    """A populated row has the same length as the name tuple."""
    evidence = tuple(_evidence(scorer) for scorer in SCORER_ORDER)
    assert len(feature_row(evidence)) == len(feature_names())


def test_feature_row_empty_evidence_is_all_zeros() -> None:
    """Absent scorers project to a full-length zero vector."""
    row = feature_row(())
    assert len(row) == _EXPECTED_FEATURE_COUNT
    assert set(row) == {0.0}


def test_feature_row_skipped_flag_set() -> None:
    """A skipped scorer flips its ``__skipped`` flag and zeroes its score."""
    names = feature_names()
    evidence = (_evidence("title.token_set", score=80.0, skipped=True),)
    row = feature_row(evidence)
    score_index = names.index("title.token_set")
    flag_index = names.index("title.token_set__skipped")
    assert row[score_index] == 0.0
    assert row[flag_index] == 1.0


def test_feature_row_present_scorer_carries_normalized_score() -> None:
    """A present scorer contributes its normalized score and a 0 skipped flag."""
    names = feature_names()
    evidence = (_evidence("title.token_set", score=50.0, skipped=False),)
    row = feature_row(evidence)
    assert row[names.index("title.token_set")] == 0.5
    assert row[names.index("title.token_set__skipped")] == 0.0


def test_feature_row_named_subfeature_projected() -> None:
    """A named sub-feature value lands in its namespaced column."""
    names = feature_names()
    evidence = (
        _evidence(
            "name.author",
            score=10.0,
            features=(("token_overlap", 3.0),),
        ),
    )
    row = feature_row(evidence)
    assert row[names.index("name.author.token_overlap")] == 3.0


def test_feature_row_absent_subfeature_is_zero() -> None:
    """A scorer that omits a named sub-feature yields 0.0 for that column."""
    names = feature_names()
    evidence = (_evidence("name.author", score=10.0, features=()),)
    row = feature_row(evidence)
    assert row[names.index("name.author.normalized_marc_len")] == 0.0


def test_feature_row_presence_flag_set_when_value_present() -> None:
    """A present-flagged sub-feature with a value sets its ``__present`` flag."""
    names = feature_names()
    evidence = (
        _evidence(
            "extent.page_count",
            score=10.0,
            skipped=False,
            features=(("marc_pages", 200.0),),
        ),
    )
    row = feature_row(evidence)
    assert row[names.index("extent.page_count.marc_pages")] == 200.0
    assert row[names.index("extent.page_count.marc_pages__present")] == 1.0


def test_feature_row_presence_flag_clear_when_skipped() -> None:
    """A skipped present-flagged scorer clears the ``__present`` flag."""
    names = feature_names()
    evidence = (_evidence("extent.page_count", skipped=True, features=()),)
    row = feature_row(evidence)
    assert row[names.index("extent.page_count.marc_pages__present")] == 0.0


def test_title_len_ratio_derived_from_token_lengths() -> None:
    """The pair ratio divides the two title token-length sub-features."""
    names = feature_names()
    evidence = (
        _evidence(
            "title.token_set",
            score=80.0,
            features=(("marc_token_len", 6.0), ("nypl_token_len", 3.0)),
        ),
    )
    row = feature_row(evidence)
    assert row[names.index("pair.title_len_ratio")] == 2.0


def test_title_len_ratio_zero_when_denominator_zero() -> None:
    """A zero NYPL token length yields a 0.0 ratio rather than dividing."""
    names = feature_names()
    evidence = (
        _evidence(
            "title.token_set",
            features=(("marc_token_len", 6.0), ("nypl_token_len", 0.0)),
        ),
    )
    row = feature_row(evidence)
    assert row[names.index("pair.title_len_ratio")] == 0.0


def test_title_len_ratio_zero_when_title_absent() -> None:
    """No title Evidence at all yields a 0.0 ratio."""
    names = feature_names()
    row = feature_row((_evidence("name.author"),))
    assert row[names.index("pair.title_len_ratio")] == 0.0


def test_feature_row_full_evidence_happy_path() -> None:
    """A full per-scorer Evidence set projects to a finite, correct-length row."""
    names = feature_names()
    evidence = (
        _evidence(
            "title.token_set",
            score=90.0,
            features=(
                ("token_overlap", 4.0),
                ("avg_token_idf", 7.5),
                ("marc_token_len", 5.0),
                ("nypl_token_len", 5.0),
            ),
        ),
        _evidence("name.author", score=70.0, features=(("token_overlap", 2.0),)),
        _evidence("name.publisher", score=60.0, features=(("token_overlap", 1.0),)),
        _evidence("edition.compat", skipped=True),
        _evidence("lccn.exact", skipped=True),
        _evidence("isbn.exact", skipped=True),
        _evidence(
            "extent.page_count",
            score=80.0,
            features=(("marc_pages", 200.0), ("cce_pages", 198.0), ("delta", 2.0)),
        ),
        _evidence("volume.compat", skipped=True),
    )
    row = feature_row(evidence)
    assert len(row) == len(names)
    assert row[names.index("title.token_set")] == 0.9
    assert row[names.index("name.author.token_overlap")] == 2.0
    assert row[names.index("name.publisher.token_overlap")] == 1.0
    assert row[names.index("pair.title_len_ratio")] == 1.0
    assert row[names.index("edition.compat__skipped")] == 1.0
