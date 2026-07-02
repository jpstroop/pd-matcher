"""Tests for :mod:`pd_matcher.match.combiners.weighted_mean`.

These tests pin down the post-walk-back behaviour of the combiner: an
exact-LCCN ``Evidence`` is **not** a short-circuit; it is just a heavily
weighted contributor like the other scorers. The combiner is a plain
weighted mean over present Evidence, end of story.
"""

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners.weighted_mean import _RENEWAL_CLAIMANT_WEIGHT
from pd_matcher.match.combiners.weighted_mean import _RENEWAL_NAME_WEIGHT
from pd_matcher.match.combiners.weighted_mean import _RENEWAL_OREG_WEIGHT
from pd_matcher.match.combiners.weighted_mean import _WHOLE_PART_PENALTY_CAP
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.evidence import Evidence


def _ev(
    scorer: str,
    score: float,
    *,
    skipped: bool = False,
    decisive: bool = False,
    weight_multiplier: float = 1.0,
) -> Evidence:
    return Evidence(
        scorer=scorer,
        score=score,
        max=100.0,
        skipped=skipped,
        decisive=decisive,
        features=(),
        weight_multiplier=weight_multiplier,
    )


def test_weighted_mean_combines_present_evidence(matching_config: MatchingConfig) -> None:
    """The combiner averages over present Evidence weighted by config."""
    combiner = WeightedMeanCombiner(config=matching_config)
    evidence = (
        _ev("title.token_set", 100.0),
        _ev("name.author", 100.0),
        _ev("name.publisher", 100.0),
        _ev("year.delta", 100.0),
        _ev("edition.compat", 100.0),
        _ev("lccn.exact", 100.0, decisive=True),
        _ev("isbn.exact", 100.0, decisive=True),
    )
    combined = combiner.combine(evidence)
    assert combined.raw == 100.0
    assert combined.calibrated == 1.0


def test_weighted_mean_partial_evidence(matching_config: MatchingConfig) -> None:
    """The mean of a single 50/100 Evidence (the rest skipped) is 50."""
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 50.0),
            _ev("name.author", 100.0, skipped=True),
        )
    )
    assert combined.raw == 50.0


def test_weighted_mean_lccn_decisive_does_not_short_circuit(
    matching_config: MatchingConfig,
) -> None:
    """An exact-LCCN Evidence at max is weighted, not a short-circuit.

    With title=0, lccn=100 and the default weights (title=0.40,
    lccn=0.10), the raw score is the weighted mean over those two present
    scorers: ``(0.40*0 + 0.10*1.0) / (0.40 + 0.10) * 100 = 20``. It is
    emphatically not 100.
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 0.0),
            _ev("lccn.exact", 100.0, decisive=True),
        )
    )
    expected = (
        (matching_config.lccn_weight * 1.0)
        / (matching_config.title_weight + matching_config.lccn_weight)
        * 100.0
    )
    assert combined.raw == expected
    assert combined.raw < 100.0
    assert combined.calibrated == combined.raw / 100.0


def test_weighted_mean_lccn_match_with_conflicting_title_is_moderate(
    matching_config: MatchingConfig,
) -> None:
    """A perfect LCCN match plus a zero-scoring title yields a moderate raw.

    The whole point of walking back the short-circuit: a transcription
    error in either side of the LCCN comparison would otherwise have
    promoted a clear mismatch to 100% confidence. Year Evidence is supplied
    at 100 to confirm it is now ignored (issue #88): it contributes to
    neither the numerator nor the denominator, so only LCCN carries the
    score.
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 0.0),
            _ev("name.author", 0.0),
            _ev("name.publisher", 0.0),
            _ev("year.delta", 100.0),
            _ev("edition.compat", 0.0),
            _ev("lccn.exact", 100.0, decisive=True),
        )
    )
    # Year is dropped as a combiner weight; only LCCN carries the score:
    # lccn_weight / (every weight except year).
    cfg = matching_config
    denom = (
        cfg.title_weight
        + cfg.author_weight
        + cfg.publisher_weight
        + cfg.edition_weight
        + cfg.lccn_weight
    )
    expected = (cfg.lccn_weight / denom) * 100.0
    assert combined.raw == expected
    assert combined.raw < 50.0


def test_weighted_mean_isbn_contributes_its_share(matching_config: MatchingConfig) -> None:
    """A perfect ISBN match contributes its 0.05 share, not 100%."""
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 0.0),
            _ev("isbn.exact", 100.0, decisive=True),
        )
    )
    expected = (
        (matching_config.isbn_weight * 1.0)
        / (matching_config.title_weight + matching_config.isbn_weight)
        * 100.0
    )
    assert combined.raw == expected
    assert combined.raw < 100.0


def test_weighted_mean_skipped_lccn_does_not_contribute(
    matching_config: MatchingConfig,
) -> None:
    """A skipped LCCN Evidence is excluded entirely from the mean."""
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 50.0),
            _ev("lccn.exact", 100.0, skipped=True, decisive=True),
        )
    )
    assert combined.raw == 50.0


def test_weighted_mean_all_skipped_returns_zero(matching_config: MatchingConfig) -> None:
    """If every Evidence is skipped the raw and calibrated scores are zero."""
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 0.0, skipped=True),
            _ev("name.author", 0.0, skipped=True),
        )
    )
    assert combined.raw == 0.0
    assert combined.calibrated == 0.0


def test_weighted_mean_ignores_unknown_scorer(matching_config: MatchingConfig) -> None:
    """Evidence from an unknown scorer is excluded from the mean."""
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 100.0),
            _ev("mystery.scorer", 0.0),
        )
    )
    assert combined.raw == 100.0


def test_weighted_mean_ignores_year_evidence(matching_config: MatchingConfig) -> None:
    """Year Evidence is excluded from the mean (issue #88).

    Year carries no weight in the combiner, so a perfect-scoring year
    Evidence neither raises nor lowers the combined score: the result is the
    title alone, exactly as if the year Evidence were absent.
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 40.0),
            _ev("year.delta", 100.0),
        )
    )
    assert combined.raw == 40.0


def test_weight_multiplier_default_one_preserves_old_behaviour(
    matching_config: MatchingConfig,
) -> None:
    """An Evidence with the default ``weight_multiplier=1.0`` is unchanged."""
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 100.0, weight_multiplier=1.0),
            _ev("name.author", 0.0, weight_multiplier=1.0),
        )
    )
    cfg = matching_config
    expected = (cfg.title_weight * 1.0) / (cfg.title_weight + cfg.author_weight) * 100.0
    assert combined.raw == expected


def test_weight_multiplier_halves_author_share(matching_config: MatchingConfig) -> None:
    """A ``weight_multiplier=0.5`` on the author halves its share of the mean.

    With title=100 (multiplier 1.0) and author=0 (multiplier 0.5), the
    effective weights are title_weight and 0.5*author_weight; the author
    contributes proportionally less to the denominator, so the combined
    score is pulled toward the title's 100 (relative to the unscaled case).
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 100.0),
            _ev("name.author", 0.0, weight_multiplier=0.5),
        )
    )
    cfg = matching_config
    effective_author = cfg.author_weight * 0.5
    expected = (cfg.title_weight * 1.0) / (cfg.title_weight + effective_author) * 100.0
    assert combined.raw == expected


def test_weight_multiplier_zero_drops_evidence(matching_config: MatchingConfig) -> None:
    """An effective weight of zero excludes the Evidence entirely."""
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 100.0),
            _ev("name.author", 0.0, weight_multiplier=0.0),
        )
    )
    # The author Evidence is dropped, so the mean is title alone -> 100.
    assert combined.raw == 100.0


def test_weighted_mean_incorporates_renewal_domain_scorers(
    matching_config: MatchingConfig,
) -> None:
    """The three renewal-domain scorers contribute with their fixed weights.

    A perfect title alone is 100; adding the three domain scorers all reading
    0.0 pulls the mean down by exactly their combined weight share, proving the
    combiner carries a weight for each of the three ``renewal.*`` scorers.
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    combined = combiner.combine(
        (
            _ev("title.token_set", 100.0),
            _ev("renewal.oreg_class", 0.0),
            _ev("renewal.claimant_class", 0.0),
            _ev("renewal.name_conditioned", 0.0),
        )
    )
    cfg = matching_config
    denom = (
        cfg.title_weight + _RENEWAL_OREG_WEIGHT + _RENEWAL_CLAIMANT_WEIGHT + _RENEWAL_NAME_WEIGHT
    )
    expected = (cfg.title_weight * 1.0) / denom * 100.0
    assert combined.raw == expected
    assert combined.raw < 100.0


def test_whole_part_penalty_caps_uncorroborated_incompatibility(
    matching_config: MatchingConfig,
) -> None:
    """An uncorroborated whole/part incompatibility caps the combined score.

    Title/author/publisher all agree (the whole/part pairs share them by
    nature) and ``volume.compat`` fired the incompatibility at 0.0 with no
    LCCN veto. Without the cap the lone 0.0 would be averaged away (volume
    carries zero weight in the default config); with it the calibrated score is
    pinned to the penalty ceiling and the raw tracks it.
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    evidence = (
        _ev("title.token_set", 64.0),
        _ev("name.author", 100.0),
        _ev("name.publisher", 100.0),
        _ev("volume.compat", 0.0),
    )
    uncapped = combiner.combine(
        (
            _ev("title.token_set", 64.0),
            _ev("name.author", 100.0),
            _ev("name.publisher", 100.0),
        )
    )
    assert uncapped.calibrated > _WHOLE_PART_PENALTY_CAP
    combined = combiner.combine(evidence)
    assert combined.calibrated == _WHOLE_PART_PENALTY_CAP
    assert combined.raw == _WHOLE_PART_PENALTY_CAP * 100.0


def test_whole_part_penalty_spared_by_exact_lccn(
    matching_config: MatchingConfig,
) -> None:
    """An exact LCCN vetoes the penalty: the corroborated pair is not capped.

    This is the guardrail protecting LCCN-confirmed true matches (e.g. reg
    A63607) that happen to carry a misleading ``volume.compat=0.0``.
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    evidence = (
        _ev("title.token_set", 64.0),
        _ev("name.author", 100.0),
        _ev("name.publisher", 100.0),
        _ev("volume.compat", 0.0),
        _ev("lccn.exact", 100.0, decisive=True),
    )
    combined = combiner.combine(evidence)
    assert combined.calibrated > _WHOLE_PART_PENALTY_CAP


def test_whole_part_penalty_not_applied_below_cap(
    matching_config: MatchingConfig,
) -> None:
    """A pair already below the cap is left untouched by the penalty.

    The penalty only lowers; it never raises a score up to the cap.
    """
    combiner = WeightedMeanCombiner(config=matching_config)
    evidence = (
        _ev("title.token_set", 10.0),
        _ev("name.author", 0.0),
        _ev("volume.compat", 0.0),
    )
    uncapped = combiner.combine(
        (
            _ev("title.token_set", 10.0),
            _ev("name.author", 0.0),
        )
    )
    assert uncapped.calibrated < _WHOLE_PART_PENALTY_CAP
    combined = combiner.combine(evidence)
    assert combined.calibrated == uncapped.calibrated


def test_whole_part_penalty_not_applied_when_volume_compatible(
    matching_config: MatchingConfig,
) -> None:
    """A compatible volume signal does not trigger the penalty cap."""
    combiner = WeightedMeanCombiner(config=matching_config)
    evidence = (
        _ev("title.token_set", 64.0),
        _ev("name.author", 100.0),
        _ev("name.publisher", 100.0),
        _ev("volume.compat", 100.0),
    )
    combined = combiner.combine(evidence)
    assert combined.calibrated > _WHOLE_PART_PENALTY_CAP
