"""Unit tests for active-learning selection (issue #81).

Covers the pure selection policy in isolation: per-language target resolution,
vault exclusion, in-scope filtering, language weighting, and deterministic
reservoir sampling — all over an in-memory per-language record source so no
pool on disk is touched.
"""

from collections.abc import Iterator

from pytest import raises

from pd_groundtruth.active_select import DEFAULT_LANGUAGE_WEIGHTS
from pd_groundtruth.active_select import LanguagePlan
from pd_groundtruth.active_select import RecordSource
from pd_groundtruth.active_select import resolve_language_targets
from pd_groundtruth.active_select import select_records
from pd_matcher.models import MarcRecord


def _marc(control_id: str, *, language: str = "eng", year: int | None = 1953) -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A Title",
        title_main="A Title",
        publication_year=year,
        language_code=language,
    )


def _source(records_by_lang: dict[str, list[MarcRecord]]) -> RecordSource:
    def source(language: str) -> Iterator[MarcRecord]:
        yield from records_by_lang.get(language, [])

    return source


def test_default_weights_are_english_heavy_and_sum_to_one() -> None:
    assert DEFAULT_LANGUAGE_WEIGHTS["eng"] == 0.70
    assert abs(sum(DEFAULT_LANGUAGE_WEIGHTS.values()) - 1.0) < 1e-9


def test_resolve_language_targets_rounds_and_floors() -> None:
    targets = resolve_language_targets(100, {"eng": 0.7, "fre": 0.3})
    assert targets == {"eng": 70, "fre": 30}


def test_resolve_language_targets_floors_tiny_positive_weight_to_one() -> None:
    targets = resolve_language_targets(10, {"eng": 0.99, "fre": 0.001})
    assert targets["fre"] == 1


def test_resolve_language_targets_zero_weight_yields_zero() -> None:
    targets = resolve_language_targets(100, {"eng": 1.0, "fre": 0.0})
    assert targets == {"eng": 100, "fre": 0}


def test_resolve_language_targets_rejects_non_positive_target() -> None:
    with raises(ValueError, match="target must be positive"):
        resolve_language_targets(0, {"eng": 1.0})


def test_resolve_language_targets_rejects_empty_weights() -> None:
    with raises(ValueError, match="weights must not be empty"):
        resolve_language_targets(10, {})


def test_resolve_language_targets_rejects_negative_weight() -> None:
    with raises(ValueError, match="weights must not be negative"):
        resolve_language_targets(10, {"eng": -0.5})


def test_select_excludes_vault_marcs() -> None:
    source = _source({"eng": [_marc("a"), _marc("b"), _marc("c")]})
    result = select_records(
        source=source,
        weights={"eng": 1.0},
        target=10,
        excluded_marc_ids=frozenset({"b"}),
        seed=1,
    )
    selected_ids = {record.control_id for record in result.records}
    assert "b" not in selected_ids
    assert selected_ids == {"a", "c"}
    assert result.excluded == 1


def test_select_drops_out_of_scope_records() -> None:
    source = _source({"eng": [_marc("a"), _marc("b", year=None)]})
    result = select_records(
        source=source,
        weights={"eng": 1.0},
        target=10,
        excluded_marc_ids=frozenset(),
        seed=1,
    )
    assert {record.control_id for record in result.records} == {"a"}
    assert result.out_of_scope == 1


def test_select_caps_at_language_target() -> None:
    source = _source({"eng": [_marc(str(i)) for i in range(20)]})
    result = select_records(
        source=source,
        weights={"eng": 1.0},
        target=5,
        excluded_marc_ids=frozenset(),
        seed=7,
    )
    assert len(result.records) == 5
    assert result.plans == (LanguagePlan(language="eng", target=5, selected=5),)


def test_select_is_deterministic_for_a_seed() -> None:
    source_records = {"eng": [_marc(str(i)) for i in range(20)]}
    first = select_records(
        source=_source(source_records),
        weights={"eng": 1.0},
        target=5,
        excluded_marc_ids=frozenset(),
        seed=7,
    )
    second = select_records(
        source=_source(source_records),
        weights={"eng": 1.0},
        target=5,
        excluded_marc_ids=frozenset(),
        seed=7,
    )
    assert [r.control_id for r in first.records] == [r.control_id for r in second.records]


def test_select_interleaves_languages_weighted() -> None:
    source = _source(
        {
            "eng": [_marc(f"e{i}", language="eng") for i in range(10)],
            "fre": [_marc(f"f{i}", language="fre") for i in range(10)],
        }
    )
    result = select_records(
        source=source,
        weights={"eng": 0.8, "fre": 0.2},
        target=10,
        excluded_marc_ids=frozenset(),
        seed=3,
    )
    by_lang = {plan.language: plan.selected for plan in result.plans}
    assert by_lang == {"eng": 8, "fre": 2}
    assert len(result.records) == 10


def test_select_skips_zero_weight_language_without_consuming_source() -> None:
    consumed: list[str] = []

    def source(language: str) -> Iterator[MarcRecord]:
        consumed.append(language)
        yield _marc("a", language=language)

    result = select_records(
        source=source,
        weights={"eng": 1.0, "fre": 0.0},
        target=5,
        excluded_marc_ids=frozenset(),
        seed=1,
    )
    assert "fre" not in consumed
    assert any(plan.language == "fre" and plan.target == 0 for plan in result.plans)


def test_select_reports_fewer_selected_than_target_when_pool_is_small() -> None:
    source = _source({"eng": [_marc("a"), _marc("b")]})
    result = select_records(
        source=source,
        weights={"eng": 1.0},
        target=10,
        excluded_marc_ids=frozenset(),
        seed=1,
    )
    assert result.plans == (LanguagePlan(language="eng", target=10, selected=2),)
