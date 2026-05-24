"""Unit tests for the controlled reason vocabulary."""

from pd_groundtruth.review.reasons import NO_MATCH_REASONS
from pd_groundtruth.review.reasons import UNSURE_REASONS
from pd_groundtruth.review.reasons import is_valid_reason
from pd_groundtruth.review.reasons import normalize_reasons
from pd_groundtruth.review.reasons import reasons_for
from pd_groundtruth.review.reasons import summarize_reasons
from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.review_db import VERDICT_UNSURE


def test_reasons_for_returns_vocab_per_verdict() -> None:
    assert reasons_for(VERDICT_NO_MATCH) == NO_MATCH_REASONS
    assert reasons_for(VERDICT_UNSURE) == UNSURE_REASONS


def test_reasons_for_match_is_empty() -> None:
    assert reasons_for(VERDICT_MATCH) == ()


def test_is_valid_reason_checks_membership_per_verdict() -> None:
    assert is_valid_reason(VERDICT_NO_MATCH, "diff_work")
    assert not is_valid_reason(VERDICT_NO_MATCH, "insufficient_data")
    assert not is_valid_reason(VERDICT_MATCH, "diff_work")


def test_normalize_reasons_keeps_only_valid_codes() -> None:
    assert normalize_reasons(VERDICT_NO_MATCH, ["diff_work", "nonsense"]) == ("diff_work",)
    assert normalize_reasons(VERDICT_NO_MATCH, ["nonsense"]) == ()
    assert normalize_reasons(VERDICT_MATCH, ["diff_work"]) == ()


def test_normalize_reasons_empty_input_returns_empty() -> None:
    assert normalize_reasons(VERDICT_NO_MATCH, []) == ()


def test_normalize_reasons_dedupes_and_preserves_vocab_order() -> None:
    submitted = ["wrong_year_edition", "diff_work", "diff_work"]
    assert normalize_reasons(VERDICT_NO_MATCH, submitted) == ("diff_work", "wrong_year_edition")


def test_normalize_reasons_drops_cross_verdict_codes() -> None:
    submitted = ["edition_unsure", "diff_work"]
    assert normalize_reasons(VERDICT_UNSURE, submitted) == ("edition_unsure",)
    assert normalize_reasons(VERDICT_NO_MATCH, submitted) == ("diff_work",)


def test_edition_unsure_validates_per_verdict() -> None:
    assert is_valid_reason(VERDICT_UNSURE, "edition_unsure")
    assert not is_valid_reason(VERDICT_NO_MATCH, "edition_unsure")


def test_generic_title_valid_for_no_match_only() -> None:
    assert is_valid_reason(VERDICT_NO_MATCH, "generic_title")
    assert not is_valid_reason(VERDICT_UNSURE, "generic_title")


def test_translation_valid_for_both_no_match_and_unsure() -> None:
    assert is_valid_reason(VERDICT_NO_MATCH, "translation")
    assert is_valid_reason(VERDICT_UNSURE, "translation")


def test_new_unsure_chips_valid_for_unsure_only() -> None:
    for code in ("pub_differs", "reprint_or_format", "whole_or_part", "periodical_issue"):
        assert is_valid_reason(VERDICT_UNSURE, code), code
        assert not is_valid_reason(VERDICT_NO_MATCH, code), code


def test_summarize_reasons_orders_by_vocab_and_drops_zero() -> None:
    counts = {
        (VERDICT_NO_MATCH, "wrong_year_edition"): 3,
        (VERDICT_NO_MATCH, "diff_work"): 5,
        (VERDICT_UNSURE, "insufficient_data"): 2,
    }
    summary = summarize_reasons(counts)
    assert [(r.verdict, r.code, r.count) for r in summary] == [
        (VERDICT_NO_MATCH, "diff_work", 5),
        (VERDICT_NO_MATCH, "wrong_year_edition", 3),
        (VERDICT_UNSURE, "insufficient_data", 2),
    ]
    assert summary[0].label == "Different work / title collision"


def test_summarize_reasons_empty_when_no_counts() -> None:
    assert summarize_reasons({}) == ()


def test_summarize_reasons_keeps_no_match_before_unsure_with_new_codes() -> None:
    counts = {
        (VERDICT_UNSURE, "pub_differs"): 4,
        (VERDICT_NO_MATCH, "generic_title"): 2,
        (VERDICT_UNSURE, "translation"): 3,
        (VERDICT_NO_MATCH, "translation"): 1,
    }
    summary = summarize_reasons(counts)
    verdicts_in_order = [row.verdict for row in summary]
    no_match_count = verdicts_in_order.count(VERDICT_NO_MATCH)
    assert verdicts_in_order[:no_match_count] == [VERDICT_NO_MATCH] * no_match_count
    assert verdicts_in_order[no_match_count:] == [VERDICT_UNSURE] * (
        len(verdicts_in_order) - no_match_count
    )
    pairs = {(row.verdict, row.code) for row in summary}
    assert (VERDICT_NO_MATCH, "translation") in pairs
    assert (VERDICT_UNSURE, "translation") in pairs
    assert (VERDICT_NO_MATCH, "generic_title") in pairs
    assert (VERDICT_UNSURE, "pub_differs") in pairs
