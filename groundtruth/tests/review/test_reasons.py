"""Unit tests for the controlled reason vocabulary."""

from pd_groundtruth.review.reasons import NO_MATCH_REASONS
from pd_groundtruth.review.reasons import UNSURE_REASONS
from pd_groundtruth.review.reasons import is_valid_reason
from pd_groundtruth.review.reasons import normalize_reason
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


def test_normalize_reason_drops_invalid_and_empty() -> None:
    assert normalize_reason(VERDICT_NO_MATCH, "diff_work") == "diff_work"
    assert normalize_reason(VERDICT_NO_MATCH, "nonsense") is None
    assert normalize_reason(VERDICT_NO_MATCH, "") is None
    assert normalize_reason(VERDICT_NO_MATCH, None) is None
    assert normalize_reason(VERDICT_MATCH, "diff_work") is None


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
