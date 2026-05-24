"""Unit tests for the per-field annotation vocabulary."""

from pd_groundtruth.review.field_annotations import ALL_JUDGMENTS
from pd_groundtruth.review.field_annotations import ANNOTATABLE_FIELDS
from pd_groundtruth.review.field_annotations import JUDGMENT_CORRECT
from pd_groundtruth.review.field_annotations import JUDGMENT_NA
from pd_groundtruth.review.field_annotations import JUDGMENT_OVERSCORED
from pd_groundtruth.review.field_annotations import JUDGMENT_UNDERSCORED
from pd_groundtruth.review.field_annotations import FieldAnnotation
from pd_groundtruth.review.field_annotations import field_index
from pd_groundtruth.review.field_annotations import field_label
from pd_groundtruth.review.field_annotations import judgment_label
from pd_groundtruth.review.field_annotations import judgment_symbol
from pd_groundtruth.review.field_annotations import normalize_annotations
from pd_groundtruth.review.field_annotations import summarize_field_annotations


def test_annotatable_fields_and_judgments_are_fixed() -> None:
    assert ANNOTATABLE_FIELDS == ("title", "author", "publisher", "year", "edition")
    assert ALL_JUDGMENTS == (
        JUDGMENT_CORRECT,
        JUDGMENT_OVERSCORED,
        JUDGMENT_UNDERSCORED,
        JUDGMENT_NA,
    )


def test_normalize_annotations_orders_results_by_vocab_not_input() -> None:
    out = normalize_annotations(
        {
            "edition": JUDGMENT_NA,
            "title": JUDGMENT_CORRECT,
            "author": JUDGMENT_OVERSCORED,
        }
    )
    assert out == (
        FieldAnnotation(field="title", judgment=JUDGMENT_CORRECT),
        FieldAnnotation(field="author", judgment=JUDGMENT_OVERSCORED),
        FieldAnnotation(field="edition", judgment=JUDGMENT_NA),
    )


def test_normalize_annotations_drops_unknown_fields_and_judgments() -> None:
    out = normalize_annotations(
        {
            "title": JUDGMENT_CORRECT,
            "garbage": JUDGMENT_OVERSCORED,
            "author": "nonsense",
        }
    )
    assert out == (FieldAnnotation(field="title", judgment=JUDGMENT_CORRECT),)


def test_normalize_annotations_drops_blank_and_missing_values() -> None:
    out = normalize_annotations({"title": "", "author": JUDGMENT_CORRECT})
    assert out == (FieldAnnotation(field="author", judgment=JUDGMENT_CORRECT),)


def test_normalize_annotations_empty_input_returns_empty() -> None:
    assert normalize_annotations({}) == ()


def test_judgment_label_returns_human_readable() -> None:
    assert judgment_label(JUDGMENT_CORRECT) == "Correct"
    assert judgment_label(JUDGMENT_OVERSCORED) == "Overscored"
    assert judgment_label(JUDGMENT_UNDERSCORED) == "Underscored"
    assert judgment_label(JUDGMENT_NA) == "n/a"


def test_judgment_label_falls_back_to_raw_code_when_unknown() -> None:
    assert judgment_label("unknown_code") == "unknown_code"


def test_field_label_returns_human_readable() -> None:
    assert field_label("title") == "title"
    assert field_label("publisher") == "publisher"


def test_field_label_falls_back_to_raw_code_when_unknown() -> None:
    assert field_label("nonsense") == "nonsense"


def test_judgment_symbol_returns_compact_form() -> None:
    assert judgment_symbol(JUDGMENT_CORRECT) == "OK"
    assert judgment_symbol(JUDGMENT_OVERSCORED) == "over"
    assert judgment_symbol(JUDGMENT_UNDERSCORED) == "under"
    assert judgment_symbol(JUDGMENT_NA) == "n/a"


def test_judgment_symbol_falls_back_to_raw_code_when_unknown() -> None:
    assert judgment_symbol("mystery") == "mystery"


def test_field_index_returns_canonical_position() -> None:
    assert field_index("title") == 0
    assert field_index("edition") == 4


def test_summarize_field_annotations_orders_by_vocab_and_drops_empty() -> None:
    counts = {
        ("author", JUDGMENT_CORRECT): 5,
        ("author", JUDGMENT_OVERSCORED): 1,
        ("title", JUDGMENT_OVERSCORED): 3,
    }
    summary = summarize_field_annotations(counts)
    assert [(row.field, row.total) for row in summary] == [("title", 3), ("author", 6)]
    title_row = summary[0]
    assert title_row.counts == (0, 3, 0, 0)
    assert title_row.field_label == "title"
    author_row = summary[1]
    assert author_row.counts == (5, 1, 0, 0)


def test_summarize_field_annotations_empty_when_no_counts() -> None:
    assert summarize_field_annotations({}) == ()
