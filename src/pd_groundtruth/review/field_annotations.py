"""Per-field annotation vocabulary for scorer-tuning training signal.

Layered atop the verdict + reason chips, the field-annotation grid lets a
reviewer flag *why* the scorer was wrong on a specific MARC/CCE field
("title", "author", "publisher", "year", "edition"). Each annotation is a
``(field, judgment)`` pair drawn from this module's fixed vocabularies, so the
labels feed structured training signal to the future learned scorer
(jpstroop/pd-matcher#4) rather than living in free-text notes.

Annotations are *optional* — the labeler only marks the fields they actively
noticed a problem with — and orthogonal to the verdict: a ``match`` may still
carry annotations (e.g. the scorer underscored the author it ultimately got
right), and a ``no_match`` may carry none at all.
"""

from msgspec import Struct

ANNOTATABLE_FIELDS: tuple[str, ...] = ("title", "author", "publisher", "year", "edition")

JUDGMENT_CORRECT: str = "correct"
JUDGMENT_OVERSCORED: str = "overscored"
JUDGMENT_UNDERSCORED: str = "underscored"
JUDGMENT_NA: str = "n_a"

ALL_JUDGMENTS: tuple[str, ...] = (
    JUDGMENT_CORRECT,
    JUDGMENT_OVERSCORED,
    JUDGMENT_UNDERSCORED,
    JUDGMENT_NA,
)

_ALLOWED_FIELDS: frozenset[str] = frozenset(ANNOTATABLE_FIELDS)
_ALLOWED_JUDGMENTS: frozenset[str] = frozenset(ALL_JUDGMENTS)

_FIELD_INDEX: dict[str, int] = {field: index for index, field in enumerate(ANNOTATABLE_FIELDS)}

_JUDGMENT_LABELS: dict[str, str] = {
    JUDGMENT_CORRECT: "Correct",
    JUDGMENT_OVERSCORED: "Overscored",
    JUDGMENT_UNDERSCORED: "Underscored",
    JUDGMENT_NA: "n/a",
}

_FIELD_LABELS: dict[str, str] = {
    "title": "title",
    "author": "author",
    "publisher": "publisher",
    "year": "year",
    "edition": "edition",
}


class FieldAnnotation(Struct, frozen=True, forbid_unknown_fields=True):
    """One reviewer annotation: a field name and the scorer's judgment for it."""

    field: str
    judgment: str


def normalize_annotations(items: dict[str, str]) -> tuple[FieldAnnotation, ...]:
    """Filter ``items`` to allowed ``(field, judgment)`` pairs in vocabulary order.

    Pairs whose field or judgment is not in the controlled sets are dropped;
    the result is ordered by :data:`ANNOTATABLE_FIELDS` so storage is
    deterministic regardless of the order the reviewer clicked the buttons.
    Empty / blank judgments collapse to "no annotation for this field" and
    are not emitted.
    """
    out: list[FieldAnnotation] = []
    for field in ANNOTATABLE_FIELDS:
        judgment = items.get(field)
        if judgment is None or judgment == "":
            continue
        if field not in _ALLOWED_FIELDS or judgment not in _ALLOWED_JUDGMENTS:
            continue
        out.append(FieldAnnotation(field=field, judgment=judgment))
    return tuple(out)


def judgment_label(judgment: str) -> str:
    """Return the human-readable label for ``judgment`` or the raw code if unknown."""
    return _JUDGMENT_LABELS.get(judgment, judgment)


def field_label(field: str) -> str:
    """Return the human-readable label for ``field`` or the raw code if unknown."""
    return _FIELD_LABELS.get(field, field)


class FieldAnnotationSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """One render-ready row of the per-field annotation tally on ``/stats``."""

    field: str
    field_label: str
    counts: tuple[int, ...]
    total: int


def summarize_field_annotations(
    counts: dict[tuple[str, str], int],
) -> tuple[FieldAnnotationSummary, ...]:
    """Project ``(field, judgment) -> count`` into ordered per-field rows.

    Rows follow :data:`ANNOTATABLE_FIELDS` order; each row's ``counts`` tuple
    follows :data:`ALL_JUDGMENTS` order so the stats table can render the
    header once and let each row provide its cells in lockstep. Fields with
    zero total are dropped so the table shows only fields that carry signal.
    """
    rows: list[FieldAnnotationSummary] = []
    for field in ANNOTATABLE_FIELDS:
        per_judgment = tuple(counts.get((field, judgment), 0) for judgment in ALL_JUDGMENTS)
        total = sum(per_judgment)
        if total == 0:
            continue
        rows.append(
            FieldAnnotationSummary(
                field=field,
                field_label=field_label(field),
                counts=per_judgment,
                total=total,
            )
        )
    return tuple(rows)


_JUDGMENT_SYMBOLS: dict[str, str] = {
    JUDGMENT_CORRECT: "OK",
    JUDGMENT_OVERSCORED: "over",
    JUDGMENT_UNDERSCORED: "under",
    JUDGMENT_NA: "n/a",
}


def judgment_symbol(judgment: str) -> str:
    """Return the compact symbol used in the ``/labels`` table cell."""
    return _JUDGMENT_SYMBOLS.get(judgment, judgment)


def field_index(field: str) -> int:
    """Return the canonical render order index for ``field``.

    Used to sort annotations consistently across views; raises ``KeyError``
    when ``field`` is outside :data:`ANNOTATABLE_FIELDS` (a caller bug, since
    only normalized annotations should ever reach the renderer).
    """
    return _FIELD_INDEX[field]


__all__ = [
    "ALL_JUDGMENTS",
    "ANNOTATABLE_FIELDS",
    "JUDGMENT_CORRECT",
    "JUDGMENT_NA",
    "JUDGMENT_OVERSCORED",
    "JUDGMENT_UNDERSCORED",
    "FieldAnnotation",
    "FieldAnnotationSummary",
    "field_index",
    "field_label",
    "judgment_label",
    "judgment_symbol",
    "normalize_annotations",
    "summarize_field_annotations",
]
