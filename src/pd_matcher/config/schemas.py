"""msgspec Struct schemas for all configuration files.

These are frozen structs with ``forbid_unknown_fields=True``: they reject
unknown keys at load time and cannot be mutated after construction. Later
phases (matching, indexing) consume these as their single source of truth.
msgspec was chosen over pydantic because its strictly-typed metaclass leaks
no ``Any`` into our type checker, and it generates ``__slots__`` by default,
matching the project's memory-efficiency policy.
"""

from pathlib import Path
from typing import Annotated
from typing import Literal

from msgspec import Meta
from msgspec import Struct

_WEIGHT_SUM_TOLERANCE: float = 1e-3
_DEFAULT_LMDB_MAP_SIZE_BYTES: int = 16 * 1024 * 1024 * 1024


class MatchingConfig(Struct, frozen=True, forbid_unknown_fields=True):
    """Per-field weights and thresholds used by the scoring pipeline.

    The seven field weights (``title_weight``, ``author_weight``,
    ``publisher_weight``, ``year_weight``, ``edition_weight``,
    ``lccn_weight``, ``isbn_weight``) must sum to ``1.0`` within
    :data:`_WEIGHT_SUM_TOLERANCE`. Identifier scorers (LCCN, ISBN) are
    weighted alongside the heuristic scorers rather than short-circuiting
    the combiner: in this corpus, transcription/OCR errors give standard
    identifiers a non-trivial false-positive rate, so the Platt calibrator
    learns the empirical ``P(true match)`` for the resulting raw scores.
    """

    title_weight: Annotated[float, Meta(ge=0.0, le=1.0)]
    author_weight: Annotated[float, Meta(ge=0.0, le=1.0)]
    publisher_weight: Annotated[float, Meta(ge=0.0, le=1.0)]
    year_weight: Annotated[float, Meta(ge=0.0, le=1.0)]
    edition_weight: Annotated[float, Meta(ge=0.0, le=1.0)]
    lccn_weight: Annotated[float, Meta(ge=0.0, le=1.0)]
    isbn_weight: Annotated[float, Meta(ge=0.0, le=1.0)]
    year_window: Annotated[int, Meta(ge=0)]
    min_combined_score: Annotated[float, Meta(ge=0.0, le=100.0)]
    scorer: Literal["weighted_mean", "learned"] = "weighted_mean"

    def __post_init__(self) -> None:
        """Reject weight tuples that do not sum to 1.0 within tolerance."""
        total = (
            self.title_weight
            + self.author_weight
            + self.publisher_weight
            + self.year_weight
            + self.edition_weight
            + self.lccn_weight
            + self.isbn_weight
        )
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                "title_weight + author_weight + publisher_weight + year_weight + "
                "edition_weight + lccn_weight + isbn_weight must sum to 1.0 "
                f"(got {total!r})"
            )


class IndexConfig(Struct, frozen=True, forbid_unknown_fields=True):
    """LMDB index location and sizing parameters."""

    lmdb_path: Path
    map_size_bytes: Annotated[int, Meta(ge=1)] = _DEFAULT_LMDB_MAP_SIZE_BYTES
    schema_version: Annotated[int, Meta(ge=1)] = 3


class FieldSpec(Struct, frozen=True, forbid_unknown_fields=True):
    """A named composition of one or more raw record subfields.

    ``fields`` lists raw-field-registry names (resolved at compile time
    against :data:`pd_matcher.match.pairing_compiler.MARC_FIELDS` or
    ``CCE_FIELDS`` depending on the section). ``combine`` selects the
    closed-vocabulary operation applied to the concatenated raw values:

    * ``first`` — the first non-empty value, else ``None``.
    * ``concat`` / ``join`` — every non-empty value joined by
      ``separator`` (synonyms; both drop empties and yield ``None`` when
      all values are empty).

    The vocabulary is deliberately finite so configuration can compose
    already-extracted subfields without expressing arbitrary logic.
    """

    fields: tuple[str, ...]
    combine: Literal["first", "concat", "join"]
    separator: str = " "


class PairingSpec(Struct, frozen=True, forbid_unknown_fields=True):
    """One ``(MARC field, CCE field)`` pairing routed to a scorer group.

    ``group`` selects the scorer family (``title``, ``author``, or
    ``publisher``) and therefore the combiner weight bucket. ``marc`` and
    ``cce`` name :class:`FieldSpec` entries in the enclosing
    :class:`PairingConfig`'s ``marc_fields`` / ``cce_fields`` maps.
    """

    group: Literal["title", "author", "publisher"]
    marc: str
    cce: str


class PairingConfig(Struct, frozen=True, forbid_unknown_fields=True):
    """The full configurable field-pairing specification.

    ``marc_fields`` and ``cce_fields`` define named compositions of raw
    subfields; ``pairings`` lists which composed MARC field is compared
    against which composed CCE field, and under which scorer group. Every
    name is validated at compile time
    (:func:`pd_matcher.match.pairing_compiler.compile_pairings`).
    """

    marc_fields: dict[str, FieldSpec]
    cce_fields: dict[str, FieldSpec]
    pairings: tuple[PairingSpec, ...]


__all__ = [
    "FieldSpec",
    "IndexConfig",
    "MatchingConfig",
    "PairingConfig",
    "PairingSpec",
]
