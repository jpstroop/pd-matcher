"""msgspec Struct schemas for all configuration files.

These are frozen structs with ``forbid_unknown_fields=True``: they reject
unknown keys at load time and cannot be mutated after construction. Later
phases (matching, copyright rules, indexing) consume these as their single
source of truth. msgspec was chosen over pydantic because its strictly-typed
metaclass leaks no ``Any`` into our type checker, and it generates ``__slots__``
by default, matching the project's memory-efficiency policy.
"""

from pathlib import Path
from typing import Annotated
from typing import Literal

from msgspec import Meta
from msgspec import Struct
from msgspec import field

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


class CopyrightRule(Struct, frozen=True, forbid_unknown_fields=True):
    """One row of the Cornell public-domain matrix in declarative form.

    ``when`` predicates are stored as strings here; the Phase 5 rule engine
    parses and evaluates them. ``then`` likewise references a
    ``CopyrightStatus`` member by name and is resolved in Phase 5.
    """

    name: Annotated[str, Meta(min_length=1)]
    then: Annotated[str, Meta(min_length=1)]
    explanation: Annotated[str, Meta(min_length=1)]
    when: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject blank predicate strings inside the ``when`` list."""
        for index, predicate in enumerate(self.when):
            if not predicate.strip():
                raise ValueError(f"when[{index}] is empty or whitespace-only")


class CopyrightRuleSet(Struct, frozen=True, forbid_unknown_fields=True):
    """A versioned, ordered collection of :class:`CopyrightRule` entries."""

    version: Annotated[str, Meta(min_length=1)]
    rules: list[CopyrightRule] = field(default_factory=list)


class IndexConfig(Struct, frozen=True, forbid_unknown_fields=True):
    """LMDB index location and sizing parameters."""

    lmdb_path: Path
    map_size_bytes: Annotated[int, Meta(ge=1)] = _DEFAULT_LMDB_MAP_SIZE_BYTES
    schema_version: Annotated[int, Meta(ge=1)] = 1


__all__ = [
    "CopyrightRule",
    "CopyrightRuleSet",
    "IndexConfig",
    "MatchingConfig",
]
