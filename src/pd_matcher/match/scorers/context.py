"""Shared per-MARC-record context passed to every scorer.

Building one :class:`ScorerContext` per ``MarcRecord`` and then handing the
same instance to every scorer call keeps the per-candidate code path small
and allocation-free: the stopword set, the stemmer callable, and the IDF
table are resolved once for the record's language rather than once per
candidate.
"""

from collections.abc import Callable

from msgspec import Struct

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.idf import IdfTable
from pd_matcher.normalize.stopwords import StopwordSet


class ScorerContext(Struct, frozen=True, forbid_unknown_fields=True):
    """Read-only handles that every scorer needs for a single MARC record.

    Attributes:
        language: 3-letter MARC language code resolved from the record;
            ``"eng"`` is used as the fallback for missing/unknown codes.
        stopwords: Per-field stopword sets for ``language``.
        stemmer: Single-token stemmer callable for ``language``.
        idf: IDF table built once over the NYPL corpus.
        config: The active :class:`MatchingConfig`.
        publisher_alias_index: Optional ``{normalized_name: human_canonical}``
            lookup used by the publisher scorer to lift the score on
            curated imprint / alias hits. The value is the human-readable
            canonical (e.g. ``"McGraw-Hill Book Company"``) so it can be
            surfaced verbatim in the review UI. ``None`` disables the
            alias path.
    """

    language: str
    stopwords: StopwordSet
    stemmer: Callable[[str], str]
    idf: IdfTable
    config: MatchingConfig
    publisher_alias_index: dict[str, str] | None = None


__all__ = [
    "ScorerContext",
]
