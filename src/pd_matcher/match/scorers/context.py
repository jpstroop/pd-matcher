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
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.script import dominant_script
from pd_matcher.normalize.stopwords import StopwordSet


class ScorerContext(Struct, frozen=True, forbid_unknown_fields=True):
    """Read-only handles that every scorer needs for a single MARC record.

    Attributes:
        language: 3-letter MARC language code resolved from the record;
            ``"eng"`` is used as the fallback for missing/unknown codes.
        stopwords: Per-field stopword sets for ``language``.
        stemmer: Single-token stemmer callable for ``language``.
        idf: Title IDF table built once over the NYPL corpus titles.
        author_idf: IDF table built once over the NYPL corpus
            ``author_name`` tokens; used by the author scorer to discount
            generic-word overlap relative to distinctive name tokens.
        publisher_idf: IDF table built once over the NYPL corpus
            ``publisher_names`` tokens; used by the publisher scorer to
            discount generic-word overlap ("university", "press") relative
            to distinctive house tokens ("knopf", "macmillan").
        config: The active :class:`MatchingConfig`.
        publisher_alias_index: Optional ``{normalized_name: human_canonical}``
            lookup used by the publisher scorer to lift the score on
            curated imprint / alias hits. The value is the human-readable
            canonical (e.g. ``"McGraw-Hill Book Company"``) so it can be
            surfaced verbatim in the review UI. ``None`` disables the
            alias path.
        cross_field_title_stems: Stemmed tokens drawn from the MARC fields
            the CCE routinely embeds *inside its title* string — the
            ``publisher``, ``publication_place``, and
            ``statement_of_responsibility`` (#90). The title scorer strips any
            of these from the CCE-title comparand (unless the stem also belongs
            to the genuine MARC title), so cross-field contamination no longer
            deflates the title score. Prepared with the same normalize/stem
            pipeline the title scorer uses, so the comparison is like-with-like.
            Empty by default, which disables the strip.
        marc_title_scripts: Per-record memo of
            :func:`pd_matcher.normalize.script.dominant_script` over each
            distinct MARC title string the title pairings produce. The
            script-mismatch guard is a pure function of the raw title, so the
            MARC side is resolved at most once per distinct title per record
            instead of for every candidate (it was the dominant
            ``unicodedata.name`` cost alongside the CCE side, now precomputed
            on :attr:`~pd_matcher.models.IndexedNyplRegRecord.title_script`).
            Use :meth:`marc_title_script` rather than reading the dict.
        normalized_numbers: Per-record memo of
            :func:`pd_matcher.normalize.numbers.normalize_numbers` keyed by raw
            string. ``normalize_numbers`` depends only on the string and the
            record's (fixed) :attr:`language`, so the MARC title / author /
            publisher — re-scored against every candidate — are normalized at
            most once per distinct string per record instead of once per
            candidate (it was the dominant per-candidate cost). Distinct CCE
            strings simply miss the cache at no correctness cost. Use
            :meth:`normalize_numbers` rather than reading the dict.
    """

    language: str
    stopwords: StopwordSet
    stemmer: Callable[[str], str]
    idf: IdfTable
    author_idf: IdfTable
    publisher_idf: IdfTable
    config: MatchingConfig
    publisher_alias_index: dict[str, str] | None = None
    cross_field_title_stems: frozenset[str] = frozenset()
    marc_title_scripts: dict[str, str | None] = {}
    normalized_numbers: dict[str, str] = {}

    def marc_title_script(self, marc_title: str) -> str | None:
        """Return the memoized dominant script of ``marc_title`` for this record.

        Computes :func:`dominant_script` on first sight of a given title
        string and caches it, so re-scoring the same MARC title against every
        candidate costs one script resolution per distinct title rather than
        one per candidate. The result is byte-identical to calling
        :func:`dominant_script` directly.
        """
        if marc_title not in self.marc_title_scripts:
            self.marc_title_scripts[marc_title] = dominant_script(marc_title)
        return self.marc_title_scripts[marc_title]

    def normalize_numbers(self, value: str) -> str:
        """Return the memoized number-normalized form of ``value``.

        Computes :func:`pd_matcher.normalize.numbers.normalize_numbers` on
        first sight of a given string under this record's :attr:`language`
        and caches it. Re-scoring the same MARC field against every candidate
        then costs one normalization per distinct string rather than one per
        candidate. The result is byte-identical to calling
        :func:`pd_matcher.normalize.numbers.normalize_numbers` directly.
        """
        cached = self.normalized_numbers.get(value)
        if cached is None:
            cached = normalize_numbers(value, self.language)
            self.normalized_numbers[value] = cached
        return cached


__all__ = [
    "ScorerContext",
]
