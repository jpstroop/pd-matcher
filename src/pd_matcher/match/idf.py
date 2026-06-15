"""IDF (inverse document frequency) table builder and on-disk cache.

The title scorer in Phase 4 weights token overlap by per-token IDF so that
rare distinguishing tokens (``"Albuquerque"``) outweigh stopwords-adjacent
filler (``"American"``). Computing IDF requires one full scan of the NYPL
corpus's titles; the result is small (one ``dict[str, float]`` keyed on
stems) and is persisted via :mod:`msgspec.msgpack` so subsequent matcher
runs reuse it without re-scanning the LMDB env.

The cache file embeds the source hash recorded in the index's ``meta``
sub-DB; if either changes (rebuilt index or upstream source mutation) the
table is rebuilt automatically.
"""

from collections.abc import Callable
from math import log
from pathlib import Path

from msgspec import Struct
from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder

from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.stemming import stem_tokens
from pd_matcher.normalize.stopwords import load_stopwords
from pd_matcher.normalize.text import tokenize


class IdfTable(Struct, frozen=True, forbid_unknown_fields=True):
    """Cached IDF lookup with a default for unseen tokens."""

    document_count: int
    default_idf: float
    source_hash: str
    language: str
    idf: dict[str, float]

    def score(self, token: str) -> float:
        """Return the IDF score for ``token`` (default for unknowns)."""
        return self.idf.get(token, self.default_idf)


_ENCODER: Encoder = Encoder()
_DECODER: Decoder[IdfTable] = Decoder(IdfTable)


def _prepare_tokens(
    title: str,
    *,
    language: str,
    title_stopwords: frozenset[str],
) -> tuple[str, ...]:
    """Run the title normalization pipeline used by both build and score."""
    normalized = normalize_numbers(title, language)
    tokens = tokenize(normalized)
    filtered = tuple(token for token in tokens if token not in title_stopwords)
    return stem_tokens(filtered, language)


def _prepare_name_tokens(
    value: str,
    *,
    language: str,
    name_stopwords: frozenset[str],
) -> tuple[str, ...]:
    """Run the name normalization pipeline used by both build and score.

    Mirrors :func:`pd_matcher.match.scorers.name._prepare`: normalize
    numbers, tokenize, drop the field's stopwords. Names are not stemmed —
    the name scorer compares raw normalized tokens via rapidfuzz — so the
    per-field IDF statistics must be computed over the same unstemmed tokens.
    """
    normalized = normalize_numbers(value, language)
    tokens = tokenize(normalized)
    return tuple(token for token in tokens if token not in name_stopwords)


def _build_name_idf_table(
    lookup: NyplIndexLookup,
    *,
    language: str,
    name_stopwords: frozenset[str],
    values_of: Callable[[IndexedNyplRegRecord], tuple[str, ...]],
) -> IdfTable:
    """Scan the corpus once and return an :class:`IdfTable` over name tokens.

    ``values_of`` extracts the field's string values from a registration
    (one for the author, possibly several for publishers). Each registration
    is one document; a token's document frequency is the count of records in
    which it appears at least once across that record's values.
    """
    document_count = 0
    df: dict[str, int] = {}
    for record in lookup.iter_registrations():
        document_count += 1
        record_tokens: set[str] = set()
        for value in values_of(record):
            record_tokens.update(
                _prepare_name_tokens(
                    value,
                    language=language,
                    name_stopwords=name_stopwords,
                )
            )
        for token in record_tokens:
            df[token] = df.get(token, 0) + 1
    idf: dict[str, float] = {
        token: log((document_count + 1) / (count + 1)) + 1.0 for token, count in df.items()
    }
    default_idf = log((document_count + 1) / 1) + 1.0
    source_hash = lookup.stats().source_hash
    return IdfTable(
        document_count=document_count,
        default_idf=default_idf,
        source_hash=source_hash,
        language=language,
        idf=idf,
    )


def _author_values(record: IndexedNyplRegRecord) -> tuple[str, ...]:
    """Return the author name strings carried by a registration record."""
    return (record.author_name,) if record.author_name else ()


def _publisher_values(record: IndexedNyplRegRecord) -> tuple[str, ...]:
    """Return the publisher name strings carried by a registration record."""
    return record.publisher_names


def build_author_idf_table(lookup: NyplIndexLookup, *, language: str = "eng") -> IdfTable:
    """Scan the corpus and return an :class:`IdfTable` over CCE author tokens.

    Mirrors :func:`build_idf_table` but ranges over each registration's
    ``author_name`` using the language's *author* stopword set and no
    stemming, so the table lines up with the author scorer's token pipeline.
    """
    stopwords = load_stopwords(language)
    return _build_name_idf_table(
        lookup,
        language=language,
        name_stopwords=stopwords.author,
        values_of=_author_values,
    )


def build_publisher_idf_table(lookup: NyplIndexLookup, *, language: str = "eng") -> IdfTable:
    """Scan the corpus and return an :class:`IdfTable` over CCE publisher tokens.

    Mirrors :func:`build_idf_table` but ranges over each registration's
    ``publisher_names`` using the language's *publisher* stopword set and no
    stemming, so the table lines up with the publisher scorer's token
    pipeline. The IDF mass of a publisher's shared tokens is what lets the
    scorer discount generic-word overlap ("university", "press") relative to
    distinctive house tokens ("knopf", "macmillan").
    """
    stopwords = load_stopwords(language)
    return _build_name_idf_table(
        lookup,
        language=language,
        name_stopwords=stopwords.publisher,
        values_of=_publisher_values,
    )


def build_idf_table(lookup: NyplIndexLookup, *, language: str = "eng") -> IdfTable:
    """Scan the entire NYPL corpus and return an :class:`IdfTable`.

    Args:
        lookup: Open :class:`NyplIndexLookup` over the LMDB env.
        language: Language whose stopwords/stemmer drive tokenization. The
            IDF table is single-language by design — Phase 4's title scorer
            tokenizes both sides through the same pipeline so the per-token
            statistics line up.

    Returns:
        A fully populated :class:`IdfTable` whose ``source_hash`` matches
        the index's current build.
    """
    stopwords = load_stopwords(language)
    document_count = 0
    df: dict[str, int] = {}
    for record in lookup.iter_registrations():
        document_count += 1
        tokens = _prepare_tokens(
            record.title,
            language=language,
            title_stopwords=stopwords.title,
        )
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    idf: dict[str, float] = {
        token: log((document_count + 1) / (count + 1)) + 1.0 for token, count in df.items()
    }
    default_idf = log((document_count + 1) / 1) + 1.0
    source_hash = lookup.stats().source_hash
    return IdfTable(
        document_count=document_count,
        default_idf=default_idf,
        source_hash=source_hash,
        language=language,
        idf=idf,
    )


def save_idf_table(table: IdfTable, path: Path) -> None:
    """Serialize ``table`` to ``path`` via msgspec msgpack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ENCODER.encode(table))


def load_idf_table(path: Path) -> IdfTable:
    """Deserialize an :class:`IdfTable` previously persisted via :func:`save_idf_table`."""
    return _DECODER.decode(path.read_bytes())


def _load_or_build(
    cache_path: Path,
    lookup_factory: Callable[[], NyplIndexLookup],
    builder: Callable[[NyplIndexLookup], IdfTable],
    *,
    language: str,
) -> IdfTable:
    """Return a cached :class:`IdfTable`, rebuilding when the source drifts.

    Shared cache spine for every IDF flavour (title, author, publisher): the
    only thing that varies is ``builder``, the field-specific table
    constructor. The cache is keyed by ``cache_path``, so each flavour must
    pass a distinct path or it will collide with another's table.
    """
    if cache_path.exists():
        cached = load_idf_table(cache_path)
        with lookup_factory() as lookup:
            current_hash = lookup.stats().source_hash
        if cached.source_hash == current_hash and cached.language == language:
            return cached
    with lookup_factory() as lookup:
        table = builder(lookup)
    save_idf_table(table, cache_path)
    return table


def load_or_build_idf(
    cache_path: Path,
    lookup_factory: Callable[[], NyplIndexLookup],
    *,
    language: str = "eng",
) -> IdfTable:
    """Return a cached title :class:`IdfTable`, rebuilding on source drift.

    Args:
        cache_path: Filesystem location of the msgpack-encoded cache.
        lookup_factory: Zero-arg callable returning a fresh
            :class:`NyplIndexLookup`. The callable is invoked at most once —
            only when a (re)build is required — and the resulting lookup is
            closed before the function returns.
        language: Language whose stopwords/stemmer drive tokenization.
    """
    return _load_or_build(
        cache_path,
        lookup_factory,
        lambda lookup: build_idf_table(lookup, language=language),
        language=language,
    )


def load_or_build_author_idf(
    cache_path: Path,
    lookup_factory: Callable[[], NyplIndexLookup],
    *,
    language: str = "eng",
) -> IdfTable:
    """Return a cached author :class:`IdfTable`, rebuilding on source drift.

    Same caching contract as :func:`load_or_build_idf` but over CCE
    ``author_name`` tokens (see :func:`build_author_idf_table`).
    """
    return _load_or_build(
        cache_path,
        lookup_factory,
        lambda lookup: build_author_idf_table(lookup, language=language),
        language=language,
    )


def load_or_build_publisher_idf(
    cache_path: Path,
    lookup_factory: Callable[[], NyplIndexLookup],
    *,
    language: str = "eng",
) -> IdfTable:
    """Return a cached publisher :class:`IdfTable`, rebuilding on source drift.

    Same caching contract as :func:`load_or_build_idf` but over CCE
    ``publisher_names`` tokens (see :func:`build_publisher_idf_table`).
    """
    return _load_or_build(
        cache_path,
        lookup_factory,
        lambda lookup: build_publisher_idf_table(lookup, language=language),
        language=language,
    )


__all__ = [
    "IdfTable",
    "build_author_idf_table",
    "build_idf_table",
    "build_publisher_idf_table",
    "load_idf_table",
    "load_or_build_author_idf",
    "load_or_build_idf",
    "load_or_build_publisher_idf",
    "save_idf_table",
]
