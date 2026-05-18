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


def load_or_build_idf(
    cache_path: Path,
    lookup_factory: Callable[[], NyplIndexLookup],
    *,
    language: str = "eng",
) -> IdfTable:
    """Return a cached :class:`IdfTable`, rebuilding when source hash drifts.

    Args:
        cache_path: Filesystem location of the msgpack-encoded cache.
        lookup_factory: Zero-arg callable returning a fresh
            :class:`NyplIndexLookup`. The callable is invoked at most once —
            only when a (re)build is required — and the resulting lookup is
            closed before the function returns.
        language: Language whose stopwords/stemmer drive tokenization.
    """
    if cache_path.exists():
        cached = load_idf_table(cache_path)
        with lookup_factory() as lookup:
            current_hash = lookup.stats().source_hash
        if cached.source_hash == current_hash and cached.language == language:
            return cached
    with lookup_factory() as lookup:
        table = build_idf_table(lookup, language=language)
    save_idf_table(table, cache_path)
    return table


__all__ = [
    "IdfTable",
    "build_idf_table",
    "load_idf_table",
    "load_or_build_idf",
    "save_idf_table",
]
