"""Shared blob-matcher primitives: field extraction, tokenization, IDF, scoring.

MEASUREMENT ONLY. Nothing here is shipped. Reuses the project's own
normalization primitives verbatim (normalize_numbers / tokenize / stem_tokens /
title stopwords) so a blob token lines up with the production title-IDF token
pipeline (number-normalized, stemmed, title-stopwords dropped).
"""

from math import log

from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.stemming import stem_tokens
from pd_matcher.normalize.stopwords import load_stopwords
from pd_matcher.normalize.text import tokenize

_LANG = "eng"
_STOP = load_stopwords(_LANG).title


def _marc_text_values(marc: MarcRecord) -> list[str]:
    """Every descriptive-text string a MarcRecord exposes (identifiers/dates excluded)."""
    vals: list[str] = [marc.title, marc.title_main]
    for opt in (
        marc.title_part_number,
        marc.title_part_name,
        marc.main_author,
        marc.statement_of_responsibility,
        marc.edition,
        marc.publication_place,
        marc.publisher,
        marc.extent,
    ):
        if opt:
            vals.append(opt)
    vals.extend(marc.added_authors)
    vals.extend(marc.series_titles)
    vals.extend(marc.notes)
    return vals


def _cce_text_values(cce: IndexedNyplRegRecord) -> list[str]:
    """Every descriptive-text string an IndexedNyplRegRecord exposes.

    Registration's OWN descriptive text only. Identifiers (uuid/regnum/lccn/
    prev_regnums), dates/years, bools, script, and the renewal_* projection are
    excluded on purpose — the renewal join is a separate arm and would leak.
    """
    vals: list[str] = [cce.title]
    for opt in (
        cce.author_name,
        cce.author_place,
        cce.edition,
        cce.copies,
        cce.desc,
        cce.new_matter_claimed,
    ):
        if opt:
            vals.append(opt)
    vals.extend(cce.publisher_names)
    vals.extend(cce.publication_places)
    vals.extend(cce.claimants)
    vals.extend(cce.notes)
    return vals


def blob_tokens(values: list[str]) -> frozenset[str]:
    """Normalize -> tokenize -> drop title stopwords -> stem; union to one set.

    number-normalized (matches production) and stemmed (matches production
    title behavior), eng pipeline throughout so tokens align with the eng
    blob-IDF table.
    """
    raw: set[str] = set()
    for value in values:
        if not value:
            continue
        normalized = normalize_numbers(value, _LANG)
        for token in tokenize(normalized):
            if token not in _STOP:
                raw.add(token)
    if not raw:
        return frozenset()
    return frozenset(stem_tokens(tuple(raw), _LANG))


def marc_blob(marc: MarcRecord) -> frozenset[str]:
    return blob_tokens(_marc_text_values(marc))


def cce_blob(cce: IndexedNyplRegRecord) -> frozenset[str]:
    return blob_tokens(_cce_text_values(cce))


class BlobIdf:
    """IDF over CCE registration full-blobs. Unknown tokens get default_idf."""

    __slots__ = ("idf", "default_idf", "document_count")

    def __init__(self, idf: dict[str, float], default_idf: float, document_count: int) -> None:
        self.idf = idf
        self.default_idf = default_idf
        self.document_count = document_count

    def w(self, token: str) -> float:
        return self.idf.get(token, self.default_idf)


def idf_from_df(df: dict[str, int], document_count: int) -> BlobIdf:
    idf = {tok: log((document_count + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}
    default_idf = log((document_count + 1) / 1) + 1.0
    return BlobIdf(idf, default_idf, document_count)


def weighted_jaccard(a: frozenset[str], b: frozenset[str], idf: BlobIdf) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    num = sum(idf.w(t) for t in inter)
    den = sum(idf.w(t) for t in union)
    return num / den if den else 0.0


def unweighted_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def top_k(tokens: frozenset[str], idf: BlobIdf, k: int) -> frozenset[str]:
    if len(tokens) <= k:
        return tokens
    ranked = sorted(tokens, key=lambda t: idf.w(t), reverse=True)
    return frozenset(ranked[:k])
