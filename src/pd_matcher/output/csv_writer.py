"""Streaming CSV writer for matcher output rows.

The output schema is a flat linkage row: MARC metadata, matched CCE metadata,
and per-field plus combined scores. :class:`CsvResultWriter` is a
context-manager :class:`ResultWriter` that buffers a single underlying file
handle and flushes after every row so a partial run is always readable.

Per-record normalization/stemming is recomputed at write time to keep
this module independent from the matcher: it expects only a
:class:`MarcRecord` and a :class:`MatchResult`. This is intentionally a
per-row cost — a single CSV row's normalize+stem work is dwarfed by the
matcher pipeline that produced it.
"""

from csv import DictWriter
from pathlib import Path
from types import TracebackType
from typing import IO
from typing import Protocol
from typing import Self

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.stemming import stem_tokens
from pd_matcher.normalize.text import normalize_text
from pd_matcher.normalize.text import tokenize

_DEFAULT_LANGUAGE: str = "eng"

CSV_COLUMNS: tuple[str, ...] = (
    "marc_id",
    "marc_title_original",
    "marc_title_normalized",
    "marc_title_stemmed",
    "marc_author_original",
    "marc_author_normalized",
    "marc_author_stemmed",
    "marc_main_author_original",
    "marc_main_author_normalized",
    "marc_main_author_stemmed",
    "marc_publisher_original",
    "marc_publisher_normalized",
    "marc_publisher_stemmed",
    "marc_year",
    "marc_lccn",
    "marc_lccn_normalized",
    "marc_country_code",
    "marc_language_code",
    "match_type",
    "match_title",
    "match_title_normalized",
    "match_author",
    "match_author_normalized",
    "match_publisher",
    "match_publisher_normalized",
    "match_year",
    "match_source_id",
    "match_date",
    "title_score",
    "author_score",
    "publisher_score",
    "combined_score",
    "year_difference",
)


class ResultWriter(Protocol):
    """Streaming writer for one linkage row per processed MARC record."""

    def __enter__(self) -> Self:  # pragma: no cover
        """Open the underlying sink."""
        ...

    def __exit__(  # pragma: no cover
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying sink."""
        ...

    def write(  # pragma: no cover
        self,
        marc: MarcRecord,
        match: MatchResult | None,
        matched_nypl: IndexedNyplRegRecord | None = None,
    ) -> None:
        """Emit one CSV row for the supplied triple."""
        ...


def _normalize_and_stem(value: str | None, language: str) -> tuple[str, str, str]:
    """Return ``(original, normalized, stemmed)`` strings for ``value``.

    The empty string is returned for every slot when ``value`` is ``None`` or
    normalizes to an empty form. ``stemmed`` is a space-joined string in the
    same shape as the historic ground truth column.
    """
    if value is None:
        return "", "", ""
    normalized = normalize_text(value)
    if not normalized:
        return value, "", ""
    stems = stem_tokens(tokenize(normalized), language)
    return value, normalized, " ".join(stems)


def _evidence_score(evidence: tuple[Evidence, ...], scorer: str) -> str:
    """Return the integer string of the named scorer's Evidence score.

    The combined CSV uses integers for per-field scores; we round half-to-even
    because the underlying scorers emit float scores in ``[0, 100]``. When the
    Evidence is missing or skipped, an empty string is returned so the column
    survives joins against records that have no match.
    """
    for ev in evidence:
        if ev.scorer == scorer:
            if ev.skipped:
                return ""
            return f"{round(ev.score)}"
    return ""


def _format_match_date(record: IndexedNyplRegRecord) -> str:
    """Return ISO ``reg_date`` text, or year-only when only the year is known."""
    if record.reg_date is not None:
        return record.reg_date.isoformat()
    if record.reg_year is not None:
        return str(record.reg_year)
    return ""


def _build_row(
    marc: MarcRecord,
    match: MatchResult | None,
    matched_nypl: IndexedNyplRegRecord | None,
) -> dict[str, str]:
    """Translate the input triple into a flat ``dict[str, str]`` row."""
    language = marc.language_code or _DEFAULT_LANGUAGE
    title_o, title_n, title_s = _normalize_and_stem(marc.title, language)
    sor_o, sor_n, sor_s = _normalize_and_stem(marc.statement_of_responsibility, language)
    main_o, main_n, main_s = _normalize_and_stem(marc.main_author, language)
    pub_o, pub_n, pub_s = _normalize_and_stem(marc.publisher, language)
    row: dict[str, str] = {
        "marc_id": marc.control_id,
        "marc_title_original": title_o,
        "marc_title_normalized": title_n,
        "marc_title_stemmed": title_s,
        "marc_author_original": sor_o,
        "marc_author_normalized": sor_n,
        "marc_author_stemmed": sor_s,
        "marc_main_author_original": main_o,
        "marc_main_author_normalized": main_n,
        "marc_main_author_stemmed": main_s,
        "marc_publisher_original": pub_o,
        "marc_publisher_normalized": pub_n,
        "marc_publisher_stemmed": pub_s,
        "marc_year": "" if marc.publication_year is None else str(marc.publication_year),
        "marc_lccn": marc.lccn or "",
        "marc_lccn_normalized": marc.lccn or "",
        "marc_country_code": marc.country_code or "",
        "marc_language_code": marc.language_code or "",
        "match_type": "",
        "match_title": "",
        "match_title_normalized": "",
        "match_author": "",
        "match_author_normalized": "",
        "match_publisher": "",
        "match_publisher_normalized": "",
        "match_year": "",
        "match_source_id": "",
        "match_date": "",
        "title_score": "",
        "author_score": "",
        "publisher_score": "",
        "combined_score": "",
        "year_difference": "",
    }
    if match is None or match.best is None or matched_nypl is None:
        return row
    best = match.best
    match_title_original = matched_nypl.title
    match_title_normalized = normalize_text(matched_nypl.title)
    match_author_original = matched_nypl.author_name or ""
    match_author_normalized = (
        normalize_text(matched_nypl.author_name) if matched_nypl.author_name else ""
    )
    publisher_joined = " ".join(matched_nypl.publisher_names)
    match_publisher_normalized = normalize_text(publisher_joined) if publisher_joined else ""
    row["match_type"] = "registration"
    row["match_title"] = match_title_original
    row["match_title_normalized"] = match_title_normalized
    row["match_author"] = match_author_original
    row["match_author_normalized"] = match_author_normalized
    row["match_publisher"] = publisher_joined
    row["match_publisher_normalized"] = match_publisher_normalized
    row["match_year"] = "" if matched_nypl.reg_year is None else str(matched_nypl.reg_year)
    row["match_source_id"] = matched_nypl.uuid
    row["match_date"] = _format_match_date(matched_nypl)
    row["title_score"] = _evidence_score(best.evidence, "title.token_set")
    row["author_score"] = _evidence_score(best.evidence, "name.author")
    row["publisher_score"] = _evidence_score(best.evidence, "name.publisher")
    row["combined_score"] = f"{best.combined.calibrated * 100.0:.2f}"
    if marc.publication_year is not None and matched_nypl.reg_year is not None:
        row["year_difference"] = str(marc.publication_year - matched_nypl.reg_year)
    return row


class CsvResultWriter:
    """:class:`ResultWriter` that emits the verified-linkage CSV schema."""

    __slots__ = ("_fp", "_path", "_writer")

    def __init__(self, path: Path) -> None:
        """Capture ``path``; the file is opened on context-manager entry."""
        self._path = path
        self._fp: IO[str] | None = None
        self._writer: DictWriter[str] | None = None

    def __enter__(self) -> Self:
        """Open the destination file and write the header row."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("w", encoding="utf-8", newline="")
        self._writer = DictWriter(self._fp, fieldnames=list(CSV_COLUMNS))
        self._writer.writeheader()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Flush and close the underlying file handle."""
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
            self._writer = None

    def write(
        self,
        marc: MarcRecord,
        match: MatchResult | None,
        matched_nypl: IndexedNyplRegRecord | None = None,
    ) -> None:
        """Emit one CSV row for the supplied triple.

        Args:
            marc: The MARC record being matched.
            match: The matcher's verdict, or ``None`` when no match was made.
            matched_nypl: The CCE registration corresponding to
                ``match.best``. When omitted the row's ``match_*`` columns
                are blank even if ``match.best`` is set, because the writer
                cannot resolve the indexed record on its own.
        """
        if self._writer is None or self._fp is None:
            raise RuntimeError("CsvResultWriter not entered; use as a context manager")
        row = _build_row(marc, match, matched_nypl)
        self._writer.writerow(row)
        self._fp.flush()


__all__ = [
    "CSV_COLUMNS",
    "CsvResultWriter",
    "ResultWriter",
]
