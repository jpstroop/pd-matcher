"""Volume cardinality compatibility scorer (whole-vs-part detector).

The labeler's notes flag three cases where a MARC record describes a
whole multi-volume work while the CCE registration describes a single
volume of it (or vice versa). The page-count scorer catches the most
extreme of these (Δ > 22) implicitly, but cases where a CCE registration
explicitly marks itself as ``"v. 1"`` or ``"pt. 2"`` (and the MARC side
describes a single-volume edition) need a dedicated signal.

The scorer classifies each side as one of:

* ``whole`` — one record describes an entire multi-volume set or a
  collected/complete edition with a known volume count. MARC indicators:
  extent like ``"5 v."``, ``"v. 1-3"``, ``"3 v. in 1"``; title
  containing ``"collected"``, ``"complete"``, ``"selected"``. CCE
  indicators: same patterns on ``desc``, plus the same title cues. A CCE
  title that extends the MARC work title with a volume RANGE covering all
  of the MARC's volumes (``"A critical history. Vol.1-2."`` against a
  ``"2 v."`` MARC) is the whole set, not a part.
* ``whole_open`` — a MARC record cataloged at the series level as an
  open/ongoing multipart monograph. AACR2 codes this with bare
  ``300 ‡a v.`` (no number, no count) — the period gets stripped at
  parse time so we see ``"v"`` in :attr:`MarcRecord.extent`. RDA codes
  the same shape as ``300 ‡a volumes``. The open-date convention
  ``260/264 ‡c [1945-]`` (square brackets = cataloger-supplied date,
  trailing hyphen = ongoing publication) is the second signal. Either
  cue is sufficient. Such a record describes the abstract serial
  entity rather than any particular piece, so it is essentially never
  the correct linkage target for a CCE registration (which is always
  for one specific volume).
* ``part`` — one record describes a single part/volume of a larger
  work. MARC indicators: ``title_part_number`` populated, or title
  starting with ``"vol."``/``"pt."``. CCE indicators: ``desc`` or a
  ``note`` containing ``"v. 1"`` / ``"pt. 2"`` / ``"book one"`` or
  matching title prefixes; a CCE title whose tokens strictly extend the
  MARC work title (a named/numbered subdivision such as
  ``"Guide to art museums in the United States, east coast"``).
  Multilingual coverage: the part-detector also matches
  French/Italian/Spanish/Portuguese (``"T. 1"``, ``"tome I"``,
  ``"tomo II"``), German (``"Bd. 3"``, ``"Band III"``, ``"Tl. 2"``,
  ``"Teil 2"``, ``"Heft 4"``), Dutch (``"Dl. 1"``, ``"Deel 2"``), and
  Latin (``"Lib. III"``, ``"Pars II"``, ``"tomus IV"``).
* ``unknown`` — neither cue fires.

A part-designator found in a CCE *note* that takes the
``<series name>, <designator>`` shape and whose ``<series name>``
token-overlaps the MARC ``series_titles`` is a monographic-series
statement (``"Die Grundlehren der mathematischen Wissenschaften,
Bd.66"`` — a standalone book in a numbered series), not a whole/part
designator, and is suppressed.

Score:

* 100.0 when both sides agree (``whole↔whole``, ``whole_open↔whole``,
  ``whole_open↔whole_open``, or both ``part`` with the same part number).
* 25.0 when both sides agree on ``part`` but the part numbers differ
  (``Vol. 1`` vs. ``Vol. 2``).
* 0.0 (soft penalty) when one side is ``whole`` and the other ``part``;
  also ``whole ↔ unknown`` and ``whole_open ↔ unknown`` when the CCE
  ``desc`` parses to a concrete page count (single-volume registration);
  also ``whole_open ↔ part``.
* ``skipped=True`` when either side is ``unknown`` and no cross-reference
  fires; ``whole ↔ unknown`` / ``whole_open ↔ unknown`` skip when the CCE
  ``desc`` is not a parseable page-count statement (no useful signal
  either way).

The penalty is soft on purpose: the matching architecture treats every
scorer as a downweighted feature, no hard rejects. A calibrator can
learn the right magnitude from a larger labeled corpus.
"""

from re import IGNORECASE
from re import Match
from re import compile as re_compile

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.extent import extract_page_count
from pd_matcher.match.signals.multipart import is_series_level
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.text import tokenize

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "volume.compat"

_PART_NUMBER_RE = re_compile(
    r"\b(?:"
    r"v(?:ol(?:ume)?)?"  # English: v, vol, volume
    r"|pt|part"  # English: pt, part
    r"|bk|book"  # English: bk, book
    r"|t(?:omes?|omos?|omus)?"  # FR tome(s) / IT-ES-PT tomo(s) / Latin tomus / bare T.
    r"|bd|band"  # German: Bd., Band
    r"|tl|teil"  # German: Tl., Teil
    r"|dl|deel"  # Dutch: Dl., Deel
    r"|l(?:ivres?|ibros?|iber)?"  # FR livre(s) / IT-ES libro(s) / Latin liber / bare L.
    r"|lib"  # Latin: Lib.
    r"|pars"  # Latin: Pars
    r"|heft"  # German: Heft  (bare 'h' excluded vs initials)
    r")(?:\.\s*|\s+|(?=\d))"  # separator: period, whitespace, or zero-width before a digit
    r"(?![a-z]\.\s)"  # negative lookahead: reject 'L. M. Montgomery'-shape initials
    r"([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    IGNORECASE,
)
_WHOLE_VOLUME_COUNT_RE = re_compile(
    r"\b(\d+)\s*(?:v(?:ol(?:ume)?)?s?)\.?\b",
    IGNORECASE,
)
_VOLUME_RANGE_RE = re_compile(r"\bv(?:ol)?\.?\s*(\d+)\s*-\s*(\d+)\b", IGNORECASE)
_MULTIVOLUME_IN_ONE_RE = re_compile(r"\b\d+\s*v\.?\s*in\s*\d+\b", IGNORECASE)
_COLLECTED_TITLE_RE = re_compile(
    r"\b(collected|complete|selected)\s+(works|writings|poems|essays|letters)\b",
    IGNORECASE,
)

_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def _canonical_part_number(text: str) -> str:
    """Normalise a raw part-number token to a canonical comparand."""
    lower = text.lower()
    return _NUMBER_WORDS.get(lower, lower)


def _detect_part_match(value: str | None) -> Match[str] | None:
    """Return the raw part-designator match for ``value`` (or ``None``)."""
    if not value:
        return None
    return _PART_NUMBER_RE.search(value)


def _detect_part(value: str | None) -> str | None:
    """Return a canonical part-number string when ``value`` looks like a part."""
    match = _detect_part_match(value)
    if match is None:
        return None
    return _canonical_part_number(match.group(1))


def _is_multivolume_whole(value: str | None) -> bool:
    """Return ``True`` when ``value`` describes a multi-volume whole."""
    if not value:
        return False
    if _VOLUME_RANGE_RE.search(value) is not None:
        return True
    if _MULTIVOLUME_IN_ONE_RE.search(value) is not None:
        return True
    count_match = _WHOLE_VOLUME_COUNT_RE.search(value)
    return count_match is not None and int(count_match.group(1)) >= 2


def _is_collected_title(value: str | None) -> bool:
    """Return ``True`` when the title flags a collected/complete edition."""
    if not value:
        return False
    return _COLLECTED_TITLE_RE.search(value) is not None


def _marc_whole_volume_count(value: str) -> int | None:
    """Return the explicit multi-volume count in a MARC extent, when present."""
    count_match = _WHOLE_VOLUME_COUNT_RE.search(value)
    if count_match is None:
        return None
    count = int(count_match.group(1))
    return count if count >= 2 else None


def _range_covers_marc(cce_title: str, marc_extent: str | None) -> bool:
    """Return ``True`` when a CCE volume range covers all the MARC's volumes."""
    if not marc_extent:
        return False
    range_match = _VOLUME_RANGE_RE.search(cce_title)
    marc_count = _marc_whole_volume_count(marc_extent)
    if range_match is None or marc_count is None:
        return False
    lower = int(range_match.group(1))
    upper = int(range_match.group(2))
    return lower <= 1 and upper >= marc_count


class _Cardinality:
    """Sentinel kind labels — kept here to avoid leaking into the public API."""

    WHOLE = "whole"
    WHOLE_OPEN = "whole_open"
    PART = "part"
    UNKNOWN = "unknown"


def _title_tokens(value: str, ctx: ScorerContext) -> frozenset[str]:
    """Return the content-token set of ``value`` (stopwords dropped, stemmed).

    Mirrors the title scorer's preparation pipeline so the subset test in
    :func:`_cce_title_extends_marc` compares like with like.
    """
    normalized = normalize_numbers(value, ctx.language)
    tokens = tokenize(normalized)
    return frozenset(ctx.stemmer(token) for token in tokens if token not in ctx.stopwords.title)


def _cce_title_extends_marc(
    marc: MarcRecord,
    cce: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> bool:
    """Return ``True`` when the CCE title strictly extends the MARC work title."""
    marc_tokens = _title_tokens(marc.title_main, ctx)
    if not marc_tokens:
        return False
    cce_tokens = _title_tokens(cce.title, ctx)
    return marc_tokens < cce_tokens


def _detect_part_in_notes(
    notes: tuple[str, ...],
    marc_series_titles: tuple[str, ...],
) -> str | None:
    """Return a canonical part number from CCE notes, suppressing series members.

    A part designator in ``<series name>, <designator>`` shape whose
    ``<series name>`` token-overlaps any MARC ``series_titles`` value is a
    monographic-series statement (a standalone book in a numbered series),
    not a whole/part designator, and is skipped.
    """
    series_tokens = {
        token for series_title in marc_series_titles for token in tokenize(series_title)
    }
    for note in notes:
        match = _detect_part_match(note)
        if match is None:
            continue
        if series_tokens and _note_is_series_member(note, match, series_tokens):
            continue
        return _canonical_part_number(match.group(1))
    return None


def _note_is_series_member(
    note: str,
    match: Match[str],
    series_tokens: set[str],
) -> bool:
    """Return ``True`` when the designator's preceding name names a MARC series."""
    prefix = note[: match.start()].rstrip().rstrip(",")
    prefix_tokens = set(tokenize(prefix))
    return bool(prefix_tokens & series_tokens)


def _classify_marc(marc: MarcRecord) -> tuple[str, str | None]:
    """Return ``(cardinality, part_number)`` for the MARC side."""
    if is_series_level(marc):
        return _Cardinality.WHOLE_OPEN, None
    if _is_multivolume_whole(marc.extent) or _is_collected_title(marc.title):
        return _Cardinality.WHOLE, None
    if marc.title_part_number:
        return _Cardinality.PART, _canonical_part_number(marc.title_part_number)
    part = _detect_part(marc.title) or _detect_part(marc.extent)
    if part is not None:
        return _Cardinality.PART, part
    return _Cardinality.UNKNOWN, None


def _classify_cce(cce: IndexedNyplRegRecord, marc: MarcRecord) -> tuple[str, str | None]:
    """Return ``(cardinality, part_number)`` for the CCE side."""
    if _is_multivolume_whole(cce.desc) or _is_collected_title(cce.title):
        return _Cardinality.WHOLE, None
    part = _detect_part(cce.desc) or _detect_part(cce.title)
    if part is not None:
        return _Cardinality.PART, part
    note_part = _detect_part_in_notes(cce.notes, marc.series_titles)
    if note_part is not None:
        return _Cardinality.PART, note_part
    return _Cardinality.UNKNOWN, None


def _refine_cce_kind(
    marc: MarcRecord,
    cce: IndexedNyplRegRecord,
    marc_kind: str,
    cce_kind: str,
    ctx: ScorerContext,
) -> str:
    """Reclassify the CCE side from a title-extension signal against a whole MARC.

    When the MARC describes a whole/open set and the CCE title strictly
    extends the MARC work title, the CCE is a named/numbered subdivision
    (a part) — unless that extension carries a volume range that covers all
    of a closed MARC set, in which case the CCE is the whole set.
    """
    if marc_kind not in (_Cardinality.WHOLE, _Cardinality.WHOLE_OPEN):
        return cce_kind
    if cce_kind not in (_Cardinality.UNKNOWN, _Cardinality.PART):
        return cce_kind
    if not _cce_title_extends_marc(marc, cce, ctx):
        return cce_kind
    if marc_kind == _Cardinality.WHOLE and _range_covers_marc(cce.title, marc.extent):
        return _Cardinality.WHOLE
    return _Cardinality.PART


def _build_features(
    marc_kind: str,
    cce_kind: str,
) -> tuple[tuple[str, float], ...]:
    """Project per-side classification onto the Evidence feature tuple."""
    return (
        ("marc_is_whole", 1.0 if marc_kind == _Cardinality.WHOLE else 0.0),
        ("marc_is_whole_open", 1.0 if marc_kind == _Cardinality.WHOLE_OPEN else 0.0),
        ("marc_is_part", 1.0 if marc_kind == _Cardinality.PART else 0.0),
        ("cce_is_whole", 1.0 if cce_kind == _Cardinality.WHOLE else 0.0),
        ("cce_is_part", 1.0 if cce_kind == _Cardinality.PART else 0.0),
    )


def _evidence(
    score: float,
    skipped: bool,
    features: tuple[tuple[str, float], ...],
) -> Evidence:
    """Construct the scorer's :class:`Evidence` with shared constant fields."""
    return Evidence(
        scorer=_SCORER_NAME,
        score=score,
        max=_MAX_SCORE,
        skipped=skipped,
        decisive=False,
        features=features,
    )


def _score_whole(
    cce_kind: str,
    cce: IndexedNyplRegRecord,
    features: tuple[tuple[str, float], ...],
) -> Evidence:
    """Return Evidence for a MARC-side ``whole`` (closed multi-volume) record."""
    if cce_kind == _Cardinality.WHOLE:
        return _evidence(_MAX_SCORE, False, features)
    if cce_kind == _Cardinality.PART:
        return _evidence(0.0, False, features)
    if extract_page_count(cce.desc) is not None:
        return _evidence(0.0, False, features)
    return _evidence(0.0, True, features)


def _score_whole_open(
    cce_kind: str,
    cce: IndexedNyplRegRecord,
    features: tuple[tuple[str, float], ...],
) -> Evidence:
    """Return Evidence for a MARC-side ``whole_open`` classification."""
    if cce_kind == _Cardinality.PART:
        return _evidence(0.0, False, features)
    if cce_kind in (_Cardinality.WHOLE, _Cardinality.WHOLE_OPEN):
        return _evidence(_MAX_SCORE, False, features)
    if extract_page_count(cce.desc) is not None:
        return _evidence(0.0, False, features)
    return _evidence(0.0, True, features)


def score_volume(
    marc: MarcRecord,
    cce: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> Evidence:
    """Return :class:`Evidence` comparing volume cardinality on both sides."""
    marc_kind, marc_part = _classify_marc(marc)
    cce_kind, cce_part = _classify_cce(cce, marc)
    cce_kind = _refine_cce_kind(marc, cce, marc_kind, cce_kind, ctx)
    features = _build_features(marc_kind, cce_kind)
    if marc_kind == _Cardinality.WHOLE_OPEN:
        return _score_whole_open(cce_kind, cce, features)
    if marc_kind == _Cardinality.WHOLE:
        return _score_whole(cce_kind, cce, features)
    if marc_kind == _Cardinality.UNKNOWN or cce_kind == _Cardinality.UNKNOWN:
        return _evidence(0.0, True, features)
    if marc_kind != cce_kind:
        score = 0.0
    elif marc_kind == _Cardinality.PART and marc_part != cce_part:
        score = 25.0
    else:
        score = _MAX_SCORE
    return _evidence(score, False, features)


__all__ = [
    "score_volume",
]
