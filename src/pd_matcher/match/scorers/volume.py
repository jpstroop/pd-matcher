"""Volume cardinality compatibility scorer (whole-vs-part detector).

A common false-high pattern in the labeled corpus is a MARC record that
describes an entire multi-volume *set* matched against a CCE registration
that covers only a single *part* of that set (``"...Pt. 1"``,
``"Vol. 1"``, a Roman-numeral ``"I:"`` subtitle, a mid-title ``"v. 1"``).
Title, author, and year all agree, so the pair scores high — but the
records describe different bibliographic units. This scorer is a clean,
multi-field detection FEATURE that flags such whole-vs-part mismatches so
the combiner (and the learned model, which carries ``volume.compat`` as a
column) can weight them.

The scorer classifies each side as one of:

* ``whole`` — one record describes an entire CLOSED multi-volume set or a
  collected/complete edition with a known volume count. MARC indicators:
  extent like ``"5 v."``, ``"v. 1-3"``, ``"5 v. in 10"``; title
  containing ``"collected"`` / ``"complete"`` / ``"selected"``. CCE
  indicators: the same patterns on ``desc`` or ``title``, plus a CCE
  part-designator RANGE (``"Vol. 1-2"``, ``"T.1-3"``) that registers the
  whole/range of pieces rather than a single piece.
* ``whole_open`` — a MARC record cataloged at the series level as an
  open/ongoing multipart monograph (AACR2 bare ``300 ‡a v.`` / RDA
  ``volumes`` / the open-date convention ``[1945-]``). Leading
  punctuation noise on the extent (``". v"``) is tolerated. Such a record
  describes the abstract serial entity, never one specific volume.
* ``part`` — one record describes a SINGLE part/volume of a larger work.
  MARC indicators: ``title_part_number`` populated, or a single
  designator in the title/extent. CCE indicators: a SINGLE part
  designator (``"Pt. 1"``, ``"Vol. 1"``, mid-title ``"v. 1"``, a bare
  Roman-numeral ``"I:"`` subtitle, ``"book one"``) in the title, the
  ``desc``, or the ``notes`` — explicitly NOT a range. Multilingual
  coverage spans French/Italian/Spanish/Portuguese (``"T. 1"``,
  ``"tome I"``, ``"tomo II"``), German (``"Bd. 3"``, ``"Band III"``,
  ``"Tl. 2"``, ``"Teil 2"``, ``"Heft 4"``), Dutch (``"Dl. 1"``,
  ``"Deel 2"``), and Latin (``"Lib. III"``, ``"Pars II"``).
* ``unknown`` — neither cue fires.

A single part designator that the CCE registers takes precedence over a
multi-volume ``desc`` count: a ``"Vol. 1"`` in the CCE notes alongside a
``desc`` of ``"2 v."`` (one of several physical volumes that make up the
registered part) is still a single part of the larger MARC whole —
containment of one part in a multi-volume whole is INCOMPATIBLE, not a
set match. Only a covering RANGE reads as the CCE-whole direction.

A part-designator found in a CCE *note* that takes the
``<series name>, <designator>`` shape and whose ``<series name>``
token-overlaps the MARC ``series_titles`` is a monographic-series
statement (``"Die Grundlehren der mathematischen Wissenschaften,
Bd.66"`` — a standalone book in a numbered series), not a whole/part
designator, and is suppressed.

When the MARC describes a whole/open set and the CCE title strictly
extends the MARC work title, the CCE is a named/numbered subdivision (a
part) — unless that extension carries a covering volume range, in which
case the CCE is the whole set.

Score:

* 100.0 when both sides agree (``whole↔whole``, ``whole_open↔whole``,
  ``whole_open↔whole_open``, or both ``part`` with the same part number).
* 25.0 when both sides agree on ``part`` but the part numbers differ.
* 0.0 when one side is ``whole``/``whole_open`` and the other is a
  ``part`` (the marc_whole_cce_part / cce_whole_marc_part mismatch); also
  ``whole``/``whole_open`` vs an ``unknown`` CCE whose ``desc`` parses to
  a concrete single-volume page count.
* ``skipped=True`` when there is no usable volume signal on either side.

``decisive`` is never set: a false detection must never veto a true
match, so this is a downweighted feature, not a hard reject.
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

_PART_PREFIX: str = (
    r"(?:"
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
    r")"
)
_PART_NUMBER_RE = re_compile(
    r"\b" + _PART_PREFIX + r"(?:\.\s*|\s+|(?=\d))"  # separator: period, whitespace, before digit
    r"(?![a-z]\.\s)"  # negative lookahead: reject 'L. M. Montgomery'-shape initials
    r"([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    IGNORECASE,
)
# A designator RANGE ("Vol. 1-2", "T.1-3"): the CCE registered the whole
# set of pieces, not a single piece. Generalised over every part prefix.
_PART_RANGE_RE = re_compile(
    r"\b" + _PART_PREFIX + r"\.?\s*(\d+)\s*-\s*(\d+)\b",
    IGNORECASE,
)
# A bare Roman-numeral / digit volume designator used as a subtitle marker
# ("Kontakia ..., I: On the person of Christ"). The leading [,.;] separator
# avoids the far commoner "Main title: subtitle" colon.
_BARE_DESIGNATOR_RE = re_compile(
    r"[,.;]\s*([ivxlcdm]+|\d+)\s*:\s",
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
_LEADING_NOISE_RE = re_compile(r"^[^0-9a-z]+", IGNORECASE)

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


def _is_part_range(value: str | None) -> bool:
    """Return ``True`` when ``value`` carries a designator RANGE (a set/whole)."""
    if not value:
        return False
    return _PART_RANGE_RE.search(value) is not None


def _detect_part_match(value: str | None) -> Match[str] | None:
    """Return the raw single-part-designator match for ``value`` (or ``None``).

    A designator RANGE (``"Vol. 1-2"``) is the whole/set direction, not a
    single part, so it is rejected here.
    """
    if not value or _is_part_range(value):
        return None
    return _PART_NUMBER_RE.search(value)


def _detect_part(value: str | None) -> str | None:
    """Return a canonical part-number string when ``value`` looks like a part."""
    match = _detect_part_match(value)
    if match is None:
        return None
    return _canonical_part_number(match.group(1))


def _detect_bare_designator(value: str | None) -> str | None:
    """Return a canonical number for a bare Roman/digit subtitle designator."""
    if not value or _is_part_range(value):
        return None
    match = _BARE_DESIGNATOR_RE.search(value)
    if match is None:
        return None
    return _canonical_part_number(match.group(1))


def _normalized_extent(value: str | None) -> str | None:
    """Strip leading punctuation noise (OCR ``". v"``) before classification."""
    if not value:
        return value
    return _LEADING_NOISE_RE.sub("", value) or None


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
    extent = _normalized_extent(marc.extent)
    if is_series_level(marc) or _is_bare_volume(extent):
        return _Cardinality.WHOLE_OPEN, None
    if _is_multivolume_whole(extent) or _is_collected_title(marc.title):
        return _Cardinality.WHOLE, None
    if marc.title_part_number:
        return _Cardinality.PART, _canonical_part_number(marc.title_part_number)
    part = _detect_part(marc.title) or _detect_part(extent)
    if part is not None:
        return _Cardinality.PART, part
    return _Cardinality.UNKNOWN, None


def _is_bare_volume(value: str | None) -> bool:
    """Return ``True`` for a bare ``"v"`` / ``"volumes"`` extent (noise stripped)."""
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered in ("v", "volumes")


def _cce_whole_signal(cce: IndexedNyplRegRecord) -> bool:
    """Return ``True`` when the CCE registers a whole set/range or collected work."""
    if _is_collected_title(cce.title):
        return True
    if _is_part_range(cce.title) or _is_part_range(cce.desc):
        return True
    return any(_is_part_range(note) for note in cce.notes)


def _cce_single_part(cce: IndexedNyplRegRecord, marc: MarcRecord) -> str | None:
    """Return a single CCE part designator from title, desc, or notes."""
    part = _detect_part(cce.title) or _detect_part(cce.desc)
    if part is not None:
        return part
    note_part = _detect_part_in_notes(cce.notes, marc.series_titles)
    if note_part is not None:
        return note_part
    return _detect_bare_designator(cce.title)


def _classify_cce(cce: IndexedNyplRegRecord, marc: MarcRecord) -> tuple[str, str | None]:
    """Return ``(cardinality, part_number)`` for the CCE side.

    A single explicit part designator (in the title, desc, or notes) wins
    over a multi-volume ``desc`` count: a registered ``"Vol. 1"`` that
    spans ``"2 v."`` is still a single part of the larger MARC whole, not a
    set. Only a covering RANGE or a collected-work title reads as whole.
    """
    if _cce_whole_signal(cce):
        return _Cardinality.WHOLE, None
    single_part = _cce_single_part(cce, marc)
    if single_part is not None:
        return _Cardinality.PART, single_part
    if _is_multivolume_whole(cce.desc):
        return _Cardinality.WHOLE, None
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
    extends the MARC work title, the CCE is a named/numbered subdivision (a
    part). A covering designator range on the CCE has already classified it
    as the whole set in :func:`_classify_cce`, so this only ever upgrades an
    ``UNKNOWN``/``PART`` CCE to a ``PART``.
    """
    if marc_kind not in (_Cardinality.WHOLE, _Cardinality.WHOLE_OPEN):
        return cce_kind
    if cce_kind not in (_Cardinality.UNKNOWN, _Cardinality.PART):
        return cce_kind
    if not _cce_title_extends_marc(marc, cce, ctx):
        return cce_kind
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
    """Return Evidence for a MARC-side whole / whole_open record."""
    if cce_kind in (_Cardinality.WHOLE, _Cardinality.WHOLE_OPEN):
        return _evidence(_MAX_SCORE, False, features)
    if cce_kind == _Cardinality.PART:
        return _evidence(0.0, False, features)
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
    if marc_kind in (_Cardinality.WHOLE, _Cardinality.WHOLE_OPEN):
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
