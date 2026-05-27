"""Volume cardinality compatibility scorer (whole-vs-part detector).

The labeler's notes flag three cases where a MARC record describes a
whole multi-volume work while the CCE registration describes a single
volume of it (or vice versa). The page-count scorer catches the most
extreme of these (Δ > 22) implicitly, but cases where a CCE registration
explicitly marks itself as ``"v. 1"`` or ``"pt. 2"`` (and the MARC side
describes a single-volume edition) need a dedicated signal.

The scorer classifies each side as one of:

* ``whole`` — one record describes an entire multi-volume set or a
  collected/complete edition. MARC indicators: extent like ``"5 v."``,
  ``"v. 1-3"``, ``"3 v. in 1"``; title containing ``"collected"``,
  ``"complete"``, ``"selected"``. CCE indicators: same patterns on
  ``desc``, plus the same title cues.
* ``part`` — one record describes a single part/volume of a larger
  work. MARC indicators: ``title_part_number`` populated, or title
  starting with ``"vol."``/``"pt."``. CCE indicators: ``desc``
  containing ``"v. 1"`` / ``"pt. 2"`` / ``"book one"`` or matching
  title prefixes.
* ``unknown`` — neither cue fires.

Score:

* 100.0 when both sides agree (``whole↔whole`` or both ``part`` with
  the same part number).
* 25.0 when both sides agree on ``part`` but the part numbers differ
  (``Vol. 1`` vs. ``Vol. 2``).
* 0.0 (soft penalty) when one side is ``whole`` and the other ``part``.
* ``skipped=True`` when either side is ``unknown``.

The penalty is soft on purpose: the matching architecture treats every
scorer as a downweighted feature, no hard rejects. A calibrator can
learn the right magnitude from a larger labeled corpus.
"""

from re import IGNORECASE
from re import compile as re_compile

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "volume.compat"

_PART_NUMBER_RE = re_compile(
    r"\b(?:v(?:ol)?|pt|bk|book)\.?\s*([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    IGNORECASE,
)
_WHOLE_VOLUME_COUNT_RE = re_compile(r"\b(\d+)\s*v(?:ol)?s?\.?\b", IGNORECASE)
_VOLUME_RANGE_RE = re_compile(r"\bv(?:ol)?\.?\s*\d+\s*-\s*\d+\b", IGNORECASE)
_MULTIVOLUME_IN_ONE_RE = re_compile(r"\b\d+\s*v\.?\s*in\s*\d+\b", IGNORECASE)
_COLLECTED_TITLE_RE = re_compile(
    r"\b(collected|complete|selected)\s+(works|writings|poems|essays|letters)\b",
    IGNORECASE,
)

_PART_KIND_CANONICAL = {
    "v": "v",
    "vol": "v",
    "pt": "pt",
    "bk": "bk",
    "book": "bk",
}

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


def _detect_part(value: str | None) -> str | None:
    """Return a canonical part-number string when ``value`` looks like a part."""
    if not value:
        return None
    match = _PART_NUMBER_RE.search(value)
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


class _Cardinality:
    """Sentinel kind labels — kept here to avoid leaking into the public API."""

    WHOLE = "whole"
    PART = "part"
    UNKNOWN = "unknown"


def _classify_marc(marc: MarcRecord) -> tuple[str, str | None]:
    """Return ``(cardinality, part_number)`` for the MARC side."""
    if _is_multivolume_whole(marc.extent) or _is_collected_title(marc.title):
        return _Cardinality.WHOLE, None
    if marc.title_part_number:
        return _Cardinality.PART, _canonical_part_number(marc.title_part_number)
    part = _detect_part(marc.title) or _detect_part(marc.extent)
    if part is not None:
        return _Cardinality.PART, part
    return _Cardinality.UNKNOWN, None


def _classify_cce(cce: IndexedNyplRegRecord) -> tuple[str, str | None]:
    """Return ``(cardinality, part_number)`` for the CCE side."""
    if _is_multivolume_whole(cce.desc) or _is_collected_title(cce.title):
        return _Cardinality.WHOLE, None
    part = _detect_part(cce.desc) or _detect_part(cce.title)
    if part is not None:
        return _Cardinality.PART, part
    return _Cardinality.UNKNOWN, None


def score_volume(
    marc: MarcRecord,
    cce: IndexedNyplRegRecord,
    ctx: ScorerContext,
) -> Evidence:
    """Return :class:`Evidence` comparing volume cardinality on both sides."""
    del ctx
    marc_kind, marc_part = _classify_marc(marc)
    cce_kind, cce_part = _classify_cce(cce)
    features: tuple[tuple[str, float], ...] = (
        ("marc_is_whole", 1.0 if marc_kind == _Cardinality.WHOLE else 0.0),
        ("marc_is_part", 1.0 if marc_kind == _Cardinality.PART else 0.0),
        ("cce_is_whole", 1.0 if cce_kind == _Cardinality.WHOLE else 0.0),
        ("cce_is_part", 1.0 if cce_kind == _Cardinality.PART else 0.0),
    )
    if marc_kind == _Cardinality.UNKNOWN or cce_kind == _Cardinality.UNKNOWN:
        return Evidence(
            scorer=_SCORER_NAME,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=features,
        )
    if marc_kind != cce_kind:
        score = 0.0
    elif marc_kind == _Cardinality.PART and marc_part != cce_part:
        score = 25.0
    else:
        score = _MAX_SCORE
    return Evidence(
        scorer=_SCORER_NAME,
        score=score,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=features,
    )


__all__ = [
    "score_volume",
]
