"""Page-count compatibility scorer (MARC 300$a ↔ CCE ``<desc>``).

Both sides describe physical extent in semi-structured free text. The
labeled corpus shows page count is the second most-applicable scalar
signal after title: of pairs where both sides parse, 95.7% of matches
sit at Δ ≤ 2 pages while 88.9% of no-matches sit at Δ > 10. A short
gently-decaying ramp turns that separation into a usable score.

Score curve:

* Δ ≤ 2 pages: 100.0 (matches commonly differ by 1-2 because of
  introductory pagination conventions).
* 2 < Δ ≤ 22: linear decay (100 - 5·(Δ-2)), reaching 0 at Δ = 22.
* Δ > 22: 0.0 (catches whole-vs-part records implicitly — single
  volume vs. multi-volume set).
* Unparseable on either side: ``skipped=True`` (the combiner already
  excludes skipped Evidence from numerator and denominator).

Parser heuristic: strip volume-count statements and a leading
roman-numeral pagination block, then take the LARGEST plain integer
that remains. Roman numerals at the start (``"xii, 312 p."``) are
paginated front-matter and not part of the page count. Volume counts
(``"3 v"``, ``"1 v."``, ``"2 vols."``, and the volumes-in-bindings form
``"5 v. in 10"``) are NOT page counts: a bare volume statement yields no
integer and skips, so a 3-volume set and a 1-volume work no longer read as
"3 pages" vs "1 page" and falsely match (extent bug, pair 295; the
``"in <m>"`` tail is also volumes, pair 377). Multi-volume statements that *also* carry a page
count — ``"1 v. (312 p.)"`` — strip the ``"1 v."`` and pick out 312;
``"v. (loose-leaf)"`` and ``"unpaged"`` yield no integer at all and skip.
Whether a volume-count *mismatch* (3 v. vs 1 v.) should be a negative
whole/part signal is out of scope here (#82); this scorer only declines
to manufacture a false page match.
"""

from re import IGNORECASE
from re import compile as re_compile

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "extent.page_count"
_TOLERANCE_PAGES: int = 2
_PENALTY_PER_PAGE: float = 5.0

_ROMAN_PREFIX_RE = re_compile(r"^\s*(?:\[[^\]]*\]\s*,?\s*)?[ivxlcdm]+\s*,\s*", IGNORECASE)
# A volume COUNT: "<n> v", "<n> v.", "<n> vol", "<n> vols.", "<n> volume(s)",
# and the volumes-in-bindings form "<n> v. in <m>" (both tallies are volumes,
# not pages). The negative lookahead stops the abbreviation from swallowing an
# unrelated word ("3 voluntary..."); the digits it consumes are volume tallies,
# not pages, so the whole expression is removed before page extraction.
_VOLUME_COUNT_RE = re_compile(r"\b\d+\s*v(?:ol(?:ume)?s?)?\.?(?:\s+in\s+\d+)?(?![a-z])", IGNORECASE)
_INTEGER_RE = re_compile(r"\d+")


def extract_page_count(value: str | None) -> int | None:
    """Return the largest plain page integer, or ``None`` to skip.

    Strips volume-count statements (``"3 v"``) and a leading roman-numeral
    block, then returns the largest remaining positive integer. Returns
    ``None`` when ``value`` is empty, carries only a volume count, has no
    digits, or yields only zero (avoiding ``"0 p."`` sentinels) — all cases
    where the scorer should skip rather than compare.
    """
    if not value:
        return None
    no_volumes = _VOLUME_COUNT_RE.sub(" ", value)
    stripped = _ROMAN_PREFIX_RE.sub("", no_volumes)
    integers = [int(match.group(0)) for match in _INTEGER_RE.finditer(stripped)]
    candidates = [integer for integer in integers if integer > 0]
    if not candidates:
        return None
    return max(candidates)


def score_extent(
    marc_extent: str | None,
    cce_desc: str | None,
    ctx: ScorerContext,
) -> Evidence:
    """Return :class:`Evidence` for a (MARC extent, CCE desc) pair."""
    del ctx
    marc_pages = extract_page_count(marc_extent)
    cce_pages = extract_page_count(cce_desc)
    if marc_pages is None or cce_pages is None:
        return Evidence(
            scorer=_SCORER_NAME,
            score=0.0,
            max=_MAX_SCORE,
            skipped=True,
            decisive=False,
            features=(
                ("marc_pages", float(marc_pages) if marc_pages is not None else -1.0),
                ("cce_pages", float(cce_pages) if cce_pages is not None else -1.0),
            ),
        )
    delta = abs(marc_pages - cce_pages)
    if delta <= _TOLERANCE_PAGES:
        score = _MAX_SCORE
    else:
        score = max(0.0, _MAX_SCORE - _PENALTY_PER_PAGE * (delta - _TOLERANCE_PAGES))
    return Evidence(
        scorer=_SCORER_NAME,
        score=score,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=(
            ("marc_pages", float(marc_pages)),
            ("cce_pages", float(cce_pages)),
            ("delta", float(delta)),
        ),
    )


__all__ = [
    "extract_page_count",
    "score_extent",
]
