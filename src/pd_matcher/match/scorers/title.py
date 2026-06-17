"""IDF-weighted token-set similarity over title fields.

The scorer applies the same normalization pipeline to both sides:

* :func:`pd_matcher.normalize.text.normalize_text` (already implicit in
  :func:`tokenize`) — NFKD-strip + lowercase + punctuation collapse.
* :func:`pd_matcher.normalize.numbers.normalize_numbers` — Roman/word/
  ordinal → Arabic digits in the record's language.
* Stopword removal using the language's title stopword set.
* Snowball stemming for the record's language.

Then the IDF-weighted Jaccard ratio is computed: the sum of IDF over the
stem intersection divided by the sum over the union. This down-weights
common bibliographic filler ("the", "and", "of") and up-weights rare
distinguishing tokens, so two titles that differ only in noise still score
high while two titles that happen to share filler do not.

The CCE side is OCR'd from the printed *Catalog of Copyright Entries*, so a
true match's distinctive title tokens are frequently corrupted by a single
character ("immunochemistry" → "immunocheraistry", "Histoire" → "Histolre")
or split across a compound boundary ("toymaker" → "toy maker"). Exact set
intersection is maximally brittle to this: one wrong character drops an
entire high-IDF token out of the numerator, cratering the score on a real
match (#55 — measured at ~30% of labeled matches). So the intersection is
computed over a **fuzzy alignment** instead of exact set membership: a MARC
token counts as shared when it has a CCE counterpart whose character-level
:func:`rapidfuzz.fuzz.ratio` clears :data:`_FUZZY_MIN_RATIO`. The threshold
is deliberately high so only near-identical (OCR-distance) tokens align —
distinct words ("work"/"word") stay unmatched, and IDF weighting keeps any
residual generic near-match low-impact. An exact match scores ``ratio ==
100`` and so reduces to the original Jaccard, making this a controlled
generalization.

IDF cancels in a single-token Jaccard (``idf(tok) / idf(tok) = 1.0``), so a
lone generic shared word scores a perfect match (#87). The score is therefore
scaled by an absolute-evidence confidence keyed on the shared IDF *mass* (see
:data:`_GENERIC_TITLE_MASS_FACTOR`), which is low for a thin generic overlap
and high for a rich one — the discriminating signal the ratio throws away.
"""

from rapidfuzz.fuzz import ratio

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.signals.script import is_script_mismatch
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.text import tokenize

_MAX_SCORE: float = 100.0
_SCORER_NAME: str = "title.token_set"
# Chosen by a threshold sweep over the 921 labeled matches vs labeled non-matches
# (#55): match-side title lift peaks at 80 (37 recovered, mean +0.033) — the
# plateau, since 78 adds nothing — while non-match inflation stays flat and tiny
# (2 pairs, +0.006), a ~6x match-vs-non-match asymmetry. 80 is the per-token
# OCR distance for a 5-char word (one substitution); shorter words effectively
# require identity, which keeps distinct short words ("work"/"word" = 75) apart.
_FUZZY_MIN_RATIO: float = 80.0

# Per-token fuzzy matching handles OCR *substitution* within a token but not
# token-*boundary* errors — compound splits ("toymaker"/"toy maker"), OCR
# line-break hyphens ("cru-ciale"/"cruciale"), or inserted spaces
# ("Tennessee"/"Tennes see"). Joining the prepared stems erases boundaries, so a
# whole-string character ratio catches them. It bypasses IDF (it is a raw
# string-identity claim), so the gate is high: a sweep over the same labeled sets
# recovered +34 matches beyond per-token at 0.90 while only one non-match crossed
# 0.80 (a same-title-different-work, whose title legitimately *is* identical — the
# rejection correctly comes from author/year). The boost is a ``max``, so it only
# ever rescues a low Jaccard, never lowers a high one. Loosen toward 0.85 only if
# the #84 separation AUC shows headroom.
_WHOLE_STRING_MIN_RATIO: float = 90.0

# The whole-string rescue only fires when BOTH concatenations reach this length.
# A high character ratio is a strong same-title claim on a long string (many
# characters agree) but a weak, coincidence-prone one on a short string (~1 char
# at 0.90). Boundary errors inherently span multiple tokens, so genuine rescues
# are long anyway: of the 34 matches the rescue recovers, 33 are >=16 characters;
# a length floor of 10 keeps every one while excluding the short coincidental tail
# (single generic words like "report"/"index", where per-token already suffices).
_WHOLE_STRING_MIN_LEN: int = 10

# IDF cancels in a single-token Jaccard: with one shared stem and no unique
# tokens, raw = idf(tok) / idf(tok) = 1.0 regardless of how generic the token is,
# so a lone common word ("Bridges" vs "The bridges") scores a perfect 100 — the
# same as a distinctive multi-word match. This inflates no-match separation-test
# pairs (#87 — ~4.1% of vault no_matches falsely high). The absolute shared IDF
# mass, not the ratio, is the discriminating signal: it is low for a thin generic
# overlap and high for a rich one. So the Jaccard score is multiplied by a
# confidence keyed on that mass, scaled by a single distinctive (once-seen)
# token's IDF (``default_idf``): a lone low-IDF token (mass << default) is
# discounted toward 0, a lone distinctive token (mass ~= default) clears it at
# ~1.0, and any multi-token title easily clears it. The factor is the tunable
# knob (#84 sweeps it); 1.0 means "one once-seen token is full confidence".
_GENERIC_TITLE_MASS_FACTOR: float = 1.0


def _align_tokens(
    marc_set: set[str], nypl_set: set[str]
) -> tuple[tuple[tuple[str, str], ...], frozenset[str], frozenset[str]]:
    """Align two stem sets, exact matches first then high-ratio fuzzy ones.

    Returns ``(matched_pairs, unique_marc, unique_nypl)``. Exact shared stems
    are paired with themselves; each remaining MARC stem is greedily paired
    with its single best unused CCE stem whose :func:`rapidfuzz.fuzz.ratio`
    meets :data:`_FUZZY_MIN_RATIO` (OCR/compound tolerance). Iteration is over
    sorted snapshots so the greedy choice is deterministic. A matched pair
    collapses into one shared unit (counted once in numerator and once in the
    union); unmatched stems remain unique to their side.
    """
    matched: list[tuple[str, str]] = []
    rem_marc = set(marc_set)
    rem_nypl = set(nypl_set)
    for token in sorted(marc_set & nypl_set):
        matched.append((token, token))
        rem_marc.discard(token)
        rem_nypl.discard(token)
    for marc_token in sorted(rem_marc):
        best_token: str | None = None
        best_ratio = _FUZZY_MIN_RATIO
        for nypl_token in sorted(rem_nypl):
            current = ratio(marc_token, nypl_token)
            if current >= best_ratio:
                best_ratio = current
                best_token = nypl_token
        if best_token is not None:
            matched.append((marc_token, best_token))
            rem_marc.discard(marc_token)
            rem_nypl.discard(best_token)
    return tuple(matched), frozenset(rem_marc), frozenset(rem_nypl)


def _shared_weight(marc_token: str, nypl_token: str, idf: IdfTable) -> float:
    """Return the IDF weight of one matched pair (their mean distinctiveness).

    For an exact pair the two IDFs are equal, so this is the token's own IDF
    and the whole scorer reduces to the original Jaccard. For an OCR pair the
    corrupted form is usually rarer (higher IDF); the mean keeps the shared
    unit's weight close to the clean token's true distinctiveness.
    """
    return (idf.score(marc_token) + idf.score(nypl_token)) / 2.0


def _prepare(value: str, ctx: ScorerContext) -> tuple[str, ...]:
    """Tokenize, drop stopwords, and stem ``value`` for the context language."""
    normalized = normalize_numbers(value, ctx.language)
    tokens = tokenize(normalized)
    filtered = [token for token in tokens if token not in ctx.stopwords.title]
    return tuple(ctx.stemmer(token) for token in filtered)


def _skipped() -> Evidence:
    return Evidence(
        scorer=_SCORER_NAME,
        score=0.0,
        max=_MAX_SCORE,
        skipped=True,
        decisive=False,
        features=(),
    )


def _script_mismatch_zero() -> Evidence:
    return Evidence(
        scorer=_SCORER_NAME,
        score=0.0,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=(("script_mismatch", 1.0),),
    )


def score_title(marc_title: str | None, nypl_title: str | None, ctx: ScorerContext) -> Evidence:
    """Return :class:`Evidence` for one (marc_title, nypl_title) pairing.

    Args:
        marc_title: MARC 245 ``$a$b`` value or ``None``.
        nypl_title: NYPL registration title or ``None``.
        ctx: Per-record :class:`ScorerContext`.

    Returns:
        An :class:`Evidence` whose ``score`` lies in ``[0, 100]``. The
        ``skipped`` flag is set when either input is empty or unusable.
        When the two sides use different dominant Unicode scripts, the
        scorer emits a non-skipped zero so the pair contributes to the
        combiner's denominator instead of silently dropping out.
    """
    if not marc_title or not nypl_title:
        return _skipped()
    if is_script_mismatch(marc_title, nypl_title):
        return _script_mismatch_zero()
    marc_tokens = _prepare(marc_title, ctx)
    nypl_tokens = _prepare(nypl_title, ctx)
    if not marc_tokens or not nypl_tokens:
        return _skipped()
    marc_set = set(marc_tokens)
    nypl_set = set(nypl_tokens)
    matched, unique_marc, unique_nypl = _align_tokens(marc_set, nypl_set)
    weighted_intersection = sum(_shared_weight(a, b, ctx.idf) for a, b in matched)
    weighted_union = (
        weighted_intersection
        + sum(ctx.idf.score(token) for token in unique_marc)
        + sum(ctx.idf.score(token) for token in unique_nypl)
    )
    raw = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
    mass_floor = _GENERIC_TITLE_MASS_FACTOR * ctx.idf.default_idf
    confidence = min(1.0, weighted_intersection / mass_floor) if mass_floor > 0 else 1.0
    score = raw * _MAX_SCORE * confidence
    marc_joined = "".join(marc_tokens)
    nypl_joined = "".join(nypl_tokens)
    if min(len(marc_joined), len(nypl_joined)) >= _WHOLE_STRING_MIN_LEN:
        whole_ratio = ratio(marc_joined, nypl_joined)
        if whole_ratio >= _WHOLE_STRING_MIN_RATIO:
            score = max(score, whole_ratio)
    token_total = len(matched) + len(unique_marc) + len(unique_nypl)
    avg_idf = (weighted_union / token_total) if token_total else 0.0
    features: tuple[tuple[str, float], ...] = (
        ("token_overlap", float(len(matched))),
        ("token_total", float(token_total)),
        ("unique_to_marc", float(len(unique_marc))),
        ("unique_to_nypl", float(len(unique_nypl))),
        ("avg_token_idf", avg_idf),
        ("marc_token_len", float(len(marc_set))),
        ("nypl_token_len", float(len(nypl_set))),
    )
    return Evidence(
        scorer=_SCORER_NAME,
        score=score,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=features,
    )


__all__ = [
    "score_title",
]
