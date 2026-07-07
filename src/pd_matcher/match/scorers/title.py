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

The CCE also routinely packs into its *title* string content MARC keeps in
separate fields — the publisher, the publication place, or the 245$c statement
of responsibility (#90). Those leaked tokens are unique to the CCE side and the
Jaccard penalizes them as a title difference, deflating true matches. Before
scoring, the CCE-title stems explained by a non-title MARC field (carried on the
context as :attr:`ScorerContext.cross_field_title_stems`) are stripped from the
CCE comparand — but never a stem that also belongs to the genuine MARC title,
and never to the point of emptying the comparand. Genuine *different-title*
content still penalizes, because it is not explained by any MARC field.

The symmetric Jaccard tanks when one title is much longer than the other — a
subtitle the CCE omits ("Oscar Wilde a biography" vs "Oscar Wilde"), or a
bound-with second work ("Babylone, suivi de la Vache la mort" vs "Babylone").
The longer side's extra distinctive tokens bloat the union and crater the score
even on a real match; the asymmetry is bidirectional (MARC or CCE longer). The
scorer emits an asymmetric ``coverage`` sub-feature for the learned combiner
(#85): shared IDF mass over the *smaller* side's total IDF mass — high (→1.0)
when the shorter title is a distinctive subset of the longer one, even when the
symmetric ``score`` is low. It is a FEATURE only; the ``score`` is deliberately
unchanged (a blunt coverage lift on the score regressed both arms in v1 because
it inflated coincidental-subset no_matches). The learned model weights coverage
in concert with author/year/publisher — high coverage with corroboration means
match, high coverage with an author mismatch is still rejected. The weighted
mean never sees it.
"""

from collections.abc import Callable

from rapidfuzz.fuzz import ratio

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.signals.script import scripts_mismatch
from pd_matcher.normalize.numbers import normalize_numbers
from pd_matcher.normalize.script import dominant_script
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

# The short/long title comparison the sliding window replaces (#133). When the
# two normalized token sequences differ substantially in length, one side is a
# subtitle-, translated-original-, or responsibility-statement-bloated version
# of the other and the symmetric Jaccard craters even on a true match (the extra
# distinctive tokens bloat the union). The window slides the shorter token
# sequence along the longer one and scores each position with the SAME
# IDF-weighted alignment machinery the symmetric path uses, then competes via
# ``max`` — crediting containment instead of punishing the surplus. The trigger
# is structural (a length ratio carried on the config, not a score), so the
# comparison only runs on the skewed pairs it targets, and the IDF weighting is
# the intrinsic generic-title guard: a window matched only on common tokens
# carries thin shared mass, so its confidence — and thus its score — stays near
# zero by construction ("Selected poems" containment cannot fire on filler).
# When the window wins the comparison the Evidence carries this flag (a
# diagnostic sub-feature the learned model ignores — it is absent from the
# canonical feature projection), so the pipeline can label the evidence source
# ``title_window`` and the review card shows the window fired.
TITLE_WINDOW_FEATURE: str = "title_window"


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


def _alignment_masses(
    marc_set: set[str], nypl_set: set[str], idf: IdfTable
) -> tuple[tuple[tuple[str, str], ...], frozenset[str], frozenset[str], float, float, float]:
    """Align two stem sets and return the aligned pairs plus their IDF masses.

    Returns ``(matched, unique_marc, unique_nypl, weighted_intersection,
    unique_marc_mass, unique_nypl_mass)``. The three masses are the IDF-weighted
    shared, MARC-only, and CCE-only sums that both the symmetric score and the
    sliding window (#133) build their similarity from, factored here so the two
    paths compute them identically.
    """
    matched, unique_marc, unique_nypl = _align_tokens(marc_set, nypl_set)
    weighted_intersection = sum(_shared_weight(a, b, idf) for a, b in matched)
    unique_marc_mass = sum(idf.score(token) for token in unique_marc)
    unique_nypl_mass = sum(idf.score(token) for token in unique_nypl)
    return (
        matched,
        unique_marc,
        unique_nypl,
        weighted_intersection,
        unique_marc_mass,
        unique_nypl_mass,
    )


def _similarity_from_masses(
    weighted_intersection: float,
    unique_marc_mass: float,
    unique_nypl_mass: float,
    mass_floor: float,
) -> float:
    """Return the IDF-weighted Jaccard score, mass-confidence scaled, in ``[0, 100]``.

    The IDF-weighted Jaccard ratio (shared mass over union mass) is multiplied
    by the absolute-mass confidence that keeps a thin generic overlap off a
    perfect score (see :data:`_GENERIC_TITLE_MASS_FACTOR`). Both the symmetric
    path and the sliding window score through this one function so a window
    matched only on filler is driven toward zero by the same guard.
    """
    weighted_union = weighted_intersection + unique_marc_mass + unique_nypl_mass
    raw = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
    confidence = min(1.0, weighted_intersection / mass_floor) if mass_floor > 0 else 1.0
    return raw * _MAX_SCORE * confidence


def _best_window_score(
    marc_tokens: tuple[str, ...],
    nypl_tokens: tuple[str, ...],
    idf: IdfTable,
    mass_floor: float,
    trigger_ratio: float,
) -> float:
    """Return the best sliding-window containment score, or ``0.0`` when idle (#133).

    Fires only when the two normalized token sequences differ substantially in
    length — ``len(shorter) / len(longer) <= trigger_ratio`` — a structural
    trigger that is side-agnostic (MARC or CCE may be the longer side). A window
    of ``len(shorter)`` tokens is slid across the longer sequence and each
    position scored with :func:`_similarity_from_masses` over the same
    IDF-weighted alignment the symmetric path uses, so a window that lands on the
    shorter title's distinctive tokens scores high (containment credited) while
    one matched only on common tokens stays near zero (thin shared mass). Returns
    the maximum score over all window positions; ``0.0`` when the trigger is not
    met, either side is empty, or the sequences are the same length.
    """
    if not marc_tokens or not nypl_tokens or trigger_ratio <= 0.0:
        return 0.0
    short, long = (
        (marc_tokens, nypl_tokens)
        if len(marc_tokens) <= len(nypl_tokens)
        else (nypl_tokens, marc_tokens)
    )
    short_len = len(short)
    long_len = len(long)
    if short_len == long_len or short_len / long_len > trigger_ratio:
        return 0.0
    short_set = set(short)
    best = 0.0
    for start in range(long_len - short_len + 1):
        window_set = set(long[start : start + short_len])
        _, _, _, intersection, marc_mass, nypl_mass = _alignment_masses(short_set, window_set, idf)
        best = max(best, _similarity_from_masses(intersection, marc_mass, nypl_mass, mass_floor))
    return best


def _tokenize_filter_stem(
    normalized: str,
    title_stopwords: frozenset[str],
    stemmer: Callable[[str], str],
) -> tuple[str, ...]:
    """Tokenize, drop title stopwords, and stem an already-number-normalized string."""
    tokens = tokenize(normalized)
    filtered = [token for token in tokens if token not in title_stopwords]
    return tuple(stemmer(token) for token in filtered)


def _prepare_tokens(
    value: str,
    language: str,
    title_stopwords: frozenset[str],
    stemmer: Callable[[str], str],
) -> tuple[str, ...]:
    """Normalize numbers, tokenize, drop title stopwords, and stem ``value``."""
    return _tokenize_filter_stem(normalize_numbers(value, language), title_stopwords, stemmer)


def _coverage(
    weighted_intersection: float, unique_marc_mass: float, unique_nypl_mass: float
) -> float:
    """Return the asymmetric title coverage signal (#85).

    Coverage is the shared IDF mass divided by the total IDF mass of the
    *smaller* side — ``shared / min(marc_mass, cce_mass)`` where each side's
    mass is its shared mass plus its own unique mass. It approaches ``1.0``
    when the shorter title's distinctive content is mostly present in the
    longer one (a subtitle the other side omits, or a bound-with second work),
    precisely the asymmetric shape the symmetric Jaccard ``score`` deflates.
    It is ``0.0`` when there is no shared mass or both sides are empty.

    Reuses the masses the scorer has already computed from its IDF / normalize
    / tokenize / stem pipeline, so the signal is consistent with ``score``.

    Args:
        weighted_intersection: Shared IDF mass over the aligned token pairs.
        unique_marc_mass: IDF mass of the MARC-only stems.
        unique_nypl_mass: IDF mass of the CCE-only stems.

    Returns:
        Coverage in ``[0.0, 1.0]``.
    """
    marc_mass = weighted_intersection + unique_marc_mass
    cce_mass = weighted_intersection + unique_nypl_mass
    smaller_mass = min(marc_mass, cce_mass)
    if smaller_mass <= 0.0:
        return 0.0
    return weighted_intersection / smaller_mass


def _prepare(value: str, ctx: ScorerContext) -> tuple[str, ...]:
    """Tokenize, drop stopwords, and stem ``value`` for the context language.

    Number-normalization is routed through :meth:`ScorerContext.normalize_numbers`
    so a MARC field re-scored against every candidate is normalized once per
    record; the result is byte-identical to :func:`_prepare_tokens`.
    """
    return _tokenize_filter_stem(ctx.normalize_numbers(value), ctx.stopwords.title, ctx.stemmer)


def prepare_cross_field_stems(
    values: tuple[str, ...],
    language: str,
    title_stopwords: frozenset[str],
    stemmer: Callable[[str], str],
) -> frozenset[str]:
    """Build the cross-field stem set for :class:`ScorerContext` (#90).

    Prepares the MARC publisher / publication-place / statement-of-responsibility
    strings with the **same** normalize/tokenize/stopword/stem pipeline the title
    scorer applies to titles, so the resulting stems are directly comparable to
    prepared CCE-title stems. ``None`` field values must be filtered by the caller
    before being passed in. Returns a :class:`frozenset` ready to drop straight
    into :attr:`ScorerContext.cross_field_title_stems`.
    """
    stems: set[str] = set()
    for value in values:
        stems.update(_prepare_tokens(value, language, title_stopwords, stemmer))
    return frozenset(stems)


def _strip_cross_field(
    nypl_tokens: tuple[str, ...], marc_tokens: tuple[str, ...], ctx: ScorerContext
) -> tuple[str, ...]:
    """Drop CCE-title stems explained by a non-title MARC field (#90).

    The CCE routinely packs the publisher, publication place, or statement of
    responsibility *into its title* string — fields MARC keeps separate. Those
    leaked tokens are unique to the CCE side, so the IDF-weighted similarity
    penalizes them as a title difference and deflates a true match. This removes
    any CCE-title stem present in :attr:`ScorerContext.cross_field_title_stems`
    (the prepared publisher/place/responsibility stems) **unless** the stem also
    belongs to the genuine MARC title — a token that legitimately appears in the
    title is never stripped just because it recurs in another field. The strip
    is suppressed when it would empty the CCE comparand, so an over-aggressive
    removal can never manufacture a zero-token skip.
    """
    explained = ctx.cross_field_title_stems
    if not explained:
        return nypl_tokens
    marc_set = frozenset(marc_tokens)
    kept = tuple(token for token in nypl_tokens if token not in explained or token in marc_set)
    if not kept:
        return nypl_tokens
    return kept


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


def score_title(
    marc_title: str | None,
    nypl_title: str | None,
    ctx: ScorerContext,
    *,
    nypl_title_script: str | None = None,
) -> Evidence:
    """Return :class:`Evidence` for one (marc_title, nypl_title) pairing.

    Args:
        marc_title: MARC 245 ``$a$b`` value or ``None``.
        nypl_title: NYPL registration title or ``None``.
        ctx: Per-record :class:`ScorerContext`.
        nypl_title_script: The CCE title's dominant script, precomputed at
            index build on
            :attr:`~pd_matcher.models.IndexedNyplRegRecord.title_script` and
            threaded in by the pipeline when ``nypl_title`` is the candidate's
            canonical title. ``None`` means "not supplied"; the scorer then
            derives the script itself, so the result is byte-identical whether
            or not the precomputed value is passed. (A genuinely scriptless
            CCE title also yields ``None`` from the derivation, so the
            recomputed and absent cases collapse to the same behavior.)

    Returns:
        An :class:`Evidence` whose ``score`` lies in ``[0, 100]``. The
        ``skipped`` flag is set when either input is empty or unusable.
        When the two sides use different dominant Unicode scripts, the
        scorer emits a non-skipped zero so the pair contributes to the
        combiner's denominator instead of silently dropping out.
    """
    if not marc_title or not nypl_title:
        return _skipped()
    cce_script = nypl_title_script if nypl_title_script is not None else dominant_script(nypl_title)
    if scripts_mismatch(ctx.marc_title_script(marc_title), cce_script):
        return _script_mismatch_zero()
    marc_tokens = _prepare(marc_title, ctx)
    nypl_tokens = _prepare(nypl_title, ctx)
    if not marc_tokens or not nypl_tokens:
        return _skipped()
    nypl_tokens = _strip_cross_field(nypl_tokens, marc_tokens, ctx)
    marc_set = set(marc_tokens)
    nypl_set = set(nypl_tokens)
    matched, unique_marc, unique_nypl, weighted_intersection, unique_marc_mass, unique_nypl_mass = (
        _alignment_masses(marc_set, nypl_set, ctx.idf)
    )
    weighted_union = weighted_intersection + unique_marc_mass + unique_nypl_mass
    coverage = _coverage(weighted_intersection, unique_marc_mass, unique_nypl_mass)
    mass_floor = _GENERIC_TITLE_MASS_FACTOR * ctx.idf.default_idf
    score = _similarity_from_masses(
        weighted_intersection, unique_marc_mass, unique_nypl_mass, mass_floor
    )
    marc_joined = "".join(marc_tokens)
    nypl_joined = "".join(nypl_tokens)
    if min(len(marc_joined), len(nypl_joined)) >= _WHOLE_STRING_MIN_LEN:
        whole_ratio = ratio(marc_joined, nypl_joined)
        if whole_ratio >= _WHOLE_STRING_MIN_RATIO:
            score = max(score, whole_ratio)
    window_score = _best_window_score(
        marc_tokens, nypl_tokens, ctx.idf, mass_floor, ctx.config.title_window_trigger_ratio
    )
    window_fired = window_score > score
    score = max(score, window_score)
    token_total = len(matched) + len(unique_marc) + len(unique_nypl)
    avg_idf = (weighted_union / token_total) if token_total else 0.0
    features: tuple[tuple[str, float], ...] = (
        ("token_overlap", float(len(matched))),
        ("token_total", float(token_total)),
        ("unique_to_marc", float(len(unique_marc))),
        ("unique_to_nypl", float(len(unique_nypl))),
        ("avg_token_idf", avg_idf),
        ("coverage", coverage),
        ("marc_token_len", float(len(marc_set))),
        ("nypl_token_len", float(len(nypl_set))),
    )
    if window_fired:
        features = (*features, (TITLE_WINDOW_FEATURE, 1.0))
    return Evidence(
        scorer=_SCORER_NAME,
        score=score,
        max=_MAX_SCORE,
        skipped=False,
        decisive=False,
        features=features,
    )


__all__ = [
    "TITLE_WINDOW_FEATURE",
    "prepare_cross_field_stems",
    "score_title",
]
