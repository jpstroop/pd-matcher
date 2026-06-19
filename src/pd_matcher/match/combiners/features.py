"""Canonical Evidence-to-vector projection for the learned combiner.

The learned (LightGBM) combiner consumes a fixed-shape numeric feature
vector, not the raw :class:`pd_matcher.match.evidence.Evidence` stream. This
module owns the single canonical projection: :func:`feature_names` names the
columns in a deterministic order and :func:`feature_row` produces a parallel
value tuple for one pair's winning Evidence. Training (``train-scorer``) and
inference (:class:`pd_matcher.match.combiners.learned.LearnedCombiner`) MUST
go through these two functions so the model's stored feature names always
match what inference feeds it.

The projection is the validated expanded family from the issue #4
tightening round (``docs/findings/learned_scorer_tightening_2026-06-12.md``):

* 8 per-scorer normalized scores in :data:`SCORER_ORDER`.
* 8 ``{scorer}__skipped`` flags.
* Named sub-features flattened out of each scorer's ``Evidence.features``,
  namespaced ``{scorer}.{feature}`` because author and publisher share
  sub-feature names. Where a ``-1.0`` sentinel or a skipped scorer makes a
  raw ``0.0`` ambiguous, a ``{scorer}.{feature}__present`` companion flag is
  emitted (see :data:`_PRESENCE_FLAGGED`).
* One pair-level computable, ``pair.title_len_ratio``, derived from the
  title scorer's ``marc_token_len`` / ``nypl_token_len`` sub-features
  (``0.0`` when either is ``0`` or absent).

The expanded count is **49**: 16 baseline + 32 named sub-features (incl.
presence flags and the two title token-length sub-features) + 1 pair-level.
``year.delta`` was dropped as a scoring feature in issue #88: exact-year
retrieval bucketing (``year_window = 0``) makes its delta a constant ``1.0``
for every scored pair, so it carried zero variance — uninformative to the
learned trees and pure inflation in the weighted mean. Year remains the
retrieval bucket key; it is no longer a combiner feature. The canonical
count here is authoritative and is asserted by the unit tests.
"""

from collections.abc import Sequence

from pd_matcher.match.evidence import Evidence

# Canonical per-scorer order. The matcher emits exactly one Evidence per name
# here; both the learned combiner and the eval feature matrix project the
# Evidence stream against this order. Defined in the production match package
# (not eval) so production code never imports the eval package.
SCORER_ORDER: tuple[str, ...] = (
    "title.token_set",
    "name.author",
    "name.publisher",
    "edition.compat",
    "lccn.exact",
    "isbn.exact",
    "extent.page_count",
    "volume.compat",
)

_TITLE_SCORER: str = "title.token_set"
_MARC_TOKEN_LEN: str = "marc_token_len"
_NYPL_TOKEN_LEN: str = "nypl_token_len"

# Named sub-features emitted by each scorer's ``Evidence.features``,
# namespaced by scorer name. author and publisher share the same evidence
# builder, so their sub-feature names collide; the namespace prefix keeps the
# columns distinct. Every name is read directly off the scorer source; absent
# features default to ``0.0`` plus a presence flag where the absence matters.
_NAMED_SUBFEATURES: dict[str, tuple[str, ...]] = {
    "title.token_set": (
        "token_overlap",
        "token_total",
        "unique_to_marc",
        "unique_to_nypl",
        "avg_token_idf",
        "script_mismatch",
        "marc_token_len",
        "nypl_token_len",
    ),
    "name.author": (
        "normalized_marc_len",
        "normalized_nypl_len",
        "token_overlap",
    ),
    "name.publisher": (
        "normalized_marc_len",
        "normalized_nypl_len",
        "token_overlap",
    ),
    "edition.compat": (
        "marc_edition_num",
        "nypl_edition_num",
        "explicit_mismatch",
    ),
    "lccn.exact": (
        "marc_lccn",
        "nypl_lccn_present",
    ),
    "isbn.exact": ("marc_isbn_count",),
    "extent.page_count": (
        "marc_pages",
        "cce_pages",
        "delta",
    ),
    "volume.compat": (
        "marc_is_whole",
        "marc_is_whole_open",
        "marc_is_part",
        "cce_is_whole",
        "cce_is_part",
    ),
}

# Sub-features whose value space includes a sentinel (``-1.0`` = "absent"); a
# companion ``__present`` flag disambiguates "value is genuinely -1" from
# "missing".
_PRESENCE_FLAGGED: dict[str, tuple[str, ...]] = {
    "edition.compat": ("marc_edition_num", "nypl_edition_num"),
    "extent.page_count": ("marc_pages", "cce_pages"),
}

_TITLE_LEN_RATIO: str = "pair.title_len_ratio"


def feature_names() -> tuple[str, ...]:
    """Return the canonical feature-column order for the learned combiner.

    The order is, in full: 9 normalized scorer scores in
    :data:`SCORER_ORDER`; 9 ``{scorer}__skipped`` flags; the namespaced
    named sub-features (each optionally followed by its
    ``{scorer}.{feature}__present`` flag); and finally
    ``pair.title_len_ratio``. The exact order is load-bearing: a trained
    model stores these names and inference asserts equality against them.
    """
    names: list[str] = list(SCORER_ORDER)
    for scorer in SCORER_ORDER:
        names.append(f"{scorer}__skipped")
    for scorer in SCORER_ORDER:
        for feature in _NAMED_SUBFEATURES[scorer]:
            names.append(f"{scorer}.{feature}")
            if feature in _PRESENCE_FLAGGED.get(scorer, ()):
                names.append(f"{scorer}.{feature}__present")
    names.append(_TITLE_LEN_RATIO)
    return tuple(names)


def _evidence_by_scorer(evidence: Sequence[Evidence]) -> dict[str, Evidence]:
    """Index winning Evidence by scorer name for O(1) lookup."""
    return {item.scorer: item for item in evidence}


def _title_len_ratio(title: Evidence | None) -> float:
    """Return ``marc_token_len / nypl_token_len`` from the title Evidence.

    Reads the two token-length sub-features the title scorer emits. Returns
    ``0.0`` when the title scorer is absent, skipped, or either side has zero
    (or absent) tokens — the same zero-denominator guard the tightening run
    used, expressed against the Evidence features rather than the raw titles.
    """
    if title is None:
        return 0.0
    named = dict(title.features)
    marc_tokens = named.get(_MARC_TOKEN_LEN, 0.0)
    nypl_tokens = named.get(_NYPL_TOKEN_LEN, 0.0)
    if marc_tokens == 0.0 or nypl_tokens == 0.0:
        return 0.0
    return marc_tokens / nypl_tokens


def feature_row(evidence: Sequence[Evidence]) -> tuple[float, ...]:
    """Project one pair's winning Evidence into the canonical feature vector.

    The returned tuple has the same length and order as
    :func:`feature_names`. A scorer absent from ``evidence`` contributes a
    skipped flag of ``0.0``, named sub-features of ``0.0``, and presence
    flags of ``0.0`` — the same shape an explicitly skipped scorer yields,
    so the model sees a consistent ambiguity signal either way.

    Args:
        evidence: The winning per-scorer Evidence for one candidate pair.

    Returns:
        A ``float`` tuple of length ``len(feature_names())``.
    """
    by_name = _evidence_by_scorer(evidence)
    scores: list[float] = []
    flags: list[float] = []
    for scorer in SCORER_ORDER:
        item = by_name.get(scorer)
        if item is None:
            scores.append(0.0)
            flags.append(0.0)
        else:
            scores.append(item.normalized)
            flags.append(1.0 if item.skipped else 0.0)
    values: list[float] = scores + flags
    for scorer in SCORER_ORDER:
        item = by_name.get(scorer)
        named = dict(item.features) if item is not None else {}
        present = item is not None and not item.skipped
        for feature in _NAMED_SUBFEATURES[scorer]:
            raw = named.get(feature)
            values.append(raw if raw is not None else 0.0)
            if feature in _PRESENCE_FLAGGED.get(scorer, ()):
                has_value = present and raw is not None
                values.append(1.0 if has_value else 0.0)
    values.append(_title_len_ratio(by_name.get(_TITLE_SCORER)))
    return tuple(values)


__all__ = [
    "SCORER_ORDER",
    "feature_names",
    "feature_row",
]
