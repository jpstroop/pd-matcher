"""Active-learning selection over an UNLABELED candidate pool (issue #81).

Selects in-scope MARC records that are NOT already in the label vault, sampled
per language toward an English-weighted target so the downstream dual-scoring
pass spends the human-label budget on matcher-vs-matcher disagreements rather
than on whatever a stratified sweep happened to surface.

This is the unlabeled-pool sibling of ``scripts/disagreement_scan.py`` (which
ranks the *labeled* vault by model-vs-vault disagreement). Selection here is
deliberately separated from scoring (:mod:`pd_groundtruth.active_score`) and
from IO orchestration (:mod:`pd_groundtruth.active_learning`) so the sampling
policy — vault exclusion, in-scope filtering, language weighting — is a pure,
fully unit-testable transform over a per-language record source.

In scope means: a parsed :class:`~pd_matcher.models.MarcRecord` with a
``publication_year`` (the matcher cannot retrieve candidates without one). The
moving-wall lower bound is the acquire step's job; selection trusts the pool.
"""

from collections.abc import Callable
from collections.abc import Iterator
from logging import getLogger

from msgspec import Struct

from pd_groundtruth.sampling import reservoir_sample
from pd_matcher.models import MarcRecord

_LOGGER = getLogger(__name__)

# English-weighted default mix. The non-English share mirrors the four CCE
# languages the queue budget already weights (fre/ger/ita/spa) so a single
# selection run still surfaces non-English disagreements without drowning the
# English signal. Fractions sum to 1.0; the per-language target is the fraction
# times the run's overall ``target`` (rounded, minimum 1 for any non-zero
# weight so no configured language is silently dropped).
DEFAULT_LANGUAGE_WEIGHTS: dict[str, float] = {
    "eng": 0.70,
    "fre": 0.075,
    "ger": 0.075,
    "ita": 0.075,
    "spa": 0.075,
}

# A per-language record source: given a language code, yield every parsed MARC
# record the pool holds for it. Injected so selection is testable without a
# sharded pool on disk (the orchestration binds the real pool walker).
RecordSource = Callable[[str], Iterator[MarcRecord]]


class LanguagePlan(Struct, frozen=True, forbid_unknown_fields=True):
    """One language's resolved selection target and realized count."""

    language: str
    target: int
    selected: int


class SelectionResult(Struct, frozen=True, forbid_unknown_fields=True):
    """Outcome of one selection pass.

    ``records`` is the flat, language-interleaved selection in language order;
    ``plans`` reports the per-language target vs realized count; ``excluded``
    is the number of candidate records dropped because their MARC was already
    in the vault; ``out_of_scope`` counts records skipped for lacking a
    publication year.
    """

    records: tuple[MarcRecord, ...]
    plans: tuple[LanguagePlan, ...]
    excluded: int
    out_of_scope: int


def resolve_language_targets(
    target: int,
    weights: dict[str, float],
) -> dict[str, int]:
    """Return the per-language record target for an overall ``target``.

    Each language's target is ``round(weight * target)`` with a floor of ``1``
    for any strictly-positive weight so no configured language is rounded away.

    Raises:
        ValueError: If ``target`` is not positive, ``weights`` is empty, or any
            weight is negative.
    """
    if target <= 0:
        raise ValueError(f"target must be positive (got {target!r})")
    if not weights:
        raise ValueError("weights must not be empty")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("weights must not be negative")
    resolved: dict[str, int] = {}
    for language, weight in weights.items():
        if weight <= 0.0:
            resolved[language] = 0
            continue
        resolved[language] = max(1, round(weight * target))
    return resolved


def _is_in_scope(record: MarcRecord) -> bool:
    """Return whether ``record`` can be matched (it has a publication year)."""
    return record.publication_year is not None


def select_records(
    *,
    source: RecordSource,
    weights: dict[str, float],
    target: int,
    excluded_marc_ids: frozenset[str],
    seed: int,
) -> SelectionResult:
    """Select ~``target`` unseen, in-scope records, language-weighted.

    For each language with a positive resolved target, ``source(language)`` is
    streamed once; records whose ``control_id`` is in ``excluded_marc_ids``
    (already in the vault) are dropped before sampling, records without a
    publication year are counted as out-of-scope and dropped, and the survivors
    feed a deterministic reservoir draw of the language's target size. The
    per-language seed is ``seed ^ hash(language)`` so each language samples
    independently yet reproducibly (mirrors
    :class:`pd_groundtruth.sampling.Stratifier`).

    Args:
        source: Per-language parsed-record stream.
        weights: Language -> relative weight (English-heavy default lives in
            :data:`DEFAULT_LANGUAGE_WEIGHTS`).
        target: Overall number of records to aim for across all languages.
        excluded_marc_ids: MARC control ids already in the vault; never
            selected.
        seed: Base seed for the per-language reservoir draws.

    Returns:
        A :class:`SelectionResult`.
    """
    targets = resolve_language_targets(target, weights)
    selected: list[MarcRecord] = []
    plans: list[LanguagePlan] = []
    excluded = 0
    out_of_scope = 0
    for language in targets:
        language_target = targets[language]
        if language_target <= 0:
            plans.append(LanguagePlan(language=language, target=0, selected=0))
            continue
        eligible: list[MarcRecord] = []
        for record in source(language):
            if record.control_id in excluded_marc_ids:
                excluded += 1
                continue
            if not _is_in_scope(record):
                out_of_scope += 1
                continue
            eligible.append(record)
        drawn = reservoir_sample(eligible, language_target, seed ^ hash(language))
        selected.extend(drawn)
        plans.append(LanguagePlan(language=language, target=language_target, selected=len(drawn)))
        _LOGGER.info(
            "active.select language=%s target=%d eligible=%d selected=%d",
            language,
            language_target,
            len(eligible),
            len(drawn),
        )
    return SelectionResult(
        records=tuple(selected),
        plans=tuple(plans),
        excluded=excluded,
        out_of_scope=out_of_scope,
    )


__all__ = [
    "DEFAULT_LANGUAGE_WEIGHTS",
    "LanguagePlan",
    "RecordSource",
    "SelectionResult",
    "resolve_language_targets",
    "select_records",
]
