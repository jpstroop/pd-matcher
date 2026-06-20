"""Active-learning queue builder: select, dual-score, bucket, queue (issue #81).

Orchestrates the unlabeled-pool active-learning loop end to end:

1. **Select** ~N unseen, in-scope MARC records from the candidate pool,
   excluding every MARC already in the label vault, language-weighted toward
   English (:mod:`pd_groundtruth.active_select`).
2. **Dual-score** each selected MARC: retrieve its CCE candidates from the
   index, compute per-scorer Evidence once, and apply BOTH combiners
   (weighted-mean + learned) to that single Evidence
   (:mod:`pd_groundtruth.active_score`).
3. **Bucket** by matcher-vs-matcher disagreement and rank the ``informative``
   bucket by disagreement magnitude.
4. **Queue** the informative pairs into a :class:`~pd_groundtruth.review_db.ReviewDb`
   (reviewable via ``pd-groundtruth review --db <path>``) and print a per-bucket
   distribution summary. ``--dry-run`` prints the distribution but writes nothing.

The learned model is REQUIRED here — the whole point is committee disagreement —
so a missing artifact or absent ``lightgbm`` aborts the run with a clear message
naming ``train-scorer`` / ``pdm install --group ml`` rather than silently
degrading to a single matcher.

The heavy IO (LMDB, pool walk, IDF caches) lives in :func:`run_active_learning`,
which resolves paths into the injectable callables the pure passes consume; the
record-source binding and per-bucket reporting are split out so each is unit
testable without real data.
"""

from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

from msgspec import Struct
from msgspec.structs import replace

from pd_groundtruth.active_score import BUCKET_INFORMATIVE
from pd_groundtruth.active_score import BUCKET_ORDER
from pd_groundtruth.active_score import CandidateScorer
from pd_groundtruth.active_score import ScoredRecord
from pd_groundtruth.active_score import score_record
from pd_groundtruth.active_select import DEFAULT_LANGUAGE_WEIGHTS
from pd_groundtruth.active_select import LanguagePlan
from pd_groundtruth.active_select import RecordSource
from pd_groundtruth.active_select import select_records
from pd_groundtruth.build_queue import _build_pair_insert
from pd_groundtruth.build_queue import _language_of
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.vault_pair_resolver import AUTHOR_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import PUBLISHER_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.idf import load_or_build_author_idf
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.idf import load_or_build_publisher_idf
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.result import CandidateMatch
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.parsers.marc import iter_marc_records

_LOGGER = getLogger(__name__)

_LEARNED_SCORER: str = "learned"
_DEFAULT_TARGET: int = 1000

_BUCKET_BAND: dict[str, str] = {bucket: bucket for bucket in BUCKET_ORDER}


class BucketStats(Struct, frozen=True, forbid_unknown_fields=True):
    """Per-bucket count plus the disagreement-magnitude spread within it.

    ``min_disagreement`` / ``max_disagreement`` / ``mean_disagreement`` are
    ``0.0`` when the bucket is empty.
    """

    bucket: str
    count: int
    min_disagreement: float
    max_disagreement: float
    mean_disagreement: float


class ActiveLearningSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Outcome of one active-learning run, for the CLI to render.

    ``written`` is ``0`` on a dry run (the informative pairs are computed and
    reported but never persisted).
    """

    selected: int
    excluded: int
    out_of_scope: int
    scored: int
    buckets: tuple[BucketStats, ...]
    written: int
    dry_run: bool
    language_plans: tuple[LanguagePlan, ...]

    def informative(self) -> int:
        """Return the informative-bucket count (the queue-eligible pairs)."""
        return sum(s.count for s in self.buckets if s.bucket == BUCKET_INFORMATIVE)


def _bucket_stats(scored: list[ScoredRecord]) -> tuple[BucketStats, ...]:
    """Summarize the per-bucket counts and disagreement spread of ``scored``."""
    by_bucket: dict[str, list[float]] = {bucket: [] for bucket in BUCKET_ORDER}
    for record in scored:
        by_bucket[record.bucket].append(record.disagreement)
    stats: list[BucketStats] = []
    for bucket in BUCKET_ORDER:
        magnitudes = by_bucket[bucket]
        if magnitudes:
            stats.append(
                BucketStats(
                    bucket=bucket,
                    count=len(magnitudes),
                    min_disagreement=min(magnitudes),
                    max_disagreement=max(magnitudes),
                    mean_disagreement=sum(magnitudes) / len(magnitudes),
                )
            )
        else:
            stats.append(
                BucketStats(
                    bucket=bucket,
                    count=0,
                    min_disagreement=0.0,
                    max_disagreement=0.0,
                    mean_disagreement=0.0,
                )
            )
    return tuple(stats)


def _informative_ranked(scored: list[ScoredRecord]) -> list[ScoredRecord]:
    """Return the informative records, most-disagreeing first."""
    informative = [record for record in scored if record.bucket == BUCKET_INFORMATIVE]
    informative.sort(key=lambda record: record.disagreement, reverse=True)
    return informative


def _write_informative(records: list[ScoredRecord], out_path: Path) -> int:
    """Insert the ranked informative pairs into a fresh review DB; return the count.

    Each record's WEIGHTED top-1 CCE pair is inserted UNLABELED, stamped with
    the ``informative`` band so the review UI pages the active-learning set via
    ``/?band=informative``, and annotated with both combiners' verdicts so the
    labeler sees exactly how the matchers split. Every informative record has a
    retrieved CCE by construction (a record with no candidate buckets as
    ``agree-low``), so ``record.cce`` is never ``None`` here.
    """
    written = 0
    with ReviewDb.connect(out_path) as db:
        for record in records:
            if record.cce is None:  # pragma: no cover
                continue
            pair = _build_pair_insert(
                record.marc,
                record.cce,
                record.evidence,
                language=_language_of(record.marc),
                score=record.weighted.score,
                band=_BUCKET_BAND[BUCKET_INFORMATIVE],
                source=SOURCE_BANDED,
                evidence_sources=record.evidence_sources,
                audit_note=(
                    f"weighted={record.weighted.score:.2f} "
                    f"learned={record.learned.score:.2f} "
                    f"disagreement={record.disagreement:.2f} [active-learning]"
                ),
            )
            db.insert_pair(pair)
            written += 1
        db.commit()
    return written


def _weighted_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` forced onto the weighted-mean scorer for the shared pass."""
    if config.scorer != _LEARNED_SCORER:
        return config
    return replace(config, scorer="weighted_mean")


def _learned_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` forced onto the learned scorer."""
    return replace(config, scorer=_LEARNED_SCORER)


def _pool_record_source(pool: Path) -> RecordSource:
    """Return a per-language parsed-record stream over the sharded pool.

    ``source(language)`` streams ``<pool>/<language>/*.xml`` once. A missing
    language directory yields nothing rather than raising, so a language present
    in the weights but absent from the pool simply contributes no records.
    """

    def source(language: str) -> Iterator[MarcRecord]:
        language_dir = pool / language
        if not language_dir.is_dir():
            return
        for shard in sorted(language_dir.glob("*.xml")):
            yield from iter_marc_records(shard)

    return source


def build_active_learning_summary(
    *,
    source: RecordSource,
    candidate_scorer: CandidateScorer,
    learned: Combiner,
    excluded_marc_ids: frozenset[str],
    weights: dict[str, float],
    target: int,
    seed: int,
    out_path: Path,
    dry_run: bool,
) -> ActiveLearningSummary:
    """Run select → dual-score → bucket → (optionally) write, returning a summary.

    The pure core of the loop: it consumes already-bound callables (a
    per-language record source, a candidate scorer, the learned combiner) so it
    is unit testable without LMDB or the real artifact. The selection,
    bucketing, and ranking are delegated to the dedicated modules; this function
    only sequences them and persists the informative pairs (unless ``dry_run``).
    """
    selection = select_records(
        source=source,
        weights=weights,
        target=target,
        excluded_marc_ids=excluded_marc_ids,
        seed=seed,
    )
    scored = [
        score_record(record, candidate_scorer=candidate_scorer, learned=learned)
        for record in selection.records
    ]
    buckets = _bucket_stats(scored)
    informative = _informative_ranked(scored)
    written = 0 if dry_run else _write_informative(informative, out_path)
    return ActiveLearningSummary(
        selected=len(selection.records),
        excluded=selection.excluded,
        out_of_scope=selection.out_of_scope,
        scored=len(scored),
        buckets=buckets,
        written=written,
        dry_run=dry_run,
        language_plans=selection.plans,
    )


def run_active_learning(
    *,
    pool: Path,
    index_path: Path,
    out_path: Path,
    vault_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    weights: dict[str, float] = DEFAULT_LANGUAGE_WEIGHTS,
    target: int = _DEFAULT_TARGET,
    seed: int = 42,
    dry_run: bool = False,
) -> ActiveLearningSummary:
    """Resolve resources and run one active-learning selection end to end.

    Loads the IDF caches, opens the CCE index, builds the weighted-mean per-pair
    scorer and the REQUIRED learned combiner (aborting with a clear message when
    its artifact or ``lightgbm`` is missing), binds the pool record source and
    the candidate scorer, and delegates to :func:`build_active_learning_summary`.

    Args:
        pool: Root dir whose ``<lang>/*.xml`` shards form the candidate pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination review DB for the informative pairs.
        vault_path: JSONL label vault; its MARCs are excluded from selection.
        matching_config: Active matcher config; the weighted-mean scorer drives
            the shared Evidence pass regardless of its ``scorer`` value.
        pairing_config: Active field-pairing config.
        weights: Language -> relative selection weight.
        target: Overall number of records to select.
        seed: Base seed for the per-language reservoir draws.
        dry_run: When ``True``, compute and report but write no DB.

    Raises:
        ValueError: When the learned-model artifact is missing or stale.
        ImportError: When the optional ``lightgbm`` dependency is absent.
    """
    pairings = compile_pairings(pairing_config)
    idf = load_or_build_idf(index_path.parent / IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path))
    author_idf = load_or_build_author_idf(
        index_path.parent / AUTHOR_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    publisher_idf = load_or_build_publisher_idf(
        index_path.parent / PUBLISHER_IDF_CACHE_NAME, lambda: NyplIndexLookup(index_path)
    )
    weighted_config = _weighted_config(matching_config)
    score_pair = make_pair_scorer(
        matching_config=weighted_config,
        pairings=pairings,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        calibrator=None,
    )
    learned = build_combiner(_learned_config(matching_config), learned_model_dir=index_path.parent)
    excluded_marc_ids = frozenset(marc_id for marc_id, _uuid in current_entries(vault_path))
    source = _pool_record_source(pool)

    with NyplIndexLookup(index_path) as lookup:

        def candidate_scorer(
            marc: MarcRecord,
        ) -> Iterator[tuple[IndexedNyplRegRecord, CandidateMatch]]:
            for cce in lookup.candidates_for(marc, weighted_config.year_window):
                yield cce, score_pair(marc, cce)

        return build_active_learning_summary(
            source=source,
            candidate_scorer=candidate_scorer,
            learned=learned,
            excluded_marc_ids=excluded_marc_ids,
            weights=weights,
            target=target,
            seed=seed,
            out_path=out_path,
            dry_run=dry_run,
        )


__all__ = [
    "ActiveLearningSummary",
    "BucketStats",
    "build_active_learning_summary",
    "run_active_learning",
]
