"""Lightweight precision/recall + confusion-matrix evaluator.

Drives a curated ground-truth CSV (the shape of
``data/combined_ground_truth.csv``) through the full Phase 4 matcher and
Phase 5 rule engine and returns an :class:`EvalReport` summarising how
the predictions compare against the recorded labels. Phase 7's job is to
make this runnable today; Phase 8 will layer baseline JSON, regression
gates, and per-status breakdowns on top.

The driver supports two execution modes:

* ``workers=1`` (default): a single in-process loop. Fast to start, fully
  deterministic, and the path the unit tests exercise by default.
* ``workers>=2``: a ``spawn``-based multiprocessing pool. Each worker
  opens the LMDB index read-only once at init (mmap-shared across
  workers via the OS page cache), builds the IDF table once, and then
  scores rows handed to it through ``imap_unordered``. The main process
  aggregates the small per-row result structs into the same
  :class:`EvalReport` shape the sequential path produces.

Spawn is mandatory: every other parallel entry point in this project
uses spawn so behaviour is identical across macOS and Linux, and so we
never depend on POSIX fork semantics with LMDB handles or stemmer state.
"""

from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Iterator
from csv import DictReader
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

from msgspec import Struct

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.copyright import default_ruleset
from pd_matcher.copyright.facts import build_facts
from pd_matcher.copyright.rules import assess
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord

UNRECOGNIZED_GT_STATUS: str = "UNRECOGNIZED_GT_STATUS"

_CHUNK_DIVISOR: int = 8


class EvalReport(Struct, frozen=True, forbid_unknown_fields=True):
    """Summary of one :func:`run_eval` invocation."""

    rows_evaluated: int
    rows_with_predicted_match: int
    rows_with_ground_truth_match: int
    rows_agreeing: int
    precision: float
    recall: float
    f1: float
    status_confusion: dict[str, dict[str, int]]
    elapsed_seconds: float


class _RowOutcome(Struct, frozen=True, forbid_unknown_fields=True):
    """One row's contribution to the aggregate :class:`EvalReport`.

    Workers return these instead of touching shared aggregation state.
    """

    has_predicted_match: bool
    has_ground_truth_match: bool
    agrees: bool
    predicted_status: str
    ground_truth_status: str


class _WorkerState:
    """Per-process resources opened once in the pool initializer.

    Holds the LMDB lookup, IDF table, combiner, ruleset, and configs so
    each worker only pays the open/build cost a single time, then reuses
    them across every row it processes.
    """

    __slots__ = (
        "as_of_year",
        "combiner",
        "copyright_config",
        "idf",
        "lookup",
        "matching_config",
        "pairings",
        "ruleset",
    )

    def __init__(
        self,
        *,
        index_path: Path,
        matching_config: MatchingConfig,
        copyright_config: CopyrightAssessmentConfig,
        pairing_config: PairingConfig,
        as_of_year: int,
    ) -> None:
        self.lookup = NyplIndexLookup(index_path)
        self.idf: IdfTable = build_idf_table(self.lookup)
        self.combiner = WeightedMeanCombiner(config=matching_config)
        self.ruleset = default_ruleset()
        self.matching_config = matching_config
        self.copyright_config = copyright_config
        self.pairings = compile_pairings(pairing_config)
        self.as_of_year = as_of_year


_WORKER_STATE: _WorkerState | None = None


def _parse_int(value: str) -> int | None:
    """Return ``int(value)`` when non-empty and parseable; ``None`` otherwise."""
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _maybe(value: str) -> str | None:
    """Map empty strings to ``None`` (msgspec field semantics)."""
    return value if value else None


def _marc_from_row(row: dict[str, str]) -> MarcRecord:
    """Reconstruct a minimal :class:`MarcRecord` from a ground-truth CSV row."""
    title = row.get("marc_title_original", "")
    return MarcRecord(
        control_id=row.get("marc_id", ""),
        title=title,
        title_main=title,
        lccn=_maybe(row.get("marc_lccn", "")),
        main_author=_maybe(row.get("marc_main_author_original", "")),
        statement_of_responsibility=_maybe(row.get("marc_author_original", "")),
        publisher=_maybe(row.get("marc_publisher_original", "")),
        publication_year=_parse_int(row.get("marc_year", "")),
        language_code=_maybe(row.get("marc_language_code", "")),
        country_code=_maybe(row.get("marc_country_code", "")),
    )


def _classify_gt_status(label: str) -> str:
    """Return the enum value or :data:`UNRECOGNIZED_GT_STATUS`."""
    if not label:
        return UNRECOGNIZED_GT_STATUS
    try:
        return CopyrightStatus(label).value
    except ValueError:
        return UNRECOGNIZED_GT_STATUS


def _safe_division(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator`` or ``0.0`` when the denominator is zero."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; ``0.0`` when both are zero."""
    total = precision + recall
    if total <= 0.0:
        return 0.0
    return 2.0 * precision * recall / total


def _load_rows(
    ground_truth_path: Path,
    *,
    sample: int | None,
    seed: int,
) -> list[dict[str, str]]:
    """Load CSV rows, optionally drawing a deterministic random sample.

    When ``sample`` is ``None`` the full row sequence is returned in file
    order. When ``sample`` is set, ``Random(seed).sample`` selects
    ``min(sample, len(rows))`` rows — passing a sample size larger than
    the file is a no-op (every row is returned).
    """
    with ground_truth_path.open(encoding="utf-8", newline="") as fp:
        rows = list(DictReader(fp))
    if sample is None:
        return rows
    k = min(sample, len(rows))
    return Random(seed).sample(rows, k=k)


def _eval_one_row(
    row: dict[str, str],
    *,
    state: _WorkerState,
) -> _RowOutcome:
    """Score one ground-truth row and return its contribution to the report.

    Pure function: takes the row and shared state, returns a small struct.
    Used both by the sequential path and by the spawn pool's worker
    function so the per-row logic lives in exactly one place.
    """
    marc = _marc_from_row(row)
    match = match_record(
        marc,
        lookup=state.lookup,
        config=state.matching_config,
        idf=state.idf,
        calibrator=None,
        combiner=state.combiner,
        pairings=state.pairings,
    )
    matched_nypl = None
    predicted_id: str | None = None
    if match.best is not None:
        matched_nypl = state.lookup.get_registration(match.best.nypl_uuid)
        predicted_id = match.best.nypl_uuid
    facts = build_facts(marc, match, as_of_year=state.as_of_year, matched_nypl=matched_nypl)
    assessment = assess(
        facts,
        state.ruleset,
        enable_assumptions=state.copyright_config.enable_assumptions,
    )
    gt_id = _maybe(row.get("match_source_id", ""))
    gt_status = _classify_gt_status(row.get("copyright_status", ""))
    agrees = predicted_id is not None and gt_id is not None and predicted_id == gt_id
    return _RowOutcome(
        has_predicted_match=predicted_id is not None,
        has_ground_truth_match=gt_id is not None,
        agrees=agrees,
        predicted_status=assessment.status.value,
        ground_truth_status=gt_status,
    )


def _pool_initializer(
    index_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    pairing_config: PairingConfig,
    as_of_year: int,
) -> None:
    """Spawn-pool initializer: build the per-worker :class:`_WorkerState`.

    ``multiprocessing.Pool`` only forwards positional ``initargs`` to the
    initializer, so this signature deliberately uses positional
    parameters even though the rest of the module favours keyword-only
    arguments.
    """
    global _WORKER_STATE
    _WORKER_STATE = _WorkerState(
        index_path=index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        pairing_config=pairing_config,
        as_of_year=as_of_year,
    )


def _pool_eval_row(row: dict[str, str]) -> _RowOutcome:
    """Spawn-pool worker function: scores one row using process-local state."""
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("_pool_eval_row called before _pool_initializer ran")
    return _eval_one_row(row, state=state)


def _iter_outcomes_sequential(
    rows: list[dict[str, str]],
    *,
    index_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    pairing_config: PairingConfig,
    as_of_year: int,
    limit: int | None,
) -> Iterator[_RowOutcome]:
    """Yield :class:`_RowOutcome` values by scoring every row in-process."""
    state = _WorkerState(
        index_path=index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        pairing_config=pairing_config,
        as_of_year=as_of_year,
    )
    try:
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            yield _eval_one_row(row, state=state)
    finally:
        state.lookup.close()


def _iter_outcomes_parallel(
    rows: list[dict[str, str]],
    *,
    index_path: Path,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    pairing_config: PairingConfig,
    as_of_year: int,
    limit: int | None,
    workers: int,
) -> Iterator[_RowOutcome]:
    """Yield :class:`_RowOutcome` values by fanning out across a spawn pool."""
    target_rows = rows if limit is None else rows[:limit]
    if not target_rows:
        return
    chunksize = max(1, len(target_rows) // (workers * _CHUNK_DIVISOR))
    ctx = get_context("spawn")
    init_args = (index_path, matching_config, copyright_config, pairing_config, as_of_year)
    with ctx.Pool(
        processes=workers,
        initializer=_pool_initializer,
        initargs=init_args,
    ) as pool:
        yield from pool.imap_unordered(_pool_eval_row, target_rows, chunksize=chunksize)


def _aggregate(
    outcomes: Iterable[_RowOutcome],
    *,
    started: float,
) -> EvalReport:
    """Fold a stream of :class:`_RowOutcome` values into an :class:`EvalReport`."""
    rows_evaluated = 0
    rows_with_predicted_match = 0
    rows_with_ground_truth_match = 0
    rows_agreeing = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for outcome in outcomes:
        rows_evaluated += 1
        if outcome.has_predicted_match:
            rows_with_predicted_match += 1
        if outcome.has_ground_truth_match:
            rows_with_ground_truth_match += 1
        if outcome.agrees:
            rows_agreeing += 1
        confusion[outcome.predicted_status][outcome.ground_truth_status] += 1
    precision = _safe_division(rows_agreeing, rows_with_predicted_match)
    recall = _safe_division(rows_agreeing, rows_with_ground_truth_match)
    f1 = _f1(precision, recall)
    status_confusion: dict[str, dict[str, int]] = {
        predicted: dict(counts) for predicted, counts in confusion.items()
    }
    elapsed = perf_counter() - started
    return EvalReport(
        rows_evaluated=rows_evaluated,
        rows_with_predicted_match=rows_with_predicted_match,
        rows_with_ground_truth_match=rows_with_ground_truth_match,
        rows_agreeing=rows_agreeing,
        precision=precision,
        recall=recall,
        f1=f1,
        status_confusion=status_confusion,
        elapsed_seconds=elapsed,
    )


def run_eval(
    *,
    ground_truth_path: Path,
    index_path: Path,
    as_of_year: int,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    pairing_config: PairingConfig,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    workers: int = 1,
) -> EvalReport:
    """Evaluate the matcher pipeline against ``ground_truth_path``.

    Args:
        ground_truth_path: CSV with the
            ``data/combined_ground_truth.csv`` schema.
        index_path: LMDB env produced by ``pd-matcher index build``.
        as_of_year: Reference year for the moving wall and other
            age-sensitive predicates.
        matching_config: Active :class:`MatchingConfig`.
        copyright_config: Active :class:`CopyrightAssessmentConfig`.
        pairing_config: Active :class:`PairingConfig`; compiled once per
            worker into :class:`CompiledPairings`.
        limit: Optional maximum number of rows to evaluate. ``None``
            evaluates every row. Mutually exclusive with ``sample`` at
            the CLI layer.
        sample: Optional random sample size. When set, exactly
            ``min(sample, len(rows))`` rows are drawn using
            ``Random(seed)`` and evaluated.
        seed: Seed for the random sampler. Only meaningful when
            ``sample`` is set; ignored otherwise.
        workers: Number of worker processes. ``1`` (the default) runs a
            single in-process loop; ``>= 2`` fans the per-row work out
            to a ``spawn`` pool.

    Returns:
        A populated :class:`EvalReport`.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1 (got {workers!r})")
    started = perf_counter()
    rows = _load_rows(ground_truth_path, sample=sample, seed=seed)
    if workers == 1:
        outcomes_iter: Iterator[_RowOutcome] = _iter_outcomes_sequential(
            rows,
            index_path=index_path,
            matching_config=matching_config,
            copyright_config=copyright_config,
            pairing_config=pairing_config,
            as_of_year=as_of_year,
            limit=limit,
        )
    else:
        outcomes_iter = _iter_outcomes_parallel(
            rows,
            index_path=index_path,
            matching_config=matching_config,
            copyright_config=copyright_config,
            pairing_config=pairing_config,
            as_of_year=as_of_year,
            limit=limit,
            workers=workers,
        )
    return _aggregate(outcomes_iter, started=started)


__all__ = [
    "UNRECOGNIZED_GT_STATUS",
    "EvalReport",
    "run_eval",
]
