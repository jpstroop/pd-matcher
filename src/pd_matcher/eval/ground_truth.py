"""Lightweight precision/recall evaluator for linkage accuracy.

Drives a curated ground-truth CSV (the shape of
``data/combined_ground_truth.csv``) through the matcher and returns an
:class:`EvalReport` summarising how the predicted linkage compares
against the recorded labels. The matcher's job is now linkage between
MARC and CCE; this driver measures only "did the matcher pick the same
CCE record that the human labeler did", scored as precision and recall
on ``match_source_id``.

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

from collections.abc import Iterable
from collections.abc import Iterator
from csv import DictReader
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

from msgspec import Struct

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord

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
    elapsed_seconds: float


class _RowOutcome(Struct, frozen=True, forbid_unknown_fields=True):
    """One row's contribution to the aggregate :class:`EvalReport`.

    Workers return these instead of touching shared aggregation state.
    """

    has_predicted_match: bool
    has_ground_truth_match: bool
    agrees: bool


class _WorkerState:
    """Per-process resources opened once in the pool initializer.

    Holds the LMDB lookup, IDF table, combiner, and configs so each
    worker only pays the open/build cost a single time, then reuses
    them across every row it processes.
    """

    __slots__ = (
        "combiner",
        "idf",
        "lookup",
        "matching_config",
        "pairings",
    )

    def __init__(
        self,
        *,
        index_path: Path,
        matching_config: MatchingConfig,
        pairing_config: PairingConfig,
    ) -> None:
        self.lookup = NyplIndexLookup(index_path)
        self.idf: IdfTable = build_idf_table(self.lookup)
        self.combiner = WeightedMeanCombiner(config=matching_config)
        self.matching_config = matching_config
        self.pairings = compile_pairings(pairing_config)


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
    predicted_id: str | None = None
    if match.best is not None:
        predicted_id = match.best.nypl_uuid
    gt_id = _maybe(row.get("match_source_id", ""))
    agrees = predicted_id is not None and gt_id is not None and predicted_id == gt_id
    return _RowOutcome(
        has_predicted_match=predicted_id is not None,
        has_ground_truth_match=gt_id is not None,
        agrees=agrees,
    )


def _pool_initializer(
    index_path: Path,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
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
        pairing_config=pairing_config,
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
    pairing_config: PairingConfig,
    limit: int | None,
) -> Iterator[_RowOutcome]:
    """Yield :class:`_RowOutcome` values by scoring every row in-process."""
    state = _WorkerState(
        index_path=index_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
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
    pairing_config: PairingConfig,
    limit: int | None,
    workers: int,
) -> Iterator[_RowOutcome]:
    """Yield :class:`_RowOutcome` values by fanning out across a spawn pool."""
    target_rows = rows if limit is None else rows[:limit]
    if not target_rows:
        return
    chunksize = max(1, len(target_rows) // (workers * _CHUNK_DIVISOR))
    ctx = get_context("spawn")
    init_args = (index_path, matching_config, pairing_config)
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
    for outcome in outcomes:
        rows_evaluated += 1
        if outcome.has_predicted_match:
            rows_with_predicted_match += 1
        if outcome.has_ground_truth_match:
            rows_with_ground_truth_match += 1
        if outcome.agrees:
            rows_agreeing += 1
    precision = _safe_division(rows_agreeing, rows_with_predicted_match)
    recall = _safe_division(rows_agreeing, rows_with_ground_truth_match)
    f1 = _f1(precision, recall)
    elapsed = perf_counter() - started
    return EvalReport(
        rows_evaluated=rows_evaluated,
        rows_with_predicted_match=rows_with_predicted_match,
        rows_with_ground_truth_match=rows_with_ground_truth_match,
        rows_agreeing=rows_agreeing,
        precision=precision,
        recall=recall,
        f1=f1,
        elapsed_seconds=elapsed,
    )


def run_eval(
    *,
    ground_truth_path: Path,
    index_path: Path,
    matching_config: MatchingConfig,
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
        matching_config: Active :class:`MatchingConfig`.
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
            pairing_config=pairing_config,
            limit=limit,
        )
    else:
        outcomes_iter = _iter_outcomes_parallel(
            rows,
            index_path=index_path,
            matching_config=matching_config,
            pairing_config=pairing_config,
            limit=limit,
            workers=workers,
        )
    return _aggregate(outcomes_iter, started=started)


__all__ = [
    "EvalReport",
    "run_eval",
]
