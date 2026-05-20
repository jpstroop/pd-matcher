"""Lightweight precision/recall + confusion-matrix evaluator.

Drives a curated ground-truth CSV (the shape of
``data/combined_ground_truth.csv``) through the full Phase 4 matcher and
Phase 5 rule engine and returns an :class:`EvalReport` summarising how
the predictions compare against the recorded labels. Phase 7's job is to
make this runnable today; Phase 8 will layer baseline JSON, regression
gates, and per-status breakdowns on top.
"""

from collections import Counter
from collections import defaultdict
from csv import DictReader
from pathlib import Path
from random import Random
from time import perf_counter

from msgspec import Struct

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.copyright import default_ruleset
from pd_matcher.copyright.facts import build_facts
from pd_matcher.copyright.rules import assess
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord

UNRECOGNIZED_GT_STATUS: str = "UNRECOGNIZED_GT_STATUS"


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
    return MarcRecord(
        control_id=row.get("marc_id", ""),
        title=row.get("marc_title_original", ""),
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


def run_eval(
    *,
    ground_truth_path: Path,
    index_path: Path,
    as_of_year: int,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
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
        limit: Optional maximum number of rows to evaluate. ``None``
            evaluates every row. Mutually exclusive with ``sample`` at
            the CLI layer.
        sample: Optional random sample size. When set, exactly
            ``min(sample, len(rows))`` rows are drawn using
            ``Random(seed)`` and evaluated.
        seed: Seed for the random sampler. Only meaningful when
            ``sample`` is set; ignored otherwise.

    Returns:
        A populated :class:`EvalReport`.
    """
    started = perf_counter()
    combiner = WeightedMeanCombiner(config=matching_config)
    ruleset = default_ruleset()
    rows_evaluated = 0
    rows_with_predicted_match = 0
    rows_with_ground_truth_match = 0
    rows_agreeing = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    rows = _load_rows(ground_truth_path, sample=sample, seed=seed)
    with NyplIndexLookup(index_path) as lookup:
        idf: IdfTable = build_idf_table(lookup)
        for row in rows:
            if limit is not None and rows_evaluated >= limit:
                break
            marc = _marc_from_row(row)
            match = match_record(
                marc,
                lookup=lookup,
                config=matching_config,
                idf=idf,
                calibrator=None,
                combiner=combiner,
            )
            matched_nypl = None
            predicted_id: str | None = None
            if match.best is not None:
                matched_nypl = lookup.get_registration(match.best.nypl_uuid)
                predicted_id = match.best.nypl_uuid
            facts = build_facts(marc, match, as_of_year=as_of_year, matched_nypl=matched_nypl)
            assessment = assess(
                facts,
                ruleset,
                enable_assumptions=copyright_config.enable_assumptions,
            )
            gt_id = _maybe(row.get("match_source_id", ""))
            gt_status = _classify_gt_status(row.get("copyright_status", ""))
            rows_evaluated += 1
            if predicted_id is not None:
                rows_with_predicted_match += 1
            if gt_id is not None:
                rows_with_ground_truth_match += 1
            if predicted_id is not None and gt_id is not None and predicted_id == gt_id:
                rows_agreeing += 1
            confusion[assessment.status.value][gt_status] += 1
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


__all__ = [
    "UNRECOGNIZED_GT_STATUS",
    "EvalReport",
    "run_eval",
]
