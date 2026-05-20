"""Refresh the checked-in regression baseline.

Runs the canonical 1000-row eval against the local LMDB index and writes
``tests/regression/baseline.json`` from the resulting metrics. Use this
after an intentional change to the matching or assessment pipeline (for
example once #19 reg-date parsing lands), then commit the new JSON.

Invoke via ``pdm run regression-baseline``. This is a script, not a
pytest test; it is outside the coverage source.
"""

from json import dumps
from pathlib import Path

from msgspec import to_builtins

from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.eval.regression import BaselineParams
from pd_matcher.eval.regression import baseline_from_report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _REPO_ROOT / "caches" / "nypl.lmdb"
_DATA_DIR = _REPO_ROOT / "data"
_BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"

_SAMPLE = 1000
_SEED = 42
_YEAR_WINDOW = 0
_AS_OF_YEAR = 2026
_GROUND_TRUTH = "combined_ground_truth.csv"
_TOLERANCE = 0.02
_WORKERS = 8
_NOTES = (
    "Thin-record eval (records reconstructed from GT CSV columns). Gate is "
    "precision/recall only; per-status confusion is not validated because GT "
    "status labels predate the current CopyrightStatus enum. Refreshed 2026-05-20 "
    "after #19 (reg-date fallback: reg_year now falls back regDate->copyDate->"
    f"pubDate) landed and recall improved. Generated: sample={_SAMPLE} seed={_SEED} "
    f"year_window={_YEAR_WINDOW} as_of={_AS_OF_YEAR}."
)


def main() -> None:
    """Run the canonical eval and rewrite ``baseline.json``."""
    ground_truth_path = _DATA_DIR / _GROUND_TRUTH
    if not _INDEX_PATH.exists():
        raise SystemExit(f"index not found: {_INDEX_PATH}")
    if not ground_truth_path.is_file():
        raise SystemExit(f"ground truth not found: {ground_truth_path}")
    base_matching = _load_default_matching_config()
    matching_config = MatchingConfig(
        title_weight=base_matching.title_weight,
        author_weight=base_matching.author_weight,
        publisher_weight=base_matching.publisher_weight,
        year_weight=base_matching.year_weight,
        edition_weight=base_matching.edition_weight,
        lccn_weight=base_matching.lccn_weight,
        isbn_weight=base_matching.isbn_weight,
        year_window=_YEAR_WINDOW,
        min_combined_score=base_matching.min_combined_score,
        scorer=base_matching.scorer,
    )
    pairing_config = _load_default_pairing_config()
    copyright_config = CopyrightAssessmentConfig(as_of_year=_AS_OF_YEAR)
    report = run_eval(
        ground_truth_path=ground_truth_path,
        index_path=_INDEX_PATH,
        as_of_year=_AS_OF_YEAR,
        matching_config=matching_config,
        copyright_config=copyright_config,
        pairing_config=pairing_config,
        sample=_SAMPLE,
        seed=_SEED,
        workers=_WORKERS,
    )
    params = BaselineParams(
        sample=_SAMPLE,
        seed=_SEED,
        year_window=_YEAR_WINDOW,
        as_of_year=_AS_OF_YEAR,
        ground_truth=_GROUND_TRUTH,
    )
    baseline = baseline_from_report(
        report,
        params=params,
        tolerance=_TOLERANCE,
        notes=_NOTES,
    )
    _BASELINE_PATH.write_text(dumps(to_builtins(baseline), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {_BASELINE_PATH}")
    print(f"  precision: {report.precision}")
    print(f"  recall:    {report.recall}")
    print(f"  f1:        {report.f1}")
    print(
        f"  rows_evaluated={report.rows_evaluated} "
        f"predicted={report.rows_with_predicted_match} "
        f"gt={report.rows_with_ground_truth_match} "
        f"agreeing={report.rows_agreeing}"
    )


if __name__ == "__main__":
    main()
