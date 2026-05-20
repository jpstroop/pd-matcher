"""Slow, index-dependent end-to-end regression gate.

Re-runs the canonical 1000-row eval against the local LMDB index and
asserts that precision and recall have not regressed below the
checked-in baseline by more than its tolerance. Skipped when the index
or the ground-truth CSV is absent so the default suite still passes on a
machine without the built index.

Excluded from the default ``pdm run pytest`` run by the ``regression``
marker; invoke with ``pdm run regression``.
"""

from pathlib import Path

from pytest import mark
from pytest import skip

from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.eval.regression import compare
from pd_matcher.eval.regression import load_baseline

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _REPO_ROOT / "caches" / "nypl.lmdb"
_DATA_DIR = _REPO_ROOT / "data"
_BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"

_WORKERS = 8


@mark.regression
def test_eval_meets_regression_baseline() -> None:
    baseline = load_baseline(_BASELINE_PATH)
    ground_truth_path = _DATA_DIR / baseline.params.ground_truth
    if not _INDEX_PATH.exists() or not ground_truth_path.is_file():
        skip(f"index ({_INDEX_PATH}) or ground truth ({ground_truth_path}) not available")
    base_matching = _load_default_matching_config()
    matching_config = MatchingConfig(
        title_weight=base_matching.title_weight,
        author_weight=base_matching.author_weight,
        publisher_weight=base_matching.publisher_weight,
        year_weight=base_matching.year_weight,
        edition_weight=base_matching.edition_weight,
        lccn_weight=base_matching.lccn_weight,
        isbn_weight=base_matching.isbn_weight,
        year_window=baseline.params.year_window,
        min_combined_score=base_matching.min_combined_score,
        scorer=base_matching.scorer,
    )
    pairing_config = _load_default_pairing_config()
    copyright_config = CopyrightAssessmentConfig(as_of_year=baseline.params.as_of_year)
    report = run_eval(
        ground_truth_path=ground_truth_path,
        index_path=_INDEX_PATH,
        as_of_year=baseline.params.as_of_year,
        matching_config=matching_config,
        copyright_config=copyright_config,
        pairing_config=pairing_config,
        sample=baseline.params.sample,
        seed=baseline.params.seed,
        workers=_WORKERS,
    )
    result = compare(baseline, report)
    for message in result.messages:
        print(message)
    print(f"precision_delta={result.precision_delta:+.6f} recall_delta={result.recall_delta:+.6f}")
    assert result.passed, "\n".join(result.messages)
