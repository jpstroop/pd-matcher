"""Slow, index-dependent end-to-end regression gate.

Re-runs the vault-driven eval against the local LMDB index and asserts
that precision and recall have not regressed below the checked-in
baseline by more than its tolerance. Skipped when the index, the
candidate MARC pool, or the label vault is absent so the default suite
still passes on a machine without the built corpora.

Excluded from the default ``pdm run pytest`` run by the ``regression``
marker; invoke with ``pdm run regression``.
"""

from pathlib import Path

from pytest import mark
from pytest import skip

from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.eval.regression import compare
from pd_matcher.eval.regression import load_baseline

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _REPO_ROOT / "caches" / "nypl.lmdb"
_BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


@mark.regression
def test_eval_meets_regression_baseline() -> None:
    baseline = load_baseline(_BASELINE_PATH)
    vault_path = _REPO_ROOT / baseline.params.vault
    pool_path = _REPO_ROOT / baseline.params.pool
    if not _INDEX_PATH.exists() or not vault_path.is_file() or not pool_path.is_dir():
        skip(f"index ({_INDEX_PATH}), vault ({vault_path}), or pool ({pool_path}) not available")
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
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=_INDEX_PATH,
        matching_config=matching_config,
        pairing_config=pairing_config,
    )
    result = compare(baseline, report)
    for message in result.messages:
        print(message)
    print(f"precision_delta={result.precision_delta:+.6f} recall_delta={result.recall_delta:+.6f}")
    assert result.passed, "\n".join(result.messages)
