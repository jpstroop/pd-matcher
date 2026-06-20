"""Refresh the checked-in regression baseline.

Runs the vault-driven eval against the local LMDB index and writes
``tests/regression/baseline.json`` from the resulting metrics. Use this
after an intentional change to the matching pipeline, then commit the
new JSON.

Invoke via ``pdm run regression-baseline``. This is a script, not a
pytest test; it is outside the coverage source.
"""

from json import dumps
from pathlib import Path

from msgspec import to_builtins

from pd_matcher.cli import _load_calibrator
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.eval.regression import BaselineParams
from pd_matcher.eval.regression import baseline_from_report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _REPO_ROOT / "caches" / "cce.lmdb"
_BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"

_VAULT = "data/training/label_vault.jsonl"
_POOL = "data/candidates"
_YEAR_WINDOW = 0
_TOLERANCE = 0.02
_NOTES = (
    "Vault-driven eval. Pair-level AUC/AP plus per-MARC P/R; gate is "
    "precision/recall only on linkage agreement (latest 'match' verdict "
    f"wins). Generated: vault={_VAULT} pool={_POOL} year_window={_YEAR_WINDOW}."
)


def main() -> None:
    """Run the vault-driven eval and rewrite ``baseline.json``."""
    vault_path = _REPO_ROOT / _VAULT
    pool_path = _REPO_ROOT / _POOL
    if not _INDEX_PATH.exists():
        raise SystemExit(f"index not found: {_INDEX_PATH}")
    if not vault_path.is_file():
        raise SystemExit(f"vault not found: {vault_path}")
    if not pool_path.is_dir():
        raise SystemExit(f"pool not found: {pool_path}")
    base_matching = _load_default_matching_config()
    matching_config = MatchingConfig(
        title_weight=base_matching.title_weight,
        author_weight=base_matching.author_weight,
        publisher_weight=base_matching.publisher_weight,
        edition_weight=base_matching.edition_weight,
        lccn_weight=base_matching.lccn_weight,
        isbn_weight=base_matching.isbn_weight,
        extent_weight=base_matching.extent_weight,
        volume_weight=base_matching.volume_weight,
        year_window=_YEAR_WINDOW,
        min_combined_score=base_matching.min_combined_score,
        scorer=base_matching.scorer,
    )
    pairing_config = _load_default_pairing_config()
    calibrator = _load_calibrator(_INDEX_PATH.parent)
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=_INDEX_PATH,
        matching_config=matching_config,
        pairing_config=pairing_config,
        calibrator=calibrator,
    )
    params = BaselineParams(
        vault=_VAULT,
        pool=_POOL,
        year_window=_YEAR_WINDOW,
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
    print(f"  auc_roc:   {report.auc_roc}")
    print(f"  average_precision: {report.average_precision}")
    print(
        f"  pairs_evaluated={report.pairs_evaluated} "
        f"pos={report.pairs_positive} neg={report.pairs_negative} "
        f"unsure={report.pairs_unsure_excluded}"
    )
    print(
        f"  marcs_evaluated={report.marcs_evaluated} "
        f"with_top={report.marcs_with_matcher_top} "
        f"correct={report.marcs_with_correct_top}"
    )


if __name__ == "__main__":
    main()
