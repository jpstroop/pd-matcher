"""First-fit Platt calibrator against the labeled vault (issue #70).

Throwaway one-shot research instrument. NOT shipped; ``scripts/`` is excluded
from the published package via the ``[tool.coverage.run].source`` allowlist.

For every non-``unsure`` vault entry, resolves the (MARC, CCE) pair, runs the
matcher's :func:`pd_matcher.match.pipeline._score_candidate` with
``calibrator=None`` so the resulting ``CandidateMatch.combined.raw`` is the
unmapped weighted-mean score in ``[0, 100]``, partitions the raw scores into
positives (verdict == ``match``) and negatives (verdict == ``no_match``),
then fits a :class:`PlattCalibrator` via
:func:`pd_matcher.match.combiners.calibrator.train_calibrator`. The fitted
calibrator is persisted to ``caches/calibrator.msgpack`` so the production
``_load_calibrator`` path picks it up on the next CLI invocation.

The script prints a markdown summary to stdout: training corpus counts,
the trained ``(a, b)`` coefficients, the ``trained_at`` timestamp, and a
sanity-check probe table mapping representative raw scores to their
calibrated probabilities. The probe table tells the reader at a glance
whether the calibrator is monotone-increasing in the right direction.

Usage:
    pdm run python scripts/fit_calibrator.py \\
        > docs/findings/fit_calibrator_<date>.md
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Final

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import calibrate
from pd_matcher.match.combiners.calibrator import save_calibrator
from pd_matcher.match.combiners.calibrator import train_calibrator
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.pipeline import _score_candidate

_VAULT_PATH: Final[Path] = Path("data/training/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_CALIBRATOR_PATH: Final[Path] = Path("caches/calibrator.msgpack")

_VERDICT_MATCH: Final[str] = "match"
_VERDICT_NO_MATCH: Final[str] = "no_match"
_VERDICT_UNSURE: Final[str] = "unsure"

_PROBE_RAW_SCORES: Final[tuple[float, ...]] = (50.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0, 100.0)


def _kept_entries(vault_path: Path) -> list[VaultEntry]:
    """Return all non-``unsure`` vault entries in insertion order."""
    raw = current_entries(vault_path)
    return [entry for entry in raw.values() if entry.verdict != _VERDICT_UNSURE]


def _collect_raw_scores(
    entries: list[VaultEntry],
) -> tuple[list[float], list[float], int, int]:
    """Score every resolved pair via the production combiner with ``calibrator=None``.

    Returns ``(positives_raw, negatives_raw, missing_in_pool,
    missing_in_index)``. ``positives_raw`` holds ``CandidateMatch.combined.raw``
    for every ``match`` verdict whose MARC and CCE both resolve; ``negatives_raw``
    is the same for ``no_match``. Entries that fail to resolve are counted and
    logged via the diagnostic prints but otherwise dropped.
    """
    matching_config = _load_default_matching_config()
    pairing_config = _load_default_pairing_config()
    pairings = compile_pairings(pairing_config)
    needed_marc_ids = {entry.marc_control_id for entry in entries}
    marc_by_id = build_marc_index(_POOL_PATH, needed_marc_ids)
    positives_raw: list[float] = []
    negatives_raw: list[float] = []
    missing_in_pool = 0
    missing_in_index = 0
    with NyplIndexLookup(_INDEX_PATH) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
        score_pair = make_pair_scorer(
            matching_config=matching_config,
            pairings=pairings,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            calibrator=None,
            learned_model_dir=None,
        )
        for entry in entries:
            marc = marc_by_id.get(entry.marc_control_id)
            if marc is None:
                missing_in_pool += 1
                continue
            cce = lookup.get_registration(entry.nypl_uuid)
            if cce is None:
                missing_in_index += 1
                continue
            candidate = score_pair(marc, cce)
            raw_score = candidate.combined.raw
            if entry.verdict == _VERDICT_MATCH:
                positives_raw.append(raw_score)
            elif entry.verdict == _VERDICT_NO_MATCH:
                negatives_raw.append(raw_score)
    return positives_raw, negatives_raw, missing_in_pool, missing_in_index


def _print_summary(
    calibrator: PlattCalibrator,
    *,
    positives_raw: list[float],
    negatives_raw: list[float],
    missing_in_pool: int,
    missing_in_index: int,
    output_path: Path,
) -> None:
    """Emit the markdown summary report to stdout."""
    today = datetime.now(UTC).date().isoformat()
    print(f"# Platt calibrator first-fit — {today}\n")
    print(
        "First-ever Platt calibration of the production combiner against the "
        "labeled vault (issue #70). Prior to this fit the matcher ran with "
        "`calibrator=None`, so `combined.calibrated = combined.raw / 100` "
        "(a linear pass-through). This run fits a real sigmoid against the "
        "raw weighted-mean scores from the resolved vault pairs and writes the "
        "result to the production calibrator cache.\n"
    )
    print("## 1. Training corpus\n")
    print(f"- **Vault**: `{_VAULT_PATH}`")
    print(f"- **Candidate pool**: `{_POOL_PATH}`")
    print(f"- **Index**: `{_INDEX_PATH}`")
    print(f"- **Positives** (verdict=`match`): {len(positives_raw)}")
    print(f"- **Negatives** (verdict=`no_match`): {len(negatives_raw)}")
    print(f"- **Skipped (MARC missing from pool)**: {missing_in_pool}")
    print(f"- **Skipped (CCE missing from index)**: {missing_in_index}")
    print()
    if positives_raw:
        pos_mean = sum(positives_raw) / len(positives_raw)
        pos_min = min(positives_raw)
        pos_max = max(positives_raw)
        print(
            f"- **Positives raw score**: min={pos_min:.2f} "
            f"mean={pos_mean:.2f} max={pos_max:.2f}"
        )
    if negatives_raw:
        neg_mean = sum(negatives_raw) / len(negatives_raw)
        neg_min = min(negatives_raw)
        neg_max = max(negatives_raw)
        print(
            f"- **Negatives raw score**: min={neg_min:.2f} "
            f"mean={neg_mean:.2f} max={neg_max:.2f}"
        )
    print()
    print("## 2. Fitted Platt calibrator\n")
    print(f"- **`a`** (slope): `{calibrator.a:.6f}`")
    print(f"- **`b`** (intercept): `{calibrator.b:.6f}`")
    print(f"- **`n_positive`**: {calibrator.n_positive}")
    print(f"- **`n_negative`**: {calibrator.n_negative}")
    print(f"- **`trained_at`**: `{calibrator.trained_at}`")
    print(f"- **Persisted to**: `{output_path}`")
    print()
    print("## 3. Sanity-check probe table\n")
    print(
        "Maps a representative raw weighted-mean score in `[0, 100]` to the "
        "calibrated probability returned by `calibrate(raw, calibrator)`. A "
        "well-formed calibrator is monotone-increasing: higher raw means "
        "higher probability. If this table is non-monotone or inverted the "
        "fit is broken.\n"
    )
    print("| raw | calibrated |")
    print("|---:|---:|")
    for raw in _PROBE_RAW_SCORES:
        prob = calibrate(raw, calibrator)
        print(f"| {raw:.1f} | {prob:.4f} |")
    print()


def main() -> None:
    """Fit the Platt calibrator, persist it, and print the markdown summary."""
    entries = _kept_entries(_VAULT_PATH)
    positives_raw, negatives_raw, missing_in_pool, missing_in_index = _collect_raw_scores(
        entries,
    )
    if not positives_raw or not negatives_raw:
        print("# Platt calibrator first-fit\n")
        print(
            "Cannot fit calibrator: need both positives and negatives. "
            f"Got positives={len(positives_raw)} negatives={len(negatives_raw)}."
        )
        return
    calibrator = train_calibrator(positives_raw, negatives_raw)
    save_calibrator(calibrator, _CALIBRATOR_PATH)
    _print_summary(
        calibrator,
        positives_raw=positives_raw,
        negatives_raw=negatives_raw,
        missing_in_pool=missing_in_pool,
        missing_in_index=missing_in_index,
        output_path=_CALIBRATOR_PATH,
    )


if __name__ == "__main__":
    main()
