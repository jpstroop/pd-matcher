"""Cross-matcher disagreement scan over the labeled vault (active learning).

One-off labeling-support tooling. ``scripts/*`` is gitignored (an ``!``
exception lets the maintainer commit a decision-driving proof). It does NOT
modify anything under ``src/``. The label vault ``data/label_vault.jsonl`` is
read STRICTLY read-only; the production ``caches/learned_scorer.*`` artifact is
never written (the per-fold boosters live only in memory).

WHY. The learned and weighted-mean matchers agree on almost every vault pair,
but the pairs where they DISAGREE are the high-value ones: they are either
active-learning targets (the boundary the next round of labeling should
sharpen), label errors in the vault itself (when BOTH matchers agree with each
other but contradict the recorded human verdict), or whole/part edge cases
(``volume.compat`` incompatible) worth inspecting for issue #82. This script
surfaces exactly those pairs into a review DB the maintainer can open, plus a
flat text file of links.

HOW (reuses production machinery, reinvents nothing):

* Vault / pool / index resolution copies ``scripts/learned_scorer_heldout.py``:
  ``current_entries`` for the vault, ``build_marc_index`` over
  ``data/candidates`` for the MARC side, ``NyplIndexLookup.get_registration``
  for the CCE side, ``make_pair_scorer`` (weighted-mean forced) to compute each
  pair's Evidence ONCE, and ``feature_row`` to project that Evidence.
* The weighted score is the production weighted-mean combiner's calibrated
  output on that Evidence.
* The learned score is OUT-OF-FOLD: GroupKFold by ``marc_control_id`` (5 folds,
  fixed seed), an ``LGBMClassifier`` with the locked hyperparameters fit on the
  other folds' ``(feature_row, label)`` rows, then ``predict_proba`` on this
  fold. Every pair ends with a learned score from a model that never trained on
  it. Unlike the heldout script there is NO ``match_record`` retrieval — the
  known pair's feature vector is scored directly.
* Surfaced pairs are written via ``build_queue._build_pair_insert`` /
  ``ReviewDb.insert_pair`` (the same row-builder the queue builders use) and the
  CURRENT vault verdict is pre-applied via ``ReviewDb.insert_existing_label`` so
  each review card shows what was labeled plus the disagreement reason.

Usage:
    # Smoke (caps pairs scanned; DB + text are partial):
    pdm run python scripts/disagreement_scan.py \\
        --limit 60 --out /tmp/disagree_smoke.db --rebuild

    # Full scan (the maintainer's review queue):
    pdm run python scripts/disagreement_scan.py --rebuild
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from sys import stderr
from typing import Final

from lightgbm import LGBMClassifier
from numpy import asarray
from numpy import float64
from numpy import int64
from numpy import zeros
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from pd_groundtruth.build_queue import _build_pair_insert
from pd_groundtruth.build_queue import _language_of
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import band_of
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.features import feature_row
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_DEFAULT_OUT: Final[Path] = Path("data/disagreements.db")
_PROGRESS_LOG: Final[Path] = Path("/tmp/agent-progress.log")

_VERDICT_MATCH: Final[str] = "match"
_VERDICT_UNSURE: Final[str] = "unsure"

# The review server's default bind address (pd_groundtruth.cli
# _DEFAULT_REVIEW_HOST / _DEFAULT_REVIEW_PORT) and the per-pair route registered
# in pd_groundtruth.review.app ("/pair/{pair_id}").
_DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8000"

# Locked recipe from src/pd_matcher/match/combiners/train.py, identical to
# scripts/learned_scorer_heldout.py.
_MAX_DEPTH: Final[int] = 3
_NUM_LEAVES: Final[int] = 8
_MIN_DATA_IN_LEAF: Final[int] = 10
_LAMBDA_L2: Final[float] = 1.0
_N_ESTIMATORS: Final[int] = 200
_CLASS_WEIGHT: Final[str] = "balanced"

_RANDOM_STATE: Final[int] = 20260617
_N_SPLITS: Final[int] = 5
_BOUNDARY: Final[float] = 0.5

_VOLUME_SCORER: Final[str] = "volume.compat"
_INCOMPATIBLE: Final[float] = 0.0

_BUCKET_LABEL_ERROR: Final[str] = "label-error?"
_BUCKET_MODEL_VS_MODEL: Final[str] = "model-vs-model"
_BUCKET_WHOLE_PART: Final[str] = "whole/part"
_BUCKET_ORDER: Final[tuple[str, ...]] = (
    _BUCKET_LABEL_ERROR,
    _BUCKET_MODEL_VS_MODEL,
    _BUCKET_WHOLE_PART,
)

_PROGRESS_EVERY: Final[int] = 100


def _progress(message: str) -> None:
    """Emit a one-line milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _new_classifier() -> LGBMClassifier:
    """Construct an ``LGBMClassifier`` with the locked, deterministic recipe."""
    return LGBMClassifier(
        max_depth=_MAX_DEPTH,
        num_leaves=_NUM_LEAVES,
        min_data_in_leaf=_MIN_DATA_IN_LEAF,
        reg_lambda=_LAMBDA_L2,
        n_estimators=_N_ESTIMATORS,
        class_weight=_CLASS_WEIGHT,
        objective="binary",
        verbose=-1,
        random_state=_RANDOM_STATE,
        n_jobs=1,
    )


def _scoring_config(config: MatchingConfig) -> MatchingConfig:
    """Return ``config`` forced onto the weighted-mean scorer.

    Mirrors ``pd_matcher.match.combiners.train._scoring_config``: the per-scorer
    Evidence is identical regardless of the combiner, so forcing weighted-mean
    lets the single Evidence pass run without a learned artifact and feeds both
    the weighted score and the learned feature rows.
    """
    if config.scorer == "weighted_mean":
        return config
    return MatchingConfig(
        title_weight=config.title_weight,
        author_weight=config.author_weight,
        publisher_weight=config.publisher_weight,
        year_weight=config.year_weight,
        edition_weight=config.edition_weight,
        lccn_weight=config.lccn_weight,
        isbn_weight=config.isbn_weight,
        extent_weight=config.extent_weight,
        volume_weight=config.volume_weight,
        year_window=config.year_window,
        min_combined_score=config.min_combined_score,
        scorer="weighted_mean",
    )


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """One resolved vault pair scored by both matchers.

    ``learned`` is the out-of-fold learned score, filled after the GroupKFold
    pass; the other fields are filled during the single Evidence/weighted pass.
    """

    entry: VaultEntry
    marc: MarcRecord
    cce: IndexedNyplRegRecord
    evidence: tuple[Evidence, ...]
    evidence_sources: tuple[tuple[str, str], ...]
    features: tuple[float, ...]
    weighted: float
    learned: float


@dataclass(slots=True)
class ScanResult:
    """Aggregate outcome of one disagreement scan."""

    surfaced: dict[str, int] = field(default_factory=dict)
    scanned: int = 0
    unresolved: int = 0
    agree: int = 0
    limit: int | None = None

    def surfaced_for(self, bucket: str) -> int:
        """Return the surfaced count for ``bucket`` (``0`` when none yet)."""
        return self.surfaced.get(bucket, 0)

    def total_surfaced(self) -> int:
        """Return the total surfaced pairs across all buckets."""
        return sum(self.surfaced.values())


def _is_volume_incompatible(evidence: tuple[Evidence, ...]) -> bool:
    """Return whether a present ``volume.compat`` Evidence scored incompatible.

    The whole/part slice (issue #82) is the set of pairs whose volume scorer
    actually fired (not skipped) and judged the volumes incompatible (normalized
    score ``0.0``).
    """
    for item in evidence:
        if item.scorer != _VOLUME_SCORER:
            continue
        if item.skipped:
            return False
        return item.normalized <= _INCOMPATIBLE
    return False


def _bucket_for(pair: ScoredPair) -> str | None:
    """Classify one scored pair into its most-specific disagreement bucket.

    Boundary is :data:`_BOUNDARY`. ``label-error?`` (both models agree with each
    other but contradict the human) outranks ``model-vs-model`` (models on
    opposite sides of the boundary), which outranks ``whole/part`` (a fired,
    incompatible ``volume.compat``). A pair in none of the three returns
    ``None`` and is not surfaced.
    """
    learned_match = pair.learned >= _BOUNDARY
    weighted_match = pair.weighted >= _BOUNDARY
    human_match = pair.entry.verdict == _VERDICT_MATCH
    if learned_match == weighted_match and learned_match != human_match:
        return _BUCKET_LABEL_ERROR
    if learned_match != weighted_match:
        return _BUCKET_MODEL_VS_MODEL
    if _is_volume_incompatible(pair.evidence):
        return _BUCKET_WHOLE_PART
    return None


def _label_error_starkness(pair: ScoredPair) -> float:
    """Return how confidently both models contradict the human verdict.

    For a ``label-error?`` pair both models are on the same (wrong) side of the
    boundary. Starkness is the max distance from the boundary of the two model
    scores: the wrong side is high when both models say "match" (human said no),
    low when both say "no match" (human said match).
    """
    human_match = pair.entry.verdict == _VERDICT_MATCH
    if human_match:
        return max(_BOUNDARY - pair.learned, _BOUNDARY - pair.weighted)
    return max(pair.learned - _BOUNDARY, pair.weighted - _BOUNDARY)


def _model_gap(pair: ScoredPair) -> float:
    """Return the absolute learned-minus-weighted gap for a ``model-vs-model`` pair."""
    return abs(pair.learned - pair.weighted)


def _note_for(bucket: str, pair: ScoredPair) -> str:
    """Render the per-pair disagreement summary stored on the pre-applied label."""
    return (
        f"DISAGREEMENT [{bucket}]: you={pair.entry.verdict} "
        f"learned={pair.learned:.2f} weighted={pair.weighted:.2f}"
    )


def _link_line(base_url: str, pair_id: int, bucket: str, pair: ScoredPair) -> str:
    """Render one flat text-file line: link plus how the pair disagrees."""
    return (
        f"{base_url}/pair/{pair_id} — {bucket} you={pair.entry.verdict} "
        f"learned={pair.learned:.2f} weighted={pair.weighted:.2f}"
    )


def _resolved_pairs(
    entries: list[VaultEntry],
    marc_by_id: dict[str, MarcRecord],
    lookup: NyplIndexLookup,
    weighted_config: MatchingConfig,
    idf_tables: tuple[IdfTable, IdfTable, IdfTable],
    pairings: CompiledPairings,
) -> tuple[list[ScoredPair], int]:
    """Resolve + weighted-score every labeled pair; return ``(pairs, unresolved)``.

    Each pair's Evidence is computed ONCE via ``make_pair_scorer`` (weighted-mean
    forced), the weighted score is the weighted combiner's calibrated output on
    that Evidence, and the feature row is projected for the later OOF pass. Pairs
    whose MARC (pool) or CCE (index) is unavailable are counted as unresolved.
    """
    idf, author_idf, publisher_idf = idf_tables
    score_pair = make_pair_scorer(
        matching_config=weighted_config,
        pairings=pairings,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        calibrator=None,
    )
    weighted_combiner = build_combiner(weighted_config, learned_model_dir=None)
    pairs: list[ScoredPair] = []
    unresolved = 0
    for entry in entries:
        marc = marc_by_id.get(entry.marc_control_id)
        if marc is None:
            unresolved += 1
            continue
        cce = lookup.get_registration(entry.nypl_uuid)
        if cce is None:
            unresolved += 1
            continue
        candidate = score_pair(marc, cce)
        weighted = weighted_combiner.combine(candidate.evidence).calibrated
        pairs.append(
            ScoredPair(
                entry=entry,
                marc=marc,
                cce=cce,
                evidence=candidate.evidence,
                evidence_sources=candidate.evidence_sources,
                features=feature_row(candidate.evidence),
                weighted=weighted,
                learned=0.0,
            )
        )
        if len(pairs) % _PROGRESS_EVERY == 0:
            _progress(f"resolved {len(pairs)} pairs (unresolved={unresolved})")
    return pairs, unresolved


def _fold_assignment(marc_ids: list[str]) -> dict[str, int]:
    """Assign each distinct ``marc_control_id`` to one of ``_N_SPLITS`` folds.

    GroupKFold over the distinct MARCs (each MARC its own group) gives a
    deterministic, leak-free partition for a fixed MARC order and seed, mirroring
    ``scripts/learned_scorer_heldout._fold_assignment``.
    """
    n_marcs = len(marc_ids)
    n_splits = min(_N_SPLITS, n_marcs)
    dummy_x: NDArray[int64] = zeros((n_marcs, 1), dtype=int64)
    dummy_y: NDArray[int64] = zeros(n_marcs, dtype=int64)
    groups: NDArray[int64] = asarray(range(n_marcs), dtype=int64)
    splitter = GroupKFold(n_splits=n_splits)
    fold_of: dict[str, int] = {}
    for fold_index, (_, test_idx) in enumerate(splitter.split(dummy_x, dummy_y, groups)):
        for row in test_idx:
            fold_of[marc_ids[int(row)]] = fold_index
    return fold_of


def _assign_oof_learned(pairs: list[ScoredPair]) -> list[ScoredPair]:
    """Return ``pairs`` rebuilt with an out-of-fold learned score on each.

    GroupKFold by ``marc_control_id``. For each fold a fresh ``LGBMClassifier``
    is fit on the OTHER folds' ``(feature_row, label=1 if match else 0)`` rows,
    then ``predict_proba`` produces the held-out fold's learned scores. The
    production ``caches/learned_scorer.*`` artifact is never touched.
    """
    distinct_marcs: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        marc_id = pair.entry.marc_control_id
        if marc_id not in seen:
            seen.add(marc_id)
            distinct_marcs.append(marc_id)
    fold_of = _fold_assignment(distinct_marcs)
    n_folds = max(fold_of.values()) + 1 if fold_of else 0

    indices_by_fold: dict[int, list[int]] = {}
    for index, pair in enumerate(pairs):
        fold = fold_of[pair.entry.marc_control_id]
        indices_by_fold.setdefault(fold, []).append(index)

    learned_scores: list[float] = [0.0] * len(pairs)
    for fold_index in range(n_folds):
        holdout_indices = indices_by_fold.get(fold_index, [])
        train_indices = [i for i in range(len(pairs)) if i not in set(holdout_indices)]
        if not holdout_indices or not train_indices:
            continue
        train_x: NDArray[float64] = asarray(
            [pairs[i].features for i in train_indices], dtype=float64
        )
        train_y: NDArray[int64] = asarray(
            [1 if pairs[i].entry.verdict == _VERDICT_MATCH else 0 for i in train_indices],
            dtype=int64,
        )
        model = _new_classifier()
        model.fit(train_x, train_y.astype(float64))
        holdout_x: NDArray[float64] = asarray(
            [pairs[i].features for i in holdout_indices], dtype=float64
        )
        proba = model.predict_proba(holdout_x)[:, 1]
        for slot, index in enumerate(holdout_indices):
            learned_scores[index] = float(proba[slot])
        _progress(
            f"fold {fold_index}: train={len(train_indices)} holdout={len(holdout_indices)}"
        )
    return [
        ScoredPair(
            entry=pair.entry,
            marc=pair.marc,
            cce=pair.cce,
            evidence=pair.evidence,
            evidence_sources=pair.evidence_sources,
            features=pair.features,
            weighted=pair.weighted,
            learned=learned_scores[index],
        )
        for index, pair in enumerate(pairs)
    ]


def _ordered_surfaced(pairs: list[ScoredPair]) -> dict[str, list[ScoredPair]]:
    """Group surfaced pairs by bucket and sort each group by disagreement starkness.

    ``label-error?`` is sorted by the max model confidence on the wrong side
    (most-confident contradiction first); ``model-vs-model`` by ``|learned -
    weighted|`` descending; ``whole/part`` by learned score descending (the most
    match-leaning incompatible-volume pairs first). Agreeing pairs are dropped.
    """
    grouped: dict[str, list[ScoredPair]] = {bucket: [] for bucket in _BUCKET_ORDER}
    for pair in pairs:
        bucket = _bucket_for(pair)
        if bucket is None:
            continue
        grouped[bucket].append(pair)
    grouped[_BUCKET_LABEL_ERROR].sort(key=_label_error_starkness, reverse=True)
    grouped[_BUCKET_MODEL_VS_MODEL].sort(key=_model_gap, reverse=True)
    grouped[_BUCKET_WHOLE_PART].sort(key=lambda p: p.learned, reverse=True)
    return grouped


def _write_outputs(
    grouped: dict[str, list[ScoredPair]],
    out_path: Path,
    text_path: Path,
    base_url: str,
    result: ScanResult,
) -> None:
    """Insert surfaced pairs into the review DB and write the flat link file."""
    lines: list[str] = []
    with ReviewDb.connect(out_path) as db:
        for bucket in _BUCKET_ORDER:
            bucket_pairs = grouped[bucket]
            result.surfaced[bucket] = len(bucket_pairs)
            lines.append(f"# {bucket} ({len(bucket_pairs)})")
            for pair in bucket_pairs:
                score = pair.learned
                pair_insert: PairInsert = _build_pair_insert(
                    pair.marc,
                    pair.cce,
                    pair.evidence,
                    language=_language_of(pair.marc),
                    score=score,
                    band=band_of(score),
                    source=SOURCE_BANDED,
                    evidence_sources=pair.evidence_sources,
                )
                pair_id = db.insert_pair(pair_insert)
                db.insert_existing_label(
                    pair_id=pair_id,
                    verdict=pair.entry.verdict,
                    labeled_at=pair.entry.labeled_at,
                    note=_note_for(bucket, pair),
                    categories=pair.entry.categories,
                )
                lines.append(_link_line(base_url, pair_id, bucket, pair))
        db.commit()
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scan(
    entries: dict[tuple[str, str], VaultEntry],
    out_path: Path,
    text_path: Path,
    base_url: str,
    limit: int | None,
) -> ScanResult:
    """Score every labeled pair with both matchers and surface the disagreements."""
    labeled = [e for e in entries.values() if e.verdict != _VERDICT_UNSURE]
    if limit is not None:
        labeled = labeled[:limit]
    result = ScanResult(scanned=len(labeled), limit=limit)

    matching_config = _load_default_matching_config()
    pairing_config = _load_default_pairing_config()
    weighted_config = _scoring_config(matching_config)
    pairings = compile_pairings(pairing_config)
    needed_marc_ids = {e.marc_control_id for e in labeled}
    marc_by_id = build_marc_index(_POOL_PATH, needed_marc_ids)
    _progress(f"resolved {len(marc_by_id)} MARCs from pool for {len(labeled)} pairs")

    with NyplIndexLookup(_INDEX_PATH) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
        pairs, unresolved = _resolved_pairs(
            labeled,
            marc_by_id,
            lookup,
            weighted_config,
            (idf, author_idf, publisher_idf),
            pairings,
        )
        result.unresolved = unresolved
        _progress(f"scored {len(pairs)} pairs (weighted); running OOF learned")
        scored = _assign_oof_learned(pairs)
        grouped = _ordered_surfaced(scored)
        result.agree = len(scored) - sum(len(grouped[b]) for b in _BUCKET_ORDER)
        _write_outputs(grouped, out_path, text_path, base_url, result)
    return result


def _print_summary(result: ScanResult, out_path: Path, text_path: Path, base_url: str) -> None:
    """Print the per-bucket counts, totals, and unresolved count to stdout."""
    print("=== cross-matcher disagreement scan ===")
    if result.limit is not None:
        print(
            f"\n*** SMOKE RUN — --limit {result.limit} capped pairs scanned. "
            "The DB and text file are PARTIAL and NOT the full review queue. ***\n"
        )
    print(f"review DB:   {out_path}")
    print(f"text file:   {text_path}")
    print(f"base URL:    {base_url}/pair/<id>")
    print(f"pairs scanned (non-unsure): {result.scanned}")
    print(f"unresolved (MARC/CCE gone): {result.unresolved}")
    print(f"agree (not surfaced):       {result.agree}")
    print(f"surfaced total:             {result.total_surfaced()}")
    print()
    print(f"{'bucket':<16} {'surfaced':>9}")
    print("-" * 26)
    for bucket in _BUCKET_ORDER:
        print(f"{bucket:<16} {result.surfaced_for(bucket):>9}")


def _parse_args() -> tuple[Path, Path, str, int | None, bool]:
    """Parse ``--out`` / ``--text`` / ``--base-url`` / ``--limit`` / ``--rebuild``."""
    parser = ArgumentParser(
        description="Cross-matcher disagreement scan over the labeled vault."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        metavar="PATH",
        help=f"destination review DB (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--text",
        type=Path,
        default=None,
        metavar="PATH",
        help="flat link file (default: the --out path with a .txt suffix)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=_DEFAULT_BASE_URL,
        metavar="URL",
        help=f"review server base URL for links (default: {_DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap pairs scanned (smoke run; DB + text are partial)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="overwrite the output DB if it already exists",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    out_path: Path = args.out
    text_path: Path = args.text if args.text is not None else out_path.with_suffix(".txt")
    base_url: str = args.base_url.rstrip("/")
    limit: int | None = args.limit
    rebuild: bool = args.rebuild
    return out_path, text_path, base_url, limit, rebuild


def _prepare_out_path(out_path: Path, rebuild: bool) -> None:
    """Refuse to silently overwrite the review DB without ``--rebuild``.

    Raises:
        SystemExit: If ``out_path`` exists and ``--rebuild`` was not given.
    """
    if out_path.exists():
        if not rebuild:
            raise SystemExit(f"{out_path} already exists; pass --rebuild to overwrite it.")
        out_path.unlink()


def main() -> None:
    """Run the disagreement scan and print the summary to stdout."""
    out_path, text_path, base_url, limit, rebuild = _parse_args()
    _progress(f"disagreement_scan started (out={out_path}, limit={limit})")
    _prepare_out_path(out_path, rebuild)
    entries = current_entries(_VAULT_PATH)
    result = run_scan(entries, out_path, text_path, base_url, limit)
    _print_summary(result, out_path, text_path, base_url)
    _progress(
        f"done: surfaced {result.total_surfaced()} "
        f"(label-error?={result.surfaced_for(_BUCKET_LABEL_ERROR)}, "
        f"model-vs-model={result.surfaced_for(_BUCKET_MODEL_VS_MODEL)}, "
        f"whole/part={result.surfaced_for(_BUCKET_WHOLE_PART)})"
    )


if __name__ == "__main__":
    main()
