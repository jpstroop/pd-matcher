"""Build a fresh, middle-heavy, held-out separation test queue (issue #4).

One-off tooling. NOT part of the shipped package; ``scripts/*`` is gitignored
(an ``!`` exception lets the maintainer commit it). It does NOT modify anything
under ``src/`` and it never writes the label vault — the vault is read STRICTLY
to build the held-out EXCLUDE set. It writes exactly one file:
``data/separation_test.db`` (or a ``--out`` override), a valid review-queue
SQLite database the review UI serves verbatim via
``pd-groundtruth review --db <out>``.

WHY. The learned scorer shows clean match/non-match separation on the labeled
vault (AUC 0.99), but the vault is structurally biased toward pairs the matcher
already surfaces (the vault blind-spot finding). To measure whether that
separation holds on WILD, unseen data we need a held-out test set concentrated
in the score region where the match/non-match boundary is actually decided.
This script builds that queue: ~500 FRESH (MARC, top-CCE-candidate) pairs whose
MARC has NEVER been labeled, bucketed by fine score band toward per-band targets
that over-sample the 0.60-0.70 decision region. The labeler then labels the
queue; those verdicts become the held-out separation test set.

HOW (reuses production machinery, reinvents nothing):

* The pool/index/vault idioms come from ``scripts/learned_scorer_heldout.py``
  and ``scripts/cce_part_signal.py``: ``current_entries`` for the vault,
  ``iter_pool_shards`` to stream the wild pool deterministically,
  ``NyplIndexLookup`` + ``build_idf_table`` for the index, the CLI's
  ``_load_default_matching_config`` / ``_load_default_pairing_config`` /
  ``_learned_model_dir`` so scoring matches production EXACTLY.
* Each MARC's top candidate + calibrated score + evidence comes from the SAME
  ``pd_matcher.match.pipeline.match_record`` ``build-queue`` drives (production
  config, production combiner, production calibrator).
* Each kept pair is assembled by ``build_queue._build_pair_insert`` — the exact
  row-builder ``build-queue`` uses — and inserted via
  ``ReviewDb.insert_pair`` so the output DB matches the review schema column
  for column. No ``label`` rows are written: the queue is unlabeled by design.

The fine targeting bands here are NOT ``sampling.band_of``'s coarse bands; they
add the 0.50 and sub-0.50 boundaries the separation experiment needs. Each
inserted row still stores ``sampling.band_of(score)`` in its ``band`` column so
the review UI's banding stays canonical.

Usage:
    # Smoke (caps MARCs streamed; targets will NOT fill — pipeline check only):
    pdm run python scripts/build_separation_test_queue.py \\
        --limit 200 --out /tmp/septest_smoke.db --rebuild

    # Full generation (targets fill; this is the labeler's queue):
    pdm run python scripts/build_separation_test_queue.py \\
        --out data/separation_test.db --rebuild
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from sys import stderr
from typing import Final

from msgspec import structs

from pd_groundtruth.build_queue import _build_pair_insert
from pd_groundtruth.build_queue import _language_of
from pd_groundtruth.build_queue import _load_calibrator
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import band_of
from pd_groundtruth.vault_pair_resolver import iter_pool_shards
from pd_matcher.cli import _learned_model_dir
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord
from pd_matcher.parsers.marc import iter_marc_records

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_DEFAULT_OUT: Final[Path] = Path("data/separation_test.db")
_PROGRESS_LOG: Final[Path] = Path("/tmp/agent-progress.log")

_PROGRESS_EVERY: Final[int] = 50

# Fine targeting bands for the separation experiment. Each is a half-open
# ``[lo, hi)`` calibrated-score interval with a fill target; ``hi`` of ``None``
# is the open-ended top tail. The 0.60-0.70 decision region is the priority
# (where match/non-match separation is decided); the 0.50-0.60 and 0.70-0.80
# shoulders flank it; the <0.50 (no-match-heavy) and >=0.90 (high-confidence
# anchor) tails bound the experiment. The 0.80-0.90 region is intentionally not
# targeted. Tune the per-band targets here.
_BAND_TAIL_LOW: Final[str] = "tail_below_50"
_BAND_SHOULDER_LO: Final[str] = "shoulder_50_60"
_BAND_DECISION: Final[str] = "decision_60_70"
_BAND_SHOULDER_HI: Final[str] = "shoulder_70_80"
_BAND_TAIL_HIGH: Final[str] = "anchor_ge_90"

_TARGET_TAIL_LOW: Final[int] = 50
_TARGET_SHOULDER_LO: Final[int] = 75
_TARGET_DECISION: Final[int] = 250
_TARGET_SHOULDER_HI: Final[int] = 75
_TARGET_TAIL_HIGH: Final[int] = 50


@dataclass(frozen=True, slots=True)
class FineBand:
    """One fine targeting band: a half-open score interval and its fill target."""

    label: str
    lo: float
    hi: float | None
    target: int

    def contains(self, score: float) -> bool:
        """Return whether ``score`` falls in this band's ``[lo, hi)`` interval."""
        if score < self.lo:
            return False
        if self.hi is None:
            return True
        return score < self.hi


_FINE_BANDS: Final[tuple[FineBand, ...]] = (
    FineBand(_BAND_TAIL_LOW, 0.0, 0.50, _TARGET_TAIL_LOW),
    FineBand(_BAND_SHOULDER_LO, 0.50, 0.60, _TARGET_SHOULDER_LO),
    FineBand(_BAND_DECISION, 0.60, 0.70, _TARGET_DECISION),
    FineBand(_BAND_SHOULDER_HI, 0.70, 0.80, _TARGET_SHOULDER_HI),
    FineBand(_BAND_TAIL_HIGH, 0.90, None, _TARGET_TAIL_HIGH),
)


def _progress(message: str) -> None:
    """Emit a one-line milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _fine_band_for(score: float) -> FineBand | None:
    """Return the fine targeting band ``score`` falls in, or ``None`` if untargeted.

    The 0.80-0.90 interval is intentionally absent from :data:`_FINE_BANDS`, so a
    score there returns ``None`` and the pair is dropped.
    """
    for band in _FINE_BANDS:
        if band.contains(score):
            return band
    return None


@dataclass(slots=True)
class QueueResult:
    """Aggregate outcome of one queue build."""

    kept: dict[str, int] = field(default_factory=dict)
    marcs_streamed: int = 0
    marcs_skipped_vault: int = 0
    marcs_no_top: int = 0
    pool_exhausted: bool = False
    limit: int | None = None

    def kept_for(self, label: str) -> int:
        """Return the kept count for a fine band (``0`` when none kept yet)."""
        return self.kept.get(label, 0)

    def total_kept(self) -> int:
        """Return the total pairs kept across all bands."""
        return sum(self.kept.values())

    def all_targets_met(self) -> bool:
        """Return whether every fine band has reached its target."""
        return all(self.kept_for(band.label) >= band.target for band in _FINE_BANDS)


@dataclass(frozen=True, slots=True)
class Scoring:
    """The resolved production scoring machinery, bound once per run."""

    matching_config: MatchingConfig
    pairings: CompiledPairings
    combiner: Combiner
    idf: IdfTable
    author_idf: IdfTable
    publisher_idf: IdfTable
    calibrator: PlattCalibrator | None


def _build_scoring(lookup: NyplIndexLookup) -> Scoring:
    """Resolve the production config, combiner, calibrator, idf, and pairings.

    Mirrors the CLI's match path: default ``matching.yaml`` / pairings, the
    learned model dir only when the active scorer is learned, the Platt
    calibrator from ``caches/`` if present, and an IDF table over the open
    lookup. Every value flows straight into ``match_record`` unchanged.
    """
    # Floor removed (min_combined_score=0.0) so match_record returns the top
    # candidate for EVERY MARC regardless of score. The production 0.50 floor
    # suppresses sub-0.50 pairs entirely (best=None), which made the below-0.50
    # tail — the no-match-heavy "unknown" region this test most needs — unfillable.
    # The calibrated score itself is floor-independent, so banding is unchanged.
    matching_config = structs.replace(_load_default_matching_config(), min_combined_score=0.0)
    pairing_config = _load_default_pairing_config()
    learned_model_dir = _learned_model_dir(_INDEX_PATH.parent, matching_config)
    combiner = build_combiner(matching_config, learned_model_dir=learned_model_dir)
    calibrator = _load_calibrator(_INDEX_PATH.parent)
    idf = build_idf_table(lookup)
    author_idf = build_author_idf_table(lookup)
    publisher_idf = build_publisher_idf_table(lookup)
    pairings = compile_pairings(pairing_config)
    return Scoring(
        matching_config=matching_config,
        pairings=pairings,
        combiner=combiner,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        calibrator=calibrator,
    )


def _pair_for_top_candidate(
    marc: MarcRecord, scoring: Scoring, lookup: NyplIndexLookup
) -> tuple[PairInsert, FineBand] | None:
    """Score one MARC's top candidate; return ``(pair, fine_band)`` or ``None``.

    Runs ``match_record`` with the floor removed (see ``_build_scoring``), so the
    top candidate is returned for every MARC across the full score range; bails
    only when retrieval yields no candidate at all, computes the fine targeting
    band from the calibrated score, drops untargeted scores (0.80-0.90), and
    assembles the review-schema row via the shared ``build_queue._build_pair_insert``.
    """
    result = match_record(
        marc,
        lookup=lookup,
        config=scoring.matching_config,
        idf=scoring.idf,
        author_idf=scoring.author_idf,
        publisher_idf=scoring.publisher_idf,
        calibrator=scoring.calibrator,
        combiner=scoring.combiner,
        pairings=scoring.pairings,
        top_k=1,
    )
    if result.best is None:
        return None
    score = result.best.combined.calibrated
    fine_band = _fine_band_for(score)
    if fine_band is None:
        return None
    matched_cce = lookup.get_registration(result.best.nypl_uuid)
    if matched_cce is None:
        return None
    pair = _build_pair_insert(
        marc,
        matched_cce,
        result.best.evidence,
        language=_language_of(marc),
        score=score,
        band=band_of(score),
        source=SOURCE_BANDED,
        evidence_sources=result.best.evidence_sources,
    )
    return pair, fine_band


def _iter_fresh_marcs(exclude: frozenset[str]) -> Iterator[MarcRecord]:
    """Yield wild-pool MARC records whose control id is NOT in ``exclude``.

    Streams ``data/candidates/<lang>/*.xml`` in the deterministic shard order of
    :func:`iter_pool_shards`, so a given ``--limit`` always selects the same
    prefix of MARCs. Vault MARCs are dropped before scoring so the queue is
    strictly held out.
    """
    for shard in iter_pool_shards(_POOL_PATH):
        for record in iter_marc_records(shard):
            if record.control_id in exclude:
                continue
            yield record


def build_separation_queue(
    out_path: Path, exclude: frozenset[str], limit: int | None
) -> QueueResult:
    """Stream the held-out pool into ``out_path`` until every band target fills.

    Args:
        out_path: Destination review DB (must not already exist; the caller
            enforces ``--rebuild``).
        exclude: MARC control ids to skip (the vault set) so the queue is fresh.
        limit: Optional cap on MARCs STREAMED (smoke runs); ``None`` streams
            until targets fill or the pool exhausts.

    Returns:
        A populated :class:`QueueResult`.
    """
    result = QueueResult(limit=limit)
    with NyplIndexLookup(_INDEX_PATH) as lookup:
        scoring = _build_scoring(lookup)
        _progress("scoring machinery ready; streaming pool")
        with ReviewDb.connect(out_path) as db:
            result.pool_exhausted = True
            for marc in _iter_fresh_marcs(exclude):
                if limit is not None and result.marcs_streamed >= limit:
                    result.pool_exhausted = False
                    break
                result.marcs_streamed += 1
                outcome = _pair_for_top_candidate(marc, scoring, lookup)
                if outcome is None:
                    result.marcs_no_top += 1
                    continue
                pair, band = outcome
                if result.kept_for(band.label) >= band.target:
                    continue
                db.insert_pair(pair)
                result.kept[band.label] = result.kept_for(band.label) + 1
                if result.total_kept() % _PROGRESS_EVERY == 0:
                    _progress(_fill_line(result))
                if result.all_targets_met():
                    result.pool_exhausted = False
                    break
    result.marcs_skipped_vault = _count_skipped(exclude, result.marcs_streamed)
    _progress(f"done: {_fill_line(result)}")
    return result


def _count_skipped(exclude: frozenset[str], streamed_budget: int) -> int:
    """Count vault MARCs encountered within the streamed shard prefix.

    Re-walks the pool the same way :func:`_iter_fresh_marcs` does but counts the
    excluded records, stopping once it has passed as many NON-excluded records as
    were streamed. This attributes the vault-skip count to exactly the shard
    prefix the build consumed, regardless of ``--limit`` or early target fill.
    """
    seen_fresh = 0
    skipped = 0
    for shard in iter_pool_shards(_POOL_PATH):
        for record in iter_marc_records(shard):
            if seen_fresh >= streamed_budget:
                return skipped
            if record.control_id in exclude:
                skipped += 1
                continue
            seen_fresh += 1
    return skipped


def _fill_line(result: QueueResult) -> str:
    """Render a compact per-band kept/target readout."""
    parts = [f"{band.label}={result.kept_for(band.label)}/{band.target}" for band in _FINE_BANDS]
    return f"kept {result.total_kept()} [{', '.join(parts)}] streamed={result.marcs_streamed}"


def _print_summary(result: QueueResult, out_path: Path) -> None:
    """Print the human-readable build summary to stdout."""
    print("=== separation test queue build ===")
    if result.limit is not None:
        print(
            f"SMOKE RUN: --limit {result.limit} capped MARCs streamed; "
            "targets will not fill and this DB is NOT the labeler's queue."
        )
    print(f"output: {out_path}")
    print(f"MARCs streamed (non-vault): {result.marcs_streamed}")
    print(f"MARCs skipped as vault:     {result.marcs_skipped_vault}")
    print(f"MARCs with no top candidate / untargeted band: {result.marcs_no_top}")
    print(f"pool exhausted before fill: {result.pool_exhausted}")
    print(f"pairs written total:        {result.total_kept()}")
    print()
    print(f"{'fine band':<18} {'kept':>6} {'target':>7} {'status':>8}")
    print("-" * 43)
    for band in _FINE_BANDS:
        kept = result.kept_for(band.label)
        status = "FULL" if kept >= band.target else "short"
        print(f"{band.label:<18} {kept:>6} {band.target:>7} {status:>8}")
    if result.pool_exhausted and not result.all_targets_met():
        print()
        print(
            "WARNING: pool exhausted before all targets filled; the short bands "
            "above did not reach their target (expected under --limit)."
        )


def _vault_marc_ids() -> frozenset[str]:
    """Return the set of MARC control ids appearing in the current vault.

    The vault is opened read-only; this build never writes it. Every MARC the
    model could train on is excluded so the queue stays a fresh held-out set.
    """
    entries = current_entries(_VAULT_PATH)
    return frozenset(entry.marc_control_id for entry in entries.values())


def _parse_args() -> tuple[Path, int | None, bool]:
    """Parse ``--out`` / ``--limit`` / ``--rebuild`` from argv."""
    parser = ArgumentParser(
        description="Build the middle-heavy held-out separation test queue (issue #4)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        metavar="PATH",
        help=f"destination review DB (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap MARCs STREAMED (smoke run; targets will not fill)",
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
    limit: int | None = args.limit
    rebuild: bool = args.rebuild
    return out_path, limit, rebuild


def _prepare_out_path(out_path: Path, rebuild: bool) -> None:
    """Enforce ``build-queue``-style safety: never silently overwrite the DB.

    Raises:
        SystemExit: If ``out_path`` exists and ``--rebuild`` was not given.
    """
    if out_path.exists():
        if not rebuild:
            raise SystemExit(f"{out_path} already exists; pass --rebuild to overwrite it.")
        out_path.unlink()


def main() -> None:
    """Resolve the vault exclude set, build the queue, and print the summary."""
    out_path, limit, rebuild = _parse_args()
    _progress(f"build_separation_test_queue started (out={out_path}, limit={limit})")
    _prepare_out_path(out_path, rebuild)
    exclude = _vault_marc_ids()
    _progress(f"vault exclude set: {len(exclude)} MARC ids (vault read-only)")
    result = build_separation_queue(out_path, exclude, limit)
    _print_summary(result, out_path)
    _progress("summary printed")


if __name__ == "__main__":
    main()
