"""Phase-2 pool-ranking test for the blob third-matcher (GH #128).

Production-config retrieval (candidates_for, year_window=0) + weighted arm
(make_pair_scorer, calibrator, learned_model_dir=None) + blob arm
(unweighted full Jaccard primary; IDF-weighted recorded). Registration-arm
vault only (match_source != "renewal").

MEASUREMENT ONLY. Artifacts under /tmp/blob2/. /tmp/blob/ is read-only.

Usage:
    phase2.py timing   -> 50-MARC timing sample, projects full runtime
    phase2.py full     -> full run (or seeded sample if projection > budget)
"""

import sys
from pathlib import Path
from random import Random
from statistics import quantiles
from time import time

sys.path.insert(0, "/tmp/blob")

from msgspec.msgpack import Decoder

from blob_common import BlobIdf
from blob_common import cce_blob
from blob_common import idf_from_df
from blob_common import marc_blob
from blob_common import unweighted_jaccard
from blob_common import weighted_jaccard
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_groundtruth.vault_pair_resolver import make_pair_scorer
from pd_matcher.cli import _load_calibrator
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.eval.metrics import roc_auc
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings

_INDEX = Path("caches/cce.lmdb")
_SEED = 20260705
_BUDGET_MIN = 90.0
_PROGRESS = Path("/tmp/agent-progress.log")

MODE = sys.argv[1] if len(sys.argv) > 1 else "timing"

report: list[str] = []


def emit(s: str = "") -> None:
    print(s, flush=True)
    report.append(s)


def milestone(s: str) -> None:
    from datetime import datetime

    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} blob2 {s}\n"
    with _PROGRESS.open("a") as fh:
        fh.write(line)


def q5(vals: list[float]) -> list[float]:
    vals = sorted(vals)
    if len(vals) < 2:
        return [vals[0] if vals else 0.0] * 5
    qs = quantiles(vals, n=100, method="inclusive")
    return [qs[9], qs[24], qs[49], qs[74], qs[89]]


def fmt(vals: list[float]) -> str:
    return "/".join(f"{x:.3f}" for x in vals)


def load_blob_idf() -> BlobIdf:
    payload = Decoder().decode(Path("/tmp/blob/blob_idf.msgpack").read_bytes())
    return idf_from_df(payload["df"], payload["document_count"])


# ---- vault load ----
base = _load_default_matching_config()
cfg = MatchingConfig(
    title_weight=base.title_weight, author_weight=base.author_weight,
    publisher_weight=base.publisher_weight, edition_weight=base.edition_weight,
    lccn_weight=base.lccn_weight, isbn_weight=base.isbn_weight,
    extent_weight=base.extent_weight, volume_weight=base.volume_weight,
    year_window=0, min_combined_score=base.min_combined_score, scorer=base.scorer,
)
pairings = compile_pairings(_load_default_pairing_config())

raw = current_entries(Path("data/training/label_vault.jsonl"))
entries = [
    e for e in raw.values()
    if e.verdict in ("match", "no_match") and e.match_source != "renewal"
]
n_match = sum(1 for e in entries if e.verdict == "match")
n_no = sum(1 for e in entries if e.verdict == "no_match")
emit(f"vault registration-arm entries: {len(entries)}  (match={n_match}, no_match={n_no})")

# group by MARC: true (match) uuids and wrong (no_match) uuids
by_marc: dict[str, dict[str, set[str]]] = {}
for e in entries:
    d = by_marc.setdefault(e.marc_control_id, {"true": set(), "wrong": set()})
    if e.verdict == "match":
        d["true"].add(e.nypl_uuid)
    else:
        d["wrong"].add(e.nypl_uuid)

marc_ids = sorted(by_marc)
marc_by_id = build_marc_index(Path("data/candidates"), set(marc_ids))
resolved_ids = [m for m in marc_ids if m in marc_by_id]
emit(f"distinct MARCs: {len(marc_ids)}  resolved in pool: {len(resolved_ids)}")

bidf = load_blob_idf()
emit(f"blob IDF: {bidf.document_count} CCE docs, {len(bidf.idf)} tokens")

# ---- choose sample ----
rng = Random(_SEED)
order = list(resolved_ids)
rng.shuffle(order)
if MODE == "timing":
    work_ids = order[:50]
else:
    work_ids = order  # may be trimmed after timing projection
milestone(f"{MODE}: {len(work_ids)} MARCs to score of {len(resolved_ids)} resolved")


calibrator = _load_calibrator(_INDEX.parent)
pool_sizes: list[int] = []
# per-MARC scored records
marc_results: list[dict] = []

start = time()
with NyplIndexLookup(_INDEX) as lk:
    idf = build_idf_table(lk)
    aidf = build_author_idf_table(lk)
    pidf = build_publisher_idf_table(lk)
    score = make_pair_scorer(
        matching_config=cfg, pairings=pairings, idf=idf, author_idf=aidf,
        publisher_idf=pidf, calibrator=calibrator, learned_model_dir=None,
    )
    for i, mid in enumerate(work_ids):
        marc = marc_by_id[mid]
        pool = list(lk.candidates_for(marc, cfg.year_window))
        pool_sizes.append(len(pool))
        mb = marc_blob(marc)
        scored: list[dict] = []
        for cand in pool:
            cm = score(marc, cand)
            cb = cce_blob(cand)
            scored.append({
                "uuid": cand.uuid,
                "w": cm.combined.calibrated,
                "bu": unweighted_jaccard(mb, cb),
                "bi": weighted_jaccard(mb, cb, bidf),
            })
        marc_results.append({
            "mid": mid,
            "true": by_marc[mid]["true"],
            "wrong": by_marc[mid]["wrong"],
            "marc_title": str(marc.title)[:60],
            "scored": scored,
        })
        if (i + 1) % 200 == 0:
            el = time() - start
            milestone(f"{MODE}: {i+1}/{len(work_ids)} MARCs, {el:.0f}s elapsed, avg pool {sum(pool_sizes)/len(pool_sizes):.1f}")

elapsed = time() - start
n_pairs_scored = sum(len(r["scored"]) for r in marc_results)
emit(f"\nscored {len(marc_results)} MARCs, {n_pairs_scored} candidate-pairs in {elapsed:.1f}s")
if pool_sizes:
    ps = sorted(pool_sizes)
    emit(f"pool size: min {ps[0]} p50 {ps[len(ps)//2]} p90 {ps[int(len(ps)*0.9)]} max {ps[-1]} mean {sum(ps)/len(ps):.1f}")

if MODE == "timing":
    per_marc = elapsed / max(1, len(marc_results))
    proj_full_min = per_marc * len(resolved_ids) / 60.0
    emit(f"\nper-MARC: {per_marc*1000:.0f}ms")
    emit(f"PROJECTED full run over {len(resolved_ids)} MARCs: {proj_full_min:.1f} min")
    if proj_full_min > _BUDGET_MIN:
        n_fit = int(_BUDGET_MIN * 60.0 / per_marc)
        emit(f"OVER BUDGET ({_BUDGET_MIN} min): would sample {n_fit} MARCs ({100*n_fit/len(resolved_ids):.0f}% coverage)")
    else:
        emit(f"UNDER BUDGET: full run fits in {proj_full_min:.1f} min")
    milestone(f"timing done: {per_marc*1000:.0f}ms/MARC, projected full {proj_full_min:.1f}min")
    Path("/tmp/blob2/timing.txt").write_text("\n".join(report))
    sys.exit(0)

# ============================================================
# FULL ANALYSIS
# ============================================================
coverage = 100.0 * len(marc_results) / len(resolved_ids)
emit(f"\nCOVERAGE: {len(marc_results)}/{len(resolved_ids)} resolved MARCs ({coverage:.1f}%)")

# helper: rank of a uuid under a key (1-based), and whether top-1 is in target set
def rank_of(scored: list[dict], key: str, uuid: str) -> int:
    order = sorted(scored, key=lambda r: r[key], reverse=True)
    for idx, r in enumerate(order, 1):
        if r["uuid"] == uuid:
            return idx
    return -1


def top1_uuid(scored: list[dict], key: str) -> str:
    return max(scored, key=lambda r: r[key])["uuid"]


def score_of(scored: list[dict], uuid: str, key: str) -> float:
    for r in scored:
        if r["uuid"] == uuid:
            return r[key]
    return -1.0


# ---------- A. TOP-1 COMPLEMENTARITY (match pairs) ----------
emit("\n" + "=" * 70)
emit("A. TOP-1 COMPLEMENTARITY (vault MATCH pairs, per distinct MARC)")
emit("=" * 70)
match_marcs = [r for r in marc_results if r["true"]]
in_pool = []
for r in match_marcs:
    pool_uuids = {c["uuid"] for c in r["scored"]}
    r["true_in_pool"] = r["true"] & pool_uuids
    if r["true_in_pool"]:
        in_pool.append(r)
emit(f"MATCH-labeled MARCs: {len(match_marcs)}")
emit(f"true CCE in pool: {len(in_pool)}/{len(match_marcs)} ({100*len(in_pool)/max(1,len(match_marcs)):.1f}% in-pool rate)")

# 2x2 over in-pool match marcs (primary = unweighted blob 'bu')
cell = {"both": 0, "wonly": 0, "bonly": 0, "neither": 0}
only_blob_examples: list[dict] = []
neither_ranks: list[int] = []
for r in in_pool:
    scored = r["scored"]
    w_top = top1_uuid(scored, "w")
    b_top = top1_uuid(scored, "bu")
    w_ok = w_top in r["true"]
    b_ok = b_top in r["true"]
    if w_ok and b_ok:
        cell["both"] += 1
    elif w_ok and not b_ok:
        cell["wonly"] += 1
    elif b_ok and not w_ok:
        cell["bonly"] += 1
        # recovery example: blob got it, weighted didn't
        true_u = next(iter(r["true_in_pool"]))
        beat = max((c for c in scored if c["uuid"] not in r["true"]), key=lambda c: c["w"])
        only_blob_examples.append({
            "mid": r["mid"], "true_uuid": true_u, "marc_title": r["marc_title"],
            "w_true": score_of(scored, true_u, "w"), "bu_true": score_of(scored, true_u, "bu"),
            "w_top_uuid": w_top, "w_top_score": beat["w"], "bu_of_wtop": beat["bu"],
            "pool": len(scored),
        })
    else:
        cell["neither"] += 1
        # blob's rank of the true CCE
        true_u = next(iter(r["true_in_pool"]))
        neither_ranks.append(rank_of(scored, "bu", true_u))

tot = sum(cell.values())
emit(f"\n2x2 over {tot} in-pool MATCH MARCs (blob arm = unweighted Jaccard):")
emit(f"  both top-1 correct : {cell['both']:4d} ({100*cell['both']/max(1,tot):.1f}%)")
emit(f"  only weighted      : {cell['wonly']:4d} ({100*cell['wonly']/max(1,tot):.1f}%)")
emit(f"  only blob (RECOVERY): {cell['bonly']:4d} ({100*cell['bonly']/max(1,tot):.1f}%)")
emit(f"  neither            : {cell['neither']:4d} ({100*cell['neither']/max(1,tot):.1f}%)")

if neither_ranks:
    r5 = sum(1 for x in neither_ranks if 0 < x <= 5)
    r10 = sum(1 for x in neither_ranks if 0 < x <= 10)
    emit(f"\n  'neither' set ({len(neither_ranks)}): blob rank of true CCE <=5: {r5}  <=10: {r10}")

emit(f"\n  ONLY-BLOB recovery examples (up to 15):")
for ex in only_blob_examples[:15]:
    emit(f"   MARC {ex['mid']} | pool={ex['pool']} | {ex['marc_title']!r}")
    emit(f"     true {ex['true_uuid']}: weighted={ex['w_true']:.3f} blob={ex['bu_true']:.3f}")
    emit(f"     weighted top-1 {ex['w_top_uuid']}: w={ex['w_top_score']:.3f} blob={ex['bu_of_wtop']:.3f}")
if not only_blob_examples:
    emit("   (none — blob recovered no MARC that weighted lost)")

# ---------- B. POOL-LEVEL SEPARATION ----------
emit("\n" + "=" * 70)
emit("B. POOL-LEVEL SEPARATION (true CCE = positive, other pool members = negatives)")
emit("=" * 70)
w_pool: list[tuple[float, int]] = []
bu_pool: list[tuple[float, int]] = []
bi_pool: list[tuple[float, int]] = []
w_pos, w_neg, bu_pos, bu_neg, bi_pos, bi_neg = [], [], [], [], [], []
for r in in_pool:
    for c in r["scored"]:
        lab = 1 if c["uuid"] in r["true"] else 0
        w_pool.append((c["w"], lab)); bu_pool.append((c["bu"], lab)); bi_pool.append((c["bi"], lab))
        (w_pos if lab else w_neg).append(c["w"])
        (bu_pos if lab else bu_neg).append(c["bu"])
        (bi_pos if lab else bi_neg).append(c["bi"])
emit(f"pooled pairs: {len(w_pool)}  (pos {len(w_pos)}, neg {len(w_neg)})")
emit(f"\n  weighted arm   pooled AUC: {roc_auc(w_pool):.4f}")
emit(f"  blob unweighted pooled AUC: {roc_auc(bu_pool):.4f}")
emit(f"  blob IDF-wtd   pooled AUC: {roc_auc(bi_pool):.4f}")
emit(f"\n  quantiles p10/25/50/75/90:")
emit(f"  weighted  TRUE:     {fmt(q5(w_pos))}")
emit(f"  weighted  NON-TRUE: {fmt(q5(w_neg))}")
emit(f"  blob-unw  TRUE:     {fmt(q5(bu_pos))}")
emit(f"  blob-unw  NON-TRUE: {fmt(q5(bu_neg))}")
emit(f"  blob-idf  TRUE:     {fmt(q5(bi_pos))}")
emit(f"  blob-idf  NON-TRUE: {fmt(q5(bi_neg))}")

# ---------- C. FP-VETO AT SCALE (no_match pairs) ----------
emit("\n" + "=" * 70)
emit("C. FP-VETO AT SCALE (vault NO_MATCH pairs — known hard negatives)")
emit("=" * 70)
no_match_pairs = []
for r in marc_results:
    pool_uuids = {c["uuid"] for c in r["scored"]}
    for wu in r["wrong"]:
        if wu in pool_uuids:
            no_match_pairs.append((r, wu))
n_wrong_total = sum(len(r["wrong"]) for r in marc_results)
emit(f"NO_MATCH pairs (this sample): {n_wrong_total}; wrong CCE surfaced in pool: {len(no_match_pairs)}")

fp_scenario = 0  # weighted ranks wrong #1 with cal >= 0.5
veto_03 = 0
veto_05 = 0
w_wrong_scores, bu_wrong_scores = [], []
w_ranks = []
for r, wu in no_match_pairs:
    scored = r["scored"]
    w_rank = rank_of(scored, "w", wu)
    w_ranks.append(w_rank)
    w_sc = score_of(scored, wu, "w")
    bu_sc = score_of(scored, wu, "bu")
    w_wrong_scores.append(w_sc); bu_wrong_scores.append(bu_sc)
    is_fp = (w_rank == 1 and w_sc >= 0.5)
    if is_fp:
        fp_scenario += 1
        if bu_sc < 0.3:
            veto_03 += 1
        if bu_sc < 0.5:
            veto_05 += 1
emit(f"\n  weighted ranks wrong CCE #1 with cal>=0.5 (production FP): {fp_scenario}/{len(no_match_pairs)}")
if fp_scenario:
    emit(f"    of those, blob (unw) < 0.3: {veto_03}/{fp_scenario} ({100*veto_03/fp_scenario:.0f}%)  < 0.5: {veto_05}/{fp_scenario} ({100*veto_05/fp_scenario:.0f}%)")
if w_ranks:
    r1 = sum(1 for x in w_ranks if x == 1)
    emit(f"  weighted rank of wrong CCE: #1 in {r1}/{len(w_ranks)} pools")
emit(f"  wrong-CCE score quantiles p10/25/50/75/90:")
emit(f"    weighted: {fmt(q5(w_wrong_scores))}")
emit(f"    blob-unw: {fmt(q5(bu_wrong_scores))}")

# ---------- D. TIER PREVIEW ----------
emit("\n" + "=" * 70)
emit("D. TIER PREVIEW (weighted x blob agreement, two-way; learned treated absent)")
emit("=" * 70)
# match side: among weighted-top-1-correct in-pool match marcs, does blob agree on top-1?
w_correct = [r for r in in_pool if top1_uuid(r["scored"], "w") in r["true"]]
agree_match = sum(1 for r in w_correct if top1_uuid(r["scored"], "bu") == top1_uuid(r["scored"], "w"))
emit(f"MATCH side: weighted top-1 correct: {len(w_correct)}")
emit(f"  blob agrees on same top-1 uuid: {agree_match}/{len(w_correct)} ({100*agree_match/max(1,len(w_correct)):.1f}%)")
# no_match side: both reject (weighted gives wrong <0.5 AND blob gives wrong <0.3)
both_reject = 0
for r, wu in no_match_pairs:
    w_sc = score_of(r["scored"], wu, "w")
    bu_sc = score_of(r["scored"], wu, "bu")
    if w_sc < 0.5 and bu_sc < 0.3:
        both_reject += 1
emit(f"NO_MATCH side: both arms reject wrong CCE (weighted<0.5 AND blob<0.3): {both_reject}/{len(no_match_pairs)} ({100*both_reject/max(1,len(no_match_pairs)):.1f}%)")

milestone(f"full analysis done: coverage {coverage:.0f}%, only-blob {cell['bonly']}, blob pool AUC {roc_auc(bu_pool):.4f}")
Path("/tmp/blob2/report.txt").write_text("\n".join(report))
emit("\nwrote /tmp/blob2/report.txt")
