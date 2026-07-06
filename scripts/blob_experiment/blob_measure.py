"""Blob-matcher measurement over the 1,934 registration-arm vault pairs.

Resolves each pair (MARC via pool, CCE via LMDB), computes stock title/author/
combined evidence with the production scorer, and computes blob variants:
  a. full IDF-weighted Jaccard over all blob tokens
  b. top-k IDF-weighted Jaccard, k in {8,12,16,24}
  c. unweighted Jaccard (floor)
Then AUCs, both tails, complementarity, field-bleed, same-author-trap.

MEASUREMENT ONLY. Artifacts under /tmp/blob/.
"""

import csv
import sys
from pathlib import Path
from statistics import quantiles

sys.path.insert(0, "/tmp/blob")

from msgspec.msgpack import Decoder

from blob_common import BlobIdf
from blob_common import cce_blob
from blob_common import idf_from_df
from blob_common import marc_blob
from blob_common import top_k
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
_TITLE = "title.token_set"
_AUTHOR = "name.author"
_KS = (8, 12, 16, 24)

report: list[str] = []


def emit(s: str = "") -> None:
    print(s, flush=True)
    report.append(s)


def load_blob_idf() -> BlobIdf:
    payload = Decoder().decode(Path("/tmp/blob/blob_idf.msgpack").read_bytes())
    return idf_from_df(payload["df"], payload["document_count"])


def q5(vals: list[float]) -> list[float]:
    vals = sorted(vals)
    if len(vals) < 2:
        return [vals[0] if vals else 0.0] * 5
    qs = quantiles(vals, n=100, method="inclusive")
    return [qs[9], qs[24], qs[49], qs[74], qs[89]]


def fmt(vals: list[float]) -> str:
    return "/".join(f"{x:.3f}" for x in vals)


def ev_norm(cand, scorer: str) -> float:  # noqa: ANN001
    for e in cand.evidence:
        if e.scorer == scorer:
            return e.normalized if not e.skipped else 0.0
    return 0.0


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
marc_by_id = build_marc_index(Path("data/candidates"), {e.marc_control_id for e in entries})
emit(f"vault registration-arm entries: {len(entries)}; marc resolved: {len(marc_by_id)}")

bidf = load_blob_idf()
emit(f"blob IDF: {bidf.document_count} CCE docs, {len(bidf.idf)} tokens, default_idf {bidf.default_idf:.3f}")

rows: list[dict] = []
calibrator = _load_calibrator(_INDEX.parent)
with NyplIndexLookup(_INDEX) as lk:
    idf = build_idf_table(lk)
    aidf = build_author_idf_table(lk)
    pidf = build_publisher_idf_table(lk)
    score = make_pair_scorer(
        matching_config=cfg, pairings=pairings, idf=idf, author_idf=aidf,
        publisher_idf=pidf, calibrator=calibrator, learned_model_dir=None,
    )
    n = 0
    for e in entries:
        marc = marc_by_id.get(e.marc_control_id)
        cce = lk.get_registration(e.nypl_uuid)
        if marc is None or cce is None:
            continue
        cand = score(marc, cce)
        mb = marc_blob(marc)
        cb = cce_blob(cce)
        row: dict = {
            "label": 1 if e.verdict == "match" else 0,
            "t_stock": ev_norm(cand, _TITLE),
            "a_stock": ev_norm(cand, _AUTHOR),
            "c_stock": cand.combined.calibrated,
            "blob_full": weighted_jaccard(mb, cb, bidf),
            "blob_unw": unweighted_jaccard(mb, cb),
            "n_marc_tok": len(mb),
            "n_cce_tok": len(cb),
            "marc_title": str(marc.title)[:50],
            "cce_title": str(cce.title)[:50],
        }
        for k in _KS:
            mk = top_k(mb, bidf, k)
            ck = top_k(cb, bidf, k)
            row[f"blob_k{k}"] = weighted_jaccard(mk, ck, bidf)
        rows.append(row)
        n += 1
        if n % 400 == 0:
            emit(f"  scored {n}")

emit(f"pairs scored: {len(rows)}")
pos = [r for r in rows if r["label"] == 1]
neg = [r for r in rows if r["label"] == 0]
emit(f"  matches: {len(pos)}  non-matches: {len(neg)}")

# ---------- AUC TABLE ----------
emit("\n" + "=" * 70)
emit("AUC TABLE (vs stock references on the SAME pairs)")
emit("=" * 70)
variants = ["blob_full", "blob_unw"] + [f"blob_k{k}" for k in _KS]
aucs: dict[str, float] = {}
for v in variants:
    aucs[v] = roc_auc([(r[v], r["label"]) for r in rows])
auc_c = roc_auc([(r["c_stock"], r["label"]) for r in rows])
auc_t = roc_auc([(r["t_stock"], r["label"]) for r in rows])
auc_a = roc_auc([(r["a_stock"], r["label"]) for r in rows])
emit(f"  stock combined (weighted arm)     AUC {auc_c:.4f}   [/tmp/dual harness: 0.9880]")
emit(f"  stock title.token_set evidence    AUC {auc_t:.4f}   [reference: 0.9298]")
emit(f"  stock name.author evidence        AUC {auc_a:.4f}")
emit("  ---- blob variants ----")
emit(f"  blob full IDF-weighted Jaccard    AUC {aucs['blob_full']:.4f}")
for k in _KS:
    emit(f"  blob top-{k:<2d} IDF-weighted Jaccard  AUC {aucs[f'blob_k{k}']:.4f}")
emit(f"  blob unweighted Jaccard (floor)   AUC {aucs['blob_unw']:.4f}")

best = max(variants, key=lambda v: aucs[v])
emit(f"\n  BEST blob variant: {best}  (AUC {aucs[best]:.4f})")

# ---------- BOTH TAILS for the winner ----------
emit("\n" + "=" * 70)
emit(f"BOTH TAILS — winner {best} (p10/p25/p50/p75/p90)")
emit("=" * 70)
emit(f"  MATCH     blob: {fmt(q5([r[best] for r in pos]))}")
emit(f"  NON-MATCH blob: {fmt(q5([r[best] for r in neg]))}")
emit(f"  MATCH     stock combined: {fmt(q5([r['c_stock'] for r in pos]))}")
emit(f"  NON-MATCH stock combined: {fmt(q5([r['c_stock'] for r in neg]))}")

# ---------- COMPLEMENTARITY ----------
emit("\n" + "=" * 70)
emit("COMPLEMENTARITY (does blob disagree in the RIGHT direction?)")
emit("=" * 70)
weak = [r for r in pos if r["c_stock"] < 0.8]
emit(f"\n(i) MATCH pairs where stock combined < 0.8 (weighted arm's weak spot): {len(weak)}")
if weak:
    emit(f"    blob({best}) on them  p10/25/50/75/90: {fmt(q5([r[best] for r in weak]))}")
    emit(f"    stock combined on them p10/25/50/75/90: {fmt(q5([r['c_stock'] for r in weak]))}")
    for thr in (0.3, 0.5):
        rec = sum(1 for r in weak if r[best] >= thr)
        emit(f"    blob >= {thr}: {rec}/{len(weak)} ({100*rec/len(weak):.0f}%) rescued above that bar")
fp = [r for r in neg if r["c_stock"] > 0.5]
emit(f"\n(ii) NON-MATCH pairs where stock combined > 0.5 (weighted false-positive tail): {len(fp)}")
if fp:
    emit(f"    blob({best}) on them  p10/25/50/75/90: {fmt(q5([r[best] for r in fp]))}")
    emit(f"    stock combined on them p10/25/50/75/90: {fmt(q5([r['c_stock'] for r in fp]))}")
    for thr in (0.3, 0.5):
        low = sum(1 for r in fp if r[best] < thr)
        emit(f"    blob <  {thr}: {low}/{len(fp)} ({100*low/len(fp):.0f}%) kept low (blob would not confirm)")

# ---------- FIELD-BLEED SUBSET ----------
emit("\n" + "=" * 70)
emit("FIELD-BLEED SUBSET: MATCH pairs with stock title.token_set < 0.5")
emit("(data in the wrong field / mangled title — where a blob should win)")
emit("=" * 70)
bleed = [r for r in pos if r["t_stock"] < 0.5]
emit(f"  count: {len(bleed)}")
if bleed:
    emit(f"    blob({best})   p10/25/50/75/90: {fmt(q5([r[best] for r in bleed]))}")
    emit(f"    stock title    p10/25/50/75/90: {fmt(q5([r['t_stock'] for r in bleed]))}")
    emit(f"    stock combined p10/25/50/75/90: {fmt(q5([r['c_stock'] for r in bleed]))}")
    for thr in (0.3, 0.5):
        rec = sum(1 for r in bleed if r[best] >= thr)
        emit(f"    blob >= {thr}: {rec}/{len(bleed)} ({100*rec/len(bleed):.0f}%)")

# ---------- SAME-AUTHOR TRAP ----------
emit("\n" + "=" * 70)
emit("SAME-AUTHOR TRAP: NON-MATCH pairs with stock name.author > 0.8")
emit("(same/similar author, different work — where blob should fail worst)")
emit("=" * 70)
trap = [r for r in neg if r["a_stock"] > 0.8]
emit(f"  count: {len(trap)}")
if trap:
    emit(f"    blob({best})   p10/25/50/75/90: {fmt(q5([r[best] for r in trap]))}")
    emit(f"    stock combined p10/25/50/75/90: {fmt(q5([r['c_stock'] for r in trap]))}")
    for thr in (0.3, 0.5):
        hi = sum(1 for r in trap if r[best] >= thr)
        emit(f"    blob >= {thr}: {hi}/{len(trap)} ({100*hi/len(trap):.0f}%) blob false-confirms here")

# ---------- CSV ----------
with open("/tmp/blob/pairs.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    cols = ["label", "t_stock", "a_stock", "c_stock", "blob_full", "blob_unw"] + \
        [f"blob_k{k}" for k in _KS] + ["n_marc_tok", "n_cce_tok", "marc_title", "cce_title"]
    w.writerow(cols)
    for r in rows:
        w.writerow([
            r["label"],
            f"{r['t_stock']:.4f}", f"{r['a_stock']:.4f}", f"{r['c_stock']:.4f}",
            f"{r['blob_full']:.4f}", f"{r['blob_unw']:.4f}",
            *[f"{r[f'blob_k{k}']:.4f}" for k in _KS],
            r["n_marc_tok"], r["n_cce_tok"], r["marc_title"], r["cce_title"],
        ])

Path("/tmp/blob/report.txt").write_text("\n".join(report))
emit("\nwrote /tmp/blob/pairs.csv and /tmp/blob/report.txt")
