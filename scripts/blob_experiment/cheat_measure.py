"""Title-cheat measurement over the registration-arm vault pairs (GH #128).

Keeps the blob field-free for credit but guarantees title words get a say.
Three cheats, each applied on top of the winning full unweighted-Jaccard blob
AND the IDF-weighted blob:

  1. TITLE-OVERLAP GATE   cap blob at c when title-bag overlap < t
  2. TITLE UP-WEIGHTING   multiply title-origin token weight by w in the Jaccard
  3. TWO-BAG MIN          final = min(blob, title_bag_score * k)

Title-origin tokens: MARC = title/title_main/title_part_name (NOT
statement_of_responsibility); CCE = title. Same blob_common extraction so the
tokens line up exactly with the blob.

MEASUREMENT ONLY. Artifacts under /tmp/blob/.
"""

import sys
from pathlib import Path
from statistics import quantiles

sys.path.insert(0, "/tmp/blob")

from msgspec.msgpack import Decoder

from blob_common import BlobIdf
from blob_common import blob_tokens
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
_TITLE = "title.token_set"
_AUTHOR = "name.author"

# cheat sweeps
_GATE_T = (0.1, 0.2, 0.3)
_GATE_C = (0.2, 0.3)
_UPW_W = (2, 3, 5)
_MIN_K = (1.5, 2.0)

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


def title_bag_score(mt: frozenset[str], ct: frozenset[str]) -> float:
    """max(jaccard, containment-of-smaller-in-larger).

    Titles embed subtitles / statements of responsibility on one side but not
    the other, so a strict Jaccard punishes a legitimate title whose bag is a
    subset of a fuller transcription. Containment (|inter| / min bag size)
    rewards one bag being wholly inside the other; taking the max keeps the
    metric high whenever either interpretation says the titles agree.
    """
    if not mt or not ct:
        return 0.0
    inter = len(mt & ct)
    if inter == 0:
        return 0.0
    jac = inter / len(mt | ct)
    cont = inter / min(len(mt), len(ct))
    return max(jac, cont)


def upweighted_jaccard(
    a: frozenset[str],
    b: frozenset[str],
    origin: frozenset[str],
    w: float,
    idf: BlobIdf | None,
) -> float:
    """Jaccard where title-origin tokens carry weight * w.

    idf=None -> unweighted base (base weight 1 per token); otherwise IDF base.
    """
    union = a | b
    if not union:
        return 0.0
    inter = a & b

    def wt(t: str) -> float:
        base = idf.w(t) if idf is not None else 1.0
        return base * (w if t in origin else 1.0)

    num = sum(wt(t) for t in inter)
    den = sum(wt(t) for t in union)
    return num / den if den else 0.0


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
        mt = blob_tokens([marc.title, marc.title_main, marc.title_part_name])
        ct = blob_tokens([cce.title])
        origin = mt | ct
        tbs = title_bag_score(mt, ct)
        blob_full = weighted_jaccard(mb, cb, bidf)
        blob_unw = unweighted_jaccard(mb, cb)
        row: dict = {
            "label": 1 if e.verdict == "match" else 0,
            "t_stock": ev_norm(cand, _TITLE),
            "a_stock": ev_norm(cand, _AUTHOR),
            "c_stock": cand.combined.calibrated,
            "blob_full": blob_full,
            "blob_unw": blob_unw,
            "tbs": tbs,
        }
        # variant 2: up-weighting (recomputed Jaccard) for each base
        for w in _UPW_W:
            row[f"upw_full_w{w}"] = upweighted_jaccard(mb, cb, origin, w, bidf)
            row[f"upw_unw_w{w}"] = upweighted_jaccard(mb, cb, origin, w, None)
        rows.append(row)
        n += 1
        if n % 400 == 0:
            emit(f"  scored {n}")

emit(f"pairs scored: {len(rows)}")
pos = [r for r in rows if r["label"] == 1]
neg = [r for r in rows if r["label"] == 0]
emit(f"  matches: {len(pos)}  non-matches: {len(neg)}")

# subsets (fixed once, referenced by every variant)
trap = [r for r in neg if r["a_stock"] > 0.8]
bleed = [r for r in pos if r["t_stock"] < 0.5]
fpveto = [r for r in neg if r["c_stock"] > 0.5]
emit(f"  SAME-AUTHOR TRAP subset (neg, a_stock>0.8): {len(trap)}")
emit(f"  FIELD-BLEED subset (pos, t_stock<0.5):      {len(bleed)}")
emit(f"  FP-VETO subset (neg, c_stock>0.5):          {len(fpveto)}")


def variant_scores(name: str, fn) -> dict:  # noqa: ANN001
    vals = {id(r): fn(r) for r in rows}
    auc = roc_auc([(vals[id(r)], r["label"]) for r in rows])
    trap50 = sum(1 for r in trap if vals[id(r)] >= 0.5)
    trap30 = sum(1 for r in trap if vals[id(r)] >= 0.3)
    bleed30 = sum(1 for r in bleed if vals[id(r)] >= 0.3)
    bleed50 = sum(1 for r in bleed if vals[id(r)] >= 0.5)
    veto = sum(1 for r in fpveto if vals[id(r)] < 0.5)
    return {
        "name": name, "auc": auc, "vals": vals,
        "trap50": trap50, "trap30": trap30,
        "bleed30": bleed30, "bleed50": bleed50, "veto": veto,
    }


def pct(x: int, n: int) -> str:
    return f"{x}/{n} ({100*x/n:.0f}%)" if n else "0/0"


results: list[dict] = []

# base references (no cheat)
for bname, key in (("blob_full", "blob_full"), ("blob_unw", "blob_unw")):
    results.append(variant_scores(f"BASE {bname}", (lambda k: lambda r: r[k])(key)))

# variant 1: title-overlap gate
for bname, key in (("full", "blob_full"), ("unw", "blob_unw")):
    for t in _GATE_T:
        for c in _GATE_C:
            def gate(r, key=key, t=t, c=c):  # noqa: ANN001
                return r[key] if r["tbs"] >= t else min(r[key], c)
            results.append(variant_scores(f"GATE {bname} t={t} c={c}", gate))

# variant 2: up-weighting
for bname in ("full", "unw"):
    for w in _UPW_W:
        col = f"upw_{bname}_w{w}"
        results.append(variant_scores(f"UPW {bname} w={w}", (lambda k: lambda r: r[k])(col)))

# variant 3: two-bag min
for bname, key in (("full", "blob_full"), ("unw", "blob_unw")):
    for k in _MIN_K:
        def twobag(r, key=key, k=k):  # noqa: ANN001
            return min(r[key], r["tbs"] * k)
        results.append(variant_scores(f"MIN {bname} k={k}", twobag))

# ---------- MASTER TABLE ----------
emit("\n" + "=" * 100)
emit("VARIANT TABLE  (baselines: blob_full AUC 0.9649, stock weighted 0.9879)")
emit("trap = same-author false-confirm (n={}); bleed = field-bleed survivor (n={}); veto = FP-tail kept<0.5 (n={})".format(
    len(trap), len(bleed), len(fpveto)))
emit("=" * 100)
hdr = f"{'variant':<22}{'AUC':>8}{'trap>=.5':>12}{'trap>=.3':>12}{'bleed>=.3':>14}{'bleed>=.5':>14}{'veto<.5':>12}"
emit(hdr)
emit("-" * 100)
for r in results:
    emit(
        f"{r['name']:<22}{r['auc']:>8.4f}"
        f"{pct(r['trap50'], len(trap)):>12}{pct(r['trap30'], len(trap)):>12}"
        f"{pct(r['bleed30'], len(bleed)):>14}{pct(r['bleed50'], len(bleed)):>14}"
        f"{pct(r['veto'], len(fpveto)):>12}"
    )

# ---------- TRADE-OFF FRONTIER (trap kills vs bleed survivors) ----------
emit("\n" + "=" * 100)
emit("TRADE-OFF FRONTIER  (baseline blob_full: trap>=.5 = 15%, trap>=.3 = 38%, bleed>=.3 = 76%)")
emit("trap_kill_50 = baseline_trap50 - variant_trap50 ; bleed_loss_30 = baseline_bleed30 - variant_bleed30")
emit("=" * 100)
base_full = next(r for r in results if r["name"] == "BASE blob_full")
base_unw = next(r for r in results if r["name"] == "BASE blob_unw")
emit(f"{'variant':<22}{'trap_kill50':>13}{'trap_kill30':>13}{'bleed_loss30':>14}{'net(kill50-loss30)':>20}")
emit("-" * 100)
for r in results:
    if r["name"].startswith("BASE"):
        continue
    b = base_full if " full " in r["name"] or r["name"].endswith("full") else base_unw
    # pick correct base by suffix token
    b = base_full if r["name"].split()[1] == "full" else base_unw
    kill50 = b["trap50"] - r["trap50"]
    kill30 = b["trap30"] - r["trap30"]
    loss30 = b["bleed30"] - r["bleed30"]
    emit(f"{r['name']:<22}{kill50:>13}{kill30:>13}{loss30:>14}{kill50 - loss30:>20}")

# ---------- WINNER TAILS ----------
emit("\n" + "=" * 100)
emit("WINNER SELECTION")
emit("=" * 100)


def dominates(r: dict, b: dict) -> bool:
    """Strictly dominates its own base: no worse on every axis, better on one."""
    no_worse = (
        r["auc"] >= b["auc"] - 1e-9
        and r["trap50"] <= b["trap50"]
        and r["trap30"] <= b["trap30"]
        and r["bleed30"] >= b["bleed30"]
        and r["veto"] >= b["veto"]
    )
    strictly = (
        r["auc"] > b["auc"] + 1e-9
        or r["trap50"] < b["trap50"]
        or r["trap30"] < b["trap30"]
        or r["bleed30"] > b["bleed30"]
        or r["veto"] > b["veto"]
    )
    return no_worse and strictly


emit("Variants that STRICTLY DOMINATE their base (AUC, trap50, trap30, veto no worse; bleed30 no worse; one strictly better):")
any_dom = False
for r in results:
    if r["name"].startswith("BASE"):
        continue
    b = base_full if r["name"].split()[1] == "full" else base_unw
    if dominates(r, b):
        any_dom = True
        emit(f"  DOMINATES {b['name']}: {r['name']}  "
             f"(AUC {r['auc']:.4f} vs {b['auc']:.4f}; trap50 {r['trap50']} vs {b['trap50']}; "
             f"trap30 {r['trap30']} vs {b['trap30']}; bleed30 {r['bleed30']} vs {b['bleed30']}; "
             f"veto {r['veto']} vs {b['veto']})")
if not any_dom:
    emit("  (none strictly dominate on all axes)")

# winner: best trap-kill while keeping bleed30 >= 90% of baseline and veto == full
emit("\nBEST trap-killers that keep bleed30 >= 90% of base and veto == 100%:")
cands = []
for r in results:
    if r["name"].startswith("BASE"):
        continue
    b = base_full if r["name"].split()[1] == "full" else base_unw
    if r["veto"] == len(fpveto) and r["bleed30"] >= 0.9 * b["bleed30"]:
        cands.append((b["trap50"] - r["trap50"], b["trap30"] - r["trap30"], r))
cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
for kill50, kill30, r in cands[:8]:
    emit(f"  {r['name']:<22} kill50={kill50} kill30={kill30} "
         f"AUC={r['auc']:.4f} bleed30={r['bleed30']} veto={r['veto']}/{len(fpveto)}")

winner = cands[0][2] if cands else base_full
emit(f"\nWINNER (best trap-kill under the bleed/veto guard): {winner['name']}")
wv = winner["vals"]
emit(f"  AUC {winner['auc']:.4f}")
emit(f"  same-author trap  >=0.5: {pct(winner['trap50'], len(trap))}   >=0.3: {pct(winner['trap30'], len(trap))}")
emit(f"  field-bleed       >=0.3: {pct(winner['bleed30'], len(bleed))}   >=0.5: {pct(winner['bleed50'], len(bleed))}")
emit(f"  fp-veto           <0.5:  {pct(winner['veto'], len(fpveto))}")
emit("\n  BOTH TAILS (p10/p25/p50/p75/p90):")
emit(f"    MATCH     {fmt(q5([wv[id(r)] for r in pos]))}")
emit(f"    NON-MATCH {fmt(q5([wv[id(r)] for r in neg]))}")

Path("/tmp/blob/cheat_report.txt").write_text("\n".join(report))
emit("\nwrote /tmp/blob/cheat_report.txt")
