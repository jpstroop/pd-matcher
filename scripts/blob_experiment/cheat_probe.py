"""Mechanism probe: title-bag overlap on trap vs bleed subsets (GH #128).

Explains WHY no title gate moves the same-author trap. MEASUREMENT ONLY.
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
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pairing_compiler import compile_pairings

_INDEX = Path("caches/cce.lmdb")


def q5(vals):
    vals = sorted(vals)
    if len(vals) < 2:
        return [vals[0] if vals else 0.0] * 5
    qs = quantiles(vals, n=100, method="inclusive")
    return [qs[9], qs[24], qs[49], qs[74], qs[89]]


def fmt(vals):
    return "/".join(f"{x:.3f}" for x in vals)


def tbs(mt, ct):
    if not mt or not ct:
        return 0.0
    inter = len(mt & ct)
    if inter == 0:
        return 0.0
    return max(inter / len(mt | ct), inter / min(len(mt), len(ct)))


def ev_norm(cand, scorer):
    for e in cand.evidence:
        if e.scorer == scorer:
            return e.normalized if not e.skipped else 0.0
    return 0.0


b = _load_default_matching_config()
cfg = MatchingConfig(
    title_weight=b.title_weight, author_weight=b.author_weight,
    publisher_weight=b.publisher_weight, edition_weight=b.edition_weight,
    lccn_weight=b.lccn_weight, isbn_weight=b.isbn_weight,
    extent_weight=b.extent_weight, volume_weight=b.volume_weight,
    year_window=0, min_combined_score=b.min_combined_score, scorer=b.scorer,
)
pairings = compile_pairings(_load_default_pairing_config())
raw = current_entries(Path("data/training/label_vault.jsonl"))
entries = [e for e in raw.values() if e.verdict in ("match", "no_match") and e.match_source != "renewal"]
marc_by_id = build_marc_index(Path("data/candidates"), {e.marc_control_id for e in entries})
bidf = idf_from_df(*[Decoder().decode(Path("/tmp/blob/blob_idf.msgpack").read_bytes())[k] for k in ("df", "document_count")])

rows = []
cal = _load_calibrator(_INDEX.parent)
with NyplIndexLookup(_INDEX) as lk:
    idf = build_idf_table(lk); aidf = build_author_idf_table(lk); pidf = build_publisher_idf_table(lk)
    score = make_pair_scorer(matching_config=cfg, pairings=pairings, idf=idf, author_idf=aidf, publisher_idf=pidf, calibrator=cal, learned_model_dir=None)
    for e in entries:
        marc = marc_by_id.get(e.marc_control_id); cce = lk.get_registration(e.nypl_uuid)
        if marc is None or cce is None:
            continue
        cand = score(marc, cce)
        mb = marc_blob(marc); cb = cce_blob(cce)
        mt = blob_tokens([marc.title, marc.title_main, marc.title_part_name]); ct = blob_tokens([cce.title])
        rows.append({
            "label": 1 if e.verdict == "match" else 0,
            "t": ev_norm(cand, "title.token_set"), "a": ev_norm(cand, "name.author"),
            "c": cand.combined.calibrated, "bf": weighted_jaccard(mb, cb, bidf),
            "bu": unweighted_jaccard(mb, cb), "tbs": tbs(mt, ct),
            "mtitle": str(marc.title)[:42], "ctitle": str(cce.title)[:42],
        })

pos = [r for r in rows if r["label"] == 1]; neg = [r for r in rows if r["label"] == 0]
trap = [r for r in neg if r["a"] > 0.8]
bleed = [r for r in pos if r["t"] < 0.5]
tc = [r for r in trap if r["bf"] >= 0.5]  # the 16 false-confirms
print(f"trap n={len(trap)}  bleed n={len(bleed)}  trap-false-confirms(bf>=.5) n={len(tc)}")
print(f"\ntitle_bag_score (tbs) p10/25/50/75/90:")
print(f"  trap ALL            {fmt(q5([r['tbs'] for r in trap]))}")
print(f"  trap false-confirms {fmt(q5([r['tbs'] for r in tc]))}   <-- must be LOW for a title gate to cap them")
print(f"  field-bleed         {fmt(q5([r['tbs'] for r in bleed]))}   <-- must be HIGH so the gate spares them")
print(f"\nfraction with tbs < t (would the gate even touch them?):")
for t in (0.1, 0.2, 0.3):
    tct = sum(1 for r in tc if r['tbs'] < t)
    blt = sum(1 for r in bleed if r['tbs'] < t)
    print(f"  t={t}: trap-false-confirms below t {tct}/{len(tc)} | field-bleed below t {blt}/{len(bleed)} (these get capped/hurt)")
print(f"\n16 same-author false-confirms (bf/tbs/marc-title | cce-title):")
for r in sorted(tc, key=lambda r: r['tbs']):
    print(f"  bf={r['bf']:.2f} tbs={r['tbs']:.2f}  {r['mtitle']!r:44} | {r['ctitle']!r}")
