"""Publisher scoring: joined-blob (current) vs best-of-split, over the vault."""
from pathlib import Path

from msgspec.structs import replace

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

def publisher_ev(cand):
    for ev in cand.evidence:
        if ev.scorer == "name.publisher":
            return ev.normalized
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
entries = [e for e in raw.values() if e.verdict in ("match", "no_match") and e.match_source != "renewal"]
marc_by_id = build_marc_index(Path("data/candidates"), {e.marc_control_id for e in entries})

joined_scores, split_scores = [], []
lifted_pos = lifted_neg = 0
multi = 0
examples = []
with NyplIndexLookup(_INDEX) as lk:
    idf = build_idf_table(lk)
    aidf = build_author_idf_table(lk)
    pidf = build_publisher_idf_table(lk)
    score = make_pair_scorer(
        matching_config=cfg, pairings=pairings, idf=idf, author_idf=aidf,
        publisher_idf=pidf, calibrator=_load_calibrator(_INDEX.parent), learned_model_dir=None,
    )
    for e in entries:
        marc = marc_by_id.get(e.marc_control_id)
        cce = lk.get_registration(e.nypl_uuid)
        if marc is None or cce is None:
            continue
        label = 1 if e.verdict == "match" else 0
        j = publisher_ev(score(marc, cce))
        s = j
        if len(cce.publisher_names) > 1:
            multi += 1
            s = max(
                (publisher_ev(score(marc, replace(cce, publisher_names=(name,)))) for name in cce.publisher_names),
                default=j,
            )
            s = max(s, j)
            if s > j + 0.05:
                if label == 1:
                    lifted_pos += 1
                    if len(examples) < 5:
                        examples.append((marc.publisher, cce.publisher_names, j, s))
                else:
                    lifted_neg += 1
        joined_scores.append((j, label))
        split_scores.append((s, label))

print(f"pairs scored: {len(joined_scores)} (multi-name CCE: {multi})")
print(f"lifted >0.05: positives {lifted_pos}, negatives {lifted_neg}")
print(f"publisher-evidence AUC joined: {roc_auc(joined_scores):.4f}")
print(f"publisher-evidence AUC split:  {roc_auc(split_scores):.4f}")
print("\nexamples (marc_pub | cce_names | joined -> split):")
for m, names, j, s in examples:
    print(f"  {str(m)[:40]!r} | {list(names)} | {j:.2f} -> {s:.2f}")
